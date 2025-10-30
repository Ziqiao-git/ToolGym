"""
Fetch tool descriptions for new reachable servers.

Usage:
    python mcp_data/fetch_new_tools.py          # Fetch all servers
    python mcp_data/fetch_new_tools.py --retry  # Retry only failed servers

This script reads the new_reachable_servers.ndjson file and fetches tool descriptions
for each server, saving the results to new_tool_descriptions.ndjson.

With --retry flag, it will read the output file and retry only servers that failed.

Requirements:
    * `SMITHERY_API_KEY` exported in your environment.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "Orchestrator"

# Add paths for imports
for path in (ORCHESTRATOR_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import load_dotenv
from mcpuniverse.mcp.manager import MCPManager


def add_api_key_to_url(url: str, api_key: str) -> str:
    """Add API key to URL query parameters."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query['api_key'] = api_key
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


async def fetch_tools_for_server(
    manager: MCPManager,
    qualified_name: str,
    url: str,
    api_key: str,
    timeout: int = 15,
) -> dict:
    """Fetch tool descriptions for a single server."""
    result = {
        "qualifiedName": qualified_name,
        "url": url,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "unknown",
        "tools": [],
        "error": None,
    }

    client = None
    try:
        # Build server config for this URL
        auth_url = add_api_key_to_url(url, api_key)

        server_config = {
            "streamable_http": {
                "url": auth_url,
                "headers": {}
            }
        }

        # Add server config if not already present
        if qualified_name not in manager.list_server_names():
            manager.add_server_config(qualified_name, server_config)

        # Build client and fetch tools with timeout
        print(f"Fetching tools for {qualified_name}...", flush=True)

        # Wrap in timeout to avoid hanging connections
        client = await asyncio.wait_for(
            manager.build_client(qualified_name, transport="streamable_http", timeout=timeout),
            timeout=timeout + 5
        )

        # Fetch tools with timeout
        tools = await asyncio.wait_for(
            client.list_tools(),
            timeout=timeout
        )

        result["status"] = "ok"
        result["toolCount"] = len(tools)
        result["tools"] = [
            {
                "name": tool.name,
                "description": tool.description if hasattr(tool, "description") else None,
                "inputSchema": tool.inputSchema if hasattr(tool, "inputSchema") else None,
            }
            for tool in tools
        ]
        print(f"  ✓ {qualified_name}: {len(tools)} tools", flush=True)

    except asyncio.TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Operation timed out after {timeout}s"
        print(f"  ✗ {qualified_name}: timeout", flush=True)
    except asyncio.CancelledError:
        result["status"] = "cancelled"
        result["error"] = "Operation cancelled"
        print(f"  ✗ {qualified_name}: cancelled", flush=True)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        print(f"  ✗ {qualified_name}: {exc}", flush=True)
    finally:
        # Clean up client in the same task context
        if client is not None:
            try:
                await asyncio.wait_for(client.cleanup(), timeout=5)
            except asyncio.TimeoutError:
                print(f"  ⚠ {qualified_name}: cleanup timed out", flush=True)
            except Exception as cleanup_exc:
                # Suppress cleanup errors - they're often caused by already-closed connections
                print(f"  ⚠ {qualified_name}: cleanup error (suppressed): {cleanup_exc}", flush=True)

    return result


def load_servers(filepath: Path) -> list[dict]:
    """Load servers from NDJSON file."""
    servers = []
    with filepath.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                servers.append(json.loads(line))
    return servers


async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fetch tool descriptions for MCP servers")
    parser.add_argument("--retry", action="store_true", help="Retry only failed servers from previous run")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout in seconds (default: 15, retry: 30)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Define file paths
    input_file = script_dir / "usable" / "new_reachable_servers.ndjson"
    output_file = script_dir / "raw" / "new_tool_descriptions.ndjson"

    # Load environment variables
    load_dotenv(str(ORCHESTRATOR_DIR / ".env"))

    # Check for API key
    api_key = os.getenv("SMITHERY_API_KEY")
    if not api_key:
        print("Error: SMITHERY_API_KEY not found in environment", file=sys.stderr)
        return 1

    # Retry mode: load failed servers from output file
    if args.retry:
        if not output_file.exists():
            print(f"Error: Output file not found: {output_file}", file=sys.stderr)
            print("Please run without --retry first.", file=sys.stderr)
            return 1

        print(f"Loading previous results from {output_file}...")
        previous_results = load_servers(output_file)
        print(f"Loaded {len(previous_results)} previous results")

        # Filter for failed servers
        servers = [
            {"qualifiedName": r["qualifiedName"], "url": r["url"]}
            for r in previous_results
            if r.get("status") != "ok"
        ]

        if len(servers) == 0:
            print("\n✓ No failed servers found. All servers were successful!")
            return 0

        print(f"\nFound {len(servers)} failed servers to retry:")
        for server in servers:
            print(f"  - {server['qualifiedName']}")

        # Use longer timeout for retry if not specified
        if args.timeout == 15:
            args.timeout = 30

        print(f"\nRetrying with timeout={args.timeout}s...")

        # Create a mapping of previous results for later merging
        previous_results_map = {r["qualifiedName"]: r for r in previous_results}

    else:
        # Normal mode: load from input file
        # Check input file
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            print("Please run filter_new_servers.py first.", file=sys.stderr)
            return 1

        # Load servers
        print(f"Loading servers from {input_file}...")
        servers = load_servers(input_file)
        print(f"Loaded {len(servers)} servers")

        if len(servers) == 0:
            print("No servers to process.")
            return 0

        previous_results_map = None

    # Initialize MCP manager
    manager = MCPManager()

    # Process servers and write results incrementally
    print(f"\nFetching tool descriptions...")
    print(f"Output will be written to {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    retry_results = []

    # In retry mode, collect results; in normal mode, write directly
    if args.retry:
        # Retry mode: collect new results
        for i, server in enumerate(servers, 1):
            qualified_name = server.get("qualifiedName", "<unknown>")
            url = server.get("url")

            if not url:
                print(f"  ⚠ {qualified_name}: no URL found, skipping")
                continue

            # Fetch tools with custom timeout
            result = await fetch_tools_for_server(manager, qualified_name, url, api_key, timeout=args.timeout)
            retry_results.append(result)

            if result["status"] == "ok":
                success_count += 1
            else:
                error_count += 1

            # Small delay between requests
            await asyncio.sleep(1.0)

        # Merge retry results with previous results
        print(f"\nMerging retry results with previous results...")
        retry_map = {r["qualifiedName"]: r for r in retry_results}

        # Write merged results back to file
        with output_file.open("w", encoding="utf-8") as out_f:
            for name, prev_result in previous_results_map.items():
                if name in retry_map:
                    # Use retry result
                    out_f.write(json.dumps(retry_map[name]) + "\n")
                else:
                    # Keep previous result
                    out_f.write(json.dumps(prev_result) + "\n")

    else:
        # Normal mode: write results directly
        with output_file.open("w", encoding="utf-8") as out_f:
            for i, server in enumerate(servers, 1):
                qualified_name = server.get("qualifiedName", "<unknown>")
                url = server.get("url")

                if not url:
                    print(f"  ⚠ {qualified_name}: no URL found, skipping")
                    continue

                # Fetch tools with custom timeout
                result = await fetch_tools_for_server(manager, qualified_name, url, api_key, timeout=args.timeout)

                # Write result immediately
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                if result["status"] == "ok":
                    success_count += 1
                else:
                    error_count += 1

                # Progress update every 10 servers
                if i % 10 == 0:
                    print(f"\nProgress: {i}/{len(servers)} ({success_count} ok, {error_count} errors)")

                # Small delay between requests to avoid overwhelming servers
                await asyncio.sleep(0.5)

    print(f"\n{'='*60}")
    if args.retry:
        print(f"Retry completed!")
        print(f"  Total retried: {len(servers)}")
        print(f"  Successful: {success_count}")
        print(f"  Still failing: {error_count}")
    else:
        print(f"Completed!")
        print(f"  Total processed: {len(servers)}")
        print(f"  Successful: {success_count}")
        print(f"  Errors: {error_count}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")

    # Show remaining failures if any
    if args.retry and error_count > 0:
        print(f"\n⚠ {error_count} server(s) still failing:")
        for r in retry_results:
            if r["status"] != "ok":
                print(f"  - {r['qualifiedName']}: {r['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
