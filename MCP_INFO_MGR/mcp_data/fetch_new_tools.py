"""
Fetch tool descriptions for new servers using Smithery OAuth.

Usage:
    python MCP_INFO_MGR/mcp_data/fetch_new_tools.py          # Fetch all servers
    python MCP_INFO_MGR/mcp_data/fetch_new_tools.py --retry  # Retry only failed servers

This script reads working/new_remote_servers.json (simple JSON array of server names)
and fetches tool descriptions for each server via Smithery OAuth.

With --retry flag, it will read the output file and retry only servers that failed.

On first run, a browser will open for Smithery OAuth authentication.
Tokens are cached under ~/.mcp/smithery_tokens/.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "Orchestrator"

# Add paths for imports
for path in (ORCHESTRATOR_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcpuniverse.mcp.oauth import create_smithery_auth


async def fetch_tools_for_server(
    qualified_name: str,
    url: str,
    timeout: int = 30,
) -> dict:
    """Fetch tool descriptions using OAuth authentication."""
    result = {
        "qualifiedName": qualified_name,
        "url": url,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "unknown",
        "tools": [],
        "error": None,
    }

    try:
        print(f"Fetching tools for {qualified_name} (OAuth)...", flush=True)

        auth_provider, callback_handler = create_smithery_auth(
            server_url=url,
            client_name="MCP Tool Fetcher",
        )

        async with callback_handler:
            async with streamablehttp_client(url=url, auth=auth_provider) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=timeout)

                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=timeout)
                    tools = tools_result.tools

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

    return result


def load_servers_from_json(filepath: Path) -> list[dict]:
    """Load servers from simple JSON array of server names."""
    with filepath.open('r', encoding='utf-8') as f:
        server_names = json.load(f)
    # Convert to list of dicts with qualifiedName and url
    return [
        {
            "qualifiedName": name,
            "url": f"https://server.smithery.ai/{name}"
        }
        for name in server_names
    ]


def load_servers_from_ndjson(filepath: Path) -> list[dict]:
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
    parser = argparse.ArgumentParser(description="Fetch tool descriptions for MCP servers via OAuth")
    parser.add_argument("--retry", action="store_true", help="Retry only failed servers from previous run")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds (default: 30)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Define file paths
    input_file = script_dir / "working" / "new_remote_servers.json"
    output_file = script_dir / "working" / "new_tool_descriptions.ndjson"

    print("Using Smithery OAuth authentication")

    # Retry mode: load failed servers from output file
    if args.retry:
        if not output_file.exists():
            print(f"Error: Output file not found: {output_file}", file=sys.stderr)
            print("Please run without --retry first.", file=sys.stderr)
            return 1

        print(f"Loading previous results from {output_file}...")
        previous_results = load_servers_from_ndjson(output_file)
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
            return 1

        # Load servers from simple JSON array
        print(f"Loading servers from {input_file}...")
        servers = load_servers_from_json(input_file)
        print(f"Loaded {len(servers)} servers")

        if len(servers) == 0:
            print("No servers to process.")
            return 0

        previous_results_map = None

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

            # Fetch tools with OAuth
            result = await fetch_tools_for_server(qualified_name, url, timeout=args.timeout)
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

                # Fetch tools with OAuth
                result = await fetch_tools_for_server(qualified_name, url, timeout=args.timeout)

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
