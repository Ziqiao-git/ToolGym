"""
Fetch tool descriptions for all reachable MCP servers with automatic retry.

Usage:
    # Basic usage (with automatic retry)
    python MCP_INFO_MGR/fetch_tool_descriptions.py \
        --input MCP_INFO_MGR/reachability_ok_servers.ndjson \
        --config MCP_INFO_MGR/remote_server_configs.json \
        --output MCP_INFO_MGR/tool_descriptions.ndjson

    # Retry ONLY failed servers (skip reprocessing all servers)
    python MCP_INFO_MGR/fetch_tool_descriptions.py \
        --retry-only \
        --retry-failed 3 \
        --retry-timeout 90

    # Custom retry settings
    python MCP_INFO_MGR/fetch_tool_descriptions.py \
        --input MCP_INFO_MGR/reachability_ok_servers.ndjson \
        --config MCP_INFO_MGR/remote_server_configs.json \
        --output MCP_INFO_MGR/tool_descriptions.ndjson \
        --retry-failed 3 \
        --retry-timeout 90

    # Disable automatic retry
    python MCP_INFO_MGR/fetch_tool_descriptions.py \
        --input MCP_INFO_MGR/reachability_ok_servers.ndjson \
        --config MCP_INFO_MGR/remote_server_configs.json \
        --output MCP_INFO_MGR/tool_descriptions.ndjson \
        --no-auto-retry

Features:
    * Automatic detection of failed servers (error/cancelled status)
    * Retry-only mode to avoid reprocessing all servers
    * Configurable retry attempts with increasing timeouts
    * Updates existing results in-place when retrying
    * Detailed statistics for initial run and retries

Requirements:
    * `SMITHERY_API_KEY` exported in your environment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "Orchestrator"

# Add paths for imports
for path in (ORCHESTRATOR_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import load_dotenv
from mcpuniverse.mcp.manager import MCPManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch tool descriptions for reachable MCP servers."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("MCP_INFO_MGR/reachability_ok_servers.ndjson"),
        help="NDJSON file containing reachable servers",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("MCP_INFO_MGR/remote_server_configs.json"),
        help="JSON file with server configurations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("MCP_INFO_MGR/tool_descriptions.ndjson"),
        help="Output NDJSON file for tool descriptions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of servers to process (for testing)",
    )
    parser.add_argument(
        "--retry-failed",
        type=int,
        default=2,
        help="Number of retry attempts for failed servers (default: 2)",
    )
    parser.add_argument(
        "--retry-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for retry attempts (default: 60)",
    )
    parser.add_argument(
        "--no-auto-retry",
        action="store_true",
        help="Disable automatic retry of failed servers at the end",
    )
    parser.add_argument(
        "--retry-only",
        action="store_true",
        help="Only retry failed servers from existing output file (skip initial processing)",
    )
    return parser.parse_args()


def load_reachable_servers(input_path: Path) -> list[dict]:
    """Load reachable servers from NDJSON file."""
    servers = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                servers.append(json.loads(line))
    return servers


def load_server_configs(config_path: Path) -> dict:
    """Load server configurations from JSON file."""
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_failed_servers(output_path: Path) -> list[str]:
    """Load list of failed server names from existing output file."""
    if not output_path.exists():
        return []

    failed = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                if result.get("status") in ["error", "cancelled"]:
                    failed.append(result["qualifiedName"])
    return failed


def update_result_in_file(output_path: Path, result: dict) -> None:
    """Update or append a result in the output file."""
    if not output_path.exists():
        # File doesn't exist, just write the result
        with output_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        return

    # Read all existing results
    results = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                results[r["qualifiedName"]] = r

    # Update with new result
    results[result["qualifiedName"]] = result

    # Write back all results
    with output_path.open("w", encoding="utf-8") as f:
        for r in results.values():
            f.write(json.dumps(r) + "\n")


async def fetch_tools_for_server(
    manager: MCPManager,
    server_name: str,
    server_config: dict,
    timeout: int = 30,
) -> dict:
    """Fetch tool descriptions for a single server."""
    result = {
        "qualifiedName": server_name,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "unknown",
        "tools": [],
        "error": None,
    }

    try:
        # Add server config if not already present
        if server_name not in manager.list_server_names():
            manager.add_server_config(server_name, server_config)

        # Build client and fetch tools with timeout
        print(f"Fetching tools for {server_name}...", flush=True)
        client = await manager.build_client(server_name, transport="streamable_http")

        try:
            # Add timeout to the list_tools operation
            tools = await asyncio.wait_for(client.list_tools(), timeout=timeout)
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
            print(f"  ✓ {server_name}: {len(tools)} tools", flush=True)
        finally:
            await client.cleanup()

    except asyncio.TimeoutError:
        result["status"] = "error"
        result["error"] = f"Timeout after {timeout}s"
        print(f"  ✗ {server_name}: timeout after {timeout}s", flush=True)
    except asyncio.CancelledError:
        result["status"] = "cancelled"
        result["error"] = "Operation cancelled"
        print(f"  ✗ {server_name}: cancelled", flush=True)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        print(f"  ✗ {server_name}: {exc}", flush=True)

    return result


async def fetch_with_task_isolation(
    manager: MCPManager,
    server_name: str,
    server_config: dict,
    timeout: int = 30,
) -> dict:
    """
    Wrap fetch in its own task to ensure clean asyncio context.

    This prevents the "Attempted to exit cancel scope in a different task" error
    by ensuring each server connection lifecycle happens within a single task context.
    """
    return await asyncio.create_task(
        fetch_tools_for_server(manager, server_name, server_config, timeout)
    )


async def retry_failed_servers(
    manager: MCPManager,
    configs: dict,
    output_path: Path,
    max_retries: int,
    timeout: int,
) -> dict:
    """Retry failed servers from the output file."""
    failed_servers = load_failed_servers(output_path)

    if not failed_servers:
        print("\n✓ No failed servers to retry")
        return {"total": 0, "success": 0, "still_failed": 0}

    print(f"\n{'='*60}")
    print(f"Retrying {len(failed_servers)} failed servers")
    print(f"  Max retries: {max_retries}")
    print(f"  Timeout: {timeout}s")
    print(f"{'='*60}\n")

    retry_stats = {"total": len(failed_servers), "success": 0, "still_failed": 0}

    for server_name in failed_servers:
        server_config = configs.get(server_name)
        if not server_config:
            print(f"  ⚠ {server_name}: config not found, skipping")
            retry_stats["still_failed"] += 1
            continue

        # Try multiple times with increasing timeout
        success = False
        for attempt in range(1, max_retries + 1):
            current_timeout = timeout * attempt  # Increase timeout with each attempt
            print(f"  Retry {attempt}/{max_retries} for {server_name} (timeout: {current_timeout}s)")

            result = await fetch_with_task_isolation(
                manager, server_name, server_config, current_timeout
            )

            if result["status"] == "ok":
                print(f"  ✓ {server_name}: Success on retry {attempt}!")
                update_result_in_file(output_path, result)
                retry_stats["success"] += 1
                success = True
                break
            else:
                print(f"  ✗ {server_name}: Failed retry {attempt} - {result['error']}")

        if not success:
            print(f"  ✗ {server_name}: All retries exhausted")
            retry_stats["still_failed"] += 1

    return retry_stats


async def main():
    """Main entry point."""
    args = parse_args()

    # Load environment variables
    load_dotenv(str(ORCHESTRATOR_DIR / ".env"))

    # Check for API key
    if not os.getenv("SMITHERY_API_KEY"):
        print("Error: SMITHERY_API_KEY not found in environment", file=sys.stderr)
        return 1

    print(f"Loading server configs from {args.config}...")
    configs = load_server_configs(args.config)
    print(f"Loaded {len(configs)} server configurations")

    # Initialize MCP manager
    manager = MCPManager()

    # RETRY-ONLY MODE: Skip initial processing, only retry failed servers
    if args.retry_only:
        if not args.output.exists():
            print(f"Error: Output file not found: {args.output}")
            print("Cannot use --retry-only without existing output file")
            return 1

        print(f"\n{'='*60}")
        print("RETRY-ONLY MODE")
        print(f"Skipping initial processing, only retrying failed servers")
        print(f"{'='*60}\n")

        retry_stats = await retry_failed_servers(
            manager, configs, args.output, args.retry_failed, args.retry_timeout
        )

        print(f"\n{'='*60}")
        print(f"Retry Complete!")
        print(f"  Total failed servers: {retry_stats['total']}")
        print(f"  Successfully recovered: {retry_stats['success']}")
        print(f"  Still failed: {retry_stats['still_failed']}")
        if retry_stats['total'] > 0:
            print(f"  Recovery rate: {retry_stats['success'] / retry_stats['total'] * 100:.1f}%")
        print(f"  Output: {args.output}")
        print(f"{'='*60}")

        return 0

    # NORMAL MODE: Process all servers from input
    # Load input data
    print(f"Loading reachable servers from {args.input}...")
    servers = load_reachable_servers(args.input)

    if args.limit:
        servers = servers[:args.limit]
        print(f"Limited to {args.limit} servers")

    print(f"Loaded {len(servers)} reachable servers")

    # Process servers and write results incrementally
    print(f"\nFetching tool descriptions...")
    print(f"Output will be written to {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out_f:
        success_count = 0
        error_count = 0

        for i, server in enumerate(servers, 1):
            server_name = server["qualifiedName"]

            # Get config for this server
            server_config = configs.get(server_name)
            if not server_config:
                print(f"  ⚠ {server_name}: config not found, skipping")
                continue

            # Fetch tools with task isolation to prevent asyncio context errors
            result = await fetch_with_task_isolation(manager, server_name, server_config)

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

    print(f"\n{'='*60}")
    print(f"Initial Run Completed!")
    print(f"  Total processed: {len(servers)}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")

    # Retry failed servers if enabled
    if not args.no_auto_retry and error_count > 0:
        retry_stats = await retry_failed_servers(
            manager, configs, args.output, args.retry_failed, args.retry_timeout
        )

        print(f"\n{'='*60}")
        print(f"Retry Summary:")
        print(f"  Total failed servers: {retry_stats['total']}")
        print(f"  Successfully recovered: {retry_stats['success']}")
        print(f"  Still failed: {retry_stats['still_failed']}")
        print(f"{'='*60}")

        # Update final stats
        final_success = success_count + retry_stats['success']
        final_errors = retry_stats['still_failed']

        print(f"\n{'='*60}")
        print(f"Final Results:")
        print(f"  Total processed: {len(servers)}")
        print(f"  Successful: {final_success}")
        print(f"  Errors: {final_errors}")
        print(f"  Success rate: {final_success / len(servers) * 100:.1f}%")
        print(f"  Output: {args.output}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
