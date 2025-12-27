"""
Check reachability of remote Smithery MCP endpoints using OAuth authentication.

Usage:
    python MCP_INFO_MGR/check_servers_oauth.py                    # Check all servers
    python MCP_INFO_MGR/check_servers_oauth.py --failed-only      # Only check previously failed servers
    python MCP_INFO_MGR/check_servers_oauth.py --limit 10         # Check first 10 servers

This script reads remote_servers.json and probes each server via OAuth.
On first run a browser opens for Smithery OAuth; tokens are cached under ~/.mcp/smithery_tokens/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "Orchestrator"

# Add paths for imports
for path in (ORCHESTRATOR_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcpuniverse.mcp.oauth import create_smithery_auth


async def probe_server(qualified_name: str, url: str, timeout: int = 30) -> dict:
    """Probe a single server using OAuth authentication."""
    result = {
        "qualifiedName": qualified_name,
        "url": url,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "unknown",
        "toolCount": None,
        "error": None,
    }

    try:
        auth_provider, callback_handler = create_smithery_auth(
            server_url=url,
            client_name="MCP Server Checker",
        )

        async with callback_handler:
            async with streamablehttp_client(url=url, auth=auth_provider) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=timeout)
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=timeout)
                    tools = tools_result.tools

                    result["status"] = "ok"
                    result["toolCount"] = len(tools)

    except asyncio.TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Timed out after {timeout}s"
    except asyncio.CancelledError:
        result["status"] = "cancelled"
        result["error"] = "Operation cancelled"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


async def main():
    parser = argparse.ArgumentParser(description="Check MCP server reachability via OAuth")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per server in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of servers to check")
    parser.add_argument("--failed-only", action="store_true", help="Only check previously failed servers")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests")
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "mcp_data/working/remote_servers.json",
        help="Input JSON file with server names"
    )
    parser.add_argument(
        "--tool-desc",
        type=Path,
        default=SCRIPT_DIR / "mcp_data/working/tool_descriptions.ndjson",
        help="Tool descriptions NDJSON (for --failed-only mode)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "mcp_data/working/check_results.ndjson",
        help="Output NDJSON file"
    )
    args = parser.parse_args()

    # Load server list
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    with args.input.open() as f:
        all_servers = json.load(f)

    print(f"Loaded {len(all_servers)} servers from {args.input}")

    # Filter to failed only if requested
    if args.failed_only:
        if not args.tool_desc.exists():
            print(f"Error: Tool descriptions file not found: {args.tool_desc}", file=sys.stderr)
            return 1

        # Load previously successful servers
        ok_servers = set()
        with args.tool_desc.open() as f:
            for line in f:
                data = json.loads(line)
                if data.get("status") == "ok":
                    ok_servers.add(data["qualifiedName"])

        # Filter to only failed/missing servers
        servers_to_check = [s for s in all_servers if s not in ok_servers]
        print(f"Filtering to {len(servers_to_check)} failed/missing servers")
    else:
        servers_to_check = all_servers

    # Apply limit
    if args.limit:
        servers_to_check = servers_to_check[:args.limit]
        print(f"Limited to {len(servers_to_check)} servers")

    if not servers_to_check:
        print("No servers to check!")
        return 0

    # Check servers
    print(f"\nChecking {len(servers_to_check)} servers...")
    print(f"Output will be written to {args.output}")
    print("=" * 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    results = []

    with args.output.open("w", encoding="utf-8") as out_f:
        for i, server_name in enumerate(servers_to_check, 1):
            url = f"https://server.smithery.ai/{server_name}"

            print(f"[{i}/{len(servers_to_check)}] {server_name}...", end=" ", flush=True)

            result = await probe_server(server_name, url, timeout=args.timeout)
            results.append(result)

            # Write immediately
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            if result["status"] == "ok":
                print(f"✓ {result['toolCount']} tools")
                success_count += 1
            else:
                print(f"✗ {result['status']}: {result['error'][:50] if result['error'] else ''}")
                error_count += 1

            # Sleep between requests
            if i < len(servers_to_check):
                await asyncio.sleep(args.sleep)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total checked: {len(servers_to_check)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Output: {args.output}")

    # Show failed servers
    if error_count > 0:
        print(f"\nFailed servers ({error_count}):")
        for r in results:
            if r["status"] != "ok":
                print(f"  - {r['qualifiedName']}: [{r['status']}] {r['error'][:60] if r['error'] else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
