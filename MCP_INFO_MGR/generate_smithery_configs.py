"""
Generate server configs for Smithery remote MCP servers.

This script creates MCP-Universe configuration entries for Smithery remote servers
using the standard Smithery URL pattern: https://server.smithery.ai/{qualifiedName}/mcp

Usage:
    python MCP_INFO_MGR/generate_smithery_configs.py \
        --input mcp_data/raw/smithery_remote_servers.ndjson \
        --output MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MCP configs for Smithery remote servers"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("mcp_data/raw/smithery_remote_servers.ndjson"),
        help="Input NDJSON file with remote servers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json"),
        help="Output JSON file for server configs",
    )
    parser.add_argument(
        "--api-key-variable",
        type=str,
        default="SMITHERY_API_KEY",
        help="Environment variable name for API key (default: SMITHERY_API_KEY)",
    )
    parser.add_argument(
        "--profile-variable",
        type=str,
        default="SMITHERY_PROFILE",
        help="Environment variable name for profile (default: SMITHERY_PROFILE)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    configs: Dict[str, dict] = {}
    processed = 0

    # Read all remote servers
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                server = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

            qualified_name = server.get("qualifiedName")
            if not qualified_name:
                print(f"Warning: Server missing qualifiedName: {server}")
                continue

            # Smithery remote servers follow standard URL pattern
            url = f"https://server.smithery.ai/{qualified_name}/mcp?api_key={{{{{args.api_key_variable}}}}}"

            # Add profile parameter if profile variable is specified
            if args.profile_variable:
                url += f"&profile={{{{{args.profile_variable}}}}}"

            configs[qualified_name] = {
                "streamable_http": {
                    "url": url,
                    "headers": {}
                },
                "env": {}
            }

            processed += 1

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("SERVER CONFIG GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Processed servers: {processed}")
    print(f"Output file: {args.output}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
