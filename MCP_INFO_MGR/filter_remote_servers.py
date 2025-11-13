"""
Filter Smithery servers to keep only remote MCP servers.

This script reads the raw smithery_servers.ndjson and filters for servers
where "remote": true, which indicates they are accessible via SSE endpoints
rather than local stdio execution.

Usage:
    python MCP_INFO_MGR/filter_remote_servers.py \
        --input mcp_data/raw/smithery_servers.ndjson \
        --output mcp_data/raw/smithery_remote_servers.ndjson
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Smithery servers to keep only remote MCPs"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("mcp_data/raw/smithery_servers.ndjson"),
        help="Input NDJSON file with all Smithery servers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mcp_data/raw/smithery_remote_servers.ndjson"),
        help="Output NDJSON file with only remote servers",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed statistics about filtering",
    )
    return parser.parse_args()


def filter_remote_servers(
    input_path: Path,
    output_path: Path,
    show_stats: bool = False
) -> tuple[int, int]:
    """
    Filter servers to keep only remote MCPs.

    Returns:
        (total_count, remote_count) tuple
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    total_count = 0
    remote_count = 0
    local_count = 0
    missing_field_count = 0

    # Statistics for detailed output
    remote_servers = []
    local_servers = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse JSON: {e}")
                continue

            total_count += 1

            # Check if "remote" field exists
            if "remote" not in record:
                missing_field_count += 1
                if show_stats:
                    print(f"⚠️  Server missing 'remote' field: {record.get('qualifiedName', 'unknown')}")
                continue

            # Filter for remote servers
            if record.get("remote") is True:
                remote_count += 1
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

                if show_stats:
                    remote_servers.append(record.get("qualifiedName", "unknown"))
            else:
                local_count += 1
                if show_stats:
                    local_servers.append(record.get("qualifiedName", "unknown"))

    # Print statistics
    print(f"\n{'='*70}")
    print("FILTERING RESULTS")
    print(f"{'='*70}")
    print(f"Total servers processed: {total_count}")
    print(f"Remote servers (saved):  {remote_count} ({remote_count/total_count*100:.1f}%)")
    print(f"Local servers (filtered out): {local_count} ({local_count/total_count*100:.1f}%)")
    if missing_field_count > 0:
        print(f"Servers missing 'remote' field: {missing_field_count}")
    print(f"\nOutput saved to: {output_path}")
    print(f"{'='*70}\n")

    # Detailed statistics if requested
    if show_stats:
        print("\n--- REMOTE SERVERS (sample) ---")
        for server in remote_servers[:10]:
            print(f"  ✓ {server}")
        if len(remote_servers) > 10:
            print(f"  ... and {len(remote_servers) - 10} more")

        print("\n--- LOCAL SERVERS (sample) ---")
        for server in local_servers[:10]:
            print(f"  ✗ {server}")
        if len(local_servers) > 10:
            print(f"  ... and {len(local_servers) - 10} more")
        print()

    return total_count, remote_count


def main() -> int:
    args = parse_args()

    try:
        total, remote = filter_remote_servers(
            args.input,
            args.output,
            show_stats=args.stats
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
