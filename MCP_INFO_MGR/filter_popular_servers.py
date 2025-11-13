#!/usr/bin/env python3
"""
Filter servers by popularity (useCount).

This script filters servers based on usage statistics to prioritize
testing and indexing the most commonly used servers.

Usage:
    # Get top 100 most popular servers
    python filter_popular_servers.py --top 100

    # Get servers with at least 1000 uses
    python filter_popular_servers.py --min-uses 1000

    # Get verified servers only
    python filter_popular_servers.py --verified-only
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter servers by popularity/usage"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("mcp_data/raw/smithery_remote_servers.ndjson"),
        help="Input NDJSON file with servers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mcp_data/raw/smithery_popular_servers.ndjson"),
        help="Output NDJSON file for filtered servers",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Keep only top N most popular servers",
    )
    parser.add_argument(
        "--min-uses",
        type=int,
        help="Minimum useCount required",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Keep only verified servers",
    )
    return parser.parse_args()


def load_servers(input_path: Path) -> List[Dict]:
    """Load servers and deduplicate."""
    servers = []
    seen = set()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            server = json.loads(line)
            name = server.get("qualifiedName")

            # Skip duplicates
            if name in seen:
                continue

            seen.add(name)
            servers.append(server)

    return servers


def filter_servers(servers: List[Dict], args: argparse.Namespace) -> List[Dict]:
    """Filter servers based on criteria."""
    filtered = servers[:]

    # Filter by verification
    if args.verified_only:
        filtered = [s for s in filtered if s.get("verified", False)]

    # Filter by minimum uses
    if args.min_uses is not None:
        filtered = [s for s in filtered if s.get("useCount", 0) >= args.min_uses]

    # Sort by useCount
    filtered.sort(key=lambda s: s.get("useCount", 0), reverse=True)

    # Keep top N
    if args.top is not None:
        filtered = filtered[:args.top]

    return filtered


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Load and deduplicate
    print(f"Loading servers from {args.input}...")
    servers = load_servers(args.input)
    print(f"Loaded {len(servers)} unique servers")

    # Filter
    filtered = filter_servers(servers, args)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for server in filtered:
            f.write(json.dumps(server, ensure_ascii=False) + "\n")

    # Stats
    print("\n" + "=" * 70)
    print("FILTERING RESULTS")
    print("=" * 70)
    print(f"Input servers:           {len(servers)}")
    print(f"Filtered servers:        {len(filtered)}")

    if args.verified_only:
        print(f"Filter: Verified only")
    if args.min_uses:
        print(f"Filter: Minimum {args.min_uses:,} uses")
    if args.top:
        print(f"Filter: Top {args.top}")

    if filtered:
        total_usage = sum(s.get("useCount", 0) for s in filtered)
        avg_usage = total_usage / len(filtered)
        verified_count = sum(1 for s in filtered if s.get("verified", False))

        print(f"\nFiltered set statistics:")
        print(f"  Total use count:       {total_usage:,}")
        print(f"  Average use count:     {avg_usage:,.1f}")
        print(f"  Verified servers:      {verified_count} ({verified_count/len(filtered)*100:.1f}%)")

        print(f"\nTop 10 in filtered set:")
        for i, server in enumerate(filtered[:10], 1):
            verified = "✓" if server.get("verified") else " "
            use_count = server.get("useCount", 0)
            print(f"  {i:2}. {use_count:>8,} {verified} {server.get('qualifiedName')}")

    print(f"\nOutput: {args.output}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
