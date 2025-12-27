#!/usr/bin/env python3
"""
Deduplicate and merge remote server lists.

Usage:
    # Merge new_remote_servers.json into remote_servers.json (deduplicated)
    python MCP_INFO_MGR/dedupe_servers.py

    # Preview only (don't write)
    python MCP_INFO_MGR/dedupe_servers.py --dry-run

    # Custom paths
    python MCP_INFO_MGR/dedupe_servers.py \
        --existing path/to/remote_servers.json \
        --new path/to/new_remote_servers.json \
        --output path/to/merged.json
"""

import json
import argparse
from pathlib import Path


def load_json_list(path: Path) -> list:
    """Load a JSON file containing a list of server names."""
    if not path.exists():
        print(f"  ⚠️ File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def dedupe_and_merge(existing: list, new: list) -> tuple[list, dict]:
    """
    Merge two lists, removing duplicates.
    Returns (merged_list, stats_dict).
    """
    existing_set = set(existing)
    new_set = set(new)

    # Find duplicates (in new that already exist)
    duplicates = new_set & existing_set

    # Find truly new entries
    truly_new = new_set - existing_set

    # Merge: existing + truly new (sorted)
    merged = sorted(existing_set | new_set)

    stats = {
        "existing_count": len(existing),
        "new_count": len(new),
        "duplicates_found": len(duplicates),
        "truly_new": len(truly_new),
        "merged_count": len(merged),
    }

    return merged, stats


def main():
    parser = argparse.ArgumentParser(description="Deduplicate and merge server lists")
    parser.add_argument(
        "--existing",
        type=Path,
        default=Path("MCP_INFO_MGR/mcp_data/working/remote_servers.json"),
        help="Path to existing server list"
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=Path("MCP_INFO_MGR/mcp_data/working/new_remote_servers.json"),
        help="Path to new server list to merge"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite existing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only, don't write"
    )
    parser.add_argument(
        "--show-new",
        action="store_true",
        help="Print the truly new servers"
    )
    parser.add_argument(
        "--show-duplicates",
        action="store_true",
        help="Print the duplicate servers"
    )

    args = parser.parse_args()

    # Resolve paths
    existing_path = args.existing
    new_path = args.new
    output_path = args.output or existing_path

    print(f"📂 Existing: {existing_path}")
    print(f"📂 New:      {new_path}")
    print(f"📂 Output:   {output_path}")
    print()

    # Load
    existing = load_json_list(existing_path)
    new = load_json_list(new_path)

    if not existing and not new:
        print("❌ Both files are empty or missing")
        return

    # Merge
    merged, stats = dedupe_and_merge(existing, new)

    # Print stats
    print("📊 Statistics:")
    print(f"   Existing servers:    {stats['existing_count']}")
    print(f"   New servers:         {stats['new_count']}")
    print(f"   Duplicates found:    {stats['duplicates_found']}")
    print(f"   Truly new:           {stats['truly_new']}")
    print(f"   Merged total:        {stats['merged_count']}")
    print()

    # Show details if requested
    if args.show_new:
        truly_new = sorted(set(new) - set(existing))
        if truly_new:
            print(f"🆕 Truly new servers ({len(truly_new)}):")
            for s in truly_new[:20]:
                print(f"   + {s}")
            if len(truly_new) > 20:
                print(f"   ... and {len(truly_new) - 20} more")
            print()

    if args.show_duplicates:
        duplicates = sorted(set(new) & set(existing))
        if duplicates:
            print(f"🔄 Duplicate servers ({len(duplicates)}):")
            for s in duplicates[:20]:
                print(f"   = {s}")
            if len(duplicates) > 20:
                print(f"   ... and {len(duplicates) - 20} more")
            print()

    # Write
    if args.dry_run:
        print("🔍 Dry run - no changes written")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"✅ Wrote {len(merged)} servers to {output_path}")


if __name__ == "__main__":
    main()
