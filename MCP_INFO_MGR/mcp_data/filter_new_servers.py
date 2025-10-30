"""
Filter out duplicate servers and keep only newly discovered ones.

Usage:
    python mcp_data/filter_new_servers.py

This script compares the newly discovered reachable servers against the existing
pool and extracts only the new ones that aren't already in the pool.

Deduplication is done by normalized URL to ensure we're comparing actual endpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Set
from urllib.parse import urlparse, urlunparse, parse_qsl


def normalize_url(url: str) -> str:
    """
    Normalize a URL for comparison by:
    - Converting to lowercase
    - Removing query parameters (api_key, etc.)
    - Removing trailing slashes
    - Normalizing scheme to https
    """
    if not url:
        return ""

    parsed = urlparse(url.lower())

    # Remove query parameters (they're auth-related, not part of the endpoint identity)
    # Keep the path and domain
    normalized = urlunparse((
        'https',  # Normalize all to https
        parsed.netloc,
        parsed.path.rstrip('/'),  # Remove trailing slash
        '',  # params
        '',  # query (removed)
        ''   # fragment
    ))

    return normalized


def load_server_urls(filepath: Path) -> Set[str]:
    """Load normalized server URLs from an NDJSON file."""
    server_urls = set()
    if not filepath.exists():
        print(f"Warning: {filepath} not found", file=sys.stderr)
        return server_urls

    with filepath.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                url = data.get('url')
                if url:
                    normalized = normalize_url(url)
                    if normalized:
                        server_urls.add(normalized)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)

    return server_urls


def filter_new_servers(
    existing_file: Path,
    new_file: Path,
    output_file: Path,
    status_filter: str = "ok"
) -> tuple[int, int, int]:
    """
    Filter new servers that aren't in the existing pool.

    Args:
        existing_file: Path to existing servers NDJSON
        new_file: Path to newly discovered servers NDJSON
        output_file: Path to output file for new servers
        status_filter: Only include servers with this status (default: "ok")

    Returns:
        Tuple of (new_count, duplicate_count, error_count)
    """
    # Load existing server URLs (normalized)
    print(f"Loading existing servers from {existing_file}...")
    existing_servers = load_server_urls(existing_file)
    print(f"  Loaded {len(existing_servers)} existing server URLs")

    # Process new servers
    print(f"\nProcessing new servers from {new_file}...")
    new_count = 0
    duplicate_count = 0
    error_count = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with new_file.open('r', encoding='utf-8') as infile:
        with output_file.open('w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    status = data.get('status')
                    url = data.get('url')

                    # Count errors for statistics
                    if status != status_filter:
                        error_count += 1
                        continue

                    # Normalize URL for comparison
                    normalized_url = normalize_url(url) if url else None
                    if not normalized_url:
                        continue

                    # Check if this is a new server (by URL)
                    if normalized_url not in existing_servers:
                        outfile.write(line + '\n')
                        new_count += 1
                    else:
                        duplicate_count += 1

                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)

    return new_count, duplicate_count, error_count


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent

    # Define file paths
    existing_file = script_dir / "usable" / "reachability_ok_servers.ndjson"
    new_file = script_dir / "filtered_reachable.ndjson"
    output_file = script_dir / "usable" / "new_reachable_servers.ndjson"

    print("=" * 60)
    print("Filtering New Servers")
    print("=" * 60)
    print(f"Existing pool: {existing_file}")
    print(f"New results:   {new_file}")
    print(f"Output:        {output_file}")
    print("=" * 60)

    # Check if required files exist
    if not new_file.exists():
        print(f"\nError: New results file not found: {new_file}", file=sys.stderr)
        print("Please run check_remote_reachability.py first.", file=sys.stderr)
        return 1

    if not existing_file.exists():
        print(f"\nWarning: Existing pool file not found: {existing_file}")
        print("Will treat all servers as new.")

    # Filter servers
    new_count, duplicate_count, error_count = filter_new_servers(
        existing_file,
        new_file,
        output_file,
        status_filter="ok"
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  New servers found:      {new_count}")
    print(f"  Duplicate servers:      {duplicate_count}")
    print(f"  Servers with errors:    {error_count}")
    print(f"  Total processed:        {new_count + duplicate_count + error_count}")
    print("=" * 60)
    print(f"\nNew servers saved to: {output_file}")

    if new_count > 0:
        print(f"\n✓ Successfully found {new_count} new reachable servers!")
    else:
        print(f"\n⚠ No new servers found. All {duplicate_count} reachable servers already exist in the pool.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
