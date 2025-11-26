#!/usr/bin/env python3
"""
Filter out problematic servers from working set and create a clean verified set.

Based on verification results, removes servers that:
1. Have connection issues
2. Have malformed tool descriptions (None/null)
3. Are missing from configs
4. Cause runtime errors
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Servers to remove based on verification issues
PROBLEMATIC_SERVERS = {
    # Connection issues
    "@Kastalien-Research/clear-thought-two",  # Timeout/cancellation errors

    # Tool description issues (NoneType) + returns empty results
    "@amalinakurniasari/figmamcp",  # None description causes AttributeError
    "@coinranking/coinranking-mcp",  # Related to NoneType errors
    "@dravidsajinraj-iex/code-runner-mcp",  # Part of failed query
    "@abdqum/supabase-mcp-selfhosted",  # All 43 tools have no descriptions, returns empty results

    # Note: 'exa' server is fine - the error was from malformed reference 'exa/get_code_context_exa'
}

def main():
    working_dir = PROJECT_ROOT / "MCP_INFO_MGR" / "mcp_data" / "working"

    # Load current working configs
    configs_path = working_dir / "remote_server_configs.json"
    with open(configs_path, "r") as f:
        configs = json.load(f)

    print(f"Original server count: {len(configs)}")
    print(f"\nRemoving {len(PROBLEMATIC_SERVERS)} problematic servers:")
    for server in sorted(PROBLEMATIC_SERVERS):
        if server in configs:
            print(f"  ✓ Removing: {server}")
            del configs[server]
        else:
            print(f"  ⚠ Not found: {server}")

    print(f"\nFiltered server count: {len(configs)}")

    # Save filtered configs
    with open(configs_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"\n✓ Saved filtered configs to: {configs_path}")

    # Also filter tool descriptions
    tool_desc_path = working_dir / "tool_descriptions.ndjson"
    if tool_desc_path.exists():
        filtered_tools = []
        removed_count = 0

        with open(tool_desc_path, "r") as f:
            for line in f:
                data = json.loads(line)
                server_name = data.get("server_name")

                if server_name not in PROBLEMATIC_SERVERS:
                    filtered_tools.append(line)
                else:
                    removed_count += 1

        with open(tool_desc_path, "w") as f:
            f.writelines(filtered_tools)

        print(f"✓ Filtered tool descriptions: removed {removed_count} entries")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Verified working servers: {len(configs)}")
    print(f"Removed problematic servers: {len(PROBLEMATIC_SERVERS)}")
    print(f"\nNext steps:")
    print(f"1. Rebuild semantic index with verified servers")
    print(f"2. Re-run query generation with clean server set")

if __name__ == "__main__":
    main()
