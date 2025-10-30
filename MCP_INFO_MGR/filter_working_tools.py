#!/usr/bin/env python3
"""
Filter Working Tools from Probe Results

This script:
1. Reads tool_descriptions.ndjson (all 5000 tools)
2. Reads tool_probe_result.json (test results)
3. Filters out broken tools (connection errors, timeouts, real failures)
4. Keeps working tools (including those with validation errors)
5. Categorizes removed tools by failure reason
6. Creates new indexed/tool_descriptions_working.ndjson

Categories for removed tools:
- connection_failed: Server unreachable
- timeout: Tool took too long
- execution_failed: Real tool bugs (excluding validation errors)
- validation_error: Argument validation (these stay in WORKING set)

Usage:
    python filter_working_tools.py [--probe PROBE_RESULTS.json]
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def is_argument_validation_error(error_msg, error_traceback):
    """Check if error is argument validation (means tool is working)."""
    if not error_msg:
        error_msg = ""
    if not error_traceback:
        error_traceback = ""

    error_text = (str(error_msg) + " " + str(error_traceback)).lower()

    validation_patterns = [
        r"validation error",
        r"invalid.*argument",
        r"invalid.*parameter",
        r"required.*missing",
        r"missing required",
        r"must be.*type",
        r"expected.*got",
        r"invalid.*value",
        r"out of range",
        r"must be one of",
        r"does not match.*pattern",
        r"schema.*validation",
        r"additional properties",
        r"enum",
        r"invalid.*format",
        r"cannot be empty",
        r"malformed.*input",
        r"mcp error -32602",
    ]

    for pattern in validation_patterns:
        if re.search(pattern, error_text, re.IGNORECASE):
            return True
    return False


def categorize_tool_status(result):
    """
    Categorize tool test result.

    Returns: (is_working, category)
    - is_working: True if tool should be kept
    - category: failure reason if not working
    """
    status = result.get('status')
    error = result.get('error')
    traceback = result.get('error_traceback')

    # Success = working
    if status == 'success':
        return (True, 'success')

    # Argument validation = working (tool validates correctly)
    if status == 'execution_failed' and is_argument_validation_error(error, traceback):
        return (True, 'working_validates_args')

    # Connection failures = broken
    if status == 'connection_failed':
        return (False, 'connection_failed')

    # Timeouts = broken
    if status == 'timeout':
        return (False, 'timeout')

    # Execution failures (non-validation) = broken
    if status == 'execution_failed':
        return (False, 'execution_failed')

    # Critical errors = broken
    if status == 'critical_error':
        return (False, 'critical_error')

    # Unknown = broken (to be safe)
    return (False, 'unknown')


def filter_tools(tool_descriptions_path, probe_results_path):
    """Filter tools based on probe results."""

    # Load tool descriptions
    print(f"Loading tool descriptions from: {tool_descriptions_path}")
    with open(tool_descriptions_path, 'r') as f:
        tool_descriptions = []
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get('status') == 'ok':
                    tool_descriptions.append(entry)

    print(f"Loaded {len(tool_descriptions)} servers with tools")

    # Load probe results
    print(f"Loading probe results from: {probe_results_path}")
    with open(probe_results_path, 'r') as f:
        probe_data = json.load(f)
        probe_results = probe_data.get('results', [])

    print(f"Loaded {len(probe_results)} test results")

    # Index probe results by (server, tool)
    probe_index = {}
    for result in probe_results:
        key = (result.get('server'), result.get('tool'))
        probe_index[key] = result

    # Filter tools
    working_servers = []
    failed_servers = []
    removed_tools_by_category = defaultdict(list)

    stats = {
        'total_servers': 0,
        'total_tools_before': 0,
        'total_tools_after': 0,
        'servers_fully_working': 0,
        'servers_partially_working': 0,
        'servers_fully_broken': 0,
        'tools_by_status': defaultdict(int)
    }

    for server_entry in tool_descriptions:
        server_name = server_entry.get('qualifiedName')
        original_tools = server_entry.get('tools', [])

        stats['total_servers'] += 1
        stats['total_tools_before'] += len(original_tools)

        working_tools = []
        failed_tools = []
        removed_tools = []

        for tool in original_tools:
            tool_name = tool.get('name')
            key = (server_name, tool_name)

            # Check if we have probe results for this tool
            if key not in probe_index:
                # No probe result = assume working (better safe than sorry)
                working_tools.append(tool)
                stats['tools_by_status']['no_probe_data'] += 1
                continue

            probe_result = probe_index[key]
            is_working, category = categorize_tool_status(probe_result)

            if is_working:
                working_tools.append(tool)
                stats['tools_by_status'][category] += 1
            else:
                # Create failed tool entry with reason
                failed_tool_info = tool.copy()
                failed_tool_info['failure_reason'] = category
                failed_tool_info['error_message'] = probe_result.get('error', '')
                failed_tool_info['status'] = probe_result.get('status')

                # Add traceback if not too long
                error_trace = probe_result.get('error_traceback', '')
                if error_trace and len(error_trace) < 2000:
                    failed_tool_info['error_traceback'] = error_trace

                failed_tools.append(failed_tool_info)

                removed_tools.append({
                    'server': server_name,
                    'tool': tool,
                    'category': category,
                    'error': probe_result.get('error'),
                    'original_status': probe_result.get('status')
                })
                stats['tools_by_status'][category] += 1
                removed_tools_by_category[category].append({
                    'server': server_name,
                    'tool': tool_name,
                    'description': tool.get('description', ''),
                    'error': probe_result.get('error')
                })

        # Create filtered server entry for working tools
        if working_tools:
            filtered_entry = server_entry.copy()
            filtered_entry['tools'] = working_tools
            filtered_entry['toolCount'] = len(working_tools)
            filtered_entry['original_toolCount'] = len(original_tools)
            filtered_entry['removed_toolCount'] = len(removed_tools)
            filtered_entry['filtered_timestamp'] = datetime.now().isoformat()

            working_servers.append(filtered_entry)
            stats['total_tools_after'] += len(working_tools)

            if len(working_tools) == len(original_tools):
                stats['servers_fully_working'] += 1
            else:
                stats['servers_partially_working'] += 1
        else:
            stats['servers_fully_broken'] += 1

        # Create failed server entry if there are failed tools
        if failed_tools:
            failed_entry = server_entry.copy()
            failed_entry['tools'] = failed_tools
            failed_entry['toolCount'] = len(failed_tools)
            failed_entry['working_toolCount'] = len(working_tools)
            failed_entry['total_toolCount'] = len(original_tools)
            failed_entry['status'] = 'has_failures'
            failed_entry['filtered_timestamp'] = datetime.now().isoformat()

            failed_servers.append(failed_entry)

    return {
        'working_servers': working_servers,
        'failed_servers': failed_servers,
        'removed_tools_by_category': dict(removed_tools_by_category),
        'stats': dict(stats)
    }


def save_results(results, output_dir):
    """Save filtered results to files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save working tools to tool_descriptions.ndjson
    working_path = output_dir / 'tool_descriptions.ndjson'
    print(f"\nSaving working tools to: {working_path}")
    with open(working_path, 'w') as f:
        for entry in results['working_servers']:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved {len(results['working_servers'])} servers with working tools")

    # Save failed tools to tool_descriptions_failed.ndjson
    failed_path = output_dir / 'tool_descriptions_failed.ndjson'
    print(f"\nSaving failed tools to: {failed_path}")
    with open(failed_path, 'w') as f:
        for entry in results['failed_servers']:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved {len(results['failed_servers'])} servers with failed tools")

    # Save removed tools by category
    removed_dir = output_dir / 'removed_tools'
    removed_dir.mkdir(exist_ok=True)

    for category, tools in results['removed_tools_by_category'].items():
        category_path = removed_dir / f'{category}.json'
        print(f"Saving {len(tools)} tools removed for '{category}' to: {category_path}")
        with open(category_path, 'w') as f:
            json.dump({
                'category': category,
                'count': len(tools),
                'tools': tools
            }, f, indent=2, ensure_ascii=False)

    # Save summary statistics
    stats_path = output_dir / 'filtering_stats.json'
    print(f"\nSaving statistics to: {stats_path}")

    stats = results['stats']
    summary = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_servers_before': stats['total_servers'],
            'servers_fully_working': stats['servers_fully_working'],
            'servers_partially_working': stats['servers_partially_working'],
            'servers_fully_broken': stats['servers_fully_broken'],
            'total_tools_before': stats['total_tools_before'],
            'total_tools_after': stats['total_tools_after'],
            'tools_removed': stats['total_tools_before'] - stats['total_tools_after'],
            'removal_rate': (stats['total_tools_before'] - stats['total_tools_after']) / stats['total_tools_before'] * 100
        },
        'tools_by_status': dict(stats['tools_by_status']),
        'removed_by_category': {cat: len(tools) for cat, tools in results['removed_tools_by_category'].items()}
    }

    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def print_summary(summary):
    """Print human-readable summary."""

    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)

    s = summary['summary']

    print(f"\n📊 SERVERS:")
    print(f"  Total: {s['total_servers_before']}")
    print(f"  ✅ Fully working: {s['servers_fully_working']} ({s['servers_fully_working']/s['total_servers_before']*100:.1f}%)")
    print(f"  ⚠️  Partially working: {s['servers_partially_working']} ({s['servers_partially_working']/s['total_servers_before']*100:.1f}%)")
    print(f"  ❌ Fully broken: {s['servers_fully_broken']} ({s['servers_fully_broken']/s['total_servers_before']*100:.1f}%)")

    print(f"\n📊 TOOLS:")
    print(f"  Before filtering: {s['total_tools_before']}")
    print(f"  After filtering: {s['total_tools_after']}")
    print(f"  Removed: {s['tools_removed']} ({s['removal_rate']:.1f}%)")

    print(f"\n📊 TOOLS BY STATUS:")
    for status, count in sorted(summary['tools_by_status'].items(), key=lambda x: x[1], reverse=True):
        pct = count / s['total_tools_before'] * 100
        emoji = "✅" if status in ['success', 'working_validates_args'] else "❌"
        print(f"  {emoji} {status}: {count} ({pct:.1f}%)")

    print(f"\n📊 REMOVED TOOLS BY CATEGORY:")
    for category, count in sorted(summary['removed_by_category'].items(), key=lambda x: x[1], reverse=True):
        pct = count / s['tools_removed'] * 100 if s['tools_removed'] > 0 else 0
        print(f"  • {category}: {count} ({pct:.1f}%)")

    print(f"\n💡 NEXT STEPS:")
    print(f"  1. Review removed tools in: mcp_data/indexed/removed_tools/")
    print(f"  2. Rebuild FAISS index with working tools:")
    print(f"     cd MCP_INFO_MGR/semantic_search")
    print(f"     python build_search_index.py --input ../mcp_data/indexed/tool_descriptions_working.ndjson")


def main():
    parser = argparse.ArgumentParser(
        description="Filter working tools based on probe results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--tool-descriptions",
        default="mcp_data/indexed/tool_descriptions.ndjson",
        help="Input tool descriptions NDJSON (default: mcp_data/indexed/tool_descriptions.ndjson)"
    )
    parser.add_argument(
        "--probe",
        default="tool_probe_result.json",
        help="Probe results JSON (default: tool_probe_result.json)"
    )
    parser.add_argument(
        "--output-dir",
        default="mcp_data/indexed",
        help="Output directory (default: mcp_data/indexed)"
    )

    args = parser.parse_args()

    # Check input files exist
    for path, name in [(args.tool_descriptions, "Tool descriptions"), (args.probe, "Probe results")]:
        if not Path(path).exists():
            print(f"Error: {name} not found: {path}")
            return 1

    print("="*80)
    print("FILTERING WORKING TOOLS")
    print("="*80)

    # Filter tools
    results = filter_tools(args.tool_descriptions, args.probe)

    # Save results
    summary = save_results(results, args.output_dir)

    # Print summary
    print_summary(summary)

    return 0


if __name__ == "__main__":
    exit(main())
