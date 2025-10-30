#!/usr/bin/env python3
"""
Analyze MCP Server Failure Patterns

This script analyzes tool probe results to determine if server failures are:
1. All-or-nothing (if one tool fails, all tools in that server fail)
2. Partial failures (some tools work, some don't)

Usage:
    python analyze_server_failure_patterns.py [--input PROBE_RESULTS.json] [--output REPORT.json]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def analyze_failure_patterns(probe_results_path: str):
    """
    Analyze failure patterns in probe results.

    Args:
        probe_results_path: Path to tool_probe_result.json

    Returns:
        Dictionary with analysis results
    """
    # Load probe results
    with open(probe_results_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])

    # Group results by server
    servers = defaultdict(lambda: {
        'tools': [],
        'statuses': defaultdict(int),
        'total': 0,
        'success': 0,
        'failed': 0
    })

    for result in results:
        server_name = result.get('server')
        tool_name = result.get('tool')
        status = result.get('status')

        servers[server_name]['tools'].append({
            'name': tool_name,
            'status': status,
            'error': result.get('error')
        })
        servers[server_name]['statuses'][status] += 1
        servers[server_name]['total'] += 1

        if status == 'success':
            servers[server_name]['success'] += 1
        else:
            servers[server_name]['failed'] += 1

    # Classify servers by failure pattern
    all_success = []  # All tools succeeded
    all_failed = []   # All tools failed (all-or-nothing failure)
    partial_failure = []  # Some tools work, some don't

    for server_name, server_data in servers.items():
        if server_data['success'] == server_data['total']:
            all_success.append({
                'server': server_name,
                'total_tools': server_data['total']
            })
        elif server_data['failed'] == server_data['total']:
            # Analyze why all failed - same reason or different?
            failure_reasons = defaultdict(int)
            for tool in server_data['tools']:
                if tool['status'] != 'success':
                    failure_reasons[tool['status']] += 1

            all_failed.append({
                'server': server_name,
                'total_tools': server_data['total'],
                'failure_types': dict(server_data['statuses']),
                'same_failure': len(failure_reasons) == 1
            })
        else:
            # Partial failure - some tools work, some don't
            partial_failure.append({
                'server': server_name,
                'total_tools': server_data['total'],
                'success': server_data['success'],
                'failed': server_data['failed'],
                'success_rate': server_data['success'] / server_data['total'] * 100,
                'failure_types': dict(server_data['statuses']),
                'failed_tools': [t['name'] for t in server_data['tools'] if t['status'] != 'success']
            })

    # Sort partial failures by success rate
    partial_failure.sort(key=lambda x: x['success_rate'], reverse=True)

    return {
        'summary': {
            'total_servers': len(servers),
            'all_success_servers': len(all_success),
            'all_failed_servers': len(all_failed),
            'partial_failure_servers': len(partial_failure),
            'total_tools_tested': len(results)
        },
        'all_success': all_success,
        'all_failed': all_failed,
        'partial_failure': partial_failure,
        'detailed_stats': {
            'all_or_nothing_failures': len([s for s in all_failed if s['same_failure']]),
            'mixed_failure_types': len([s for s in all_failed if not s['same_failure']]),
            'servers_with_partial_success': len(partial_failure)
        }
    }


def print_analysis(analysis: dict):
    """Print human-readable analysis."""
    summary = analysis['summary']
    stats = analysis['detailed_stats']

    print("="*80)
    print("MCP SERVER FAILURE PATTERN ANALYSIS")
    print("="*80)

    print(f"\n📊 SUMMARY:")
    print(f"  Total servers tested: {summary['total_servers']}")
    print(f"  Total tools tested: {summary['total_tools_tested']}")

    print(f"\n✅ All tools successful: {summary['all_success_servers']} servers")
    print(f"   ({summary['all_success_servers']/summary['total_servers']*100:.1f}%)")

    print(f"\n❌ All tools failed: {summary['all_failed_servers']} servers")
    print(f"   ({summary['all_failed_servers']/summary['total_servers']*100:.1f}%)")

    print(f"\n⚠️  Partial failures: {summary['partial_failure_servers']} servers")
    print(f"   ({summary['partial_failure_servers']/summary['total_servers']*100:.1f}%)")

    print(f"\n" + "="*80)
    print("FAILURE PATTERN ANALYSIS")
    print("="*80)

    print(f"\n🔴 All-or-Nothing Failures:")
    print(f"   {stats['all_or_nothing_failures']} servers where ALL tools failed with the SAME error")
    print(f"   (This suggests server-level issues, not tool-specific problems)")

    print(f"\n🟠 Mixed Failure Types:")
    print(f"   {stats['mixed_failure_types']} servers where all tools failed but with DIFFERENT errors")
    print(f"   (Less common - might indicate multiple issues)")

    print(f"\n🟡 Partial Failures:")
    print(f"   {stats['servers_with_partial_success']} servers where SOME tools work and SOME don't")
    print(f"   (This suggests tool-specific issues, not server-level problems)")

    # Show examples of each pattern
    if analysis['all_failed']:
        print(f"\n" + "="*80)
        print("EXAMPLES: All-or-Nothing Failures")
        print("="*80)
        for server in analysis['all_failed'][:5]:
            print(f"\n  Server: {server['server']}")
            print(f"  Tools: {server['total_tools']} (all failed)")
            print(f"  Failure types: {server['failure_types']}")
            if server['same_failure']:
                print(f"  ⚠️  ALL tools failed with the SAME error type (all-or-nothing)")

    if analysis['partial_failure']:
        print(f"\n" + "="*80)
        print("EXAMPLES: Partial Failures (Tools Can Fail Independently)")
        print("="*80)
        for server in analysis['partial_failure'][:5]:
            print(f"\n  Server: {server['server']}")
            print(f"  Tools: {server['total_tools']} total")
            print(f"  Success: {server['success']} ({server['success_rate']:.1f}%)")
            print(f"  Failed: {server['failed']}")
            print(f"  Failure types: {server['failure_types']}")
            if len(server['failed_tools']) <= 5:
                print(f"  Failed tools: {', '.join(server['failed_tools'])}")
            else:
                print(f"  Failed tools: {', '.join(server['failed_tools'][:3])}, ... ({len(server['failed_tools'])} total)")

    # Key findings
    print(f"\n" + "="*80)
    print("🔍 KEY FINDINGS")
    print("="*80)

    partial_pct = summary['partial_failure_servers'] / summary['total_servers'] * 100
    all_failed_pct = summary['all_failed_servers'] / summary['total_servers'] * 100

    if partial_pct > 10:
        print(f"\n✓ TOOLS CAN FAIL INDEPENDENTLY:")
        print(f"  {partial_pct:.1f}% of servers have partial failures.")
        print(f"  This means individual tools can fail while other tools in the same server work.")
        print(f"  Conclusion: Tool failures are often tool-specific, not server-wide.")

    if all_failed_pct > 10:
        print(f"\n✓ ALL-OR-NOTHING FAILURES EXIST:")
        print(f"  {all_failed_pct:.1f}% of servers have all tools failing.")
        all_or_nothing_pct = stats['all_or_nothing_failures'] / summary['all_failed_servers'] * 100 if summary['all_failed_servers'] > 0 else 0
        print(f"  Of these, {all_or_nothing_pct:.1f}% failed with the same error (true all-or-nothing).")
        print(f"  Conclusion: Server connection failures affect all tools equally.")

    if summary['all_success_servers'] > summary['total_servers'] * 0.5:
        print(f"\n✓ MAJORITY OF SERVERS FULLY FUNCTIONAL:")
        print(f"  {summary['all_success_servers']/summary['total_servers']*100:.1f}% of servers have all tools working.")
        print(f"  Conclusion: Most servers are reliable when they work.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MCP server failure patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input",
        default="tool_probe_result.json",
        help="Input probe results JSON file (default: tool_probe_result.json)"
    )
    parser.add_argument(
        "--output",
        default="failure_pattern_analysis.json",
        help="Output analysis JSON file (default: failure_pattern_analysis.json)"
    )
    parser.add_argument(
        "--show-all-partial",
        action="store_true",
        help="Show all servers with partial failures (not just top 5)"
    )

    args = parser.parse_args()

    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    print(f"Analyzing probe results from: {args.input}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    # Run analysis
    analysis = analyze_failure_patterns(args.input)

    # Print results
    print_analysis(analysis)

    # Show all partial failures if requested
    if args.show_all_partial and analysis['partial_failure']:
        print(f"\n" + "="*80)
        print(f"ALL SERVERS WITH PARTIAL FAILURES ({len(analysis['partial_failure'])})")
        print("="*80)
        for i, server in enumerate(analysis['partial_failure'], 1):
            print(f"\n{i}. {server['server']}")
            print(f"   Success: {server['success']}/{server['total_tools']} ({server['success_rate']:.1f}%)")
            print(f"   Failed tools: {', '.join(server['failed_tools'][:10])}")
            if len(server['failed_tools']) > 10:
                print(f"   ... and {len(server['failed_tools']) - 10} more")

    # Save detailed analysis
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'input_file': args.input,
        'analysis': analysis
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\n📄 Detailed analysis saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
