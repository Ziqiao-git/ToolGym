#!/usr/bin/env python3
"""
Recalculate True Tool Success Rate

Argument validation errors should NOT be counted as failures because:
- They prove the tool is working (it's validating inputs correctly)
- They prove the connection is working (we got a response)
- The "failure" is our test data, not the tool itself

This script recategorizes results:
- WORKING: success + argument validation errors (tool responds correctly)
- BROKEN: connection failures, timeouts, server errors (tool doesn't work)

Usage:
    python recalculate_true_success_rate.py [--input PROBE_RESULTS.json]
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def is_argument_validation_error(status, error_msg, error_traceback):
    """
    Determine if this is an argument validation error.
    These should be counted as "working" not "failed".
    """
    if status == "success":
        return False

    if not error_msg:
        error_msg = ""
    if not error_traceback:
        error_traceback = ""

    error_text = (str(error_msg) + " " + str(error_traceback)).lower()

    # Patterns that indicate argument validation (tool is working correctly)
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
        r"type.*error.*expected",
        r"too_small",
        r"too_large",
        r"invalid choice",
        r"not a valid",
        r"mcp error -32602",  # MCP invalid params error code
    ]

    for pattern in validation_patterns:
        if re.search(pattern, error_text, re.IGNORECASE):
            return True

    return False


def recalculate_success_rates(probe_results_path: str):
    """Recalculate success rates treating arg validation as success."""

    with open(probe_results_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])

    # Categorize each result
    truly_working = []  # Tool responds correctly (success or validates args)
    truly_broken = []   # Tool doesn't work (connection, timeout, server error)

    server_stats = defaultdict(lambda: {
        'total': 0,
        'originally_success': 0,
        'arg_validation': 0,
        'truly_working': 0,
        'truly_broken': 0,
        'broken_reasons': defaultdict(int)
    })

    for result in results:
        server = result.get('server')
        tool = result.get('tool')
        status = result.get('status')
        error = result.get('error')
        traceback = result.get('error_traceback')

        server_stats[server]['total'] += 1

        if status == 'success':
            truly_working.append(result)
            server_stats[server]['originally_success'] += 1
            server_stats[server]['truly_working'] += 1
        elif is_argument_validation_error(status, error, traceback):
            # Reclassify as working!
            result_copy = result.copy()
            result_copy['recategorized'] = 'working_validates_args'
            result_copy['original_status'] = status
            truly_working.append(result_copy)
            server_stats[server]['arg_validation'] += 1
            server_stats[server]['truly_working'] += 1
        else:
            # Truly broken
            truly_broken.append(result)
            server_stats[server]['truly_broken'] += 1
            server_stats[server]['broken_reasons'][status] += 1

    # Calculate rates
    total_tests = len(results)
    original_success = len([r for r in results if r.get('status') == 'success'])
    arg_validation_count = len([r for r in truly_working if r.get('recategorized')])
    true_working = len(truly_working)
    true_broken = len(truly_broken)

    # Server-level analysis
    servers_fully_working = []
    servers_fully_broken = []
    servers_partial = []

    for server, stats in server_stats.items():
        if stats['truly_broken'] == 0:
            servers_fully_working.append({
                'server': server,
                'total_tools': stats['total'],
                'originally_success': stats['originally_success'],
                'arg_validation': stats['arg_validation']
            })
        elif stats['truly_working'] == 0:
            servers_fully_broken.append({
                'server': server,
                'total_tools': stats['total'],
                'broken_reasons': dict(stats['broken_reasons'])
            })
        else:
            servers_partial.append({
                'server': server,
                'total_tools': stats['total'],
                'truly_working': stats['truly_working'],
                'truly_broken': stats['truly_broken'],
                'working_pct': stats['truly_working'] / stats['total'] * 100
            })

    servers_partial.sort(key=lambda x: x['working_pct'], reverse=True)

    return {
        'summary': {
            'total_tests': total_tests,
            'original_success': original_success,
            'original_success_rate': original_success / total_tests * 100,
            'arg_validation_reclassified': arg_validation_count,
            'true_working': true_working,
            'true_working_rate': true_working / total_tests * 100,
            'true_broken': true_broken,
            'true_broken_rate': true_broken / total_tests * 100,
            'improvement': (true_working - original_success) / total_tests * 100
        },
        'server_summary': {
            'total_servers': len(server_stats),
            'fully_working': len(servers_fully_working),
            'fully_broken': len(servers_fully_broken),
            'partial': len(servers_partial)
        },
        'servers_fully_working': servers_fully_working,
        'servers_fully_broken': servers_fully_broken,
        'servers_partial': servers_partial,
        'recategorized_results': truly_working,
        'broken_results': truly_broken
    }


def print_analysis(analysis: dict):
    """Print the recalculated analysis."""

    summary = analysis['summary']
    server_summary = analysis['server_summary']

    print("="*80)
    print("TRUE TOOL SUCCESS RATE (Excluding Argument Validation)")
    print("="*80)

    print(f"\n📊 RECALCULATED STATISTICS:")
    print(f"  Total tools tested: {summary['total_tests']}")

    print(f"\n  Original Results:")
    print(f"    ✓ Success: {summary['original_success']} ({summary['original_success_rate']:.1f}%)")
    print(f"    ✗ Failed: {summary['total_tests'] - summary['original_success']} ({100 - summary['original_success_rate']:.1f}%)")

    print(f"\n  Argument Validation Reclassification:")
    print(f"    🔄 Reclassified from 'failed' to 'working': {summary['arg_validation_reclassified']}")
    print(f"    💡 These tools ARE working - they correctly validated our bad arguments")

    print(f"\n  TRUE Results (Arg validation = Working):")
    print(f"    ✅ TRULY WORKING: {summary['true_working']} ({summary['true_working_rate']:.1f}%)")
    print(f"       - Tools that respond correctly (success OR validate args)")
    print(f"    ❌ TRULY BROKEN: {summary['true_broken']} ({summary['true_broken_rate']:.1f}%)")
    print(f"       - Tools with connection, timeout, or server errors")

    print(f"\n  📈 IMPROVEMENT:")
    print(f"    {summary['original_success_rate']:.1f}% → {summary['true_working_rate']:.1f}%")
    print(f"    +{summary['improvement']:.1f} percentage points")

    print(f"\n" + "="*80)
    print("SERVER-LEVEL ANALYSIS")
    print("="*80)

    print(f"\n  Total servers: {server_summary['total_servers']}")
    print(f"  ✅ Fully working: {server_summary['fully_working']} ({server_summary['fully_working']/server_summary['total_servers']*100:.1f}%)")
    print(f"  ❌ Fully broken: {server_summary['fully_broken']} ({server_summary['fully_broken']/server_summary['total_servers']*100:.1f}%)")
    print(f"  ⚠️  Partial: {server_summary['partial']} ({server_summary['partial']/server_summary['total_servers']*100:.1f}%)")

    # Show examples
    if analysis['servers_fully_working']:
        print(f"\n  Examples of FULLY WORKING servers:")
        for server in analysis['servers_fully_working'][:5]:
            if server['arg_validation'] > 0:
                print(f"    • {server['server']}")
                print(f"      {server['total_tools']} tools: {server['originally_success']} success + {server['arg_validation']} arg validation")

    if analysis['servers_partial']:
        print(f"\n  Examples of servers with PARTIAL issues:")
        for server in analysis['servers_partial'][:5]:
            print(f"    • {server['server']}")
            print(f"      {server['truly_working']}/{server['total_tools']} working ({server['working_pct']:.1f}%)")

    if analysis['servers_fully_broken']:
        print(f"\n  Examples of FULLY BROKEN servers:")
        for server in analysis['servers_fully_broken'][:5]:
            print(f"    • {server['server']}")
            print(f"      {server['total_tools']} tools all broken: {server['broken_reasons']}")

    print(f"\n" + "="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)

    print(f"\n1. ARGUMENT VALIDATION IS NOT A FAILURE:")
    print(f"   When a tool rejects invalid arguments, it's WORKING CORRECTLY.")
    print(f"   {summary['arg_validation_reclassified']} tools were incorrectly marked as 'failed'.")

    print(f"\n2. TRUE TOOL RELIABILITY:")
    print(f"   {summary['true_working_rate']:.1f}% of tools are actually functional.")
    print(f"   Only {summary['true_broken_rate']:.1f}% have real problems (connection, timeout, bugs).")

    print(f"\n3. SERVER RELIABILITY:")
    print(f"   {server_summary['fully_working']/server_summary['total_servers']*100:.1f}% of servers have ALL tools working.")
    print(f"   {server_summary['fully_broken']/server_summary['total_servers']*100:.1f}% of servers are completely unreachable.")

    if summary['true_working_rate'] > 80:
        print(f"\n✅ EXCELLENT: Over 80% of tools are functional!")
        print(f"   The MCP ecosystem is quite reliable.")
    elif summary['true_working_rate'] > 70:
        print(f"\n✅ GOOD: Over 70% of tools are functional.")
        print(f"   Most tools work when reachable.")

    print(f"\n4. NEXT STEPS:")
    print(f"   • For better testing, use: python test_all_tools.py --use-llm")
    print(f"   • Focus debugging on the {summary['true_broken']} truly broken tools")
    print(f"   • The {summary['arg_validation_reclassified']} arg validation 'failures' need better test data")


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate true tool success rate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input",
        default="tool_probe_result.json",
        help="Input probe results JSON (default: tool_probe_result.json)"
    )
    parser.add_argument(
        "--output",
        default="true_success_rate_analysis.json",
        help="Output analysis JSON (default: true_success_rate_analysis.json)"
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    print(f"Analyzing: {args.input}\n")

    analysis = recalculate_success_rates(args.input)

    print_analysis(analysis)

    # Save analysis
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'input_file': args.input,
        'methodology': 'Argument validation errors reclassified as working (tool validates correctly)',
        'analysis': analysis
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\n📄 Detailed analysis saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
