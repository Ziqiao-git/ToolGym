#!/usr/bin/env python3
"""
Analyze Failure Reasons in MCP Tool Tests

This script categorizes test failures by their root cause:
1. Connection/Server issues (connection_failed, timeout)
2. Invalid arguments (execution_failed with argument-related errors)
3. Tool bugs/implementation issues
4. Authentication/permission issues

Usage:
    python analyze_failure_reasons.py [--input PROBE_RESULTS.json]
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime


def categorize_error(status, error_msg, error_traceback):
    """
    Categorize the error into root cause categories.

    Returns:
        (category, subcategory, confidence)
    """
    if not error_msg:
        error_msg = ""
    if not error_traceback:
        error_traceback = ""

    error_text = (str(error_msg) + " " + str(error_traceback)).lower()

    # Connection/Infrastructure issues
    if status == "connection_failed":
        return ("infrastructure", "connection_failed", "high")
    if status == "timeout":
        return ("infrastructure", "timeout", "high")
    if "401" in error_text or "unauthorized" in error_text:
        return ("infrastructure", "auth_error", "high")
    if "404" in error_text or "not found" in error_text:
        return ("infrastructure", "not_found", "high")
    if "connection" in error_text and "refused" in error_text:
        return ("infrastructure", "connection_refused", "high")

    # Argument validation errors (likely our fault)
    if status == "execution_failed":
        # Check for common argument validation patterns
        arg_patterns = [
            r"required.*missing",
            r"missing required",
            r"invalid.*argument",
            r"invalid.*parameter",
            r"validation.*failed",
            r"schema.*validation",
            r"type.*error.*expected",
            r"must be.*type",
            r"expected.*got",
            r"invalid.*value",
            r"out of range",
            r"must be one of",
            r"does not match",
            r"pattern.*failed",
            r"additional properties",
            r"enum",
            r"invalid.*format",
            r"cannot be empty",
            r"malformed",
            r"invalid.*input",
        ]

        for pattern in arg_patterns:
            if re.search(pattern, error_text, re.IGNORECASE):
                return ("invalid_arguments", "validation_error", "high")

        # Check for type errors
        if "typeerror" in error_text or "type error" in error_text:
            return ("invalid_arguments", "type_error", "medium")

        # Check for missing required fields
        if "required" in error_text and ("field" in error_text or "property" in error_text):
            return ("invalid_arguments", "missing_required", "high")

        # Check for authentication/permission in tool execution
        if "permission" in error_text or "forbidden" in error_text or "403" in error_text:
            return ("infrastructure", "permission_denied", "high")

        # Check for rate limiting
        if "rate limit" in error_text or "too many requests" in error_text or "429" in error_text:
            return ("infrastructure", "rate_limited", "high")

        # Check for tool implementation errors
        if "internal server error" in error_text or "500" in error_text:
            return ("tool_implementation", "server_error", "medium")

        if "not implemented" in error_text or "notimplementederror" in error_text:
            return ("tool_implementation", "not_implemented", "high")

        if "attributeerror" in error_text or "nameerror" in error_text:
            return ("tool_implementation", "code_error", "medium")

        # Generic execution failure - could be our arguments or tool bug
        return ("unknown", "execution_failed", "low")

    # Critical errors from test harness
    if status == "critical_error":
        return ("test_harness", "critical_error", "high")

    return ("unknown", status, "low")


def analyze_failure_reasons(probe_results_path: str):
    """Analyze and categorize all failures."""

    with open(probe_results_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])

    # Categorize all failures
    failures_by_category = defaultdict(list)
    failures_by_server = defaultdict(lambda: defaultdict(list))
    error_message_samples = defaultdict(list)

    total_tests = len(results)
    success_count = 0

    for result in results:
        status = result.get('status')
        server = result.get('server')
        tool = result.get('tool')
        error = result.get('error')
        traceback = result.get('error_traceback')

        if status == 'success':
            success_count += 1
            continue

        # Categorize this failure
        category, subcategory, confidence = categorize_error(status, error, traceback)

        failure_info = {
            'server': server,
            'tool': tool,
            'status': status,
            'error': error,
            'subcategory': subcategory,
            'confidence': confidence
        }

        failures_by_category[category].append(failure_info)
        failures_by_server[server][category].append(failure_info)

        # Store error message samples (first 5 per subcategory)
        key = f"{category}:{subcategory}"
        if len(error_message_samples[key]) < 5:
            error_message_samples[key].append({
                'server': server,
                'tool': tool,
                'error': str(error)[:200] if error else "No error message"
            })

    # Count subcategories
    subcategory_counts = Counter()
    for category, failures in failures_by_category.items():
        for failure in failures:
            subcategory_counts[f"{category}:{failure['subcategory']}"] += 1

    # Analyze servers with invalid_arguments failures
    servers_with_arg_issues = {}
    for server, categories in failures_by_server.items():
        if 'invalid_arguments' in categories:
            total_tools = sum(len(f) for f in categories.values())
            arg_failures = len(categories['invalid_arguments'])
            servers_with_arg_issues[server] = {
                'total_failed': total_tools,
                'arg_failures': arg_failures,
                'arg_failure_pct': arg_failures / total_tools * 100 if total_tools > 0 else 0,
                'other_failures': {cat: len(f) for cat, f in categories.items() if cat != 'invalid_arguments'}
            }

    return {
        'summary': {
            'total_tests': total_tests,
            'success': success_count,
            'success_rate': success_count / total_tests * 100,
            'failed': total_tests - success_count,
            'failure_rate': (total_tests - success_count) / total_tests * 100
        },
        'failures_by_category': {
            cat: len(failures) for cat, failures in failures_by_category.items()
        },
        'subcategory_counts': dict(subcategory_counts.most_common()),
        'error_samples': error_message_samples,
        'servers_with_arg_issues': servers_with_arg_issues,
        'detailed_failures': failures_by_category
    }


def print_analysis(analysis: dict):
    """Print human-readable analysis."""

    summary = analysis['summary']

    print("="*80)
    print("FAILURE REASON ANALYSIS")
    print("="*80)

    print(f"\n📊 OVERALL SUMMARY:")
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Success: {summary['success']} ({summary['success_rate']:.1f}%)")
    print(f"  Failed: {summary['failed']} ({summary['failure_rate']:.1f}%)")

    print(f"\n" + "="*80)
    print("FAILURE BREAKDOWN BY ROOT CAUSE")
    print("="*80)

    failures_by_cat = analysis['failures_by_category']
    total_failures = sum(failures_by_cat.values())

    # Sort by count
    sorted_categories = sorted(failures_by_cat.items(), key=lambda x: x[1], reverse=True)

    for category, count in sorted_categories:
        pct = count / total_failures * 100 if total_failures > 0 else 0
        print(f"\n📌 {category.upper()}: {count} failures ({pct:.1f}%)")

        # Show subcategories
        subcats = {k: v for k, v in analysis['subcategory_counts'].items() if k.startswith(f"{category}:")}
        for subcat_key, subcat_count in sorted(subcats.items(), key=lambda x: x[1], reverse=True):
            _, subcat_name = subcat_key.split(':', 1)
            sub_pct = subcat_count / count * 100 if count > 0 else 0
            print(f"  ├─ {subcat_name}: {subcat_count} ({sub_pct:.1f}%)")

    # Detailed analysis of invalid_arguments
    if 'invalid_arguments' in failures_by_cat:
        print(f"\n" + "="*80)
        print("🔍 INVALID ARGUMENTS ANALYSIS")
        print("="*80)

        arg_failures = failures_by_cat['invalid_arguments']
        print(f"\nTotal argument-related failures: {arg_failures}")
        print(f"Percentage of all failures: {arg_failures/total_failures*100:.1f}%")
        print(f"Percentage of all tests: {arg_failures/summary['total_tests']*100:.1f}%")

        print(f"\n⚠️  These failures are likely due to SIMPLE ARGUMENT GENERATION")
        print(f"   The test script used generic arguments like:")
        print(f"   - String parameters: 'test query', 'https://example.com'")
        print(f"   - Number parameters: 1")
        print(f"   - Boolean parameters: false")
        print(f"\n   These may not satisfy tool-specific requirements:")
        print(f"   - Enum values (must be from specific set)")
        print(f"   - Format requirements (email, URL patterns)")
        print(f"   - Range constraints (min/max values)")
        print(f"   - Complex object structures")

        # Show examples
        print(f"\n📋 Example Argument Validation Errors:")
        for key, samples in list(analysis['error_samples'].items())[:3]:
            if key.startswith('invalid_arguments:'):
                _, subcat = key.split(':', 1)
                print(f"\n  {subcat}:")
                for i, sample in enumerate(samples[:2], 1):
                    print(f"    {i}. {sample['server']} → {sample['tool']}")
                    print(f"       Error: {sample['error'][:150]}")
                    if len(sample['error']) > 150:
                        print(f"              ...")

    # Infrastructure issues
    if 'infrastructure' in failures_by_cat:
        print(f"\n" + "="*80)
        print("🌐 INFRASTRUCTURE ISSUES")
        print("="*80)

        infra_failures = failures_by_cat['infrastructure']
        print(f"\nTotal infrastructure failures: {infra_failures}")
        print(f"Percentage of all failures: {infra_failures/total_failures*100:.1f}%")

        print(f"\n⚠️  These are NOT due to argument generation:")
        print(f"   - Server connection issues")
        print(f"   - Authentication/authorization problems")
        print(f"   - Network timeouts")
        print(f"   - Rate limiting")

        # Show examples
        print(f"\n📋 Example Infrastructure Errors:")
        infra_keys = [k for k in analysis['error_samples'].keys() if k.startswith('infrastructure:')]
        for key in infra_keys[:3]:
            _, subcat = key.split(':', 1)
            samples = analysis['error_samples'][key]
            print(f"\n  {subcat} ({len(samples)} samples):")
            for i, sample in enumerate(samples[:2], 1):
                print(f"    {i}. {sample['server']} → {sample['tool']}")

    # Tool implementation issues
    if 'tool_implementation' in failures_by_cat:
        print(f"\n" + "="*80)
        print("🐛 TOOL IMPLEMENTATION ISSUES")
        print("="*80)

        impl_failures = failures_by_cat['tool_implementation']
        print(f"\nTotal tool bugs: {impl_failures}")
        print(f"Percentage of all failures: {impl_failures/total_failures*100:.1f}%")

        print(f"\n⚠️  These are bugs in the MCP tool implementations:")
        print(f"   - Server-side errors")
        print(f"   - Not implemented features")
        print(f"   - Code errors in tool logic")

    # Key conclusions
    print(f"\n" + "="*80)
    print("💡 KEY CONCLUSIONS")
    print("="*80)

    if 'invalid_arguments' in failures_by_cat:
        arg_pct = failures_by_cat['invalid_arguments'] / total_failures * 100
        if arg_pct > 30:
            print(f"\n✓ ARGUMENT GENERATION IS A MAJOR ISSUE:")
            print(f"  {arg_pct:.1f}% of failures are likely due to simple argument generation.")
            print(f"  → Recommendation: Use --use-llm mode for realistic arguments")
            print(f"  → Or improve simple argument generation with better type handling")

    if 'infrastructure' in failures_by_cat:
        infra_pct = failures_by_cat['infrastructure'] / total_failures * 100
        if infra_pct > 20:
            print(f"\n✓ INFRASTRUCTURE IS ALSO SIGNIFICANT:")
            print(f"  {infra_pct:.1f}% of failures are connection/auth issues.")
            print(f"  → These are not fixable by better arguments")
            print(f"  → These indicate real server unavailability")

    # Calculate "true" success rate if we exclude arg validation errors
    if 'invalid_arguments' in failures_by_cat:
        arg_failures = failures_by_cat['invalid_arguments']
        adjusted_success = summary['success'] + arg_failures
        adjusted_rate = adjusted_success / summary['total_tests'] * 100

        print(f"\n✓ POTENTIAL SUCCESS RATE WITH BETTER ARGUMENTS:")
        print(f"  Current success rate: {summary['success_rate']:.1f}%")
        print(f"  If arg validation errors were fixed: {adjusted_rate:.1f}%")
        print(f"  Potential improvement: +{adjusted_rate - summary['success_rate']:.1f} percentage points")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze failure reasons in MCP tool tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        default="tool_probe_result.json",
        help="Input probe results JSON (default: tool_probe_result.json)"
    )
    parser.add_argument(
        "--output",
        default="failure_reasons_analysis.json",
        help="Output analysis JSON (default: failure_reasons_analysis.json)"
    )
    parser.add_argument(
        "--show-all-samples",
        action="store_true",
        help="Show more error samples"
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    print(f"Analyzing failure reasons from: {args.input}\n")

    analysis = analyze_failure_reasons(args.input)

    print_analysis(analysis)

    # Save analysis
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
