#!/usr/bin/env python3
"""
Identify problematic trajectories that should be regenerated.

Categories of issues:
1. LLM provider errors (timeout, rate limit, insufficient funds) - CRITICAL
2. Empty or minimal trajectories (no tool calls, no meaningful response)
3. Tool-level errors (external API failures) - NOT critical, just logged
4. Very low goal completion rate with no tool calls

Key distinction:
- LLM PROVIDER errors (OpenRouter, model API) → should regenerate
- TOOL errors (external APIs like flight data, weather) → normal, don't regenerate
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# ============================================================================
# LLM PROVIDER ERROR PATTERNS - These indicate the trajectory should be regenerated
# These are errors from OpenRouter/model providers, NOT from external tool APIs
# ============================================================================
LLM_PROVIDER_ERROR_PATTERNS = {
    'llm_timeout': [
        'llm request timed out after multiple retries',  # Exact message from agent
        'llm request timed out',
    ],
    'llm_rate_limit': [
        '"code":429',  # HTTP 429 from LLM provider
        'status code 429',
        'rate limit exceeded',  # Specific rate limit message
    ],
    'llm_insufficient_funds': [
        '"code":402',  # HTTP 402 from LLM provider
        'status code 402',
        'insufficient credits',
        'payment required',
    ],
    'llm_model_error': [
        'model not found',
        'invalid model',
        'model is not available',
    ],
    'llm_content_filter': [
        'content filter',
        'content policy violation',
        'flagged by our safety system',
    ],
}

# ============================================================================
# TOOL ERROR PATTERNS - These are from external APIs, NOT critical for regeneration
# Logged for informational purposes but don't trigger regeneration
# ============================================================================
TOOL_ERROR_PATTERNS = {
    'tool_auth_error': [
        '401 client error',
        '403 client error',
        'unauthorized for url',
        'forbidden for url',
    ],
    'tool_server_error': [
        '500 server error',
        '502 bad gateway',
        '503 service unavailable',
    ],
    'tool_timeout': [
        'connection timeout',
        'read timeout',
        'connect timeout',
    ],
}


def check_agent_response_for_llm_errors(agent_response) -> List[Tuple[str, str]]:
    """
    Check agent_response field specifically for LLM provider errors.
    This is the most reliable indicator of LLM-level failures.
    """
    if not agent_response:
        return []

    # Handle case where agent_response is a dict (convert to string)
    if isinstance(agent_response, dict):
        agent_response = json.dumps(agent_response)

    response_lower = str(agent_response).lower()
    found_errors = []

    for category, patterns in LLM_PROVIDER_ERROR_PATTERNS.items():
        for pattern in patterns:
            if pattern in response_lower:
                found_errors.append((category, pattern))
                break

    return found_errors


def check_for_tool_errors(content: str) -> List[Tuple[str, str]]:
    """
    Check for tool-level errors (external API failures).
    These are informational only - don't trigger regeneration.
    """
    content_lower = content.lower()
    found_errors = []

    for category, patterns in TOOL_ERROR_PATTERNS.items():
        for pattern in patterns:
            if pattern in content_lower:
                found_errors.append((category, pattern))
                break

    return found_errors

def analyze_trajectory(filepath: Path) -> Dict[str, Any]:
    """Analyze a single trajectory file for issues."""
    issues = []
    tool_issues = []  # Informational only, don't trigger regeneration
    stats = {
        'total_turns': 0,
        'total_tool_calls': 0,
        'successful_tool_calls': 0,
        'failed_tool_calls': 0,
        'goal_completion_rate': 0.0,
        'has_meaningful_response': False,
    }

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            'filepath': str(filepath),
            'issues': [('parse_error', f'JSON parse error: {e}')],
            'tool_issues': [],
            'stats': stats,
            'should_regenerate': True,
        }
    except Exception as e:
        return {
            'filepath': str(filepath),
            'issues': [('read_error', f'File read error: {e}')],
            'tool_issues': [],
            'stats': stats,
            'should_regenerate': True,
        }

    # Check metadata
    metadata = data.get('metadata', {})
    stats['goal_completion_rate'] = metadata.get('goal_completion_rate', 0.0)

    # Check turns
    turns = data.get('turns', [])
    stats['total_turns'] = len(turns)

    if len(turns) == 0:
        issues.append(('empty_trajectory', 'No turns in trajectory'))

    # Track if we found LLM-level errors
    found_llm_errors = set()

    for turn in turns:
        tool_calls = turn.get('tool_calls', [])
        stats['total_tool_calls'] += len(tool_calls)

        for tc in tool_calls:
            if tc.get('status') == 'success':
                stats['successful_tool_calls'] += 1
            else:
                stats['failed_tool_calls'] += 1

        # =====================================================================
        # KEY CHECK: Look for LLM timeout in agent_response field
        # This is the most reliable indicator of LLM-level failure
        # =====================================================================
        agent_response = turn.get('agent_response', '')

        # Check if this is a meaningful response or an error message
        llm_errors = check_agent_response_for_llm_errors(agent_response)
        for category, pattern in llm_errors:
            if category not in found_llm_errors:
                issues.append((category, f'In agent_response: "{pattern}"'))
                found_llm_errors.add(category)

        # Check if response is meaningful (not an error message)
        # Convert to string if needed
        agent_response_str = json.dumps(agent_response) if isinstance(agent_response, dict) else str(agent_response) if agent_response else ''
        if agent_response_str and len(agent_response_str) > 100:
            # Make sure it's not just an error message
            if not any(p in agent_response_str.lower() for p in ['timed out', 'error', 'failed']):
                stats['has_meaningful_response'] = True

        # Check reasoning trace for tool-level errors (informational only)
        reasoning_trace = turn.get('reasoning_trace', [])
        for trace in reasoning_trace:
            if trace.get('type') == 'result':
                result_content = str(trace.get('content', ''))
                # Tool errors - informational only
                tool_errs = check_for_tool_errors(result_content)
                for category, pattern in tool_errs:
                    tool_issues.append((category, f'Tool result: {pattern}'))

    # =========================================================================
    # Determine if should regenerate - ONLY for LLM-level errors
    # =========================================================================
    should_regenerate = False

    # Critical LLM provider errors trigger regeneration
    critical_categories = ['llm_timeout', 'llm_rate_limit', 'llm_insufficient_funds', 'llm_model_error']
    for issue_cat, _ in issues:
        if issue_cat in critical_categories:
            should_regenerate = True
            break

    # Empty trajectory
    if stats['total_turns'] == 0:
        should_regenerate = True

    # No tool calls AND no meaningful response AND very low goal completion
    # This catches cases where agent refused or couldn't proceed
    if (stats['total_tool_calls'] == 0 and
        not stats['has_meaningful_response'] and
        stats['goal_completion_rate'] < 0.1):
        issues.append(('minimal_trajectory', 'No tool calls, no meaningful response, very low goal completion'))
        should_regenerate = True

    # All tool calls failed AND no meaningful response
    # Only regenerate if the agent couldn't produce ANY useful output
    if (stats['total_tool_calls'] > 0 and
        stats['successful_tool_calls'] == 0 and
        not stats['has_meaningful_response']):
        issues.append(('all_tools_failed', 'All tool calls failed with no meaningful response'))
        should_regenerate = True

    return {
        'filepath': str(filepath),
        'filename': filepath.name,
        'issues': issues,
        'tool_issues': tool_issues,  # Informational only
        'stats': stats,
        'should_regenerate': should_regenerate,
    }

def analyze_directory(directory: Path, recursive: bool = True) -> List[Dict[str, Any]]:
    """Analyze all trajectory files in a directory."""
    results = []

    pattern = '**/*.json' if recursive else '*.json'
    for filepath in directory.glob(pattern):
        if filepath.name.startswith('trajectory_'):
            result = analyze_trajectory(filepath)
            results.append(result)

    return results

def print_summary(results: List[Dict[str, Any]], verbose: bool = False):
    """Print a summary of the analysis."""
    total = len(results)
    problematic = [r for r in results if r['should_regenerate']]

    print(f"\n{'='*80}")
    print(f"TRAJECTORY ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total trajectories analyzed: {total}")
    print(f"Problematic (should regenerate): {len(problematic)} ({100*len(problematic)/total:.1f}%)" if total > 0 else "No trajectories found")

    # Group by issue category (critical issues that trigger regeneration)
    issue_counts = defaultdict(int)
    issue_files = defaultdict(list)

    for r in problematic:
        for category, _ in r['issues']:
            issue_counts[category] += 1
            issue_files[category].append(r['filename'])

    if issue_counts:
        print(f"\n{'='*80}")
        print("CRITICAL ISSUES (trigger regeneration):")
        print(f"{'='*80}")
        for category, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count} files")
            if verbose:
                for fname in issue_files[category][:5]:
                    print(f"    - {fname}")
                if len(issue_files[category]) > 5:
                    print(f"    ... and {len(issue_files[category]) - 5} more")

    # Group tool issues (informational only)
    tool_issue_counts = defaultdict(int)
    for r in results:
        for category, _ in r.get('tool_issues', []):
            tool_issue_counts[category] += 1

    if tool_issue_counts and verbose:
        print(f"\n{'='*80}")
        print("TOOL ISSUES (informational only, do NOT trigger regeneration):")
        print(f"{'='*80}")
        for category, count in sorted(tool_issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count} occurrences")

    # Group by directory
    dir_counts = defaultdict(lambda: {'total': 0, 'problematic': 0})
    for r in results:
        dir_path = str(Path(r['filepath']).parent)
        dir_counts[dir_path]['total'] += 1
        if r['should_regenerate']:
            dir_counts[dir_path]['problematic'] += 1

    print(f"\n{'='*80}")
    print("ISSUES BY DIRECTORY:")
    print(f"{'='*80}")
    for dir_path, counts in sorted(dir_counts.items()):
        if counts['problematic'] > 0:
            print(f"  {dir_path}:")
            print(f"    {counts['problematic']}/{counts['total']} problematic ({100*counts['problematic']/counts['total']:.1f}%)")

    return problematic

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Identify problematic trajectories')
    parser.add_argument('directory', nargs='?', default='trajectories/goaloriented',
                        help='Directory to analyze')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--list', action='store_true',
                        help='List all problematic files')
    parser.add_argument('--delete', action='store_true',
                        help='Delete problematic files (use with caution!)')
    parser.add_argument('--category', type=str,
                        help='Filter by specific issue category')

    args = parser.parse_args()

    # Find the base directory
    script_dir = Path(__file__).parent
    # The script is in MCP-R directory, so use script_dir directly as base
    base_dir = script_dir
    target_dir = base_dir / args.directory

    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        sys.exit(1)

    print(f"Analyzing trajectories in: {target_dir}")
    results = analyze_directory(target_dir)

    problematic = print_summary(results, args.verbose)

    # Filter by category if specified
    if args.category:
        problematic = [r for r in problematic
                       if any(cat == args.category for cat, _ in r['issues'])]
        print(f"\nFiltered to category '{args.category}': {len(problematic)} files")

    if args.list:
        print(f"\n{'='*80}")
        print("PROBLEMATIC FILES:")
        print(f"{'='*80}")
        for r in problematic:
            print(f"\n{r['filepath']}")
            print(f"  Issues: {', '.join(f'{cat}' for cat, _ in r['issues'])}")
            print(f"  Stats: {r['stats']['total_tool_calls']} tool calls, "
                  f"{r['stats']['goal_completion_rate']*100:.1f}% goal completion")

    if args.delete:
        if not problematic:
            print("\nNo files to delete.")
            return

        confirm = input(f"\nAre you sure you want to delete {len(problematic)} files? (yes/no): ")
        if confirm.lower() == 'yes':
            for r in problematic:
                try:
                    os.remove(r['filepath'])
                    print(f"Deleted: {r['filepath']}")
                except Exception as e:
                    print(f"Error deleting {r['filepath']}: {e}")
            print(f"\nDeleted {len(problematic)} files.")
        else:
            print("Deletion cancelled.")

if __name__ == '__main__':
    main()
