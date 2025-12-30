#!/usr/bin/env python3
"""
Identify problematic trajectories that should be regenerated.

Categories of issues:
1. API errors (insufficient funds, rate limits, etc.)
2. Empty or minimal trajectories (no tool calls, no meaningful response)
3. Server errors (500, 503, etc.)
4. Timeout errors
5. Authentication errors
6. Very low goal completion rate with no tool calls
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Error patterns to detect in trajectory content
ERROR_PATTERNS = {
    'insufficient_funds': [
        'insufficient credits',
        'insufficient funds',
        'payment required',
        'credit limit',
        '"code":402',
        'status code 402',
    ],
    'rate_limit': [
        'rate limit',
        'too many requests',
        '"code":429',
        'status code 429',
    ],
    'server_error': [
        'internal server error',
        '"code":500',
        '"code":502',
        '"code":503',
        'status code 500',
        'status code 502',
        'status code 503',
        'service unavailable',
        'bad gateway',
    ],
    'timeout': [
        'timed out',
        'timeout',
        'connection timeout',
        'read timeout',
    ],
    'auth_error': [
        'unauthorized',
        'authentication failed',
        'invalid api key',
        'forbidden',
        '"code":401',
        '"code":403',
    ],
    'model_error': [
        'model not found',
        'invalid model',
        'model is not available',
    ],
    'content_filter': [
        'content filter',
        'content policy',
        'flagged by',
    ],
}

def check_for_error_patterns(content: str) -> List[Tuple[str, str]]:
    """Check content for error patterns. Returns list of (category, pattern) tuples."""
    content_lower = content.lower()
    found_errors = []
    for category, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            if pattern in content_lower:
                found_errors.append((category, pattern))
                break  # Only need one match per category
    return found_errors

def analyze_trajectory(filepath: Path) -> Dict[str, Any]:
    """Analyze a single trajectory file for issues."""
    issues = []
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
            'stats': stats,
            'should_regenerate': True,
        }
    except Exception as e:
        return {
            'filepath': str(filepath),
            'issues': [('read_error', f'File read error: {e}')],
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

    # Analyze each turn
    full_content = json.dumps(data)  # For pattern matching

    for turn in turns:
        tool_calls = turn.get('tool_calls', [])
        stats['total_tool_calls'] += len(tool_calls)

        for tc in tool_calls:
            if tc.get('status') == 'success':
                stats['successful_tool_calls'] += 1
            else:
                stats['failed_tool_calls'] += 1

        # Check agent response
        agent_response = turn.get('agent_response', '')
        if agent_response and len(agent_response) > 100:
            stats['has_meaningful_response'] = True

        # Check reasoning trace for errors
        reasoning_trace = turn.get('reasoning_trace', [])
        for trace in reasoning_trace:
            if trace.get('type') == 'result':
                result_content = str(trace.get('content', ''))
                errors = check_for_error_patterns(result_content)
                for category, pattern in errors:
                    issues.append((category, f'Found in result: {pattern}'))

    # Check full content for error patterns
    errors = check_for_error_patterns(full_content)
    for category, pattern in errors:
        if (category, f'Found in result: {pattern}') not in issues:
            issues.append((category, f'Found in content: {pattern}'))

    # Determine if should regenerate
    should_regenerate = False

    # Definite regeneration triggers
    critical_categories = ['insufficient_funds', 'rate_limit', 'auth_error', 'model_error']
    for issue_cat, _ in issues:
        if issue_cat in critical_categories:
            should_regenerate = True
            break

    # Empty or minimal trajectory
    if stats['total_turns'] == 0:
        should_regenerate = True
    elif stats['total_tool_calls'] == 0 and stats['goal_completion_rate'] < 0.1:
        issues.append(('minimal_trajectory', 'No tool calls and very low goal completion'))
        should_regenerate = True

    # All tool calls failed
    if stats['total_tool_calls'] > 0 and stats['successful_tool_calls'] == 0:
        issues.append(('all_tools_failed', 'All tool calls failed'))
        should_regenerate = True

    return {
        'filepath': str(filepath),
        'filename': filepath.name,
        'issues': issues,
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
    print(f"Problematic trajectories: {len(problematic)} ({100*len(problematic)/total:.1f}%)" if total > 0 else "No trajectories found")

    # Group by issue category
    issue_counts = defaultdict(int)
    issue_files = defaultdict(list)

    for r in problematic:
        for category, _ in r['issues']:
            issue_counts[category] += 1
            issue_files[category].append(r['filename'])

    if issue_counts:
        print(f"\n{'='*80}")
        print("ISSUES BY CATEGORY:")
        print(f"{'='*80}")
        for category, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count} files")
            if verbose:
                for fname in issue_files[category][:5]:
                    print(f"    - {fname}")
                if len(issue_files[category]) > 5:
                    print(f"    ... and {len(issue_files[category]) - 5} more")

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
