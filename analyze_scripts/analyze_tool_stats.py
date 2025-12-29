#!/usr/bin/env python3
"""
Multiturn Trajectory Analysis for Goal-Oriented Tasks

This script analyzes multiturn trajectory files from the goaloriented directory,
providing detailed statistics on turns, server usage, tool calls, and constraint completion.

Usage:
    # Basic usage - analyze all trajectories in goaloriented directory
    python analyze_scripts/analyze_tool_stats.py

    # Filter by model
    python analyze_scripts/analyze_tool_stats.py --model gemini-3-pro-preview

    # Use custom trajectories directory
    python analyze_scripts/analyze_tool_stats.py --traj-dir /path/to/trajectories/goaloriented

Data Source:
    - Reads from trajectories/goaloriented/**/*.json files
    - Each trajectory contains multiple turns with tool_calls

Output:
    - Number of turns per trajectory
    - Number of servers used (total and unique)
    - Number of tools called (with duplicates and deduplicated)
    - Constraint completion status (from final turn)
    - Tool call success rates
    - Subgoal completion rates and goal achievement statistics

Multiturn Structure:
    - Each trajectory has multiple turns
    - Each turn has tool_calls with server/tool/status
    - Constraints are evaluated at the end (final turn metadata)
"""
from __future__ import annotations

import sys
import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

# Default trajectories directory
DEFAULT_TRAJECTORIES_DIR = Path(__file__).parent.parent / "trajectories" / "goaloriented"


def load_multiturn_trajectories(traj_dir: Path, model_filter: str = None) -> List[Dict[str, Any]]:
    """
    Load all multiturn trajectory files from the goaloriented directory.

    Args:
        traj_dir: Path to trajectories/goaloriented directory
        model_filter: Optional model name filter (case-insensitive)

    Returns:
        List of trajectory dictionaries
    """
    trajectories = []

    if not traj_dir.exists():
        print(f"Warning: Directory does not exist: {traj_dir}")
        return trajectories

    # Find all JSON files recursively
    for json_file in traj_dir.rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Apply model filter if specified
            if model_filter:
                agent_model = data.get("metadata", {}).get("agent_model", "")
                if model_filter.lower() not in agent_model.lower():
                    continue

            trajectories.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return trajectories


def check_result_has_error(result_content: Any) -> bool:
    """
    Check if a result content indicates an error.

    Args:
        result_content: The content from reasoning_trace result

    Returns:
        True if the result indicates an error, False otherwise
    """
    if result_content is None:
        return False

    content_str = str(result_content).lower()

    # Error indicators
    error_patterns = [
        'error executing tool',
        'error occurred during executing tool',
        'failed to',
        'request failed',
        'status code 4',  # 400, 401, 402, 403, 404, 429, etc.
        'status code 5',  # 500, 502, 503, etc.
        'timed out',
        'timeout',
        'could not be loaded',
        'validation error',
        'unauthorized',
        'forbidden',
        '"error"',
        '"status":403',
        '"status":429',
        '"status":500',
    ]

    return any(pattern in content_str for pattern in error_patterns)


def analyze_trajectory_stats(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze multiturn trajectory statistics.

    Returns:
        Dictionary containing various statistics
    """
    stats = {
        "total_trajectories": len(trajectories),
        "turn_counts": [],
        "server_usage": defaultdict(int),  # server -> count
        "unique_servers_per_traj": [],
        "tool_calls_total": [],
        "tool_calls_unique": [],
        "tool_calls_real_total": [],  # Excluding meta-mcp::search_tools
        "tool_calls_real_unique": [],  # Excluding meta-mcp::search_tools
        "tool_success_rate": [],
        "constraint_satisfaction": [],
        "tools_by_server": defaultdict(set),  # server -> set of tools
        "tool_call_details": [],  # All tool calls for detailed analysis
        "subgoal_completion_rates": [],  # Subgoal completion rates
        "subgoal_achieved_count": 0,  # Number of trajectories with goal achieved
    }

    for traj in trajectories:
        turns = traj.get("turns", [])
        num_turns = len(turns)
        stats["turn_counts"].append(num_turns)

        # Track servers and tools across all turns
        servers_in_traj = set()
        tools_in_traj = []
        tools_unique_in_traj = set()
        tools_real_in_traj = []  # Excluding meta-mcp::search_tools
        tools_real_unique_in_traj = set()  # Excluding meta-mcp::search_tools
        successful_calls = 0
        total_calls = 0

        for turn in turns:
            tool_calls = turn.get("tool_calls", [])
            reasoning_trace = turn.get("reasoning_trace", [])

            # Extract results from reasoning_trace for error checking
            results = [t.get("content") for t in reasoning_trace if t.get("type") == "result"]

            for idx, call in enumerate(tool_calls):
                server = call.get("server", "unknown")
                tool = call.get("tool", "unknown")
                status = call.get("status", "unknown")

                # Track servers
                servers_in_traj.add(server)
                stats["server_usage"][server] += 1

                # Track tools
                tool_full_name = f"{server}::{tool}"
                tools_in_traj.append(tool_full_name)
                tools_unique_in_traj.add(tool_full_name)
                stats["tools_by_server"][server].add(tool)

                # Track real tools (excluding meta-mcp::search_tools)
                if not (server == "meta-mcp" and tool == "search_tools"):
                    tools_real_in_traj.append(tool_full_name)
                    tools_real_unique_in_traj.add(tool_full_name)

                # Check if the result indicates an error
                # Note: results may not align 1-1 with tool_calls, so we check all results
                has_error = False
                if idx < len(results):
                    has_error = check_result_has_error(results[idx])

                is_success = status == "success" and not has_error

                # Track success
                total_calls += 1
                if is_success:
                    successful_calls += 1

                # Store for detailed analysis
                stats["tool_call_details"].append({
                    "server": server,
                    "tool": tool,
                    "status": status,
                    "has_error": has_error,
                    "is_success": is_success
                })

        stats["unique_servers_per_traj"].append(len(servers_in_traj))
        stats["tool_calls_total"].append(len(tools_in_traj))
        stats["tool_calls_unique"].append(len(tools_unique_in_traj))
        stats["tool_calls_real_total"].append(len(tools_real_in_traj))
        stats["tool_calls_real_unique"].append(len(tools_real_unique_in_traj))

        if total_calls > 0:
            stats["tool_success_rate"].append(successful_calls / total_calls)

        # Get constraint satisfaction from final turn or metadata
        metadata = traj.get("metadata", {})
        constraint_rate = metadata.get("overall_constraint_satisfaction_rate")
        if constraint_rate is not None:
            stats["constraint_satisfaction"].append(constraint_rate)
        else:
            # Try to get from final turn
            if turns:
                final_turn = turns[-1]
                constraint_rate = final_turn.get("constraint_satisfaction_rate")
                if constraint_rate is not None:
                    stats["constraint_satisfaction"].append(constraint_rate)

        # Get subgoal completion rate and achievement status from metadata
        goal_completion_rate = metadata.get("goal_completion_rate")
        if goal_completion_rate is not None:
            stats["subgoal_completion_rates"].append(goal_completion_rate)

        goal_achieved = metadata.get("goal_achieved")
        if goal_achieved:
            stats["subgoal_achieved_count"] += 1

    return stats


def print_multiturn_analysis(stats: Dict[str, Any]):
    """Print comprehensive analysis of multiturn trajectories."""

    print("\n" + "=" * 80)
    print("MULTITURN TRAJECTORY ANALYSIS")
    print("=" * 80)

    print(f"\nTotal Trajectories Analyzed: {stats['total_trajectories']}")

    if stats['total_trajectories'] == 0:
        print("\nNo trajectories found!")
        return

    # Turn statistics
    print("\n" + "-" * 80)
    print("TURN STATISTICS")
    print("-" * 80)
    turn_counts = stats['turn_counts']
    if turn_counts:
        print(f"Average turns per trajectory: {statistics.mean(turn_counts):.2f}")
        print(f"Median turns: {statistics.median(turn_counts):.1f}")
        print(f"Min turns: {min(turn_counts)}")
        print(f"Max turns: {max(turn_counts)}")
        print(f"Total turns across all trajectories: {sum(turn_counts)}")

    # Server usage statistics
    print("\n" + "-" * 80)
    print("SERVER USAGE STATISTICS")
    print("-" * 80)

    unique_servers = stats['unique_servers_per_traj']
    if unique_servers:
        print(f"Average unique servers per trajectory: {statistics.mean(unique_servers):.2f}")
        print(f"Median unique servers: {statistics.median(unique_servers):.1f}")
        print(f"Min servers: {min(unique_servers)}")
        print(f"Max servers: {max(unique_servers)}")

    print(f"\nTotal unique servers used: {len(stats['server_usage'])}")
    print("\nTop 10 Most Used Servers:")
    sorted_servers = sorted(stats['server_usage'].items(), key=lambda x: x[1], reverse=True)
    print(f"{'Server':<50} {'Tool Calls':<12} {'Unique Tools'}")
    print("-" * 80)
    for server, count in sorted_servers[:10]:
        server_display = server[:48] if len(server) > 48 else server
        num_tools = len(stats['tools_by_server'][server])
        print(f"{server_display:<50} {count:<12} {num_tools}")

    # Tool call statistics
    print("\n" + "-" * 80)
    print("TOOL CALL STATISTICS")
    print("-" * 80)

    total_calls = stats['tool_calls_total']
    unique_calls = stats['tool_calls_unique']
    real_total_calls = stats['tool_calls_real_total']
    real_unique_calls = stats['tool_calls_real_unique']

    if total_calls:
        print(f"Average total tool calls per trajectory: {statistics.mean(total_calls):.2f}")
        print(f"Average unique tool calls per trajectory: {statistics.mean(unique_calls):.2f}")
        print(f"Total tool calls (with duplicates): {sum(total_calls)}")
        print(f"Average duplication rate: {(sum(total_calls) - sum(unique_calls)) / sum(total_calls) * 100:.1f}%")

        print(f"\n--- Real Tool Calls (excluding meta-mcp::search_tools) ---")
        print(f"Average real tool calls per trajectory: {statistics.mean(real_total_calls):.2f}")
        print(f"Average unique real tool calls per trajectory: {statistics.mean(real_unique_calls):.2f}")
        print(f"Total real tool calls (with duplicates): {sum(real_total_calls)}")
        if sum(real_total_calls) > 0:
            print(f"Real tool duplication rate: {(sum(real_total_calls) - sum(real_unique_calls)) / sum(real_total_calls) * 100:.1f}%")

    # Tool success rate
    print("\n" + "-" * 80)
    print("TOOL CALL SUCCESS RATE")
    print("-" * 80)

    all_calls = stats['tool_call_details']
    if all_calls:
        total = len(all_calls)
        successful = sum(1 for call in all_calls if call['is_success'])
        failed = total - successful

        print(f"Total tool calls: {total}")
        print(f"Successful calls: {successful}")
        print(f"Failed calls: {failed}")
        print(f"Overall success rate: {successful / total * 100:.2f}%")

        # Breakdown of failures
        if failed > 0:
            status_success_but_error = sum(1 for call in all_calls
                                          if call['status'] == 'success' and call['has_error'])
            status_not_success = sum(1 for call in all_calls
                                    if call['status'] != 'success')

            print(f"\nFailure breakdown:")
            print(f"  - Status='success' but result has error: {status_success_but_error}")
            print(f"  - Status != 'success': {status_not_success}")

        # Success rate by server
        print("\nSuccess Rate by Server:")
        server_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0})
        for call in all_calls:
            server_stats[call['server']]['total'] += 1
            if call['is_success']:
                server_stats[call['server']]['success'] += 1
            else:
                server_stats[call['server']]['failed'] += 1

        server_success_rates = []
        for server, stats_dict in server_stats.items():
            rate = stats_dict['success'] / stats_dict['total'] * 100
            server_success_rates.append((server, stats_dict['success'],
                                        stats_dict['failed'], stats_dict['total'], rate))

        server_success_rates.sort(key=lambda x: x[3], reverse=True)  # Sort by total calls

        print(f"{'Server':<50} {'Success':<10} {'Failed':<8} {'Total':<8} {'Rate'}")
        print("-" * 85)
        for server, success, failed, total, rate in server_success_rates[:15]:
            server_display = server[:48] if len(server) > 48 else server
            print(f"{server_display:<50} {success:<10} {failed:<8} {total:<8} {rate:.1f}%")

    # Constraint satisfaction
    print("\n" + "-" * 80)
    print("CONSTRAINT SATISFACTION (from final turn)")
    print("-" * 80)

    constraint_rates = stats['constraint_satisfaction']
    if constraint_rates:
        print(f"Trajectories with constraint data: {len(constraint_rates)}")
        print(f"Average constraint satisfaction rate: {statistics.mean(constraint_rates) * 100:.1f}%")
        print(f"Median constraint satisfaction rate: {statistics.median(constraint_rates) * 100:.1f}%")
        print(f"Min satisfaction rate: {min(constraint_rates) * 100:.1f}%")
        print(f"Max satisfaction rate: {max(constraint_rates) * 100:.1f}%")

        # Distribution
        perfect = sum(1 for r in constraint_rates if r >= 1.0)
        high = sum(1 for r in constraint_rates if 0.8 <= r < 1.0)
        medium = sum(1 for r in constraint_rates if 0.5 <= r < 0.8)
        low = sum(1 for r in constraint_rates if r < 0.5)

        print(f"\nConstraint Satisfaction Distribution:")
        print(f"  Perfect (100%):     {perfect:3d} ({perfect/len(constraint_rates)*100:.1f}%)")
        print(f"  High (80-99%):      {high:3d} ({high/len(constraint_rates)*100:.1f}%)")
        print(f"  Medium (50-79%):    {medium:3d} ({medium/len(constraint_rates)*100:.1f}%)")
        print(f"  Low (<50%):         {low:3d} ({low/len(constraint_rates)*100:.1f}%)")
    else:
        print("No constraint satisfaction data available")

    # Subgoal completion
    print("\n" + "-" * 80)
    print("SUBGOAL COMPLETION ANALYSIS")
    print("-" * 80)

    subgoal_rates = stats['subgoal_completion_rates']
    if subgoal_rates:
        print(f"Trajectories with subgoal data: {len(subgoal_rates)}")
        print(f"Average subgoal completion rate: {statistics.mean(subgoal_rates) * 100:.1f}%")
        print(f"Median subgoal completion rate: {statistics.median(subgoal_rates) * 100:.1f}%")
        print(f"Min completion rate: {min(subgoal_rates) * 100:.1f}%")
        print(f"Max completion rate: {max(subgoal_rates) * 100:.1f}%")

        # Goal achievement
        total_with_data = len(subgoal_rates)
        achieved = stats['subgoal_achieved_count']
        print(f"\nGoal Achievement:")
        print(f"  Goals fully achieved: {achieved:3d} ({achieved/total_with_data*100:.1f}%)")
        print(f"  Goals not achieved:   {total_with_data - achieved:3d} ({(total_with_data - achieved)/total_with_data*100:.1f}%)")

        # Distribution
        perfect = sum(1 for r in subgoal_rates if r >= 1.0)
        high = sum(1 for r in subgoal_rates if 0.8 <= r < 1.0)
        medium = sum(1 for r in subgoal_rates if 0.5 <= r < 0.8)
        low = sum(1 for r in subgoal_rates if r < 0.5)

        print(f"\nSubgoal Completion Distribution:")
        print(f"  Perfect (100%):     {perfect:3d} ({perfect/len(subgoal_rates)*100:.1f}%)")
        print(f"  High (80-99%):      {high:3d} ({high/len(subgoal_rates)*100:.1f}%)")
        print(f"  Medium (50-79%):    {medium:3d} ({medium/len(subgoal_rates)*100:.1f}%)")
        print(f"  Low (<50%):         {low:3d} ({low/len(subgoal_rates)*100:.1f}%)")
    else:
        print("No subgoal completion data available")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multiturn trajectory files from goaloriented directory"
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=DEFAULT_TRAJECTORIES_DIR,
        help=f"Trajectories directory (default: {DEFAULT_TRAJECTORIES_DIR})",
    )
    parser.add_argument(
        "--model",
        help="Filter to specific model (case-insensitive)",
    )

    args = parser.parse_args()

    if not args.traj_dir.exists():
        print(f"Error: Trajectories directory not found: {args.traj_dir}")
        return 1

    print(f"Loading trajectory data from: {args.traj_dir}")
    trajectories = load_multiturn_trajectories(args.traj_dir, args.model)
    print(f"Found {len(trajectories)} trajectories")

    if args.model:
        print(f"Filtered to model: {args.model}")

    if not trajectories:
        print("\nNo trajectories found matching criteria!")
        return 1

    stats = analyze_trajectory_stats(trajectories)
    print_multiturn_analysis(stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
