#!/usr/bin/env python3
"""
Tool Retry Behavior Analysis

This script analyzes how models behave when tool calls fail - whether they
retry the same tool, switch to a different tool, or give up entirely.

Usage:
    # Basic usage - show retry behavior analysis
    python analyze_scripts/analyze_retry.py

    # Filter by model
    python analyze_scripts/analyze_retry.py --model claude-3.5

    # Use custom trajectories directory
    python analyze_scripts/analyze_retry.py --traj-dir /path/to/trajectories

Data Source:
    - Reads from trajectories/*.json files
    - Analyzes tool_sequence in each trajectory

Output:
    - Behavior after tool failure (retry same, try different, give up)
    - Retry rate percentages by model
    - Consecutive same-tool call patterns
    - Tool usage diversity
    - Queries with most/fewest tool calls
    - Hardest/easiest queries across models

Methodology:
    1. For each trajectory, build a tool_sequence from tool_calls
    2. When a tool call fails (is_error=True), check what happens next:
       - Same tool called again → "retry_same_tool"
       - Different tool called → "retry_different_tool" (tool switch)
       - No more calls → "gave_up"
    3. Track consecutive same-tool calls to measure persistence
    4. Calculate tool diversity (unique tools per trajectory)
    5. Identify hardest queries (require most tool calls on average)

Key Metrics:
    - Retry Same %: How often model retries the exact same tool after failure
    - Switch Tool %: How often model tries a different tool after failure
    - Give Up %: How often trajectory ends after a failure
    - Avg Consecutive: Average times same tool called in a row
    - Tool Diversity: Average unique tools per trajectory
"""
from __future__ import annotations

import sys
import argparse
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

# Import shared data loading functions
from data_loaders import (
    TRAJECTORIES_DIR,
    load_trajectory_data,
)


def analyze_retry_behavior(trajectories: List[Dict[str, Any]]) -> None:
    """
    Analyze how models behave when tool calls fail.

    Tracks:
    - Retry same tool: Called same tool again after failure
    - Try different tool: Called different tool after failure
    - Gave up: No more tool calls after failure
    - Consecutive same-tool calls
    - Tool diversity per trajectory
    """
    print("\n" + "=" * 80)
    print("TOOL RETRY BEHAVIOR ANALYSIS")
    print("=" * 80)

    if not trajectories:
        print("No trajectory data found.")
        return

    # Analyze per-model retry behavior
    by_model = defaultdict(lambda: {
        "total_trajectories": 0,
        "trajectories_with_errors": 0,
        "retry_same_tool": 0,  # Called same tool again after failure
        "retry_different_tool": 0,  # Called different tool after failure
        "gave_up": 0,  # No more tool calls after failure
        "consecutive_same_tool": [],  # Track consecutive calls to same tool
        "tool_switches_after_error": 0,
    })

    for traj in trajectories:
        model = traj["model"]
        sequence = traj["tool_sequence"]
        by_model[model]["total_trajectories"] += 1

        if not sequence:
            continue

        has_error = any(tc["is_error"] for tc in sequence)
        if has_error:
            by_model[model]["trajectories_with_errors"] += 1

        # Analyze consecutive tool calls
        consecutive_count = 1
        prev_tool_key = None

        for i, tc in enumerate(sequence):
            tool_key = f"{tc['server']}::{tc['tool']}"

            if prev_tool_key == tool_key:
                consecutive_count += 1
            else:
                if consecutive_count > 1:
                    by_model[model]["consecutive_same_tool"].append(consecutive_count)
                consecutive_count = 1

            # Check behavior after errors
            if i > 0 and sequence[i-1]["is_error"]:
                prev_tool_key_check = f"{sequence[i-1]['server']}::{sequence[i-1]['tool']}"
                if tool_key == prev_tool_key_check:
                    by_model[model]["retry_same_tool"] += 1
                else:
                    by_model[model]["retry_different_tool"] += 1
                    by_model[model]["tool_switches_after_error"] += 1

            prev_tool_key = tool_key

        # Record last consecutive sequence
        if consecutive_count > 1:
            by_model[model]["consecutive_same_tool"].append(consecutive_count)

        # Check if trajectory ended after an error (gave up)
        if sequence and sequence[-1]["is_error"]:
            by_model[model]["gave_up"] += 1

    # Print results
    print("\n--- Behavior After Tool Failure ---")
    print(f"{'Model':<20} {'Retry Same':<12} {'Try Different':<15} {'Gave Up':<10} {'Total Errors'}")
    print("-" * 75)

    for model in sorted(by_model.keys()):
        stats = by_model[model]
        retry_same = stats["retry_same_tool"]
        retry_diff = stats["retry_different_tool"]
        gave_up = stats["gave_up"]
        total_after_error = retry_same + retry_diff + gave_up

        if total_after_error > 0:
            print(f"{model:<20} {retry_same:<12} {retry_diff:<15} {gave_up:<10} {total_after_error}")

    # Print retry rates as percentages
    print("\n--- Retry Rate After Failure (%) ---")
    print(f"{'Model':<20} {'Retry Same%':<15} {'Switch Tool%':<15} {'Give Up%':<12}")
    print("-" * 65)

    for model in sorted(by_model.keys()):
        stats = by_model[model]
        retry_same = stats["retry_same_tool"]
        retry_diff = stats["retry_different_tool"]
        gave_up = stats["gave_up"]
        total = retry_same + retry_diff + gave_up

        if total > 0:
            retry_same_pct = retry_same / total * 100
            retry_diff_pct = retry_diff / total * 100
            gave_up_pct = gave_up / total * 100
            print(f"{model:<20} {retry_same_pct:<15.1f} {retry_diff_pct:<15.1f} {gave_up_pct:<12.1f}")

    # Consecutive tool call analysis
    print("\n--- Consecutive Same-Tool Calls ---")
    print("(How many times models call the same tool repeatedly)")
    print(f"{'Model':<20} {'Avg Consecutive':<18} {'Max Consecutive':<18} {'Sequences'}")
    print("-" * 75)

    for model in sorted(by_model.keys()):
        stats = by_model[model]
        consecutive = stats["consecutive_same_tool"]
        if consecutive:
            avg_consec = statistics.mean(consecutive)
            max_consec = max(consecutive)
            print(f"{model:<20} {avg_consec:<18.2f} {max_consec:<18} {len(consecutive)}")
        else:
            print(f"{model:<20} {'N/A':<18} {'N/A':<18} 0")

    # Tool diversity analysis
    print("\n--- Tool Usage Diversity ---")
    print(f"{'Model':<20} {'Avg Tools/Traj':<18} {'Max Tools/Traj':<18} {'Trajectories'}")
    print("-" * 75)

    for model in sorted(by_model.keys()):
        tools_per_traj = []
        for traj in trajectories:
            if traj["model"] == model and traj["tool_sequence"]:
                unique_tools = set(f"{tc['server']}::{tc['tool']}" for tc in traj["tool_sequence"])
                tools_per_traj.append(len(unique_tools))

        if tools_per_traj:
            avg_tools = statistics.mean(tools_per_traj)
            max_tools = max(tools_per_traj)
            print(f"{model:<20} {avg_tools:<18.2f} {max_tools:<18} {len(tools_per_traj)}")

    # Top queries by tool call count
    print("\n--- Queries with Most Tool Calls ---")
    traj_by_calls = sorted(trajectories, key=lambda t: t["total_tool_calls"], reverse=True)

    print(f"{'Model':<18} {'Calls':<8} {'Unique':<8} {'Query UUID':<40}")
    print("-" * 80)
    for traj in traj_by_calls[:15]:
        if traj["total_tool_calls"] > 0:
            unique_tools = len(set(f"{tc['server']}::{tc['tool']}" for tc in traj["tool_sequence"]))
            print(f"{traj['model']:<18} {traj['total_tool_calls']:<8} {unique_tools:<8} {traj['query_uuid'][:38]}")

    # Queries grouped by UUID to see which queries are hardest across models
    print("\n--- Hardest Queries (Avg Tool Calls Per Trajectory) ---")
    by_query = defaultdict(lambda: {"call_counts": [], "models": []})
    for traj in trajectories:
        uuid = traj["query_uuid"]
        if uuid:
            by_query[uuid]["call_counts"].append(traj["total_tool_calls"])
            by_query[uuid]["models"].append((traj["model"], traj["total_tool_calls"]))

    # Calculate averages and sort
    query_stats = []
    for uuid, stats in by_query.items():
        avg_calls = statistics.mean(stats["call_counts"]) if stats["call_counts"] else 0
        max_calls = max(stats["call_counts"]) if stats["call_counts"] else 0
        num_trajs = len(stats["call_counts"])
        query_stats.append((uuid, avg_calls, max_calls, num_trajs, stats["models"]))

    sorted_queries = sorted(query_stats, key=lambda x: x[1], reverse=True)

    print(f"{'Query UUID':<40} {'Avg Calls':<12} {'Max':<8} {'#Traj':<8} {'Top Models'}")
    print("-" * 90)
    for uuid, avg_calls, max_calls, num_trajs, models in sorted_queries[:10]:
        models_str = ", ".join(f"{m}({c})" for m, c in sorted(models, key=lambda x: x[1], reverse=True)[:2])
        print(f"{uuid[:38]:<40} {avg_calls:<12.1f} {max_calls:<8} {num_trajs:<8} {models_str[:30]}")

    # Easiest queries (fewest tool calls)
    print("\n--- Easiest Queries (Fewest Tool Calls Per Trajectory) ---")
    # Filter out queries with 0 calls and sort ascending
    queries_with_calls = [q for q in sorted_queries if q[1] > 0]
    easiest_queries = sorted(queries_with_calls, key=lambda x: x[1])

    print(f"{'Query UUID':<40} {'Avg Calls':<12} {'Min':<8} {'#Traj':<8} {'Models'}")
    print("-" * 90)
    for uuid, avg_calls, max_calls, num_trajs, models in easiest_queries[:10]:
        min_calls = min(c for _, c in models) if models else 0
        models_str = ", ".join(f"{m}({c})" for m, c in sorted(models, key=lambda x: x[1])[:2])
        print(f"{uuid[:38]:<40} {avg_calls:<12.1f} {min_calls:<8} {num_trajs:<8} {models_str[:30]}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tool retry behavior from trajectory files"
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=TRAJECTORIES_DIR,
        help=f"Trajectories directory (default: {TRAJECTORIES_DIR})",
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
    tool_calls, trajectories = load_trajectory_data(args.traj_dir, args.model)
    print(f"Found {len(tool_calls)} tool calls in {len(trajectories)} trajectories")

    if args.model:
        print(f"Filtered to model: {args.model}")

    analyze_retry_behavior(trajectories)

    return 0


if __name__ == "__main__":
    sys.exit(main())
