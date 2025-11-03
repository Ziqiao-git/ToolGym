#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trajectory_stats.py

Unified script to analyze trajectory statistics - supports both single file and directory analysis.

Usage:
  # Analyze single trajectory
  python evaluation/stats/trajectory_stats.py trajectories/trajectory_20251102_184025.json

  # Analyze all trajectories in directory
  python evaluation/stats/trajectory_stats.py --dir trajectories

  # JSON output for scripting
  python evaluation/stats/trajectory_stats.py trajectories/trajectory_20251102_184025.json --json
  python evaluation/stats/trajectory_stats.py --dir trajectories --json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List


def analyze_single_trajectory(trajectory_path: Path) -> Dict[str, Any]:
    """
    Analyze a single trajectory file.

    Args:
        trajectory_path: Path to trajectory JSON file

    Returns:
        Dictionary with statistics

    Note:
        - 'executed_successfully': Tool call completed without errors (technical success)
        - 'execution_errors': Tool call failed with errors (technical failure)
        - 'unknown_status': Tool call has no status field (old format or incomplete)
        - This does NOT measure task quality - use LLM judge evaluation for that
    """
    with open(trajectory_path, 'r', encoding='utf-8') as f:
        trajectory = json.load(f)

    execution = trajectory.get("execution", {})
    tool_calls = execution.get("tool_calls", [])
    servers = trajectory.get("servers", {})

    # Basic counts - renamed for clarity
    total_calls = len(tool_calls)
    executed_successfully = sum(1 for tc in tool_calls if tc.get("status") == "success")
    execution_errors = sum(1 for tc in tool_calls if tc.get("status") == "error")
    unknown_status = sum(1 for tc in tool_calls if "status" not in tc)
    dynamic_loads = sum(1 for tc in tool_calls if tc.get("dynamically_loaded", False))

    # Additional metrics
    calls_with_results = sum(1 for tc in tool_calls if tc.get("result_preview") and tc.get("result_preview").strip())
    calls_without_results = sum(1 for tc in tool_calls if not tc.get("result_preview") or not tc.get("result_preview").strip())

    # Server usage
    servers_used = {}
    for tc in tool_calls:
        server = tc.get("server", "unknown")
        servers_used[server] = servers_used.get(server, 0) + 1

    # Tool usage
    tools_used = {}
    for tc in tool_calls:
        server = tc.get("server", "unknown")
        tool = tc.get("tool", "unknown")
        key = f"{server}/{tool}"
        tools_used[key] = tools_used.get(key, 0) + 1

    # Duration statistics
    durations = [tc.get("duration_seconds", 0) for tc in tool_calls if "duration_seconds" in tc]
    total_duration = sum(durations)
    avg_duration = total_duration / len(durations) if durations else 0

    return {
        "file": trajectory_path.name,
        "query": trajectory.get("metadata", {}).get("query", "")[:100] + "...",
        "total_calls": total_calls,
        "executed_successfully": executed_successfully,
        "execution_errors": execution_errors,
        "unknown_status": unknown_status,
        "dynamic_loads": dynamic_loads,
        "calls_with_results": calls_with_results,
        "calls_without_results": calls_without_results,
        "total_duration": round(total_duration, 2),
        "servers_used": servers.get("total_servers_used", 0),
        "dynamically_loaded_servers": len(servers.get("dynamically_loaded", [])),
        "servers": {
            "initially_loaded": servers.get("initially_loaded", []),
            "dynamically_loaded": servers.get("dynamically_loaded", []),
        },
        "server_usage": servers_used,
        "tool_usage": tools_used,
        "duration": {
            "total_seconds": round(total_duration, 2),
            "average_seconds": round(avg_duration, 2),
            "min_seconds": round(min(durations), 2) if durations else 0,
            "max_seconds": round(max(durations), 2) if durations else 0,
        }
    }


def compute_aggregate_stats(all_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics from all trajectories."""
    if not all_stats:
        return {
            "total_trajectories": 0,
            "error": "No trajectories found"
        }

    total_trajectories = len(all_stats)
    total_calls = sum(s["total_calls"] for s in all_stats)
    total_executed_successfully = sum(s["executed_successfully"] for s in all_stats)
    total_execution_errors = sum(s["execution_errors"] for s in all_stats)
    total_unknown_status = sum(s["unknown_status"] for s in all_stats)
    total_dynamic_loads = sum(s["dynamic_loads"] for s in all_stats)
    total_calls_with_results = sum(s["calls_with_results"] for s in all_stats)
    total_calls_without_results = sum(s["calls_without_results"] for s in all_stats)
    total_duration = sum(s["total_duration"] for s in all_stats)
    total_servers_used = sum(s["servers_used"] for s in all_stats)
    total_dynamic_servers = sum(s["dynamically_loaded_servers"] for s in all_stats)

    # Distribution of tool calls per trajectory
    calls_per_trajectory = [s["total_calls"] for s in all_stats]
    calls_per_trajectory.sort()

    return {
        "total_trajectories": total_trajectories,
        "total_tool_calls": total_calls,
        "executed_successfully": total_executed_successfully,
        "execution_errors": total_execution_errors,
        "unknown_status": total_unknown_status,
        "dynamically_loaded_calls": total_dynamic_loads,
        "calls_with_results": total_calls_with_results,
        "calls_without_results": total_calls_without_results,
        "averages": {
            "tool_calls_per_trajectory": round(total_calls / total_trajectories, 2),
            "executed_successfully_per_trajectory": round(total_executed_successfully / total_trajectories, 2),
            "execution_errors_per_trajectory": round(total_execution_errors / total_trajectories, 2),
            "unknown_status_per_trajectory": round(total_unknown_status / total_trajectories, 2),
            "dynamic_loads_per_trajectory": round(total_dynamic_loads / total_trajectories, 2),
            "calls_with_results_per_trajectory": round(total_calls_with_results / total_trajectories, 2),
            "duration_per_trajectory_seconds": round(total_duration / total_trajectories, 2),
            "servers_per_trajectory": round(total_servers_used / total_trajectories, 2),
            "dynamic_servers_per_trajectory": round(total_dynamic_servers / total_trajectories, 2),
        },
        "distribution": {
            "min_calls": min(calls_per_trajectory) if calls_per_trajectory else 0,
            "max_calls": max(calls_per_trajectory) if calls_per_trajectory else 0,
            "median_calls": calls_per_trajectory[len(calls_per_trajectory) // 2] if calls_per_trajectory else 0,
        },
        "technical_success_rate": round(total_executed_successfully / total_calls * 100, 1) if total_calls > 0 else 0,
        "calls_with_results_rate": round(total_calls_with_results / total_calls * 100, 1) if total_calls > 0 else 0,
    }


def print_single_trajectory(stats: Dict[str, Any]) -> None:
    """Print single trajectory statistics in readable format."""
    print(f"\n{'='*70}")
    print(f"Trajectory: {stats['file']}")
    print(f"{'='*70}")

    print(f"\nQuery: {stats['query']}")

    print(f"\n📊 Tool Call Summary:")
    print(f"  Total tool calls: {stats['total_calls']}")
    print(f"  ✅ Executed successfully: {stats['executed_successfully']} (technical success)")
    print(f"  ❌ Execution errors: {stats['execution_errors']}")
    print(f"  ❓ Unknown status: {stats['unknown_status']}")
    print(f"  🔄 Dynamically loaded: {stats['dynamic_loads']}")
    print(f"  📄 Calls with results: {stats['calls_with_results']}")
    print(f"  📭 Calls without results: {stats['calls_without_results']}")

    print(f"\n🖥️  Server Usage:")
    print(f"  Initially loaded: {', '.join(stats['servers']['initially_loaded'])}")
    print(f"  Dynamically loaded: {', '.join(stats['servers']['dynamically_loaded']) or 'None'}")
    print(f"  Total servers used: {stats['servers_used']}")

    if stats['server_usage']:
        print(f"\n📈 Calls per Server:")
        for server, count in sorted(stats['server_usage'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {server}: {count}")

    if stats['tool_usage']:
        print(f"\n🔧 Tools Used:")
        for tool, count in sorted(stats['tool_usage'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count}")

    print(f"\n⏱️  Duration Statistics:")
    print(f"  Total: {stats['duration']['total_seconds']}s")
    print(f"  Average: {stats['duration']['average_seconds']}s")
    print(f"  Min: {stats['duration']['min_seconds']}s")
    print(f"  Max: {stats['duration']['max_seconds']}s")

    print(f"\n{'='*70}\n")


def print_aggregate_stats(stats: Dict[str, Any], all_stats: List[Dict[str, Any]]) -> None:
    """Print aggregate statistics in readable format."""
    print(f"\n{'='*70}")
    print(f"Aggregate Trajectory Analysis")
    print(f"{'='*70}")

    if "error" in stats:
        print(f"\n⚠️  {stats['error']}")
        return

    print(f"\n📁 Total Trajectories Analyzed: {stats['total_trajectories']}")

    print(f"\n📊 Overall Tool Call Statistics:")
    print(f"  Total tool calls: {stats['total_tool_calls']}")
    print(f"  ✅ Executed successfully: {stats['executed_successfully']} ({stats['technical_success_rate']}%) - technical success")
    print(f"  ❌ Execution errors: {stats['execution_errors']}")
    print(f"  ❓ Unknown status: {stats['unknown_status']}")
    print(f"  🔄 Dynamically loaded: {stats['dynamically_loaded_calls']}")
    print(f"  📄 Calls with results: {stats['calls_with_results']} ({stats['calls_with_results_rate']}%)")
    print(f"  📭 Calls without results: {stats['calls_without_results']}")

    print(f"\n📈 Averages per Trajectory:")
    print(f"  Tool calls: {stats['averages']['tool_calls_per_trajectory']}")
    print(f"  Executed successfully: {stats['averages']['executed_successfully_per_trajectory']}")
    print(f"  Execution errors: {stats['averages']['execution_errors_per_trajectory']}")
    print(f"  Unknown status: {stats['averages']['unknown_status_per_trajectory']}")
    print(f"  Dynamic loads: {stats['averages']['dynamic_loads_per_trajectory']}")
    print(f"  Calls with results: {stats['averages']['calls_with_results_per_trajectory']}")
    print(f"  Duration: {stats['averages']['duration_per_trajectory_seconds']}s")
    print(f"  Servers used: {stats['averages']['servers_per_trajectory']}")
    print(f"  Dynamic servers: {stats['averages']['dynamic_servers_per_trajectory']}")

    print(f"\n📉 Distribution:")
    print(f"  Min calls per trajectory: {stats['distribution']['min_calls']}")
    print(f"  Median calls per trajectory: {stats['distribution']['median_calls']}")
    print(f"  Max calls per trajectory: {stats['distribution']['max_calls']}")

    print(f"\n🔍 Individual Trajectories:")
    print(f"  {'File':<50} {'Calls':<8} {'Exec✅':<8} {'Errors❌':<8} {'Dynamic':<8}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for s in sorted(all_stats, key=lambda x: x["total_calls"], reverse=True):
        print(f"  {s['file']:<50} {s['total_calls']:<8} {s['executed_successfully']:<8} {s['execution_errors']:<8} {s['dynamic_loads']:<8}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory statistics - single file or entire directory"
    )

    # Mutually exclusive: either a file or a directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "trajectory",
        nargs="?",
        help="Path to single trajectory JSON file"
    )
    group.add_argument(
        "--dir",
        help="Directory containing trajectory files (analyzes all)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable format"
    )

    args = parser.parse_args()

    try:
        if args.dir:
            # Directory mode - analyze all trajectories
            traj_dir = Path(args.dir)
            if not traj_dir.exists():
                print(f"Error: Directory not found: {traj_dir}", file=sys.stderr)
                sys.exit(1)

            trajectory_files = list(traj_dir.glob("trajectory_*.json"))
            if not trajectory_files:
                print(f"Error: No trajectory files found in {traj_dir}", file=sys.stderr)
                sys.exit(1)

            all_stats = []
            for traj_file in trajectory_files:
                try:
                    stats = analyze_single_trajectory(traj_file)
                    all_stats.append(stats)
                except Exception as e:
                    print(f"Warning: Failed to analyze {traj_file.name}: {e}", file=sys.stderr)

            aggregate_stats = compute_aggregate_stats(all_stats)

            if args.json:
                output = {
                    "aggregate": aggregate_stats,
                    "individual": all_stats
                }
                print(json.dumps(output, indent=2, ensure_ascii=False))
            else:
                print_aggregate_stats(aggregate_stats, all_stats)

        else:
            # Single file mode
            trajectory_path = Path(args.trajectory)
            if not trajectory_path.exists():
                print(f"Error: File not found: {trajectory_path}", file=sys.stderr)
                sys.exit(1)

            stats = analyze_single_trajectory(trajectory_path)

            if args.json:
                print(json.dumps(stats, indent=2, ensure_ascii=False))
            else:
                print_single_trajectory(stats)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
