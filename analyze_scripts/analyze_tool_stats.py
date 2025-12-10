#!/usr/bin/env python3
"""
Tool Call Statistics Analysis

This script analyzes tool call success rates from trajectory files, categorizing
errors by source (model vs server) and providing detailed breakdowns.

Usage:
    # Basic usage - show all tool call statistics
    python analyze_scripts/analyze_tool_stats.py

    # Filter by model
    python analyze_scripts/analyze_tool_stats.py --model gemini-2.5pro

    # Use custom trajectories directory
    python analyze_scripts/analyze_tool_stats.py --traj-dir /path/to/trajectories

Data Source:
    - Reads from trajectories/*.json files
    - Each trajectory contains tool_calls with status and result

Output:
    - Overall success rates
    - Success rates by model (ranking)
    - Success rates by server (which MCP servers are most reliable)
    - Tools with most failures
    - Dynamic loading statistics
    - Call duration statistics
    - Error analysis (MODEL_ERROR vs SERVER_ERROR breakdown)

Error Categories:
    - MODEL_ERROR: The LLM called the tool incorrectly
        - missing_required_field, wrong_type, invalid_schema, etc.
    - SERVER_ERROR: The MCP server itself has issues
        - rate_limit, quota_exceeded, null_reference, etc.
    - UNKNOWN: Cannot determine the cause

Methodology:
    1. Load all trajectory files and extract tool_calls
    2. For each tool call, check if status != "success" or result contains "isError=True"
    3. Classify errors using regex patterns on error messages
    4. Aggregate statistics by model, server, and tool
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


def print_tool_call_stats(tool_calls: List[Dict[str, Any]], model_filter: str = None):
    """
    Print comprehensive tool call success rate statistics.

    Includes:
    - Overall success rate
    - By model breakdown with ranking
    - By server breakdown
    - Tools with most failures
    - Dynamic loading stats
    - Call duration analysis
    - Error categorization (MODEL_ERROR vs SERVER_ERROR)
    """
    if not tool_calls:
        print("No tool call data found in trajectories.")
        return

    print("\n" + "=" * 80)
    print("TOOL CALL SUCCESS RATES")
    print("=" * 80)

    # Overall stats
    total = len(tool_calls)
    successful = sum(1 for tc in tool_calls if not tc["is_error"])
    success_rate = successful / total * 100 if total > 0 else 0

    print(f"\nOverall: {successful}/{total} successful ({success_rate:.1f}%)")

    # By model
    print("\n--- By Model ---")
    by_model = defaultdict(lambda: {"total": 0, "success": 0})
    for tc in tool_calls:
        by_model[tc["model"]]["total"] += 1
        if not tc["is_error"]:
            by_model[tc["model"]]["success"] += 1

    # Sort by success rate descending
    model_stats = []
    for model, stats in by_model.items():
        rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
        model_stats.append((model, stats["total"], stats["success"], rate))
    model_stats.sort(key=lambda x: x[3], reverse=True)

    print(f"{'Model':<25} {'Success':<12} {'Total':<8} {'Rate':<8}")
    print("-" * 55)
    for model, total, success, rate in model_stats:
        print(f"{model:<25} {success:<12} {total:<8} {rate:.1f}%")

    # By server
    print("\n--- By Server ---")
    by_server = defaultdict(lambda: {"total": 0, "success": 0, "tools": set()})
    for tc in tool_calls:
        by_server[tc["server"]]["total"] += 1
        by_server[tc["server"]]["tools"].add(tc["tool"])
        if not tc["is_error"]:
            by_server[tc["server"]]["success"] += 1

    server_stats = []
    for server, stats in by_server.items():
        rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
        server_stats.append((server, stats["total"], stats["success"], rate, len(stats["tools"])))
    server_stats.sort(key=lambda x: x[1], reverse=True)  # Sort by total calls

    print(f"{'Server':<40} {'Success':<10} {'Total':<8} {'Rate':<8} {'Tools'}")
    print("-" * 75)
    for server, total, success, rate, num_tools in server_stats[:20]:  # Top 20
        server_display = server[:38] if len(server) > 38 else server
        print(f"{server_display:<40} {success:<10} {total:<8} {rate:.1f}%    {num_tools}")

    if len(server_stats) > 20:
        print(f"... and {len(server_stats) - 20} more servers")

    # By tool (top failures)
    print("\n--- Tools with Most Failures ---")
    by_tool = defaultdict(lambda: {"total": 0, "failures": 0, "server": ""})
    for tc in tool_calls:
        key = f"{tc['server']}::{tc['tool']}"
        by_tool[key]["total"] += 1
        by_tool[key]["server"] = tc["server"]
        by_tool[key]["tool"] = tc["tool"]
        if tc["is_error"]:
            by_tool[key]["failures"] += 1

    # Sort by number of failures
    tool_failures = []
    for key, stats in by_tool.items():
        if stats["failures"] > 0:
            rate = (stats["total"] - stats["failures"]) / stats["total"] * 100
            tool_failures.append((stats["server"], stats["tool"], stats["failures"], stats["total"], rate))
    tool_failures.sort(key=lambda x: x[2], reverse=True)

    if tool_failures:
        print(f"{'Server':<30} {'Tool':<25} {'Fails':<8} {'Total':<8} {'Success%'}")
        print("-" * 80)
        for server, tool, failures, total, rate in tool_failures[:15]:
            server_display = server[:28] if len(server) > 28 else server
            tool_display = tool[:23] if len(tool) > 23 else tool
            print(f"{server_display:<30} {tool_display:<25} {failures:<8} {total:<8} {rate:.1f}%")
    else:
        print("No tool failures found!")

    # Dynamic loading stats
    print("\n--- Dynamic Loading ---")
    total_calls = len(tool_calls)
    dynamic_loaded = sum(1 for tc in tool_calls if tc["dynamically_loaded"])
    static_loaded = total_calls - dynamic_loaded
    print(f"Statically loaded (meta-mcp): {static_loaded} ({static_loaded/total_calls*100:.1f}%)")
    print(f"Dynamically loaded:           {dynamic_loaded} ({dynamic_loaded/total_calls*100:.1f}%)")

    # Average duration
    print("\n--- Call Durations ---")
    durations = [tc["duration_seconds"] for tc in tool_calls if tc["duration_seconds"] > 0]
    if durations:
        print(f"Average duration: {statistics.mean(durations):.2f}s")
        print(f"Median duration:  {statistics.median(durations):.2f}s")
        print(f"Max duration:     {max(durations):.2f}s")

    # Error categorization
    print_error_analysis(tool_calls)


def print_error_analysis(tool_calls: List[Dict[str, Any]]):
    """
    Print detailed error analysis categorizing errors by source.

    This helps distinguish between:
    - Model errors (LLM called the tool incorrectly)
    - Server errors (MCP server issues like rate limits, bugs)
    """
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS (Model vs Server Issues)")
    print("=" * 80)

    errors = [tc for tc in tool_calls if tc["is_error"]]
    if not errors:
        print("\nNo errors found!")
        print("=" * 80)
        return

    total_errors = len(errors)
    model_errors = [e for e in errors if e["error_category"] == "MODEL_ERROR"]
    server_errors = [e for e in errors if e["error_category"] == "SERVER_ERROR"]
    unknown_errors = [e for e in errors if e["error_category"] == "UNKNOWN"]

    print(f"\nTotal Errors: {total_errors}")
    print(f"  MODEL_ERROR  (LLM called incorrectly): {len(model_errors):4d} ({len(model_errors)/total_errors*100:.1f}%)")
    print(f"  SERVER_ERROR (Server/API issues):      {len(server_errors):4d} ({len(server_errors)/total_errors*100:.1f}%)")
    print(f"  UNKNOWN      (Cannot determine):       {len(unknown_errors):4d} ({len(unknown_errors)/total_errors*100:.1f}%)")

    # Breakdown by subcategory
    print("\n--- Error Subcategories ---")
    by_subcategory = defaultdict(int)
    for e in errors:
        key = f"{e['error_category']}/{e['error_subcategory']}"
        by_subcategory[key] += 1

    sorted_subcats = sorted(by_subcategory.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Category/Subcategory':<45} {'Count':<8} {'%':<8}")
    print("-" * 65)
    for subcat, count in sorted_subcats:
        print(f"{subcat:<45} {count:<8} {count/total_errors*100:.1f}%")

    # Model errors by model (which model makes most mistakes)
    print("\n--- Model Error Rate by LLM ---")
    print("(How often each model calls tools incorrectly)")
    by_model_errors = defaultdict(lambda: {"total_calls": 0, "model_errors": 0})
    for tc in tool_calls:
        by_model_errors[tc["model"]]["total_calls"] += 1
        if tc["error_category"] == "MODEL_ERROR":
            by_model_errors[tc["model"]]["model_errors"] += 1

    model_error_rates = []
    for model, stats in by_model_errors.items():
        rate = stats["model_errors"] / stats["total_calls"] * 100 if stats["total_calls"] > 0 else 0
        model_error_rates.append((model, stats["model_errors"], stats["total_calls"], rate))
    model_error_rates.sort(key=lambda x: x[3])  # Sort by rate ascending (best first)

    print(f"{'Model':<25} {'Model Errors':<15} {'Total Calls':<12} {'Error Rate'}")
    print("-" * 65)
    for model, model_errs, total, rate in model_error_rates:
        print(f"{model:<25} {model_errs:<15} {total:<12} {rate:.2f}%")

    # Server errors by server (which servers are most unreliable)
    print("\n--- Server Error Rate by MCP Server ---")
    print("(How often each server has issues - NOT the model's fault)")
    by_server_errors = defaultdict(lambda: {"total_calls": 0, "server_errors": 0})
    for tc in tool_calls:
        by_server_errors[tc["server"]]["total_calls"] += 1
        if tc["error_category"] == "SERVER_ERROR":
            by_server_errors[tc["server"]]["server_errors"] += 1

    server_error_rates = []
    for server, stats in by_server_errors.items():
        if stats["server_errors"] > 0:  # Only show servers with errors
            rate = stats["server_errors"] / stats["total_calls"] * 100
            server_error_rates.append((server, stats["server_errors"], stats["total_calls"], rate))
    server_error_rates.sort(key=lambda x: x[1], reverse=True)  # Sort by count

    print(f"{'Server':<40} {'Errors':<10} {'Total':<8} {'Rate'}")
    print("-" * 70)
    for server, errs, total, rate in server_error_rates[:15]:
        server_display = server[:38] if len(server) > 38 else server
        print(f"{server_display:<40} {errs:<10} {total:<8} {rate:.1f}%")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tool call success rates from trajectory files"
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

    print_tool_call_stats(tool_calls, args.model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
