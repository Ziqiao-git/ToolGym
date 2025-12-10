#!/usr/bin/env python3
"""
Auto-Correction Analysis

This script analyzes auto-correction behavior: when a model makes a tool error
and then successfully corrects it on a subsequent attempt.

Usage:
    # Basic usage - show auto-correction analysis
    python analyze_scripts/analyze_auto_correction.py

    # Filter by model
    python analyze_scripts/analyze_auto_correction.py --model gemini-2.5pro

    # Use custom trajectories directory
    python analyze_scripts/analyze_auto_correction.py --traj-dir /path/to/trajectories

Data Source:
    - Reads from trajectories/*.json files
    - Analyzes tool_sequence in each trajectory

Output:
    - Auto-correction rate by model (ranking)
    - Tools most often successfully corrected
    - Error types most often corrected
    - Example correction sequences

Definition:
    Auto-correction = A sequence where:
    1. Model calls tool T and gets an error
    2. Model calls tool T again (possibly multiple times)
    3. Eventually tool T succeeds

    This measures a model's ability to learn from errors and fix its mistakes.

Methodology:
    1. For each trajectory, iterate through tool_sequence
    2. When an error is found on tool T:
       - Look ahead for subsequent calls to the same tool T
       - Track if any of these retry calls succeed
       - Record the number of attempts needed
    3. Calculate correction rates:
       - Correction Rate = Successful corrections / Total retry attempts
    4. Break down by model, tool, and error type

Key Metrics:
    - Errors: Total tool call errors encountered by the model
    - Retried: Number of errors followed by a retry of the same tool
    - Corrected: Number of retries that eventually succeeded
    - Rate: Correction success rate (Corrected / Retried)
    - Avg Attempts: Average number of retry attempts needed to correct
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


def analyze_auto_correction(trajectories: List[Dict[str, Any]]) -> None:
    """
    Analyze auto-correction behavior: when a model makes a tool error and then
    successfully corrects it on a subsequent attempt.

    Auto-correction = Error on tool call → Retry same tool → Success
    """
    print("\n" + "=" * 80)
    print("AUTO-CORRECTION ANALYSIS")
    print("(Model makes error, then successfully retries the same tool)")
    print("=" * 80)

    if not trajectories:
        print("No trajectory data found.")
        return

    # Track per-model auto-correction stats
    by_model = defaultdict(lambda: {
        "total_errors": 0,
        "retry_attempts": 0,  # Errors followed by retry of same tool
        "successful_corrections": 0,  # Retry succeeded
        "failed_corrections": 0,  # Retry also failed
        "correction_sequences": [],  # Track attempts needed to correct
        "tools_corrected": defaultdict(lambda: {"attempts": 0, "successes": 0}),
        "error_types_corrected": defaultdict(lambda: {"attempts": 0, "successes": 0}),
    })

    # Detailed correction examples for reporting
    correction_examples = []

    for traj in trajectories:
        model = traj["model"]
        sequence = traj["tool_sequence"]

        if len(sequence) < 2:
            continue

        i = 0
        while i < len(sequence):
            tc = sequence[i]

            if tc["is_error"]:
                by_model[model]["total_errors"] += 1
                tool_key = f"{tc['server']}::{tc['tool']}"
                error_type = tc.get("error_category", "UNKNOWN")

                # Look ahead for retry attempts on the same tool
                retry_count = 0
                success_found = False
                j = i + 1

                while j < len(sequence):
                    next_tc = sequence[j]
                    next_tool_key = f"{next_tc['server']}::{next_tc['tool']}"

                    if next_tool_key == tool_key:
                        retry_count += 1
                        if not next_tc["is_error"]:
                            # Successful correction!
                            success_found = True
                            by_model[model]["successful_corrections"] += 1
                            by_model[model]["correction_sequences"].append(retry_count)
                            by_model[model]["tools_corrected"][tool_key]["successes"] += 1
                            by_model[model]["error_types_corrected"][error_type]["successes"] += 1

                            # Record example
                            if len(correction_examples) < 50:
                                correction_examples.append({
                                    "model": model,
                                    "tool": tool_key,
                                    "error_type": error_type,
                                    "attempts_to_correct": retry_count,
                                    "query_uuid": traj["query_uuid"][:20] if traj.get("query_uuid") else "N/A",
                                })
                            break
                        else:
                            # Still failing, continue looking
                            pass
                    else:
                        # Switched to different tool, correction attempt ended
                        break
                    j += 1

                if retry_count > 0:
                    by_model[model]["retry_attempts"] += 1
                    by_model[model]["tools_corrected"][tool_key]["attempts"] += 1
                    by_model[model]["error_types_corrected"][error_type]["attempts"] += 1

                    if not success_found:
                        by_model[model]["failed_corrections"] += 1

                # Skip past the correction sequence we just analyzed
                i = j if retry_count > 0 else i + 1
            else:
                i += 1

    # Print summary by model
    print("\n--- Auto-Correction Rate by Model ---")
    print(f"{'Model':<20} {'Errors':<10} {'Retried':<10} {'Corrected':<12} {'Rate':<10} {'Avg Attempts'}")
    print("-" * 80)

    model_stats = []
    for model, stats in by_model.items():
        total_errors = stats["total_errors"]
        retry_attempts = stats["retry_attempts"]
        successful = stats["successful_corrections"]
        correction_rate = successful / retry_attempts * 100 if retry_attempts > 0 else 0
        avg_attempts = statistics.mean(stats["correction_sequences"]) if stats["correction_sequences"] else 0

        model_stats.append((model, total_errors, retry_attempts, successful, correction_rate, avg_attempts))

    # Sort by correction rate descending
    model_stats.sort(key=lambda x: x[4], reverse=True)

    for model, errors, retried, corrected, rate, avg_att in model_stats:
        print(f"{model:<20} {errors:<10} {retried:<10} {corrected:<12} {rate:5.1f}%     {avg_att:.1f}")

    # Ranking table
    print("\n--- Auto-Correction Ranking ---")
    print(f"{'Rank':<6} {'Model':<20} {'Correction Rate':<18} {'Corrected/Retried'}")
    print("-" * 60)
    for rank, (model, errors, retried, corrected, rate, avg_att) in enumerate(model_stats, 1):
        print(f"{rank:<6} {model:<20} {rate:>6.1f}%            {corrected}/{retried}")

    # Tools most often corrected
    print("\n--- Tools Most Often Successfully Corrected ---")
    all_tools = defaultdict(lambda: {"attempts": 0, "successes": 0})
    for model, stats in by_model.items():
        for tool, tool_stats in stats["tools_corrected"].items():
            all_tools[tool]["attempts"] += tool_stats["attempts"]
            all_tools[tool]["successes"] += tool_stats["successes"]

    tool_correction_rates = []
    for tool, stats in all_tools.items():
        if stats["attempts"] >= 2:  # At least 2 attempts to be meaningful
            rate = stats["successes"] / stats["attempts"] * 100
            tool_correction_rates.append((tool, stats["attempts"], stats["successes"], rate))

    tool_correction_rates.sort(key=lambda x: x[2], reverse=True)  # Sort by successes

    print(f"{'Tool':<50} {'Attempts':<10} {'Corrected':<10} {'Rate'}")
    print("-" * 85)
    for tool, attempts, successes, rate in tool_correction_rates[:15]:
        tool_display = tool[:48] if len(tool) > 48 else tool
        print(f"{tool_display:<50} {attempts:<10} {successes:<10} {rate:.1f}%")

    # Error types most often corrected
    print("\n--- Error Types Most Often Corrected ---")
    all_error_types = defaultdict(lambda: {"attempts": 0, "successes": 0})
    for model, stats in by_model.items():
        for err_type, err_stats in stats["error_types_corrected"].items():
            all_error_types[err_type]["attempts"] += err_stats["attempts"]
            all_error_types[err_type]["successes"] += err_stats["successes"]

    error_type_rates = []
    for err_type, stats in all_error_types.items():
        if stats["attempts"] > 0:
            rate = stats["successes"] / stats["attempts"] * 100
            error_type_rates.append((err_type, stats["attempts"], stats["successes"], rate))

    error_type_rates.sort(key=lambda x: x[3], reverse=True)  # Sort by rate

    print(f"{'Error Type':<20} {'Retry Attempts':<15} {'Corrected':<12} {'Correction Rate'}")
    print("-" * 65)
    for err_type, attempts, successes, rate in error_type_rates:
        print(f"{err_type:<20} {attempts:<15} {successes:<12} {rate:.1f}%")

    # Show some correction examples
    if correction_examples:
        print("\n--- Recent Correction Examples ---")
        print(f"{'Model':<15} {'Tool':<35} {'Error Type':<15} {'Attempts':<10} {'Query'}")
        print("-" * 90)
        for ex in correction_examples[:10]:
            tool_short = ex["tool"].split("::")[-1][:33] if "::" in ex["tool"] else ex["tool"][:33]
            print(f"{ex['model']:<15} {tool_short:<35} {ex['error_type']:<15} {ex['attempts_to_correct']:<10} {ex['query_uuid']}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze auto-correction behavior (model makes error, retries, succeeds)"
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

    analyze_auto_correction(trajectories)

    return 0


if __name__ == "__main__":
    sys.exit(main())
