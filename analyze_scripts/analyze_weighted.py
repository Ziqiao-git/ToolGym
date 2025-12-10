#!/usr/bin/env python3
"""
Weighted Score Analysis

This script analyzes evaluation scores with weighted scoring that penalizes
trajectories without tool usage. This prevents models from "cheating" by
answering from internal knowledge without actually using MCP tools.

Usage:
    # Basic usage with default weights
    python analyze_scripts/analyze_weighted.py

    # Custom weights
    python analyze_scripts/analyze_weighted.py --fa-weight 0.6 --step-weight 0.4

    # Stronger penalty for no-tool trajectories
    python analyze_scripts/analyze_weighted.py --no-tool-penalty 0.3

    # Filter by model or judge
    python analyze_scripts/analyze_weighted.py --model claude-3.5 --judge gpt4omini

Data Source:
    - Reads from evaluation/*.json files
    - Uses final_answer_evaluation and step_by_step_evaluation scores
    - Checks actual_tools to determine if real tools were used

Output:
    - Model comparison (original vs weighted scores)
    - Ranking changes between original and weighted scoring
    - Penalized trajectories breakdown
    - Top non-penalized performers

Formula:
    base_score = (FA × fa_weight) + (Step × step_weight)
    weighted_score = base_score × penalty_multiplier

    Where penalty_multiplier = no_tool_penalty if no real tools used, else 1.0

    Default weights:
    - fa_weight: 0.5
    - step_weight: 0.5
    - no_tool_penalty: 0.5 (50% penalty)

Penalty Conditions:
    1. No actual MCP tools used (excluding search_tools which just finds tools)
    2. No reasoning steps recorded

Methodology:
    1. Load all individual evaluation files
    2. For each eval, calculate weighted score:
       - Extract final_answer_score and average_step_score
       - Check actual_tools (filter out search_tools)
       - Apply penalty if no real tools used
    3. Aggregate by model and compare to original rankings
    4. Show which trajectories were most affected by the penalty

Why This Matters:
    In MCP evaluation, we want to measure how well models can USE tools.
    A model that answers correctly from internal knowledge without calling
    any tools is not demonstrating MCP capability. The penalty ensures
    that tool usage is properly valued in the final score.
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
    EVALUATION_DIR,
    load_individual_evals,
)


def calculate_weighted_score(
    final_answer_score: float,
    step_score: float,
    total_steps: int,
    actual_tools: List[Dict[str, str]],
    *,
    fa_weight: float = 0.5,
    step_weight: float = 0.5,
    no_tool_penalty: float = 0.5,
) -> Dict[str, Any]:
    """
    Calculate a weighted combined score that penalizes trajectories without tool usage.

    The goal is to prevent models from "cheating" by answering from internal knowledge
    without actually using MCP tools - which defeats the purpose of MCP evaluation.

    Args:
        final_answer_score: Score from final answer evaluation (0-1)
        step_score: Average score from step-by-step evaluation (0-1)
        total_steps: Number of steps in the trajectory
        actual_tools: List of actual tools used (excluding search_tools)
        fa_weight: Weight for final answer score (default: 0.5)
        step_weight: Weight for step score (default: 0.5)
        no_tool_penalty: Multiplier when no tools used (default: 0.5, i.e., 50% penalty)

    Returns:
        Dict with weighted_score, components, and penalty info
    """
    # Filter out search_tools from actual_tools (they don't count as real tool usage)
    real_tools = [t for t in (actual_tools or []) if t.get("tool") != "search_tools"]

    # Base weighted score
    base_score = (fa_weight * final_answer_score) + (step_weight * step_score)

    # Apply no-tool penalty if no real tools were used
    penalty_applied = False
    penalty_reason = None

    if len(real_tools) == 0:
        weighted_score = base_score * no_tool_penalty
        penalty_applied = True
        penalty_reason = "No actual MCP tools used (excluding search_tools)"
    elif total_steps == 0:
        weighted_score = base_score * no_tool_penalty
        penalty_applied = True
        penalty_reason = "No reasoning steps recorded"
    else:
        weighted_score = base_score

    return {
        "weighted_score": weighted_score,
        "base_score": base_score,
        "final_answer_score": final_answer_score,
        "step_score": step_score,
        "penalty_applied": penalty_applied,
        "penalty_reason": penalty_reason,
        "penalty_multiplier": no_tool_penalty if penalty_applied else 1.0,
        "total_steps": total_steps,
        "real_tools_used": len(real_tools),
    }


def print_weighted_score_analysis(
    evals: List[Dict[str, Any]],
    fa_weight: float = 0.5,
    step_weight: float = 0.5,
    no_tool_penalty: float = 0.5,
):
    """
    Analyze and compare weighted scores vs original scores.

    This shows how the weighted scoring (which penalizes no-tool trajectories)
    changes the rankings compared to raw final_answer_score.
    """
    print("\n" + "=" * 90)
    print("WEIGHTED SCORE ANALYSIS (Penalizes No-Tool Trajectories)")
    print("=" * 90)
    print(f"Weights: FA={fa_weight}, Step={step_weight}, No-Tool Penalty={no_tool_penalty}")
    print(f"Formula: weighted = (FA×{fa_weight} + Step×{step_weight}) × penalty_multiplier")
    print("=" * 90)

    if not evals:
        print("No evaluation data found.")
        return

    # Calculate weighted scores for all evals
    scored_evals = []
    for e in evals:
        data = e["data"]
        fa_eval = data.get("final_answer_evaluation", {})
        step_eval = data.get("step_by_step_evaluation", {})
        actual_tools = data.get("actual_tools", [])

        fa_score = fa_eval.get("final_answer_score", 0)
        avg_step = step_eval.get("average_step_score", 0)
        total_steps = step_eval.get("total_steps", 0)

        weighted = calculate_weighted_score(
            fa_score, avg_step, total_steps, actual_tools,
            fa_weight=fa_weight, step_weight=step_weight, no_tool_penalty=no_tool_penalty
        )

        scored_evals.append({
            "model": e["model"],
            "judge": e["judge"],
            "pass_number": e["pass_number"],
            "uuid": e["uuid"],
            "query": data.get("query", "")[:60],
            **weighted,
        })

    # Summary by model
    print("\n--- Model Comparison (Weighted vs Original) ---")
    by_model = defaultdict(list)
    for se in scored_evals:
        by_model[se["model"]].append(se)

    print(f"{'Model':<20} {'Orig FA':<10} {'Weighted':<10} {'Δ':<8} {'Penalized':<12} {'Avg Tools'}")
    print("-" * 85)

    model_stats = []
    for model, model_evals in sorted(by_model.items()):
        avg_fa = statistics.mean(se["final_answer_score"] for se in model_evals)
        avg_weighted = statistics.mean(se["weighted_score"] for se in model_evals)
        delta = avg_weighted - avg_fa
        penalized_count = sum(1 for se in model_evals if se["penalty_applied"])
        penalized_pct = penalized_count / len(model_evals) * 100
        avg_tools = statistics.mean(se["real_tools_used"] for se in model_evals)

        model_stats.append((model, avg_fa, avg_weighted, delta, penalized_pct, avg_tools, len(model_evals)))
        print(f"{model:<20} {avg_fa:<10.3f} {avg_weighted:<10.3f} {delta:+.3f}    {penalized_pct:5.1f}%       {avg_tools:.1f}")

    # Ranking change
    print("\n--- Ranking Change (by Weighted Score) ---")
    original_rank = sorted(model_stats, key=lambda x: x[1], reverse=True)
    weighted_rank = sorted(model_stats, key=lambda x: x[2], reverse=True)

    print(f"{'Rank':<6} {'Original (by FA)':<25} {'Weighted':<25}")
    print("-" * 60)
    for i in range(max(len(original_rank), len(weighted_rank))):
        orig = original_rank[i][0] if i < len(original_rank) else "N/A"
        wght = weighted_rank[i][0] if i < len(weighted_rank) else "N/A"
        # Highlight if different
        marker = " <-" if orig != wght else ""
        print(f"{i+1:<6} {orig:<25} {wght:<25}{marker}")

    # Show most affected trajectories (biggest score drop from penalty)
    penalized = [se for se in scored_evals if se["penalty_applied"]]
    if penalized:
        print(f"\n--- Trajectories Penalized ({len(penalized)} total) ---")
        print(f"{'Model':<15} {'Orig FA':<10} {'Weighted':<10} {'Steps':<7} {'Tools':<6} {'Query'}")
        print("-" * 90)

        # Sort by original FA score descending (most affected = high original, now penalized)
        penalized_sorted = sorted(penalized, key=lambda x: x["final_answer_score"], reverse=True)
        for se in penalized_sorted[:15]:
            print(f"{se['model']:<15} {se['final_answer_score']:<10.3f} {se['weighted_score']:<10.3f} "
                  f"{se['total_steps']:<7} {se['real_tools_used']:<6} {se['query'][:40]}")

    # Non-penalized high performers
    non_penalized = [se for se in scored_evals if not se["penalty_applied"]]
    if non_penalized:
        print(f"\n--- Top Non-Penalized Performers ({len(non_penalized)} total) ---")
        non_penalized_sorted = sorted(non_penalized, key=lambda x: x["weighted_score"], reverse=True)[:10]
        print(f"{'Model':<15} {'Weighted':<10} {'Steps':<7} {'Tools':<6} {'Query'}")
        print("-" * 80)
        for se in non_penalized_sorted:
            print(f"{se['model']:<15} {se['weighted_score']:<10.3f} {se['total_steps']:<7} "
                  f"{se['real_tools_used']:<6} {se['query'][:40]}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze weighted scores (penalizes no-tool trajectories)"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=EVALUATION_DIR,
        help=f"Evaluation directory (default: {EVALUATION_DIR})",
    )
    parser.add_argument(
        "--model",
        help="Filter to specific model",
    )
    parser.add_argument(
        "--judge",
        help="Filter to specific judge",
    )
    parser.add_argument(
        "--fa-weight",
        type=float,
        default=0.5,
        help="Weight for final answer score (default: 0.5)",
    )
    parser.add_argument(
        "--step-weight",
        type=float,
        default=0.5,
        help="Weight for step score (default: 0.5)",
    )
    parser.add_argument(
        "--no-tool-penalty",
        type=float,
        default=0.5,
        help="Penalty multiplier for no-tool trajectories (default: 0.5 = 50%% penalty)",
    )

    args = parser.parse_args()

    if not args.eval_dir.exists():
        print(f"Error: Evaluation directory not found: {args.eval_dir}")
        return 1

    print(f"Loading evaluation data from: {args.eval_dir}")

    evals = load_individual_evals(args.eval_dir, model=args.model, judge=args.judge)
    print(f"Loaded {len(evals)} individual evaluation files")

    if args.model:
        print(f"Filtered to model: {args.model}")
    if args.judge:
        print(f"Filtered to judge: {args.judge}")

    print_weighted_score_analysis(
        evals,
        fa_weight=args.fa_weight,
        step_weight=args.step_weight,
        no_tool_penalty=args.no_tool_penalty,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
