# analyze_scripts/analyze_weighted.py
#!/usr/bin/env python3
"""
Weighted Score Analysis

This script analyzes evaluation scores with weighted scoring that penalizes
trajectories without tool usage. This prevents models from "cheating" by
answering from internal knowledge without actually using MCP tools.

It also computes Spearman rank correlation between the weighted ranking
and other external benchmark rankings, using overlap only (in canonical name space).

Usage:
    python analyze_scripts/analyze_weighted.py
    python analyze_scripts/analyze_weighted.py --fa-weight 0.6 --step-weight 0.4
    python analyze_scripts/analyze_weighted.py --no-tool-penalty 0.3
    python analyze_scripts/analyze_weighted.py --model claude-3.5 --judge gpt4omini

Notes:
    - Model names in evals are usually internal ids; external benchmarks use display names.
      We canonicalize BOTH sides via analyze_scripts/benchmark_rankings.py to ensure overlap.
"""
from __future__ import annotations

import sys
import argparse
import statistics
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

# Import shared data loading functions
from data_loaders import (
    EVALUATION_DIR,
    load_individual_evals,
)

# Import benchmark rankings + model name canonicalization
from benchmark_rankings import (
    BENCHMARK_RANKINGS,
    get_benchmark_rank_map,
    canonicalize_model_name,
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

    Args:
        final_answer_score: Score from final answer evaluation (0-1)
        step_score: Average score from step-by-step evaluation (0-1)
        total_steps: Number of steps in the trajectory
        actual_tools: List of actual tools used (excluding search_tools)
        fa_weight: Weight for final answer score
        step_weight: Weight for step score
        no_tool_penalty: Multiplier when no tools used (e.g., 0.5 = 50% penalty)

    Returns:
        Dict with weighted_score, components, and penalty info.
    """
    # Filter out search_tools from actual_tools (they don't count as real tool usage)
    real_tools = [t for t in (actual_tools or []) if t.get("tool") != "search_tools"]

    # Base weighted score
    base_score = (fa_weight * final_answer_score) + (step_weight * step_score)

    # Apply penalty if no real tools were used or no steps were recorded
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


def _rankdata_average_ties(values: List[float]) -> List[float]:
    """
    Assign ranks to values using average ranks for ties.

    Higher values are considered better and receive smaller rank numbers
    (rank 1 = best).
    """
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1], reverse=True)

    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_corr_from_lists(x: List[float], y: List[float]) -> float:
    """
    Compute Spearman rank correlation using average ranks for ties.

    This implementation does not rely on external libraries (e.g., scipy).
    """
    if len(x) != len(y) or len(x) < 2:
        return float("nan")

    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)

    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)

    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))

    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def compute_weighted_rank_map(model_stats: List[Tuple]) -> Dict[str, float]:
    """
    Build a rank map from weighted scores.

    Args:
        model_stats:
            (model, avg_fa, avg_weighted, delta, penalized_pct, avg_tools, n)

    Returns:
        Dict[canonical_model_name -> rank], where rank=1 is best.
    """
    sorted_models = sorted(model_stats, key=lambda x: x[2], reverse=True)
    canonical_models = [canonicalize_model_name(m[0]) for m in sorted_models]
    return {m: i + 1 for i, m in enumerate(canonical_models)}


def print_spearman_vs_benchmarks(model_stats: List[Tuple], *, debug: bool = False) -> None:
    """
    Compute and print Spearman correlations between the current weighted ranking
    and other benchmark rankings.

    Only the overlapping set of models is used for each comparison.
    """
    weighted_rank = compute_weighted_rank_map(model_stats)

    print("\n" + "-" * 90)
    print("SPEARMAN CORRELATION: Weighted Ranking vs Other Benchmarks (Overlap Only)")
    print("-" * 90)

    for benchmark_name in BENCHMARK_RANKINGS.keys():
        benchmark_rank = get_benchmark_rank_map(benchmark_name)

        overlap = sorted(set(weighted_rank) & set(benchmark_rank))
        if len(overlap) < 2:
            print(f"{benchmark_name:<12}: overlap too small (n={len(overlap)})")
            if debug:
                print("  weighted models (canonical):", sorted(list(weighted_rank.keys())))
                print("  bench models (canonical):   ", sorted(list(benchmark_rank.keys())))
            continue

        # Use negative ranks so that a better rank (1) corresponds to a larger value
        x = [-weighted_rank[m] for m in overlap]
        y = [-benchmark_rank[m] for m in overlap]

        rho = spearman_corr_from_lists(x, y)
        print(f"{benchmark_name:<12}: rho={rho:+.3f}  (overlap n={len(overlap)})")

    print("-" * 90)


def print_weighted_score_analysis(
    evals: List[Dict[str, Any]],
    fa_weight: float = 0.5,
    step_weight: float = 0.5,
    no_tool_penalty: float = 0.5,
    *,
    debug_overlap: bool = False,
) -> None:
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
    scored_evals: List[Dict[str, Any]] = []
    for e in evals:
        data = e["data"]
        fa_eval = data.get("final_answer_evaluation", {})
        step_eval = data.get("step_by_step_evaluation", {})
        actual_tools = data.get("actual_tools", [])

        fa_score = fa_eval.get("final_answer_score", 0)
        avg_step = step_eval.get("average_step_score", 0)
        total_steps = step_eval.get("total_steps", 0)

        weighted = calculate_weighted_score(
            fa_score,
            avg_step,
            total_steps,
            actual_tools,
            fa_weight=fa_weight,
            step_weight=step_weight,
            no_tool_penalty=no_tool_penalty,
        )

        scored_evals.append(
            {
                "model": e["model"],
                "judge": e["judge"],
                "pass_number": e["pass_number"],
                "uuid": e["uuid"],
                "query": data.get("query", "")[:60],
                **weighted,
            }
        )

    # Summary by model
    print("\n--- Model Comparison (Weighted vs Original) ---")
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for se in scored_evals:
        by_model[se["model"]].append(se)

    print(f"{'Model':<20} {'Orig FA':<10} {'Weighted':<10} {'Δ':<8} {'Penalized':<12} {'Avg Tools'}")
    print("-" * 85)

    model_stats: List[Tuple] = []
    for model, model_evals in sorted(by_model.items()):
        avg_fa = statistics.mean(se["final_answer_score"] for se in model_evals)
        avg_weighted = statistics.mean(se["weighted_score"] for se in model_evals)
        delta = avg_weighted - avg_fa
        penalized_count = sum(1 for se in model_evals if se["penalty_applied"])
        penalized_pct = penalized_count / len(model_evals) * 100
        avg_tools = statistics.mean(se["real_tools_used"] for se in model_evals)

        model_stats.append((model, avg_fa, avg_weighted, delta, penalized_pct, avg_tools, len(model_evals)))
        print(
            f"{model:<20} {avg_fa:<10.3f} {avg_weighted:<10.3f} {delta:+.3f}    "
            f"{penalized_pct:5.1f}%       {avg_tools:.1f}"
        )

    # Ranking change
    print("\n--- Ranking Change (by Weighted Score) ---")
    original_rank = sorted(model_stats, key=lambda x: x[1], reverse=True)
    weighted_rank = sorted(model_stats, key=lambda x: x[2], reverse=True)

    print(f"{'Rank':<6} {'Original (by FA)':<25} {'Weighted':<25}")
    print("-" * 60)
    for i in range(max(len(original_rank), len(weighted_rank))):
        orig = original_rank[i][0] if i < len(original_rank) else "N/A"
        wght = weighted_rank[i][0] if i < len(weighted_rank) else "N/A"
        marker = " <-" if orig != wght else ""
        print(f"{i+1:<6} {orig:<25} {wght:<25}{marker}")

    # Show most affected trajectories (penalized high FA)
    penalized = [se for se in scored_evals if se["penalty_applied"]]
    if penalized:
        print(f"\n--- Trajectories Penalized ({len(penalized)} total) ---")
        print(f"{'Model':<15} {'Orig FA':<10} {'Weighted':<10} {'Steps':<7} {'Tools':<6} {'Query'}")
        print("-" * 90)

        penalized_sorted = sorted(penalized, key=lambda x: x["final_answer_score"], reverse=True)
        for se in penalized_sorted[:15]:
            print(
                f"{se['model']:<15} {se['final_answer_score']:<10.3f} {se['weighted_score']:<10.3f} "
                f"{se['total_steps']:<7} {se['real_tools_used']:<6} {se['query'][:40]}"
            )

    # Non-penalized high performers
    non_penalized = [se for se in scored_evals if not se["penalty_applied"]]
    if non_penalized:
        print(f"\n--- Top Non-Penalized Performers ({len(non_penalized)} total) ---")
        non_penalized_sorted = sorted(non_penalized, key=lambda x: x["weighted_score"], reverse=True)[:10]
        print(f"{'Model':<15} {'Weighted':<10} {'Steps':<7} {'Tools':<6} {'Query'}")
        print("-" * 80)
        for se in non_penalized_sorted:
            print(
                f"{se['model']:<15} {se['weighted_score']:<10.3f} {se['total_steps']:<7} "
                f"{se['real_tools_used']:<6} {se['query'][:40]}"
            )

    # Spearman correlation vs external benchmark rankings (overlap only)
    print_spearman_vs_benchmarks(model_stats, debug=debug_overlap)

    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze weighted scores (penalizes no-tool trajectories) and compare rankings via Spearman."
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
    parser.add_argument(
        "--debug-overlap",
        action="store_true",
        help="Print canonical model lists when overlap is too small.",
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
        debug_overlap=args.debug_overlap,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
