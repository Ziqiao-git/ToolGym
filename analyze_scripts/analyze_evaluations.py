#!/usr/bin/env python3
"""
Analyze Evaluation Results - Main Entry Point

This is the main analysis script that provides a unified interface to all
MCP evaluation analysis functionality. Individual analysis modules can also
be run standalone.

Usage:
    # Show overall summary across all models
    python analyze_scripts/analyze_evaluations.py

    # Compare specific models
    python analyze_scripts/analyze_evaluations.py --models claude-3.5 Gemini-3pro

    # Show detailed per-query analysis
    python analyze_scripts/analyze_evaluations.py --detailed

    # Export to CSV
    python analyze_scripts/analyze_evaluations.py --export results.csv

    # Filter by judge model
    python analyze_scripts/analyze_evaluations.py --judge gpt4omini

    # Show tool/server success rates from trajectories
    python analyze_scripts/analyze_evaluations.py --tool-stats

    # Show tool stats for specific model
    python analyze_scripts/analyze_evaluations.py --tool-stats --models gemini-2.5pro

    # Show weighted scores (penalizes no-tool trajectories)
    python analyze_scripts/analyze_evaluations.py --weighted

    # Custom weights for weighted analysis
    python analyze_scripts/analyze_evaluations.py --weighted --fa-weight 0.6 --step-weight 0.4 --no-tool-penalty 0.3

    # Show auto-correction analysis (model makes error, retries, succeeds)
    python analyze_scripts/analyze_evaluations.py --auto-correction

    # Show retry behavior analysis
    python analyze_scripts/analyze_evaluations.py --retry-analysis

See also:
    - analyze_tool_stats.py: Tool call success rate analysis (standalone)
    - analyze_retry.py: Retry behavior analysis (standalone)
    - analyze_auto_correction.py: Auto-correction analysis (standalone)
    - analyze_weighted.py: Weighted score analysis (standalone)
    - data_loaders.py: Shared data loading functions
"""
from __future__ import annotations

import sys
import argparse
import statistics
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

# Import shared data loading functions
from data_loaders import (
    PROJECT_ROOT,
    EVALUATION_DIR,
    TRAJECTORIES_DIR,
    load_summary_files,
    load_individual_evals,
    load_trajectory_data,
)

# Import analysis modules
from analyze_tool_stats import print_tool_call_stats
from analyze_retry import analyze_retry_behavior
from analyze_auto_correction import analyze_auto_correction
from analyze_weighted import print_weighted_score_analysis


def print_model_comparison(summaries: List[Dict[str, Any]], judge_filter: str = None):
    """Print comparison table of models."""
    # Filter to overall summaries only
    overall = [s for s in summaries if s["is_overall"]]

    if judge_filter:
        overall = [s for s in overall if s["judge"] == judge_filter]

    if not overall:
        print("No overall summaries found.")
        return

    # Group by model
    by_model = defaultdict(list)
    for s in overall:
        by_model[s["model"]].append(s)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Overall Scores)")
    print("=" * 80)

    # Find unique judges (filter out empty ones)
    all_judges = sorted(set(s["judge"] for s in overall if s["judge"]))

    if not all_judges:
        print("No valid judge data found.")
        return

    # Header
    header = f"{'Model':<20}"
    for judge in all_judges:
        header += f" | {judge[:12]:<12} (FA/Step)"
    print(header)
    print("-" * 80)

    # Sort models by best final answer score across judges
    model_scores = []
    for model, model_summaries in by_model.items():
        valid_summaries = [s for s in model_summaries if s["judge"]]
        if not valid_summaries:
            continue
        # Use max score for ranking (best performance)
        best_fa = max(
            s["data"].get("avg_final_answer_score", 0)
            for s in valid_summaries
        )
        model_scores.append((model, best_fa, valid_summaries))

    model_scores.sort(key=lambda x: x[1], reverse=True)

    for model, _, model_summaries in model_scores:
        row = f"{model:<20}"
        for judge in all_judges:
            judge_summary = next((s for s in model_summaries if s["judge"] == judge), None)
            if judge_summary:
                fa = judge_summary["data"].get("avg_final_answer_score", 0)
                step = judge_summary["data"].get("avg_step_score", 0)
                row += f" | {fa:.3f}/{step:.3f}     "
            else:
                row += f" | {'N/A':<18}"
        print(row)

    print("=" * 80)


def print_pass_breakdown(summaries: List[Dict[str, Any]], model_filter: str = None, judge_filter: str = None):
    """Print pass@1, pass@2, pass@3 breakdown."""
    # Filter to pass-specific summaries
    pass_summaries = [s for s in summaries if not s["is_overall"] and s["pass_number"]]

    if model_filter:
        pass_summaries = [s for s in pass_summaries if s["model"] == model_filter]
    if judge_filter:
        pass_summaries = [s for s in pass_summaries if s["judge"] == judge_filter]

    if not pass_summaries:
        print("No pass-specific summaries found.")
        return

    print("\n" + "=" * 80)
    print("PASS BREAKDOWN (by Model)")
    print("=" * 80)

    # Group by model and judge
    grouped = defaultdict(lambda: defaultdict(dict))
    for s in pass_summaries:
        grouped[s["model"]][s["judge"]][s["pass_number"]] = s["data"]

    for model in sorted(grouped.keys()):
        print(f"\n{model}:")
        for judge in sorted(grouped[model].keys()):
            print(f"  Judge: {judge}")
            passes = grouped[model][judge]
            for pass_num in sorted(passes.keys()):
                data = passes[pass_num]
                fa = data.get("avg_final_answer_score", 0)
                step = data.get("avg_step_score", 0)
                total = data.get("total_evaluated", 0)
                errors = data.get("errors", 0)
                print(f"    pass@{pass_num}: FA={fa:.3f}, Step={step:.3f}, n={total}, errors={errors}")


def print_detailed_query_analysis(evals: List[Dict[str, Any]], top_n: int = 10):
    """Show detailed per-query analysis including best/worst performing."""
    if not evals:
        print("No individual evaluations found.")
        return

    print("\n" + "=" * 80)
    print("DETAILED QUERY ANALYSIS")
    print("=" * 80)

    # Calculate scores per query
    query_scores = []
    for e in evals:
        data = e["data"]
        fa_eval = data.get("final_answer_evaluation", {})
        step_eval = data.get("step_by_step_evaluation", {})

        fa_score = fa_eval.get("final_answer_score", 0)
        step_score = step_eval.get("average_step_score", 0)

        query_scores.append({
            "uuid": e["uuid"],
            "model": e["model"],
            "judge": e["judge"],
            "pass": e["pass_number"],
            "final_answer_score": fa_score,
            "step_score": step_score,
            "combined": (fa_score + step_score) / 2,
            "query": data.get("query", "")[:100],
        })

    # Sort by combined score
    query_scores.sort(key=lambda x: x["combined"], reverse=True)

    print(f"\nTop {top_n} Best Performing Queries:")
    print("-" * 60)
    for i, q in enumerate(query_scores[:top_n], 1):
        print(f"{i}. [{q['model']}/pass@{q['pass']}] FA={q['final_answer_score']:.2f} Step={q['step_score']:.2f}")
        print(f"   UUID: {q['uuid']}")
        print(f"   Query: {q['query']}...")
        print()

    print(f"\nBottom {top_n} Worst Performing Queries:")
    print("-" * 60)
    for i, q in enumerate(query_scores[-top_n:], 1):
        print(f"{i}. [{q['model']}/pass@{q['pass']}] FA={q['final_answer_score']:.2f} Step={q['step_score']:.2f}")
        print(f"   UUID: {q['uuid']}")
        print(f"   Query: {q['query']}...")
        print()


def print_score_distribution(evals: List[Dict[str, Any]]):
    """Show score distribution statistics."""
    if not evals:
        return

    fa_scores = []
    step_scores = []

    for e in evals:
        data = e["data"]
        fa_eval = data.get("final_answer_evaluation", {})
        step_eval = data.get("step_by_step_evaluation", {})

        fa = fa_eval.get("final_answer_score")
        step = step_eval.get("average_step_score")

        if fa is not None:
            fa_scores.append(fa)
        if step is not None:
            step_scores.append(step)

    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTIONS")
    print("=" * 80)

    if fa_scores:
        print(f"\nFinal Answer Scores (n={len(fa_scores)}):")
        print(f"  Mean:   {statistics.mean(fa_scores):.3f}")
        print(f"  Median: {statistics.median(fa_scores):.3f}")
        print(f"  Stdev:  {statistics.stdev(fa_scores):.3f}" if len(fa_scores) > 1 else "")
        print(f"  Min:    {min(fa_scores):.3f}")
        print(f"  Max:    {max(fa_scores):.3f}")

        # Histogram buckets
        buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        counts = [0] * (len(buckets) - 1)
        for score in fa_scores:
            for i in range(len(buckets) - 1):
                if buckets[i] <= score < buckets[i + 1] or (i == len(buckets) - 2 and score == buckets[i + 1]):
                    counts[i] += 1
                    break

        print(f"\n  Distribution:")
        for i in range(len(buckets) - 1):
            bar = "#" * int(counts[i] / max(counts) * 20) if max(counts) > 0 else ""
            print(f"    {buckets[i]:.1f}-{buckets[i+1]:.1f}: {counts[i]:4d} {bar}")

    if step_scores:
        print(f"\nStep Scores (n={len(step_scores)}):")
        print(f"  Mean:   {statistics.mean(step_scores):.3f}")
        print(f"  Median: {statistics.median(step_scores):.3f}")
        print(f"  Stdev:  {statistics.stdev(step_scores):.3f}" if len(step_scores) > 1 else "")
        print(f"  Min:    {min(step_scores):.3f}")
        print(f"  Max:    {max(step_scores):.3f}")


def export_to_csv(summaries: List[Dict[str, Any]], evals: List[Dict[str, Any]], output_path: Path):
    """Export evaluation data to CSV."""
    import csv

    # Export summaries
    summary_path = output_path.with_suffix(".summaries.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "judge", "pass_number", "is_overall", "total_evaluated",
                        "errors", "avg_final_answer_score", "avg_step_score"])
        for s in summaries:
            writer.writerow([
                s["model"],
                s["judge"],
                s["pass_number"] or "overall",
                s["is_overall"],
                s["data"].get("total_evaluated", 0),
                s["data"].get("errors", 0),
                s["data"].get("avg_final_answer_score", 0),
                s["data"].get("avg_step_score", 0),
            ])
    print(f"Summaries exported to: {summary_path}")

    # Export individual evals
    evals_path = output_path.with_suffix(".evals.csv")
    with open(evals_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["uuid", "model", "judge", "pass_number", "final_answer_score",
                        "step_score", "total_steps", "query_preview"])
        for e in evals:
            data = e["data"]
            fa_eval = data.get("final_answer_evaluation", {})
            step_eval = data.get("step_by_step_evaluation", {})
            writer.writerow([
                e["uuid"],
                e["model"],
                e["judge"],
                e["pass_number"],
                fa_eval.get("final_answer_score", 0),
                step_eval.get("average_step_score", 0),
                step_eval.get("total_steps", 0),
                data.get("query", "")[:100],
            ])
    print(f"Evaluations exported to: {evals_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results from the evaluation/ folder"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=EVALUATION_DIR,
        help=f"Evaluation directory (default: {EVALUATION_DIR})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Filter to specific models",
    )
    parser.add_argument(
        "--judge",
        help="Filter to specific judge (e.g., gpt4omini, deepseekv32)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-query analysis",
    )
    parser.add_argument(
        "--passes",
        action="store_true",
        help="Show pass@1/2/3 breakdown",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Show score distribution statistics",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export results to CSV files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top/bottom queries to show in detailed mode",
    )
    parser.add_argument(
        "--tool-stats",
        action="store_true",
        help="Show tool/server success rates from trajectory files",
    )
    parser.add_argument(
        "--retry-analysis",
        action="store_true",
        help="Analyze tool retry behavior (do models retry failed tools or give up?)",
    )
    parser.add_argument(
        "--auto-correction",
        action="store_true",
        help="Analyze auto-correction behavior (model makes error, retries, and succeeds)",
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=TRAJECTORIES_DIR,
        help=f"Trajectories directory (default: {TRAJECTORIES_DIR})",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Show weighted score analysis (penalizes no-tool trajectories)",
    )
    parser.add_argument(
        "--fa-weight",
        type=float,
        default=0.5,
        help="Weight for final answer score in weighted analysis (default: 0.5)",
    )
    parser.add_argument(
        "--step-weight",
        type=float,
        default=0.5,
        help="Weight for step score in weighted analysis (default: 0.5)",
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

    # Load summaries
    summaries = load_summary_files(args.eval_dir)
    print(f"Found {len(summaries)} summary files")

    # Filter by model if specified
    if args.models:
        summaries = [s for s in summaries if s["model"] in args.models]
        print(f"Filtered to {len(summaries)} summaries for models: {args.models}")

    # Always show model comparison
    print_model_comparison(summaries, args.judge)

    # Optional: show pass breakdown
    if args.passes:
        print_pass_breakdown(
            summaries,
            model_filter=args.models[0] if args.models and len(args.models) == 1 else None,
            judge_filter=args.judge,
        )

    # Optional: detailed analysis
    if args.detailed or args.distribution:
        evals = load_individual_evals(
            args.eval_dir,
            model=args.models[0] if args.models and len(args.models) == 1 else None,
            judge=args.judge,
        )
        print(f"Loaded {len(evals)} individual evaluation files")

        if args.detailed:
            print_detailed_query_analysis(evals, args.top_n)

        if args.distribution:
            print_score_distribution(evals)

    # Export
    if args.export:
        evals = load_individual_evals(args.eval_dir)
        export_to_csv(summaries, evals, args.export)

    # Tool call statistics, retry analysis, and auto-correction analysis
    if args.tool_stats or args.retry_analysis or args.auto_correction:
        if not args.traj_dir.exists():
            print(f"Error: Trajectories directory not found: {args.traj_dir}")
            return 1

        model_filter = args.models[0] if args.models and len(args.models) == 1 else None
        print(f"\nLoading trajectory data from: {args.traj_dir}")
        tool_calls, trajectories = load_trajectory_data(args.traj_dir, model_filter)
        print(f"Found {len(tool_calls)} tool calls in {len(trajectories)} trajectories")

        if args.tool_stats:
            print_tool_call_stats(tool_calls)

        if args.retry_analysis:
            analyze_retry_behavior(trajectories)

        if args.auto_correction:
            analyze_auto_correction(trajectories)

    # Weighted score analysis (penalizes no-tool trajectories)
    if args.weighted:
        evals = load_individual_evals(
            args.eval_dir,
            model=args.models[0] if args.models and len(args.models) == 1 else None,
            judge=args.judge,
        )
        print(f"Loaded {len(evals)} individual evaluation files for weighted analysis")
        print_weighted_score_analysis(
            evals,
            fa_weight=args.fa_weight,
            step_weight=args.step_weight,
            no_tool_penalty=args.no_tool_penalty,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
