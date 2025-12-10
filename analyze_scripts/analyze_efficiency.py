"""
Analyze Tool Diversity - Relationship Between Unique Tool Usage and Performance

This script analyzes:
1. Tool diversity (unique tools used) vs performance
2. Separates successful vs failed calls to isolate retry effects
3. Correlation between diverse tool usage and final scores
4. Model behavior patterns in tool selection

Key insight: Total calls can be misleading because failures lead to retries.
Focusing on UNIQUE tools used gives a cleaner signal of model capability.

Usage:
    python analyze_scripts/analyze_efficiency.py
    python analyze_scripts/analyze_efficiency.py --model claude-3.5
    python analyze_scripts/analyze_efficiency.py --export efficiency_report.csv
"""
from __future__ import annotations

import sys
import json
import argparse
import statistics
import re
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TRAJECTORIES_DIR = PROJECT_ROOT / "trajectories"


def load_trajectory_info(traj_dir: Path) -> List[Dict[str, Any]]:
    """Load trajectory information including detailed tool call analysis."""
    trajectories = []

    for traj_file in traj_dir.rglob("trajectory_*.json"):
        try:
            parts = traj_file.relative_to(traj_dir).parts
            if len(parts) < 1:
                continue
            model = parts[0]

            with open(traj_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            execution = data.get("execution", {})
            reasoning_trace = data.get("reasoning_trace", [])

            tool_calls = execution.get("tool_calls", [])

            # Analyze tool calls in detail
            unique_tools = set()
            unique_servers = set()
            successful_calls = 0
            failed_calls = 0
            successful_unique_tools = set()
            failed_unique_tools = set()

            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue

                server = tc.get("server", "unknown")
                tool = tc.get("tool", "unknown")
                tool_key = f"{server}::{tool}"
                result = tc.get("result", "")
                status = tc.get("status", "")

                unique_tools.add(tool_key)
                unique_servers.add(server)

                # Check if this call failed
                is_error = status != "success" or (isinstance(result, str) and "isError=True" in result)

                if is_error:
                    failed_calls += 1
                    failed_unique_tools.add(tool_key)
                else:
                    successful_calls += 1
                    successful_unique_tools.add(tool_key)

            # Check if model gave immediate answer without tools
            has_answer_in_trace = any(
                step.get("type") == "answer"
                for step in reasoning_trace
                if isinstance(step, dict)
            )

            trajectories.append({
                "model": model,
                "query_uuid": metadata.get("query_uuid", ""),
                "pass_number": metadata.get("pass_number", 1),
                "total_tool_calls": len(tool_calls),
                "unique_tools": len(unique_tools),
                "unique_servers": len(unique_servers),
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "successful_unique_tools": len(successful_unique_tools),
                "failed_unique_tools": len(failed_unique_tools),
                "retry_ratio": (len(tool_calls) / len(unique_tools)) if unique_tools else 0,
                "has_immediate_answer": has_answer_in_trace and len(tool_calls) == 0,
                "trajectory_file": str(traj_file),
            })
        except Exception as e:
            print(f"Warning: Could not load {traj_file}: {e}", file=sys.stderr)

    return trajectories


def load_evaluation_scores(eval_dir: Path) -> Dict[tuple, float]:
    """Load evaluation scores keyed by (model, query_uuid, pass_number)."""
    scores = {}

    for model_dir in eval_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        # Use first available judge
        judge_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not judge_dirs:
            continue
        judge_dir = judge_dirs[0]

        for pass_dir in judge_dir.iterdir():
            if not pass_dir.is_dir():
                continue
            pass_num = int(pass_dir.name.split("@")[1]) if "@" in pass_dir.name else 1

            for eval_file in pass_dir.glob("eval_*.json"):
                try:
                    with open(eval_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    uuid = data.get("uuid", "")
                    fa_eval = data.get("final_answer_evaluation", {})
                    score = fa_eval.get("final_answer_score", 0) if isinstance(fa_eval, dict) else 0

                    scores[(model, uuid, pass_num)] = score
                except Exception:
                    pass

    return scores


def analyze_tool_diversity(trajectories: List[Dict], scores: Dict[tuple, float]) -> Dict:
    """Analyze unique tool usage patterns and their relationship to scores."""
    print("\n" + "=" * 80)
    print("TOOL DIVERSITY ANALYSIS")
    print("=" * 80)
    print("(Focus on UNIQUE tools used, not total calls)")

    by_model = defaultdict(lambda: {
        "total_calls": [],
        "unique_tools": [],
        "unique_servers": [],
        "successful_calls": [],
        "failed_calls": [],
        "retry_ratios": [],
        "scores": [],
        "data": [],  # (unique_tools, score, total_calls, successful_calls)
    })

    for traj in trajectories:
        model = traj["model"]
        key = (model, traj["query_uuid"], traj["pass_number"])
        score = scores.get(key, None)

        by_model[model]["total_calls"].append(traj["total_tool_calls"])
        by_model[model]["unique_tools"].append(traj["unique_tools"])
        by_model[model]["unique_servers"].append(traj["unique_servers"])
        by_model[model]["successful_calls"].append(traj["successful_calls"])
        by_model[model]["failed_calls"].append(traj["failed_calls"])
        if traj["retry_ratio"] > 0:
            by_model[model]["retry_ratios"].append(traj["retry_ratio"])

        if score is not None:
            by_model[model]["scores"].append(score)
            by_model[model]["data"].append((
                traj["unique_tools"],
                score,
                traj["total_tool_calls"],
                traj["successful_calls"],
                traj["failed_calls"],
            ))

    # Print summary table
    print(f"\n{'Model':<18} {'Avg Unique':<12} {'Avg Total':<12} {'Retry Ratio':<13} {'Fail Rate':<12} {'Avg Score'}")
    print("-" * 85)

    for model in sorted(by_model.keys()):
        stats = by_model[model]
        avg_unique = statistics.mean(stats["unique_tools"]) if stats["unique_tools"] else 0
        avg_total = statistics.mean(stats["total_calls"]) if stats["total_calls"] else 0
        avg_retry = statistics.mean(stats["retry_ratios"]) if stats["retry_ratios"] else 0
        total_failed = sum(stats["failed_calls"])
        total_all = sum(stats["total_calls"])
        fail_rate = total_failed / total_all * 100 if total_all > 0 else 0
        avg_score = statistics.mean(stats["scores"]) if stats["scores"] else 0

        print(f"{model:<18} {avg_unique:<12.1f} {avg_total:<12.1f} {avg_retry:<13.2f} {fail_rate:<12.1f}% {avg_score:.3f}")

    return by_model


def analyze_diversity_vs_score(by_model: Dict) -> None:
    """Analyze correlation between tool diversity and final score."""
    print("\n" + "=" * 80)
    print("UNIQUE TOOLS vs SCORE CORRELATION")
    print("=" * 80)
    print("(Does using more DIVERSE tools lead to better scores?)")

    for model in sorted(by_model.keys()):
        data = by_model[model]["data"]
        if len(data) < 10:
            continue

        # Bin by unique tool count
        bins = {
            "0 tools": [],
            "1-2 tools": [],
            "3-4 tools": [],
            "5-6 tools": [],
            "7+ tools": [],
        }

        for unique, score, total, success, failed in data:
            if unique == 0:
                bins["0 tools"].append(score)
            elif unique <= 2:
                bins["1-2 tools"].append(score)
            elif unique <= 4:
                bins["3-4 tools"].append(score)
            elif unique <= 6:
                bins["5-6 tools"].append(score)
            else:
                bins["7+ tools"].append(score)

        print(f"\n{model}:")
        print(f"  {'Unique Tools':<15} {'Avg Score':<12} {'N':<8} {'Trend'}")
        print(f"  {'-' * 45}")

        prev_avg = None
        for bin_name, bin_scores in bins.items():
            if bin_scores:
                avg = statistics.mean(bin_scores)
                trend = ""
                if prev_avg is not None:
                    diff = avg - prev_avg
                    if diff > 0.05:
                        trend = "↑ better"
                    elif diff < -0.05:
                        trend = "↓ worse"
                    else:
                        trend = "→ same"
                print(f"  {bin_name:<15} {avg:<12.3f} {len(bin_scores):<8} {trend}")
                prev_avg = avg


def analyze_failures_vs_retries(by_model: Dict) -> None:
    """Analyze whether failures explain the negative correlation."""
    print("\n" + "=" * 80)
    print("FAILURE IMPACT ANALYSIS")
    print("=" * 80)
    print("(Do failures/retries explain why more calls = worse scores?)")

    for model in sorted(by_model.keys()):
        data = by_model[model]["data"]
        if len(data) < 10:
            continue

        # Separate trajectories by failure rate
        no_failures = [(u, s, t, succ, f) for u, s, t, succ, f in data if f == 0]
        with_failures = [(u, s, t, succ, f) for u, s, t, succ, f in data if f > 0]

        print(f"\n{model}:")
        print(f"  {'Category':<25} {'Avg Score':<12} {'Avg Unique':<12} {'Avg Total':<12} {'N'}")
        print(f"  {'-' * 70}")

        if no_failures:
            avg_score = statistics.mean([s for _, s, _, _, _ in no_failures])
            avg_unique = statistics.mean([u for u, _, _, _, _ in no_failures])
            avg_total = statistics.mean([t for _, _, t, _, _ in no_failures])
            print(f"  {'No failures':<25} {avg_score:<12.3f} {avg_unique:<12.1f} {avg_total:<12.1f} {len(no_failures)}")

        if with_failures:
            avg_score = statistics.mean([s for _, s, _, _, _ in with_failures])
            avg_unique = statistics.mean([u for u, _, _, _, _ in with_failures])
            avg_total = statistics.mean([t for _, _, t, _, _ in with_failures])
            avg_failures = statistics.mean([f for _, _, _, _, f in with_failures])
            print(f"  {'With failures (avg {:.1f})'.format(avg_failures):<25} {avg_score:<12.3f} {avg_unique:<12.1f} {avg_total:<12.1f} {len(with_failures)}")

        # Further breakdown: among trajectories WITH failures, compare by unique tools
        if len(with_failures) >= 5:
            low_unique = [(u, s) for u, s, _, _, _ in with_failures if u <= 3]
            high_unique = [(u, s) for u, s, _, _, _ in with_failures if u > 3]

            if low_unique and high_unique:
                print(f"\n  Among failed trajectories:")
                print(f"    1-3 unique tools: avg score = {statistics.mean([s for _, s in low_unique]):.3f} (n={len(low_unique)})")
                print(f"    4+ unique tools:  avg score = {statistics.mean([s for _, s in high_unique]):.3f} (n={len(high_unique)})")


def analyze_successful_calls_only(trajectories: List[Dict], scores: Dict[tuple, float]) -> None:
    """Analyze only successful tool calls to remove retry noise."""
    print("\n" + "=" * 80)
    print("SUCCESSFUL TOOLS ONLY ANALYSIS")
    print("=" * 80)
    print("(Removing failed calls to see true tool diversity impact)")

    by_model = defaultdict(lambda: {"data": []})

    for traj in trajectories:
        model = traj["model"]
        key = (model, traj["query_uuid"], traj["pass_number"])
        score = scores.get(key, None)

        if score is not None and traj["successful_calls"] > 0:
            by_model[model]["data"].append((
                traj["successful_unique_tools"],
                score,
                traj["successful_calls"],
            ))

    for model in sorted(by_model.keys()):
        data = by_model[model]["data"]
        if len(data) < 10:
            continue

        # Bin by successful unique tools
        bins = {
            "1 tool": [],
            "2 tools": [],
            "3 tools": [],
            "4 tools": [],
            "5+ tools": [],
        }

        for unique, score, _ in data:
            if unique == 1:
                bins["1 tool"].append(score)
            elif unique == 2:
                bins["2 tools"].append(score)
            elif unique == 3:
                bins["3 tools"].append(score)
            elif unique == 4:
                bins["4 tools"].append(score)
            else:
                bins["5+ tools"].append(score)

        print(f"\n{model} (successful calls only):")
        print(f"  {'Successful Unique':<18} {'Avg Score':<12} {'N':<8} {'Trend'}")
        print(f"  {'-' * 50}")

        prev_avg = None
        for bin_name, bin_scores in bins.items():
            if bin_scores:
                avg = statistics.mean(bin_scores)
                trend = ""
                if prev_avg is not None:
                    diff = avg - prev_avg
                    if diff > 0.05:
                        trend = "↑ better"
                    elif diff < -0.05:
                        trend = "↓ worse"
                    else:
                        trend = "→ same"
                print(f"  {bin_name:<18} {avg:<12.3f} {len(bin_scores):<8} {trend}")
                prev_avg = avg


def analyze_server_diversity(trajectories: List[Dict], scores: Dict[tuple, float]) -> None:
    """Analyze unique server usage (higher level than tools)."""
    print("\n" + "=" * 80)
    print("SERVER DIVERSITY ANALYSIS")
    print("=" * 80)
    print("(How many different MCP servers does each model use?)")

    by_model = defaultdict(lambda: {"data": []})

    for traj in trajectories:
        model = traj["model"]
        key = (model, traj["query_uuid"], traj["pass_number"])
        score = scores.get(key, None)

        if score is not None:
            by_model[model]["data"].append((
                traj["unique_servers"],
                score,
                traj["unique_tools"],
            ))

    for model in sorted(by_model.keys()):
        data = by_model[model]["data"]
        if len(data) < 10:
            continue

        # Bin by server count
        bins = {
            "0 servers": [],
            "1 server": [],
            "2 servers": [],
            "3 servers": [],
            "4+ servers": [],
        }

        for servers, score, tools in data:
            if servers == 0:
                bins["0 servers"].append((score, tools))
            elif servers == 1:
                bins["1 server"].append((score, tools))
            elif servers == 2:
                bins["2 servers"].append((score, tools))
            elif servers == 3:
                bins["3 servers"].append((score, tools))
            else:
                bins["4+ servers"].append((score, tools))

        print(f"\n{model}:")
        print(f"  {'Servers Used':<15} {'Avg Score':<12} {'Avg Tools':<12} {'N':<8} {'Trend'}")
        print(f"  {'-' * 55}")

        prev_avg = None
        for bin_name, bin_data in bins.items():
            if bin_data:
                avg_score = statistics.mean([s for s, _ in bin_data])
                avg_tools = statistics.mean([t for _, t in bin_data])
                trend = ""
                if prev_avg is not None:
                    diff = avg_score - prev_avg
                    if diff > 0.05:
                        trend = "↑ better"
                    elif diff < -0.05:
                        trend = "↓ worse"
                    else:
                        trend = "→ same"
                print(f"  {bin_name:<15} {avg_score:<12.3f} {avg_tools:<12.1f} {len(bin_data):<8} {trend}")
                prev_avg = avg_score


def print_summary_insights(trajectories: List[Dict], scores: Dict[tuple, float]) -> None:
    """Print high-level summary insights."""
    print("\n" + "=" * 80)
    print("SUMMARY INSIGHTS")
    print("=" * 80)

    # Collect all data
    all_data = []
    for t in trajectories:
        key = (t["model"], t["query_uuid"], t["pass_number"])
        score = scores.get(key)
        if score is not None:
            all_data.append({
                "model": t["model"],
                "unique_tools": t["unique_tools"],
                "total_calls": t["total_tool_calls"],
                "failed_calls": t["failed_calls"],
                "score": score,
            })

    print("\n1. TOOL DIVERSITY vs TOTAL CALLS:")
    # Compare correlations
    if all_data:
        # Unique tools correlation
        low_unique = [d["score"] for d in all_data if 1 <= d["unique_tools"] <= 2]
        high_unique = [d["score"] for d in all_data if d["unique_tools"] >= 5]

        if low_unique and high_unique:
            print(f"   - 1-2 unique tools: avg score = {statistics.mean(low_unique):.3f} (n={len(low_unique)})")
            print(f"   - 5+ unique tools:  avg score = {statistics.mean(high_unique):.3f} (n={len(high_unique)})")

            diff = statistics.mean(high_unique) - statistics.mean(low_unique)
            if diff > 0:
                print(f"   - MORE diverse tools → BETTER scores (+{diff:.3f})")
            else:
                print(f"   - MORE diverse tools → WORSE scores ({diff:.3f})")

    print("\n2. FAILURE IMPACT:")
    if all_data:
        no_fail = [d["score"] for d in all_data if d["failed_calls"] == 0]
        with_fail = [d["score"] for d in all_data if d["failed_calls"] > 0]

        if no_fail and with_fail:
            print(f"   - No failures: avg score = {statistics.mean(no_fail):.3f} (n={len(no_fail)})")
            print(f"   - With failures: avg score = {statistics.mean(with_fail):.3f} (n={len(with_fail)})")

            # Among failures, compare by unique tools
            fail_low = [d["score"] for d in all_data if d["failed_calls"] > 0 and d["unique_tools"] <= 3]
            fail_high = [d["score"] for d in all_data if d["failed_calls"] > 0 and d["unique_tools"] > 3]

            if fail_low and fail_high:
                print(f"\n   Among trajectories WITH failures:")
                print(f"   - Low diversity (1-3 tools): {statistics.mean(fail_low):.3f}")
                print(f"   - High diversity (4+ tools): {statistics.mean(fail_high):.3f}")

    print("\n3. KEY TAKEAWAY:")
    # Calculate the actual pattern
    if all_data:
        # Group by model and calculate diversity impact
        by_model = defaultdict(list)
        for d in all_data:
            by_model[d["model"]].append(d)

        positive_models = []
        negative_models = []

        for model, model_data in by_model.items():
            low = [d["score"] for d in model_data if 1 <= d["unique_tools"] <= 2]
            high = [d["score"] for d in model_data if d["unique_tools"] >= 4]
            if low and high:
                diff = statistics.mean(high) - statistics.mean(low)
                if diff > 0.02:
                    positive_models.append((model, diff))
                elif diff < -0.02:
                    negative_models.append((model, diff))

        if positive_models:
            print(f"   Models where MORE diversity = BETTER scores:")
            for m, d in sorted(positive_models, key=lambda x: x[1], reverse=True):
                print(f"     - {m}: +{d:.3f}")

        if negative_models:
            print(f"   Models where MORE diversity = WORSE scores:")
            for m, d in sorted(negative_models, key=lambda x: x[1]):
                print(f"     - {m}: {d:.3f}")

    print("\n" + "=" * 80)


def export_results(trajectories: List[Dict], scores: Dict[tuple, float], output_path: Path) -> None:
    """Export analysis results to CSV."""
    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "query_uuid", "pass_number", "total_calls", "unique_tools",
            "unique_servers", "successful_calls", "failed_calls", "retry_ratio", "score"
        ])

        for t in trajectories:
            key = (t["model"], t["query_uuid"], t["pass_number"])
            score = scores.get(key, "")

            writer.writerow([
                t["model"],
                t["query_uuid"],
                t["pass_number"],
                t["total_tool_calls"],
                t["unique_tools"],
                t["unique_servers"],
                t["successful_calls"],
                t["failed_calls"],
                f"{t['retry_ratio']:.2f}" if t["retry_ratio"] > 0 else "",
                score,
            ])

    print(f"\nExported results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tool diversity and its relationship to performance"
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=TRAJECTORIES_DIR,
        help=f"Trajectories directory (default: {TRAJECTORIES_DIR})",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=EVALUATION_DIR,
        help=f"Evaluation directory (default: {EVALUATION_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Filter to specific model",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export results to CSV file",
    )

    args = parser.parse_args()

    if not args.traj_dir.exists():
        print(f"Error: Trajectories directory not found: {args.traj_dir}")
        return 1

    if not args.eval_dir.exists():
        print(f"Error: Evaluation directory not found: {args.eval_dir}")
        return 1

    print("=" * 80)
    print("TOOL DIVERSITY ANALYSIS REPORT")
    print("=" * 80)
    print("Focus: UNIQUE tools used (not total calls, which include retries)")

    # Load data
    print(f"\nLoading trajectories from: {args.traj_dir}")
    trajectories = load_trajectory_info(args.traj_dir)
    print(f"Loaded {len(trajectories)} trajectories")

    print(f"\nLoading evaluation scores from: {args.eval_dir}")
    scores = load_evaluation_scores(args.eval_dir)
    print(f"Loaded {len(scores)} evaluation scores")

    # Filter by model if specified
    if args.model:
        trajectories = [t for t in trajectories if t["model"].lower() == args.model.lower()]
        print(f"\nFiltered to {len(trajectories)} trajectories for model: {args.model}")

    # Run analyses
    by_model = analyze_tool_diversity(trajectories, scores)
    analyze_diversity_vs_score(by_model)
    analyze_failures_vs_retries(by_model)
    analyze_successful_calls_only(trajectories, scores)
    analyze_server_diversity(trajectories, scores)
    print_summary_insights(trajectories, scores)

    # Export if requested
    if args.export:
        export_results(trajectories, scores, args.export)

    return 0


if __name__ == "__main__":
    sys.exit(main())
