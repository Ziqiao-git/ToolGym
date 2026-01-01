#!/usr/bin/env python3
"""
Analyze constraint verification results from goal-oriented evaluation files

This script analyzes constraint satisfaction across different models and judge models,
focusing on specific constraint types like SEQUENCE_ORDER, SERVER_DIVERSITY, RESPONSE_CONTENT, etc.

Usage:
    # Analyze all constraints
    python analyze_scripts/analyze_constraints.py

    # Analyze specific constraint types
    python analyze_scripts/analyze_constraints.py --constraints SEQUENCE_ORDER SERVER_DIVERSITY

    # Group by model
    python analyze_scripts/analyze_constraints.py --by-model

    # Filter by specific model
    python analyze_scripts/analyze_constraints.py --model deepseek-v3.2
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import statistics


DEFAULT_EVAL_DIR = Path(__file__).parent.parent / "evaluation" / "goaloriented"


def parse_directory_name(dir_name: str) -> tuple:
    """Parse directory name to extract model name and judge model name"""
    parts = dir_name.split('_by_')
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def load_eval_files(eval_dir: Path, model_filter: str = None) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Load all evaluation files grouped by model and judge model

    Returns:
        Dict[model_name][judge_model] = [eval_data, ...]
    """
    data = defaultdict(lambda: defaultdict(list))

    if not eval_dir.exists():
        print(f"Directory does not exist: {eval_dir}")
        return data

    # Iterate through all subdirectories
    for model_dir in eval_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name, judge_model = parse_directory_name(model_dir.name)
        if not model_name or not judge_model:
            continue

        # Apply model filter if specified
        if model_filter and model_filter.lower() not in model_name.lower():
            continue

        # Load all JSON files in this directory
        json_files = list(model_dir.glob("eval_trajectory_*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                    data[model_name][judge_model].append(eval_data)
            except Exception as e:
                print(f"  Error loading {json_file}: {e}")

    return dict(data)


def analyze_constraints(eval_files: List[Dict], constraint_types: List[str] = None) -> Dict[str, Any]:
    """
    Analyze constraint satisfaction statistics

    Args:
        eval_files: List of evaluation data dictionaries
        constraint_types: Optional list of specific constraint types to analyze

    Returns:
        Statistics dictionary
    """
    stats = {
        "total_files": len(eval_files),
        "files_with_constraints": 0,
        "constraint_stats": defaultdict(lambda: {
            "total": 0,
            "satisfied": 0,
            "violated": 0,
            "llm_required": 0,
            "satisfaction_rate": 0.0
        }),
        "overall_satisfaction_rates": [],
        "static_satisfaction_rates": [],
    }

    for eval_data in eval_files:
        constraint_verification = eval_data.get("trajectory_metrics", {}).get("constraint_verification", {})

        if not constraint_verification or constraint_verification.get("total_constraints", 0) == 0:
            continue

        stats["files_with_constraints"] += 1

        # Overall satisfaction rate
        verifications = constraint_verification.get("verifications", [])
        if verifications:
            # Calculate satisfaction rate (excluding LLM-required constraints)
            static_verifiable = [v for v in verifications
                               if v.get("satisfied") is not None]
            if static_verifiable:
                satisfied_count = sum(1 for v in static_verifiable if v.get("satisfied"))
                rate = satisfied_count / len(static_verifiable)
                stats["overall_satisfaction_rates"].append(rate)

        # Static satisfaction rate from the field
        static_rate = constraint_verification.get("static_satisfaction_rate")
        if static_rate is not None:
            stats["static_satisfaction_rates"].append(static_rate)

        # Per-constraint-type statistics
        for verification in verifications:
            constraint_type = verification.get("constraint_type", "UNKNOWN")

            # Filter by constraint types if specified
            if constraint_types and constraint_type not in constraint_types:
                continue

            c_stats = stats["constraint_stats"][constraint_type]
            c_stats["total"] += 1

            satisfied = verification.get("satisfied")
            if satisfied is None:
                # LLM-required constraint
                c_stats["llm_required"] += 1
            elif satisfied:
                c_stats["satisfied"] += 1
            else:
                c_stats["violated"] += 1

    # Calculate satisfaction rates for each constraint type
    for constraint_type, c_stats in stats["constraint_stats"].items():
        verifiable = c_stats["total"] - c_stats["llm_required"]
        if verifiable > 0:
            c_stats["satisfaction_rate"] = c_stats["satisfied"] / verifiable

    return stats


def print_constraint_analysis(stats: Dict[str, Any], model_name: str = None, judge_model: str = None):
    """Print constraint analysis results"""

    header = "CONSTRAINT SATISFACTION ANALYSIS"
    if model_name:
        header += f" - Model: {model_name}"
    if judge_model:
        header += f" (Judge: {judge_model})"

    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)

    print(f"\nTotal evaluation files: {stats['total_files']}")
    print(f"Files with constraints: {stats['files_with_constraints']}")

    if stats['files_with_constraints'] == 0:
        print("\nNo constraint data found!")
        return

    # Overall statistics
    if stats['overall_satisfaction_rates']:
        print(f"\nOverall Constraint Satisfaction:")
        print(f"  Average satisfaction rate: {statistics.mean(stats['overall_satisfaction_rates']) * 100:.1f}%")
        print(f"  Median satisfaction rate: {statistics.median(stats['overall_satisfaction_rates']) * 100:.1f}%")
        print(f"  Min: {min(stats['overall_satisfaction_rates']) * 100:.1f}%")
        print(f"  Max: {max(stats['overall_satisfaction_rates']) * 100:.1f}%")

    # Per-constraint-type analysis
    print("\n" + "-" * 80)
    print("SATISFACTION BY CONSTRAINT TYPE")
    print("-" * 80)

    constraint_stats = stats['constraint_stats']
    if not constraint_stats:
        print("No constraint type data available")
        return

    # Sort by constraint type name
    sorted_constraints = sorted(constraint_stats.items())

    print(f"\n{'Constraint Type':<25} {'Total':<8} {'Satisfied':<10} {'Violated':<10} {'LLM Req':<10} {'Rate'}")
    print("-" * 80)

    for constraint_type, c_stats in sorted_constraints:
        rate_str = f"{c_stats['satisfaction_rate'] * 100:.1f}%" if c_stats['satisfaction_rate'] > 0 else "N/A"
        print(f"{constraint_type:<25} {c_stats['total']:<8} {c_stats['satisfied']:<10} "
              f"{c_stats['violated']:<10} {c_stats['llm_required']:<10} {rate_str}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze constraint satisfaction from goal-oriented evaluation files"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help=f"Evaluation directory (default: {DEFAULT_EVAL_DIR})",
    )
    parser.add_argument(
        "--model",
        help="Filter to specific model (case-insensitive)",
    )
    parser.add_argument(
        "--by-model",
        action="store_true",
        help="Group results by model",
    )
    parser.add_argument(
        "--constraints",
        nargs="+",
        help="Specific constraint types to analyze (e.g., SEQUENCE_ORDER SERVER_DIVERSITY)",
    )

    args = parser.parse_args()

    if not args.eval_dir.exists():
        print(f"Error: Evaluation directory not found: {args.eval_dir}")
        return 1

    print(f"Loading evaluation data from: {args.eval_dir}")
    data = load_eval_files(args.eval_dir, args.model)

    total_files = sum(len(files) for model_data in data.values()
                     for files in model_data.values())
    print(f"Found {total_files} evaluation files")

    if args.model:
        print(f"Filtered to model: {args.model}")

    if args.constraints:
        print(f"Analyzing constraint types: {', '.join(args.constraints)}")

    if not data:
        print("\nNo evaluation data found!")
        return 1

    if args.by_model:
        # Analyze each model separately
        for model_name in sorted(data.keys()):
            # Combine all judge models for this model
            all_files = []
            for judge_model, files in data[model_name].items():
                all_files.extend(files)

            stats = analyze_constraints(all_files, args.constraints)
            print_constraint_analysis(stats, model_name=model_name)
    else:
        # Analyze all together
        all_files = []
        for model_data in data.values():
            for files in model_data.values():
                all_files.extend(files)

        stats = analyze_constraints(all_files, args.constraints)
        print_constraint_analysis(stats)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
