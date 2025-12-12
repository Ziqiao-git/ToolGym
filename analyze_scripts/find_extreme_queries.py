#!/usr/bin/env python3
"""
Find queries with the lowest and highest average scores across all evaluations.

This script analyzes all evaluation files across all models, judges, and passes
to identify:
1. The 3 queries with the lowest average scores
2. The 3 queries with the highest average scores

Average score is calculated as: (final_answer_score + average_step_score) / 2
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_avg_score(eval_data: Dict) -> float:
    """
    Calculate the average score for an evaluation.

    Average = (final_answer_score + average_step_score) / 2

    Args:
        eval_data: Evaluation data dictionary

    Returns:
        Average score between 0 and 1
    """
    final_answer_score = eval_data.get("final_answer_evaluation", {}).get("final_answer_score", 0)
    step_score = eval_data.get("step_by_step_evaluation", {}).get("average_step_score", 0)

    return (final_answer_score + step_score) / 2


def collect_all_evaluations(eval_dir: Path) -> Dict[str, List[Dict]]:
    """
    Collect all evaluations grouped by query UUID.

    Args:
        eval_dir: Path to evaluation directory

    Returns:
        Dictionary mapping UUID to list of evaluation data
    """
    query_evals = defaultdict(list)

    # Find all evaluation files
    for eval_file in eval_dir.rglob("eval_*.json"):
        if eval_file.name == "_summary.json":
            continue

        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            uuid = data.get("uuid")
            if not uuid:
                print(f"Warning: No UUID in {eval_file}")
                continue

            # Store evaluation with metadata
            query_evals[uuid].append({
                "file": str(eval_file),
                "data": data,
                "avg_score": calculate_avg_score(data)
            })

        except Exception as e:
            print(f"Error reading {eval_file}: {e}")

    return query_evals


def calculate_query_averages(query_evals: Dict[str, List[Dict]]) -> List[Tuple[str, float, Dict]]:
    """
    Calculate average score across all evaluations for each query.

    Args:
        query_evals: Dictionary mapping UUID to list of evaluations

    Returns:
        List of tuples (uuid, avg_score, sample_data) sorted by score
    """
    query_scores = []

    for uuid, evals in query_evals.items():
        # Calculate average score across all evaluations of this query
        avg_score = sum(e["avg_score"] for e in evals) / len(evals)

        # Use the first evaluation as sample data
        sample_data = evals[0]["data"]

        query_scores.append((uuid, avg_score, sample_data, len(evals)))

    return sorted(query_scores, key=lambda x: x[1])


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length characters."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def generate_report(query_evals: Dict[str, List[Dict]], output_file: Path):
    """
    Generate a report with the lowest and highest scoring queries.

    Args:
        query_evals: Dictionary mapping UUID to list of evaluations
        output_file: Path to output JSON file
    """
    # Calculate averages and sort
    sorted_queries = calculate_query_averages(query_evals)

    # Get top 3 and bottom 3
    lowest_3 = sorted_queries[:3]
    highest_3 = sorted_queries[-3:][::-1]  # Reverse to show highest first

    # Build report
    report = {
        "total_unique_queries": len(sorted_queries),
        "total_evaluations": sum(len(evals) for evals in query_evals.values()),
        "lowest_scoring_queries": [],
        "highest_scoring_queries": []
    }

    # Add lowest scoring queries
    for uuid, avg_score, data, num_evals in lowest_3:
        report["lowest_scoring_queries"].append({
            "uuid": uuid,
            "average_score": round(avg_score, 4),
            "num_evaluations": num_evals,
            "query_preview": truncate_text(data.get("query", ""), 300),
            "full_query": data.get("query", ""),
            "final_answer_score": data.get("final_answer_evaluation", {}).get("final_answer_score"),
            "step_score": data.get("step_by_step_evaluation", {}).get("average_step_score"),
            "final_answer_reasoning": truncate_text(
                data.get("final_answer_evaluation", {}).get("reasoning", ""),
                200
            )
        })

    # Add highest scoring queries
    for uuid, avg_score, data, num_evals in highest_3:
        report["highest_scoring_queries"].append({
            "uuid": uuid,
            "average_score": round(avg_score, 4),
            "num_evaluations": num_evals,
            "query_preview": truncate_text(data.get("query", ""), 300),
            "full_query": data.get("query", ""),
            "final_answer_score": data.get("final_answer_evaluation", {}).get("final_answer_score"),
            "step_score": data.get("step_by_step_evaluation", {}).get("average_step_score"),
            "final_answer_reasoning": truncate_text(
                data.get("final_answer_evaluation", {}).get("reasoning", ""),
                200
            )
        })

    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("EXTREME QUERIES ANALYSIS")
    print("=" * 80)
    print(f"\nTotal unique queries: {report['total_unique_queries']}")
    print(f"Total evaluations: {report['total_evaluations']}")

    print("\n" + "-" * 80)
    print("TOP 3 LOWEST SCORING QUERIES")
    print("-" * 80)
    for i, q in enumerate(report["lowest_scoring_queries"], 1):
        print(f"\n{i}. UUID: {q['uuid']}")
        print(f"   Average Score: {q['average_score']:.4f}")
        print(f"   Evaluations: {q['num_evaluations']}")
        print(f"   Query: {q['query_preview']}")
        print(f"   Reasoning: {q['final_answer_reasoning']}")

    print("\n" + "-" * 80)
    print("TOP 3 HIGHEST SCORING QUERIES")
    print("-" * 80)
    for i, q in enumerate(report["highest_scoring_queries"], 1):
        print(f"\n{i}. UUID: {q['uuid']}")
        print(f"   Average Score: {q['average_score']:.4f}")
        print(f"   Evaluations: {q['num_evaluations']}")
        print(f"   Query: {q['query_preview']}")
        print(f"   Reasoning: {q['final_answer_reasoning']}")

    print("\n" + "=" * 80)
    print(f"Report saved to: {output_file}")
    print("=" * 80 + "\n")


def main():
    """Main function."""
    # Setup paths - go up one level from analyze_scripts to find evaluation
    eval_dir = Path(__file__).parent.parent / "evaluation"
    output_file = eval_dir / "extreme_queries_report.json"

    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found at {eval_dir}")
        return

    print("Collecting evaluations from all models, judges, and passes...")
    query_evals = collect_all_evaluations(eval_dir)

    if not query_evals:
        print("No evaluations found!")
        return

    print(f"Found {len(query_evals)} unique queries")
    print(f"Total evaluations: {sum(len(evals) for evals in query_evals.values())}")

    print("\nGenerating report...")
    generate_report(query_evals, output_file)


if __name__ == "__main__":
    main()
