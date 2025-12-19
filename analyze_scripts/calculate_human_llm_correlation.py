#!/usr/bin/env python3
"""
Calculate Spearman correlation between human ratings and LLM-as-judge ratings.

Usage:
    python analyze_scripts/calculate_human_llm_correlation.py

Description:
    This script compares human evaluation scores with LLM-as-judge scores to measure
    alignment between human judgments and automated LLM evaluation systems.

What it does:
    1. Reads human evaluation CSV files from test_result/:
       - Final answer scores from 3 evaluators (Meshal, Rishika, Tia)
       - Step-by-step scores from 3 evaluators

    2. Reads LLM judge evaluation JSON files from test_result/:
       - gpt-oss-eval/
       - kimi-k2-eval/
       - deepseekv3.2-eval/
       - glm4.6-eval/
       Each model is evaluated by 3 judge models (deepseekv32, gpt4omini, gpt51chat)

    3. Calculates average scores across the 3 judge models for each evaluated model

    4. Computes Spearman correlation coefficients between human and LLM ratings:
       - For final answer scores
       - For step-by-step scores
       - Includes p-values for statistical significance (p<0.05)

    5. Prints detailed analysis results directly to terminal:
       - Individual correlations for each evaluator-model pair
       - Summary statistics (mean, median, std dev)
       - Aggregated results by evaluator
       - Aggregated results by model

Output:
    All results are printed to the terminal. No files are generated.

Expected Data Structure:
    test_result/
    ├── human_alignment_sheet - Meshal Final Answer Score.csv
    ├── human_alignment_sheet - Meshal Average Step-by-Step Score.csv
    ├── human_alignment_sheet - Rishika Final Answer Score.csv
    ├── human_alignment_sheet - Rishika Average Step-by-Step Score.csv
    ├── human_alignment_sheet - Tia_Final Answer Score.csv
    ├── human_alignment_sheet - Tia_Average Step-by-Step Score.csv
    ├── gpt-oss-eval/
    │   ├── gpt-oss-120b-pass3_by_deepseekv32/eval_*.json
    │   ├── gpt-oss-120b-pass3_by_gpt4omini/eval_*.json
    │   └── gpt-oss-120b-pass3_by_gpt51chat/eval_*.json
    ├── kimi-k2-eval/
    │   ├── kimi-k2-thinking_by_deepseekv32/eval_*.json
    │   ├── kimi-k2-thinking_by_gpt4omini/eval_*.json
    │   └── kimi-k2-thinking_by_gpt51chat/eval_*.json
    ├── deepseekv3.2-eval/
    │   ├── deepseek-v3.2_by_deepseekv32/eval_*.json
    │   ├── deepseek-v3.2_by_gpt4omini/eval_*.json
    │   └── deepseek-v3.2_by_gpt51chat/eval_*.json
    └── glm4.6-eval/
        ├── glm-4.6v_by_deepseekv32/eval_*.json
        ├── glm-4.6v_by_gpt4omini/eval_*.json
        └── glm-4.6v_by_gpt51chat/eval_*.json

Dependencies:
    - pandas
    - numpy
    - scipy
"""

import os
import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
from typing import Dict, List, Tuple


# Configuration
TEST_RESULT_DIR = Path("test_result")
HUMAN_EVALUATORS = ["Meshal", "Rishika", "Tia"]
EVALUATED_MODELS = ["gpt-oss", "kimi-k2", "deepseekv3.2", "glm4.6"]
EVAL_DIRS = {
    "gpt-oss": "gpt-oss-eval",
    "kimi-k2": "kimi-k2-eval",
    "deepseekv3.2": "deepseekv3.2-eval",
    "glm4.6": "glm4.6-eval"
}
JUDGE_MODELS = {
    "gpt-oss": ["gpt-oss-120b-pass3_by_deepseekv32", "gpt-oss-120b-pass3_by_gpt4omini", "gpt-oss-120b-pass3_by_gpt51chat"],
    "kimi-k2": ["kimi-k2-thinking_by_deepseekv32", "kimi-k2-thinking_by_gpt4omini", "kimi-k2-thinking_by_gpt51chat"],
    "deepseekv3.2": ["deepseek-v3.2_by_deepseekv32", "deepseek-v3.2_by_gpt4omini", "deepseek-v3.2_by_gpt51chat"],
    "glm4.6": ["glm-4.6v_by_deepseekv32", "glm-4.6v_by_gpt4omini", "glm-4.6v_by_gpt51chat"]
}


def read_human_scores() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Read all human evaluation CSV files.

    Returns:
        Tuple of (final_scores, step_scores) dictionaries
        Keys are evaluator names, values are DataFrames with model scores
    """
    final_scores = {}
    step_scores = {}

    for evaluator in HUMAN_EVALUATORS:
        # Read final answer scores - try both naming conventions
        final_file = TEST_RESULT_DIR / f"human_alignment_sheet - {evaluator} Final Answer Score.csv"
        if not final_file.exists():
            final_file = TEST_RESULT_DIR / f"human_alignment_sheet - {evaluator}_Final Answer Score.csv"

        if final_file.exists():
            df = pd.read_csv(final_file, index_col=0)
            final_scores[evaluator] = df
            print(f"Loaded {evaluator} final answer scores: {df.shape}")

        # Read step-by-step scores - try both naming conventions
        step_file = TEST_RESULT_DIR / f"human_alignment_sheet - {evaluator} Average Step-by-Step Score.csv"
        if not step_file.exists():
            step_file = TEST_RESULT_DIR / f"human_alignment_sheet - {evaluator}_Average Step-by-Step Score.csv"

        if step_file.exists():
            df = pd.read_csv(step_file, index_col=0)
            step_scores[evaluator] = df
            print(f"Loaded {evaluator} step-by-step scores: {df.shape}")

    return final_scores, step_scores


def read_llm_judge_scores() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Read all LLM judge evaluation JSON files and calculate average scores.

    Returns:
        Tuple of (final_scores, step_scores) dictionaries
        Keys are model names, values are DataFrames with UUID and averaged scores
    """
    final_scores = {}
    step_scores = {}

    for model, eval_dir_name in EVAL_DIRS.items():
        eval_dir = TEST_RESULT_DIR / eval_dir_name
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} does not exist, skipping {model}")
            continue

        # Dictionaries to accumulate scores from all judge models
        final_score_accumulator = {}  # uuid -> [scores from 3 judges]
        step_score_accumulator = {}   # uuid -> [scores from 3 judges]

        # Get all judge subdirectories for this model
        judge_dirs = JUDGE_MODELS.get(model, [])

        for judge_dir_name in judge_dirs:
            judge_dir = eval_dir / judge_dir_name
            if not judge_dir.exists():
                print(f"Warning: {judge_dir} does not exist, skipping")
                continue

            # Read all JSON evaluation files in this judge directory
            for json_file in judge_dir.glob("eval_*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    uuid = data.get("uuid")
                    if not uuid:
                        continue

                    # Extract final answer score
                    final_eval = data.get("final_answer_evaluation", {})
                    final_score = final_eval.get("final_answer_score")
                    if final_score is not None:
                        if uuid not in final_score_accumulator:
                            final_score_accumulator[uuid] = []
                        final_score_accumulator[uuid].append(final_score)

                    # Extract step-by-step average score
                    step_eval = data.get("step_by_step_evaluation", {})
                    step_score = step_eval.get("average_step_score")
                    if step_score is not None:
                        if uuid not in step_score_accumulator:
                            step_score_accumulator[uuid] = []
                        step_score_accumulator[uuid].append(step_score)

                except Exception as e:
                    print(f"Error reading {json_file}: {e}")

        # Calculate averages across the 3 judge models
        if final_score_accumulator:
            final_avg = {uuid: np.mean(scores) for uuid, scores in final_score_accumulator.items()}
            final_scores[model] = pd.DataFrame.from_dict(final_avg, orient='index', columns=['avg_final_score'])
            final_scores[model].index.name = 'uuid'
            print(f"Loaded {model} LLM final scores: {len(final_avg)} UUIDs")

        if step_score_accumulator:
            step_avg = {uuid: np.mean(scores) for uuid, scores in step_score_accumulator.items()}
            step_scores[model] = pd.DataFrame.from_dict(step_avg, orient='index', columns=['avg_step_score'])
            step_scores[model].index.name = 'uuid'
            print(f"Loaded {model} LLM step scores: {len(step_avg)} UUIDs")

    return final_scores, step_scores


def calculate_correlations(human_scores: Dict[str, pd.DataFrame],
                          llm_scores: Dict[str, pd.DataFrame],
                          score_type: str) -> pd.DataFrame:
    """
    Calculate Spearman correlations between human and LLM scores.

    Args:
        human_scores: Dictionary of human evaluator DataFrames
        llm_scores: Dictionary of LLM model DataFrames
        score_type: 'final' or 'step'

    Returns:
        DataFrame with correlation coefficients and p-values
    """
    results = []

    for evaluator, human_df in human_scores.items():
        for model in EVALUATED_MODELS:
            if model not in llm_scores:
                continue

            llm_df = llm_scores[model]

            # Get common UUIDs (column names in human_df, index in llm_df)
            human_uuids = set(human_df.columns)
            llm_uuids = set(llm_df.index)
            common_uuids = sorted(human_uuids & llm_uuids)

            if len(common_uuids) < 3:
                print(f"Warning: Only {len(common_uuids)} common UUIDs for {evaluator} vs {model}")
                continue

            # Extract scores for common UUIDs
            human_values = []
            llm_values = []

            for uuid in common_uuids:
                try:
                    human_score = human_df.loc[model, uuid]
                    llm_score = llm_df.loc[uuid, 'avg_final_score' if score_type == 'final' else 'avg_step_score']

                    # Skip NaN values
                    if pd.notna(human_score) and pd.notna(llm_score):
                        human_values.append(float(human_score))
                        llm_values.append(float(llm_score))
                except Exception as e:
                    print(f"Error extracting scores for {uuid}: {e}")
                    continue

            if len(human_values) < 3:
                print(f"Warning: Only {len(human_values)} valid scores for {evaluator} vs {model}")
                continue

            # Calculate Spearman correlation
            corr, p_value = spearmanr(human_values, llm_values)

            results.append({
                'evaluator': evaluator,
                'model': model,
                'score_type': score_type,
                'n_samples': len(human_values),
                'spearman_corr': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

            print(f"{evaluator} vs {model} ({score_type}): ρ={corr:.3f}, p={p_value:.4f}, n={len(human_values)}")

    return pd.DataFrame(results)


def print_summary_report(final_corr_df: pd.DataFrame,
                        step_corr_df: pd.DataFrame):
    """
    Print a text summary report of the correlation analysis.
    """
    print("\n" + "=" * 80)
    print("HUMAN-LLM ALIGNMENT CORRELATION ANALYSIS")
    print("=" * 80)
    print()

    # Final Answer Scores
    print("FINAL ANSWER SCORES")
    print("-" * 80)
    print(final_corr_df.to_string(index=False))
    print()

    # Summary statistics for final scores
    print("Summary Statistics (Final Answer):")
    print(f"  Mean Spearman ρ: {final_corr_df['spearman_corr'].mean():.3f}")
    print(f"  Median Spearman ρ: {final_corr_df['spearman_corr'].median():.3f}")
    print(f"  Std Dev: {final_corr_df['spearman_corr'].std():.3f}")
    print(f"  Significant correlations (p<0.05): {final_corr_df['significant'].sum()}/{len(final_corr_df)}")
    print()

    # Step-by-Step Scores
    print("STEP-BY-STEP SCORES")
    print("-" * 80)
    print(step_corr_df.to_string(index=False))
    print()

    # Summary statistics for step scores
    print("Summary Statistics (Step-by-Step):")
    print(f"  Mean Spearman ρ: {step_corr_df['spearman_corr'].mean():.3f}")
    print(f"  Median Spearman ρ: {step_corr_df['spearman_corr'].median():.3f}")
    print(f"  Std Dev: {step_corr_df['spearman_corr'].std():.3f}")
    print(f"  Significant correlations (p<0.05): {step_corr_df['significant'].sum()}/{len(step_corr_df)}")
    print()

    # By evaluator analysis
    print("CORRELATIONS BY EVALUATOR")
    print("-" * 80)
    for evaluator in HUMAN_EVALUATORS:
        final_eval = final_corr_df[final_corr_df['evaluator'] == evaluator]
        step_eval = step_corr_df[step_corr_df['evaluator'] == evaluator]

        if len(final_eval) > 0:
            print(f"{evaluator}:")
            print(f"  Final Answer - Mean ρ: {final_eval['spearman_corr'].mean():.3f}")
            print(f"  Step-by-Step - Mean ρ: {step_eval['spearman_corr'].mean():.3f}")
            print()

    # By model analysis
    print("CORRELATIONS BY MODEL")
    print("-" * 80)
    for model in EVALUATED_MODELS:
        final_model = final_corr_df[final_corr_df['model'] == model]
        step_model = step_corr_df[step_corr_df['model'] == model]

        if len(final_model) > 0:
            print(f"{model}:")
            print(f"  Final Answer - Mean ρ: {final_model['spearman_corr'].mean():.3f}")
            print(f"  Step-by-Step - Mean ρ: {step_model['spearman_corr'].mean():.3f}")
            print()

    print("=" * 80)


def main():
    """Main execution function."""
    print("=" * 80)
    print("Starting Human-LLM Alignment Correlation Analysis")
    print("=" * 80)
    print()

    # Read human scores
    print("\n[1/3] Reading human evaluation scores...")
    human_final, human_step = read_human_scores()

    # Read LLM judge scores
    print("\n[2/3] Reading LLM judge evaluation scores...")
    llm_final, llm_step = read_llm_judge_scores()

    # Calculate correlations for final answer scores
    print("\n[3/3] Calculating correlations...")
    final_corr_df = calculate_correlations(human_final, llm_final, 'final')

    # Calculate correlations for step-by-step scores
    step_corr_df = calculate_correlations(human_step, llm_step, 'step')

    # Print summary report
    print_summary_report(final_corr_df, step_corr_df)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
