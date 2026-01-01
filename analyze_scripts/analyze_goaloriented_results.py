#!/usr/bin/env python3
"""
Analyze evaluation results in evaluation/goaloriented directory
Calculate average scores for each model across different judge models

Usage:
    python analyze_scripts/analyze_goaloriented_results.py

Output:
    - Console output with detailed metrics for each model and judge

Metrics calculated:
    - Final Answer: completeness, coherence, actionability, constraint_adherence, overall_final_answer
    - Overall: agent_step_avg_score, overall_score, avg_grounding, progress_tracking, goal_decomposition
    - Agent Steps: thinking_quality, tool_selection_quality, tool_execution_quality, response_quality, grounding, step_overall
    - User LLM: subgoal_decomposition_quality, goal_tracking_coherence, follow_up_intent_quality, overall_user_quality
    - Trajectory: goal_completion_rate, overall_constraint_satisfaction_rate, avg_satisfaction_level, avg_goal_progress, avg_constraint_satisfaction
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import statistics


def parse_directory_name(dir_name: str) -> tuple:
    """Parse directory name to extract model name and judge model name"""
    parts = dir_name.split('_by_')
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def load_eval_files(base_dir: str) -> Dict:
    """Load all evaluation files"""
    data = defaultdict(lambda: defaultdict(list))

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory does not exist: {base_dir}")
        return data

    # Iterate through all subdirectories
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name, judge_model = parse_directory_name(model_dir.name)
        if not model_name or not judge_model:
            continue

        # Load all JSON files in this directory
        json_files = list(model_dir.glob("eval_trajectory_*.json"))
        print(f"Processing {model_name} by {judge_model}: found {len(json_files)} files")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                    data[model_name][judge_model].append(eval_data)
            except Exception as e:
                print(f"  Error: Cannot load {json_file}: {e}")

    return data


def calculate_averages(eval_list: List[Dict]) -> Dict:
    """Calculate average values of evaluation metrics"""
    if not eval_list:
        return {}

    # Final answer metrics
    final_answer_metrics = {
        'completeness': [],
        'coherence': [],
        'actionability': [],
        'constraint_adherence': [],
        'overall_final_answer': []
    }

    # Overall metrics
    overall_metrics = {
        'agent_step_avg_score': [],
        'overall_score': []
    }

    # Agent step evaluations (averaged across all steps)
    step_metrics = {
        'thinking_quality': [],
        'tool_selection_quality': [],
        'tool_execution_quality': [],
        'response_quality': [],
        'grounding': [],
        'step_overall': []
    }

    # User LLM quality
    user_llm_metrics = {
        'subgoal_decomposition_quality': [],
        'goal_tracking_coherence': [],
        'follow_up_intent_quality': [],
        'overall_user_quality': []
    }

    # Trajectory metrics
    trajectory_metrics = {
        'goal_completion_rate': [],
        'overall_constraint_satisfaction_rate': [],
        'avg_satisfaction_level': [],
        'avg_goal_progress': [],
        'avg_constraint_satisfaction': []
    }

    # Progress tracking & goal decomposition per trajectory
    progress_tracking_per_trajectory = []
    goal_decomposition_per_trajectory = []

    # Average grounding per trajectory
    avg_grounding_per_trajectory = []

    for eval_data in eval_list:
        # Final answer
        if 'agent_final_answer' in eval_data and eval_data['agent_final_answer'] is not None:
            for key in final_answer_metrics.keys():
                if key in eval_data['agent_final_answer']:
                    final_answer_metrics[key].append(eval_data['agent_final_answer'][key])

        # Overall
        for key in overall_metrics.keys():
            if key in eval_data:
                overall_metrics[key].append(eval_data[key])

        # Agent step evaluations
        trajectory_grounding_scores = []
        if 'agent_step_evaluations' in eval_data:
            for step_eval in eval_data['agent_step_evaluations']:
                for key in step_metrics.keys():
                    if key in step_eval:
                        step_metrics[key].append(step_eval[key])
                        # Collect grounding scores for this trajectory
                        if key == 'grounding':
                            trajectory_grounding_scores.append(step_eval[key])

        # Calculate average grounding for this trajectory
        if trajectory_grounding_scores:
            avg_grounding_per_trajectory.append(statistics.mean(trajectory_grounding_scores))

        # User LLM quality
        if 'user_llm_quality' in eval_data and eval_data['user_llm_quality'] is not None:
            for key in user_llm_metrics.keys():
                if key in eval_data['user_llm_quality']:
                    user_llm_metrics[key].append(eval_data['user_llm_quality'][key])

            # Collect progress tracking and goal decomposition separately
            if 'goal_tracking_coherence' in eval_data['user_llm_quality']:
                progress_tracking_per_trajectory.append(eval_data['user_llm_quality']['goal_tracking_coherence'])
            if 'subgoal_decomposition_quality' in eval_data['user_llm_quality']:
                goal_decomposition_per_trajectory.append(eval_data['user_llm_quality']['subgoal_decomposition_quality'])

        # Trajectory metrics
        if 'trajectory_metrics' in eval_data and eval_data['trajectory_metrics'] is not None:
            for key in trajectory_metrics.keys():
                if key in eval_data['trajectory_metrics']:
                    trajectory_metrics[key].append(eval_data['trajectory_metrics'][key])
                elif 'ground_truth' in eval_data['trajectory_metrics'] and eval_data['trajectory_metrics']['ground_truth'] is not None and key in eval_data['trajectory_metrics']['ground_truth']:
                    trajectory_metrics[key].append(eval_data['trajectory_metrics']['ground_truth'][key])

    # Calculate averages
    result = {
        'sample_count': len(eval_list),
        'final_answer': {},
        'overall': {},
        'agent_steps': {},
        'user_llm': {},
        'trajectory': {}
    }

    for key, values in final_answer_metrics.items():
        if values:
            result['final_answer'][key] = round(statistics.mean(values), 2)

    for key, values in overall_metrics.items():
        if values:
            result['overall'][key] = round(statistics.mean(values), 2)

    for key, values in step_metrics.items():
        if values:
            result['agent_steps'][key] = round(statistics.mean(values), 2)

    for key, values in user_llm_metrics.items():
        if values:
            result['user_llm'][key] = round(statistics.mean(values), 2)

    for key, values in trajectory_metrics.items():
        if values:
            result['trajectory'][key] = round(statistics.mean(values), 2)

    # Add average grounding score (average of trajectory averages)
    if avg_grounding_per_trajectory:
        result['overall']['avg_grounding'] = round(statistics.mean(avg_grounding_per_trajectory), 2)

    # Add progress tracking and goal decomposition
    if progress_tracking_per_trajectory:
        result['overall']['progress_tracking'] = round(statistics.mean(progress_tracking_per_trajectory), 2)
    if goal_decomposition_per_trajectory:
        result['overall']['goal_decomposition'] = round(statistics.mean(goal_decomposition_per_trajectory), 2)

    return result


def main():
    base_dir = "evaluation/goaloriented"

    print("=" * 80)
    print("Loading evaluation data...")
    print("=" * 80)
    data = load_eval_files(base_dir)

    if not data:
        print("No evaluation data found")
        return

    print("\n" + "=" * 80)
    print("Calculating average scores for each model across different Judge Models")
    print("=" * 80)

    # Organize results by model
    results_by_model = {}

    for model_name in sorted(data.keys()):
        print(f"\nModel: {model_name}")
        print("-" * 80)

        results_by_model[model_name] = {}

        for judge_model in sorted(data[model_name].keys()):
            eval_list = data[model_name][judge_model]
            averages = calculate_averages(eval_list)
            results_by_model[model_name][judge_model] = averages

            print(f"\n  Judge Model: {judge_model} (sample count: {averages['sample_count']})")

            # Print Final Answer metrics
            if averages.get('final_answer'):
                print("\n    Final Answer Judge:")
                for metric, value in sorted(averages['final_answer'].items()):
                    print(f"      {metric:30s}: {value:6.2f}")

            # Print other key metrics
            if averages.get('overall'):
                print("\n    Overall:")
                for metric, value in sorted(averages['overall'].items()):
                    print(f"      {metric:30s}: {value:6.2f}")

    # Calculate averages across Judge Models
    print("\n" + "=" * 80)
    print("Average scores for each model across all Judge Models")
    print("=" * 80)

    for model_name in sorted(results_by_model.keys()):
        print(f"\nModel: {model_name}")
        print("-" * 80)

        # Collect data from all judge models
        all_final_answer = defaultdict(list)
        all_overall = defaultdict(list)
        all_agent_steps = defaultdict(list)
        total_samples = 0

        for judge_model, averages in results_by_model[model_name].items():
            total_samples += averages['sample_count']

            for metric, value in averages.get('final_answer', {}).items():
                all_final_answer[metric].append(value)

            for metric, value in averages.get('overall', {}).items():
                all_overall[metric].append(value)

            for metric, value in averages.get('agent_steps', {}).items():
                all_agent_steps[metric].append(value)

        print(f"\n  Total samples: {total_samples}")
        print(f"  Number of Judge Models: {len(results_by_model[model_name])}")

        print("\n  Final Answer Judge (averaged across Judge Models):")
        for metric in sorted(all_final_answer.keys()):
            values = all_final_answer[metric]
            avg = statistics.mean(values)
            print(f"    {metric:30s}: {avg:6.2f} (range: {min(values):.2f}-{max(values):.2f})")

        print("\n  Overall (averaged across Judge Models):")
        for metric in sorted(all_overall.keys()):
            values = all_overall[metric]
            avg = statistics.mean(values)
            print(f"    {metric:30s}: {avg:6.2f} (range: {min(values):.2f}-{max(values):.2f})")

        print("\n  Agent Steps (averaged across Judge Models):")
        for metric in sorted(all_agent_steps.keys()):
            values = all_agent_steps[metric]
            avg = statistics.mean(values)
            print(f"    {metric:30s}: {avg:6.2f} (range: {min(values):.2f}-{max(values):.2f})")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
