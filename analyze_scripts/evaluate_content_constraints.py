#!/usr/bin/env python3
"""
Post-process evaluation results to add LLM-based constraint verification for:
1. RESPONSE_CONTENT constraints - Verifies if agent responses contain required content
2. TRADEOFF constraints - Verifies if agent explicitly discussed required tradeoffs

This script reads evaluation JSON files and uses LLM-as-judge to verify these constraints,
then updates the constraint verification results in place.

Background:
The goaloriented_evaluator.py marks RESPONSE_CONTENT and TRADEOFF constraints as
"Requires LLM judgment - evaluated in step/final answer evaluation" but doesn't actually
implement the evaluation logic. This script fills that gap by:
- Loading the original trajectory to get agent responses
- Using LLM-as-judge to verify if the constraints are satisfied
- Updating the evaluation JSON with the verification results
- Recalculating constraint satisfaction rates

Usage:
    # Process a single evaluation file
    python analyze_scripts/evaluate_content_constraints.py \
        --eval_file evaluation/goaloriented/grok-4_by_gpt-4o/eval_trajectory_xxx.json \
        --model openai/gpt-4o-mini

    # Process a directory of evaluation files
    python analyze_scripts/evaluate_content_constraints.py \
        --eval_dir evaluation/goaloriented \
        --model openai/gpt-4o-mini \
        --recursive

    # Dry run (don't save results)
    python analyze_scripts/evaluate_content_constraints.py \
        --eval_file evaluation/goaloriented/grok-4_by_gpt-4o/eval_trajectory_xxx.json \
        --model openai/gpt-4o-mini \
        --no-save
"""

import os
import json
import glob
import argparse
import sys
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Try to load .env from multiple locations
# First try current directory, then Orchestrator directory
load_dotenv()
orchestrator_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Orchestrator", ".env")
if os.path.exists(orchestrator_env):
    load_dotenv(orchestrator_env)

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "Missing openai client. Install: pip install openai python-dotenv"
    ) from e


def _call_llm(
    *,
    prompt: str,
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    model_name: str = "openai/gpt-4o-mini",
) -> Optional[str]:
    """Call LLM for constraint verification."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "Please set OPENAI_API_KEY and OPENAI_BASE_URL environment variables."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    tries = 3
    while tries > 0:
        tries -= 1
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
    return None


def _safe_parse_json(s: Optional[str]) -> Optional[Dict[str, Any]]:
    """Safely parse JSON from LLM response, handling markdown code blocks."""
    if not s:
        return None
    s = str(s).strip()

    if s.startswith("```"):
        parts = s.split("```")
        candidate = ""
        for chunk in parts:
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            candidate = chunk
            break
        s = candidate

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


# =============================================================================
# CONSTRAINT VERIFICATION PROMPTS
# =============================================================================

RESPONSE_CONTENT_SYSTEM_PROMPT = """You are an expert evaluator checking if an AI agent's responses satisfy a RESPONSE_CONTENT constraint.

RESPONSE_CONTENT constraints specify what the agent's final answer must include, such as:
- Specific types of content (comparisons, recommendations, prioritized lists)
- Certain keywords or topics
- Structured output format

You will be given:
1. The constraint description
2. The constraint verification rules
3. All of the agent's responses across turns

Your task is to determine if the constraint is satisfied.

Respond with a JSON object:
{
    "satisfied": true/false,
    "details": "<explanation of why satisfied or violated>",
    "expected_value": "<what was expected>",
    "actual_value": "<what was actually provided>"
}
"""

TRADEOFF_SYSTEM_PROMPT = """You are an expert evaluator checking if an AI agent's responses satisfy a TRADEOFF constraint.

TRADEOFF constraints require the agent to explicitly reason about tradeoffs, such as:
- Weighing costs vs. benefits
- Comparing different approaches
- Discussing pros and cons
- Making explicit decisions based on tradeoff analysis

You will be given:
1. The constraint description
2. The constraint verification rules
3. All of the agent's responses across turns

Your task is to determine if the agent explicitly discussed the required tradeoff.

Respond with a JSON object:
{
    "satisfied": true/false,
    "details": "<explanation of why satisfied or violated, with specific quotes if relevant>",
    "expected_value": "<what tradeoff was expected to be discussed>",
    "actual_value": "<what tradeoff discussion was actually provided, or 'none'>"
}
"""


def verify_response_content_constraint(
    constraint: Dict[str, Any],
    all_agent_responses: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Use LLM to verify a RESPONSE_CONTENT constraint.

    Args:
        constraint: The constraint dictionary
        all_agent_responses: Concatenated agent responses from all turns
        model_name: LLM model to use

    Returns:
        Dict with: satisfied, details, expected_value, actual_value
    """
    description = constraint.get("description", "")
    verification = constraint.get("verification", {})

    prompt = f"""Evaluate if the agent's responses satisfy this RESPONSE_CONTENT constraint.

## Constraint Description:
{description}

## Verification Rules:
{json.dumps(verification, indent=2)}

## Agent's Complete Responses (All Turns):
{all_agent_responses}

Does the agent's response satisfy the constraint? Provide your evaluation as a JSON object."""

    response = _call_llm(
        prompt=prompt,
        system_prompt=RESPONSE_CONTENT_SYSTEM_PROMPT,
        model_name=model_name,
    )

    parsed = _safe_parse_json(response)
    if not parsed:
        return {
            "satisfied": False,
            "details": "Failed to parse LLM response",
            "expected_value": description,
            "actual_value": "evaluation error",
        }

    return {
        "satisfied": parsed.get("satisfied", False),
        "details": parsed.get("details", ""),
        "expected_value": parsed.get("expected_value"),
        "actual_value": parsed.get("actual_value"),
    }


def verify_tradeoff_constraint(
    constraint: Dict[str, Any],
    all_agent_responses: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Use LLM to verify a TRADEOFF constraint.

    Args:
        constraint: The constraint dictionary
        all_agent_responses: Concatenated agent responses from all turns
        model_name: LLM model to use

    Returns:
        Dict with: satisfied, details, expected_value, actual_value
    """
    description = constraint.get("description", "")
    verification = constraint.get("verification", {})

    prompt = f"""Evaluate if the agent's responses satisfy this TRADEOFF constraint.

## Constraint Description:
{description}

## Verification Rules:
{json.dumps(verification, indent=2)}

## Agent's Complete Responses (All Turns):
{all_agent_responses}

Did the agent explicitly discuss the required tradeoff? Provide your evaluation as a JSON object."""

    response = _call_llm(
        prompt=prompt,
        system_prompt=TRADEOFF_SYSTEM_PROMPT,
        model_name=model_name,
    )

    parsed = _safe_parse_json(response)
    if not parsed:
        return {
            "satisfied": False,
            "details": "Failed to parse LLM response",
            "expected_value": description,
            "actual_value": "evaluation error",
        }

    return {
        "satisfied": parsed.get("satisfied", False),
        "details": parsed.get("details", ""),
        "expected_value": parsed.get("expected_value"),
        "actual_value": parsed.get("actual_value"),
    }


def load_trajectory_from_eval(eval_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Load the original trajectory file referenced in the evaluation.

    Args:
        eval_data: The evaluation result data

    Returns:
        The trajectory data, or None if not found
    """
    trajectory_file = eval_data.get("trajectory_file", "")
    if not trajectory_file:
        return None

    # Try to find the trajectory file
    # Assume it's in the same directory or a sibling directory
    eval_dir = os.path.dirname(eval_data.get("_filepath", "."))

    # Try various possible locations
    possible_paths = [
        os.path.join(eval_dir, trajectory_file),
        os.path.join(eval_dir, "..", trajectory_file),
        os.path.join(eval_dir, "..", "..", "trajectories", "goaloriented", trajectory_file),
        trajectory_file,  # Absolute path
    ]

    # Also try searching for the file in the project
    # Look for trajectories directory from eval_dir
    current = eval_dir
    for _ in range(5):  # Search up to 5 levels up
        traj_dir = os.path.join(current, "trajectories", "goaloriented")
        if os.path.isdir(traj_dir):
            # Search recursively in this directory
            import glob
            pattern = os.path.join(traj_dir, "**", trajectory_file)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                possible_paths.append(matches[0])
                break
        current = os.path.dirname(current)

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load trajectory from {path}: {e}", file=sys.stderr)
                continue

    print(f"[WARN] Could not find trajectory file: {trajectory_file}", file=sys.stderr)
    return None


def concatenate_agent_responses(trajectory: Dict[str, Any]) -> str:
    """
    Concatenate all agent responses from the trajectory.

    Args:
        trajectory: The trajectory data

    Returns:
        Concatenated responses as a single string
    """
    turns = trajectory.get("turns", [])
    concatenated = []

    for turn in turns:
        turn_number = turn.get("turn_number", 0)
        agent_response = turn.get("agent_response", "")
        if isinstance(agent_response, dict):
            agent_response = json.dumps(agent_response)

        concatenated.append(f"\n{'='*50}")
        concatenated.append(f"TURN {turn_number} RESPONSE:")
        concatenated.append(f"{'='*50}")
        concatenated.append(str(agent_response))

    return "\n".join(concatenated)


def infer_judge_model_from_path(eval_file: str, default_model: str = "openai/gpt-4o") -> str:
    """
    Infer the judge model from the evaluation file path.

    Pattern: evaluation/goaloriented/{agent_model}_by_{judge_model}/eval_*.json

    Supported mappings (only these three):
        by_gpt-4o -> openai/gpt-4o
        by_deepseek-v3.2 -> deepseek/deepseek-chat
        by_gpt-4.5 -> openai/gpt-4.5

    Args:
        eval_file: Path to the evaluation file
        default_model: Default model if no match found

    Returns:
        Judge model name
    """
    # Extract directory name pattern like "grok-4_by_gpt-4o"
    dir_name = os.path.basename(os.path.dirname(eval_file))

    # Fixed mapping - only these three models
    model_mapping = {
        "gpt-4o": "openai/gpt-4o",
        "deepseek-v3.2": "deepseek/deepseek-chat",
        "gpt-5.1": "openai/gpt-5.1",
    }

    # Check if it matches the pattern {agent}_by_{judge}
    if "_by_" not in dir_name:
        print(f"  [WARN] Path doesn't match pattern '*_by_*': {dir_name}, using default: {default_model}")
        return default_model

    # Extract judge model name
    parts = dir_name.split("_by_")
    if len(parts) != 2:
        print(f"  [WARN] Cannot parse judge model from: {dir_name}, using default: {default_model}")
        return default_model

    judge_model_short = parts[1]  # e.g., "gpt-4o", "deepseek-v3.2", "gpt-4.5"

    # Check if it's one of the three supported models
    if judge_model_short in model_mapping:
        inferred_model = model_mapping[judge_model_short]
        print(f"  [INFO] Using judge model: {inferred_model}")
        return inferred_model
    else:
        print(f"  [WARN] Unknown judge model '{judge_model_short}', using default: {default_model}")
        print(f"  [INFO] Supported models: {list(model_mapping.keys())}")
        return default_model


def evaluate_content_constraints(
    eval_file: str,
    model_name: Optional[str] = None,
    save_result: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate RESPONSE_CONTENT and TRADEOFF constraints for an evaluation file.

    Args:
        eval_file: Path to the evaluation JSON file
        model_name: LLM model to use for verification (if None, infer from path)
        save_result: Whether to save the updated evaluation back to the file

    Returns:
        Updated evaluation data
    """
    # Load evaluation data
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    eval_data["_filepath"] = eval_file

    # Infer judge model from path if not specified
    if model_name is None:
        # Temporarily suppress print for non-verbose mode
        if not verbose:
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                model_name = infer_judge_model_from_path(eval_file)
        else:
            model_name = infer_judge_model_from_path(eval_file)
    else:
        if verbose:
            print(f"  [INFO] Using specified model: {model_name}")

    # Load the original trajectory
    trajectory = load_trajectory_from_eval(eval_data)
    if not trajectory:
        if verbose:
            print(f"[ERROR] Could not load trajectory for {eval_file}", file=sys.stderr)
        return eval_data

    # Get all agent responses
    all_responses = concatenate_agent_responses(trajectory)

    # Get constraints from trajectory
    metadata = trajectory.get("metadata", {})
    constraints = metadata.get("constraints", [])

    # Get constraint verification results
    constraint_verification = eval_data.get("trajectory_metrics", {}).get("constraint_verification", {})
    verifications = constraint_verification.get("verifications", [])

    # Track if we made any updates
    updated = False
    llm_satisfied_count = 0
    llm_violated_count = 0

    # Process each constraint that requires LLM verification
    for i, verification in enumerate(verifications):
        constraint_type = verification.get("constraint_type", "")

        # Skip if already evaluated (has actual_value set)
        if verification.get("actual_value") is not None:
            continue

        # Find the original constraint for this verification
        constraint = None
        for c in constraints:
            if c.get("type") == constraint_type:
                constraint = c
                break

        if not constraint:
            continue

        # Verify based on constraint type
        if constraint_type == "RESPONSE_CONTENT":
            if verbose:
                print(f"  Verifying RESPONSE_CONTENT constraint: {verification.get('description', '')[:60]}...")
            result = verify_response_content_constraint(constraint, all_responses, model_name)
            verification["satisfied"] = result["satisfied"]
            verification["details"] = result["details"]
            verification["expected_value"] = result["expected_value"]
            verification["actual_value"] = result["actual_value"]
            updated = True

            if result["satisfied"]:
                llm_satisfied_count += 1
            else:
                llm_violated_count += 1

        elif constraint_type == "TRADEOFF":
            if verbose:
                print(f"  Verifying TRADEOFF constraint: {verification.get('description', '')[:60]}...")
            result = verify_tradeoff_constraint(constraint, all_responses, model_name)
            verification["satisfied"] = result["satisfied"]
            verification["details"] = result["details"]
            verification["expected_value"] = result["expected_value"]
            verification["actual_value"] = result["actual_value"]
            updated = True

            if result["satisfied"]:
                llm_satisfied_count += 1
            else:
                llm_violated_count += 1

    # Update the verification results in eval_data
    if updated:
        eval_data["trajectory_metrics"]["constraint_verification"]["verifications"] = verifications

        # Recalculate satisfaction rates
        total_constraints = constraint_verification.get("total_constraints", 0)
        static_satisfied = constraint_verification.get("static_satisfied", 0)
        static_violated = constraint_verification.get("static_violated", 0)

        # Overall satisfaction including LLM-verified constraints
        total_satisfied = static_satisfied + llm_satisfied_count
        total_violated = static_violated + llm_violated_count
        total_verified = total_satisfied + total_violated

        if total_verified > 0:
            overall_satisfaction_rate = total_satisfied / total_verified
            eval_data["trajectory_metrics"]["constraint_verification"]["overall_satisfaction_rate"] = overall_satisfaction_rate
            eval_data["trajectory_metrics"]["overall_constraint_satisfaction_rate"] = overall_satisfaction_rate

        # Update LLM-verified counts
        eval_data["trajectory_metrics"]["constraint_verification"]["llm_satisfied"] = llm_satisfied_count
        eval_data["trajectory_metrics"]["constraint_verification"]["llm_violated"] = llm_violated_count

        if verbose:
            print(f"  ✓ Updated {llm_satisfied_count + llm_violated_count} LLM-verified constraints")
            print(f"    Satisfied: {llm_satisfied_count}, Violated: {llm_violated_count}")

    # Save the updated evaluation
    if save_result and updated:
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  ✓ Saved updated evaluation to {eval_file}")

    # Store metadata about the evaluation
    eval_data["_evaluation_metadata"] = {
        "llm_constraints_found": llm_satisfied_count + llm_violated_count,
        "llm_satisfied": llm_satisfied_count,
        "llm_violated": llm_violated_count,
        "updated": updated,
    }

    return eval_data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RESPONSE_CONTENT and TRADEOFF constraints using LLM-as-judge"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--eval_file", "-f",
        help="Path to a single evaluation JSON file"
    )
    input_group.add_argument(
        "--eval_dir", "-d",
        help="Directory containing evaluation JSON files"
    )

    # Options
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model to use for constraint verification (if not specified, infer from path)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search subdirectories"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results back to files (dry run)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)"
    )

    args = parser.parse_args()

    # Collect evaluation files
    eval_files = []

    if args.eval_file:
        if not os.path.exists(args.eval_file):
            print(f"[ERROR] File not found: {args.eval_file}", file=sys.stderr)
            sys.exit(1)
        eval_files = [args.eval_file]
    else:
        if args.recursive:
            pattern = os.path.join(args.eval_dir, "**", "eval_*.json")
            eval_files = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(args.eval_dir, "eval_*.json")
            eval_files = glob.glob(pattern)

    if not eval_files:
        print("[ERROR] No evaluation files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(eval_files)} evaluation file(s)")
    if args.model:
        print(f"Model: {args.model} (specified)")
    else:
        print(f"Model: Auto-infer from path")
    print(f"Save results: {not args.no_save}")
    print(f"Workers: {args.workers}")
    print("")

    # Process files with parallel workers
    def process_file(eval_file: str) -> tuple:
        """Process a single file and return (filename, success, error_msg)"""
        try:
            evaluate_content_constraints(
                eval_file,
                model_name=args.model,
                save_result=not args.no_save,
                verbose=False,  # Suppress output in parallel mode
            )
            return (os.path.basename(eval_file), True, None)
        except Exception as e:
            import traceback
            error_msg = f"{e}\n{traceback.format_exc()}"
            return (os.path.basename(eval_file), False, error_msg)

    # Use ThreadPoolExecutor for parallel processing
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, f): f for f in eval_files}

        # Process results as they complete
        for future in as_completed(future_to_file):
            filename, success, error_msg = future.result()
            completed += 1

            if success:
                print(f"[{completed}/{len(eval_files)}] ✓ {filename}")
            else:
                failed += 1
                print(f"[{completed}/{len(eval_files)}] ✗ {filename}", file=sys.stderr)
                if error_msg:
                    print(f"  Error: {error_msg}", file=sys.stderr)

    print("")
    print(f"✓ Done! Processed: {completed}, Success: {completed - failed}, Failed: {failed}")


if __name__ == "__main__":
    main()
