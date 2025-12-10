#!/usr/bin/env python3
"""
Shared Data Loading Functions for MCP Evaluation Analysis

This module provides common functions for loading trajectory and evaluation data
from the MCP-R project. All analysis scripts should import from here to ensure
consistent data loading behavior.

Data Sources:
    - trajectories/: Contains trajectory_*.json files with tool call sequences
    - evaluation/: Contains eval_*.json and _summary.json files with scores

Key Data Structures:
    - tool_calls: List of individual tool call records with error classification
    - trajectories: List of trajectory records with tool sequences
    - summaries: List of evaluation summary records
    - evals: List of individual evaluation records
"""
from __future__ import annotations

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TRAJECTORIES_DIR = PROJECT_ROOT / "trajectories"


def classify_error(result: str) -> tuple[str, str, str]:
    """
    Classify a tool call error into categories.

    This function analyzes the error message content to determine whether
    the error was caused by the model (wrong parameters) or the server
    (rate limits, internal errors, etc.).

    Args:
        result: The raw result string from the tool call

    Returns:
        Tuple of (error_category, error_subcategory, error_message)

    Categories:
        - MODEL_ERROR: The LLM called the tool incorrectly
            - missing_required_field: Required parameter not provided
            - wrong_type: Parameter has wrong type (e.g., int instead of string)
            - invalid_schema: Value doesn't match expected schema
            - invalid_arguments: Unexpected or invalid argument names
            - validation_error: Generic input validation failure
            - invalid_date_range: Date parameter out of valid range

        - SERVER_ERROR: The MCP server itself has issues
            - rate_limit: API rate limiting (429 errors)
            - quota_exceeded: Usage quota/cap exceeded
            - null_reference: NoneType/AttributeError in server code
            - data_processing_error: Length mismatch, data issues
            - index_error: Array index out of bounds
            - not_found: Resource not found
            - network_error: Timeout/connection issues
            - server_unavailable: Server not in configs
            - execution_error: Generic execution failure

        - UNKNOWN: Cannot determine the cause
            - unclassified: No matching pattern found
    """
    # Extract the error message from the result
    match = re.search(r"text=['\"](.+?)['\"], annotations", result, re.DOTALL)
    error_msg = match.group(1) if match else result[:200]

    # --- MODEL ERRORS (LLM called incorrectly) ---

    # Missing required property/field
    if "is a required property" in error_msg or "Field required" in error_msg:
        return "MODEL_ERROR", "missing_required_field", error_msg

    # Wrong type (e.g., passed int instead of string)
    if re.search(r"is not of type ['\"]", error_msg) or "unable to parse string as an integer" in error_msg:
        return "MODEL_ERROR", "wrong_type", error_msg

    # Invalid value/schema
    if "is not valid under any of the given schemas" in error_msg:
        return "MODEL_ERROR", "invalid_schema", error_msg

    # Invalid arguments
    if "Invalid arguments for tool" in error_msg or "Unexpected keyword argument" in error_msg:
        return "MODEL_ERROR", "invalid_arguments", error_msg

    # Input validation error (generic - likely model's fault)
    if "Input validation error" in error_msg or "validation error" in error_msg.lower():
        return "MODEL_ERROR", "validation_error", error_msg

    # --- SERVER ERRORS (Server's issue) ---

    # Rate limiting
    if "429" in error_msg or "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
        return "SERVER_ERROR", "rate_limit", error_msg

    # Usage cap exceeded
    if "cap exceeded" in error_msg.lower() or "quota" in error_msg.lower():
        return "SERVER_ERROR", "quota_exceeded", error_msg

    # Server internal errors (Python exceptions, data processing errors)
    if "'NoneType' object" in error_msg or "AttributeError" in error_msg:
        return "SERVER_ERROR", "null_reference", error_msg

    if "Length mismatch" in error_msg or "Lengths must match" in error_msg:
        return "SERVER_ERROR", "data_processing_error", error_msg

    if "Index" in error_msg and "out of" in error_msg:
        return "SERVER_ERROR", "index_error", error_msg

    # Invalid data/not found
    if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
        return "SERVER_ERROR", "not_found", error_msg

    # Future date/invalid date range - model passed bad date values
    if "cannot be after" in error_msg.lower() or "invalid date" in error_msg.lower():
        return "MODEL_ERROR", "invalid_date_range", error_msg

    # Network/connection errors
    if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
        return "SERVER_ERROR", "network_error", error_msg

    # Server not available
    if "server_not_in_configs" in error_msg or "could not be loaded" in error_msg:
        return "SERVER_ERROR", "server_unavailable", error_msg

    # Generic server error
    if "Error calling tool" in error_msg or "Error executing tool" in error_msg:
        # Check if it's still likely a model error based on content
        if "validation" in error_msg.lower():
            return "MODEL_ERROR", "validation_error", error_msg
        return "SERVER_ERROR", "execution_error", error_msg

    return "UNKNOWN", "unclassified", error_msg


def load_trajectory_data(
    traj_dir: Path = TRAJECTORIES_DIR,
    model_filter: str = None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load tool call data and trajectory-level data from trajectory files.

    This function reads all trajectory_*.json files and extracts:
    1. Individual tool call records with error classification
    2. Trajectory-level records with tool call sequences

    Args:
        traj_dir: Path to trajectories directory
        model_filter: Optional model name to filter (case-insensitive)

    Returns:
        Tuple of (tool_calls, trajectories):
        - tool_calls: List of individual tool call records
        - trajectories: List of trajectory records with tool call sequences

    Tool Call Record Fields:
        - model: Model name (from directory structure)
        - pass_number: Pass number (1, 2, 3)
        - server: MCP server name
        - tool: Tool name
        - status: Call status
        - is_error: Whether the call resulted in error
        - error_category: MODEL_ERROR, SERVER_ERROR, or UNKNOWN
        - error_subcategory: Specific error type
        - error_message: Truncated error message
        - duration_seconds: Call duration
        - dynamically_loaded: Whether server was loaded dynamically
        - trajectory_file: Source file path
        - call_index: Position in trajectory

    Trajectory Record Fields:
        - model: Model name
        - pass_number: Pass number
        - query_uuid: Query UUID
        - trajectory_file: Source file path
        - tool_sequence: List of tool calls with server, tool, is_error, error_category
        - total_tool_calls: Number of tool calls
    """
    tool_calls = []
    trajectories = []

    for traj_file in traj_dir.rglob("trajectory_*.json"):
        try:
            # Parse path to get model name
            parts = traj_file.relative_to(traj_dir).parts
            if len(parts) < 1:
                continue
            model = parts[0]

            # Apply filter
            if model_filter and model.lower() != model_filter.lower():
                continue

            with open(traj_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract tool calls from execution section
            execution = data.get("execution", {})
            calls = execution.get("tool_calls", [])

            # Get pass number from metadata
            metadata = data.get("metadata", {})
            pass_num = metadata.get("pass_number", 1)
            query_uuid = metadata.get("query_uuid", "")

            # Build trajectory record with tool call sequence
            traj_tool_sequence = []
            for i, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue

                server = call.get("server", "unknown")
                tool = call.get("tool", "unknown")
                status = call.get("status", "unknown")
                duration = call.get("duration_seconds", 0)
                result = call.get("result", "")
                dynamically_loaded = call.get("dynamically_loaded", False)

                # Check if there's an error in the result
                is_error = False
                error_category = None
                error_subcategory = None
                error_message = None

                if status != "success":
                    is_error = True
                    error_category = "SERVER_ERROR"
                    error_subcategory = "status_not_success"
                    error_message = str(result)[:200]
                elif isinstance(result, str) and "isError=True" in result:
                    is_error = True
                    error_category, error_subcategory, error_message = classify_error(result)

                tool_call_record = {
                    "model": model,
                    "pass_number": pass_num,
                    "server": server,
                    "tool": tool,
                    "status": status,
                    "is_error": is_error,
                    "error_category": error_category,
                    "error_subcategory": error_subcategory,
                    "error_message": error_message,
                    "duration_seconds": duration,
                    "dynamically_loaded": dynamically_loaded,
                    "trajectory_file": str(traj_file),
                    "call_index": i,
                }
                tool_calls.append(tool_call_record)
                traj_tool_sequence.append({
                    "server": server,
                    "tool": tool,
                    "is_error": is_error,
                    "error_category": error_category,
                })

            # Store trajectory-level data
            trajectories.append({
                "model": model,
                "pass_number": pass_num,
                "query_uuid": query_uuid,
                "trajectory_file": str(traj_file),
                "tool_sequence": traj_tool_sequence,
                "total_tool_calls": len(traj_tool_sequence),
            })

        except Exception as e:
            print(f"Warning: Could not load {traj_file}: {e}", file=sys.stderr)

    return tool_calls, trajectories


def load_summary_files(eval_dir: Path = EVALUATION_DIR) -> List[Dict[str, Any]]:
    """
    Load all _summary.json files from the evaluation directory.

    Summary files contain aggregated scores for each model/judge/pass combination.

    Args:
        eval_dir: Path to evaluation directory

    Returns:
        List of summary records with fields:
        - model: Model name
        - judge: Judge model name
        - pass_number: Pass number (or None for overall)
        - is_overall: Whether this is an overall summary
        - filepath: Source file path
        - data: Raw summary data with avg_final_answer_score, avg_step_score, etc.
    """
    summaries = []

    for summary_file in eval_dir.rglob("_summary.json"):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Parse path to extract metadata
            # Structure: evaluation/{model}/{model}_by_{judge}/pass@{N}/_summary.json
            parts = summary_file.relative_to(eval_dir).parts

            if len(parts) >= 2:
                model = parts[0]
                judge_folder = parts[1] if len(parts) > 1 else ""

                # Extract judge from folder name like "claude-3.5_by_gpt4omini"
                judge = ""
                if "_by_" in judge_folder:
                    judge = judge_folder.split("_by_")[-1]

                # Check if this is a pass-specific or overall summary
                is_pass_summary = "pass@" in str(summary_file)
                pass_num = None
                if is_pass_summary:
                    for part in parts:
                        if part.startswith("pass@"):
                            pass_num = int(part.replace("pass@", ""))
                            break

                summaries.append({
                    "model": model,
                    "judge": judge,
                    "pass_number": pass_num,
                    "is_overall": not is_pass_summary,
                    "filepath": str(summary_file),
                    "data": data,
                })
        except Exception as e:
            print(f"Warning: Could not load {summary_file}: {e}", file=sys.stderr)

    return summaries


def load_individual_evals(
    eval_dir: Path = EVALUATION_DIR,
    model: str = None,
    judge: str = None
) -> List[Dict[str, Any]]:
    """
    Load individual eval_*.json files for detailed analysis.

    Each eval file contains the full evaluation results for a single trajectory.

    Args:
        eval_dir: Path to evaluation directory
        model: Optional model name filter
        judge: Optional judge name filter

    Returns:
        List of evaluation records with fields:
        - model: Model name
        - judge: Judge model name
        - pass_number: Pass number
        - uuid: Query UUID
        - filepath: Source file path
        - data: Raw evaluation data with final_answer_evaluation, step_by_step_evaluation, etc.
    """
    evals = []

    pattern = "eval_*.json"
    for eval_file in eval_dir.rglob(pattern):
        if eval_file.name == "_summary.json":
            continue

        try:
            # Parse path
            parts = eval_file.relative_to(eval_dir).parts
            if len(parts) < 2:
                continue

            file_model = parts[0]
            judge_folder = parts[1] if len(parts) > 1 else ""
            file_judge = ""
            if "_by_" in judge_folder:
                file_judge = judge_folder.split("_by_")[-1]

            # Apply filters
            if model and file_model != model:
                continue
            if judge and file_judge != judge:
                continue

            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract pass number from path
            pass_num = 1
            for part in parts:
                if part.startswith("pass@"):
                    pass_num = int(part.replace("pass@", ""))
                    break

            evals.append({
                "model": file_model,
                "judge": file_judge,
                "pass_number": pass_num,
                "uuid": data.get("uuid"),
                "filepath": str(eval_file),
                "data": data,
            })
        except Exception as e:
            print(f"Warning: Could not load {eval_file}: {e}", file=sys.stderr)

    return evals
