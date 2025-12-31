#!/usr/bin/env python3
"""
Goal-Oriented Multi-Turn Trajectory Evaluator

This evaluator is designed for goal-oriented multi-turn trajectories that evaluate
BOTH the User LLM (goal decomposition, tracking) and Agent LLM (execution quality).

Key Features:
1. STATIC VERIFICATION (code-based, no LLM cost):
   - Agent LLM metrics: tool_calls count, search_tools usage, tool success rate, server diversity
   - Trajectory metrics: goal_completion_rate, constraint_satisfaction_rate, total_turns, termination_reason

2. LLM-AS-JUDGE EVALUATION:
   - User LLM Quality: sub-goal decomposition quality, goal tracking coherence, follow-up intent quality
   - Agent LLM Quality:
     * Step-by-step evaluation: each turn's thinking, tool execution, and final response
     * Final answer evaluation: concatenate ALL turns' agent_responses → ONE judge call

Usage:
    # Evaluate a single trajectory
    python goaloriented_evaluator.py \
        --trajectory trajectories/goaloriented/trajectory_xxx.json \
        --model openai/gpt-4o-mini

    # Evaluate a directory of trajectories
    python goaloriented_evaluator.py \
        --traj_dir trajectories/goaloriented \
        --model openai/gpt-4o-mini \
        --recursive \
        --output-dir evaluation/results

    # Static metrics only (no LLM cost)
    python goaloriented_evaluator.py \
        --traj_dir trajectories/goaloriented \
        --static-only \
        --output-dir evaluation/results
"""

from __future__ import annotations
import os
import json
import glob
import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =========================
# OpenAI-compatible LLM I/O
# =========================
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
    max_tokens: int = 8000,
    prompt_char_limit: int = 200000,
    model_name: str = "openai/gpt-4o-mini",
) -> Optional[str]:
    """Thin wrapper around OpenAI-compatible chat.completions.create."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "Please set OPENAI_API_KEY and OPENAI_BASE_URL environment variables."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    safe_prompt = prompt[:prompt_char_limit]

    tries = 3
    while tries > 0:
        tries -= 1
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": safe_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[goaloriented_evaluator] LLM call failed: {e}", file=sys.stderr)
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
# DATA CLASSES FOR EVALUATION RESULTS
# =============================================================================

@dataclass
class ToolErrorInfo:
    """Information about a tool call that resulted in an error."""
    server: str = ""
    tool: str = ""
    arguments: str = ""  # String representation of arguments
    error_message: str = ""
    is_external_failure: bool = False  # True if clearly an external API/service issue


@dataclass
class ToolFailureBreakdown:
    """
    Breakdown of tool failures.

    We use STATIC detection only for clear external failures (auth, rate limit, etc.)
    Agent misuse evaluation is left to LLM-as-Judge in step evaluation.
    """
    total_calls: int = 0
    calls_with_status_success: int = 0
    calls_with_status_error: int = 0
    # Calls that had "success" status but returned error in result
    hidden_errors: int = 0
    # External failures (clearly not agent's fault - auth, rate limits, etc.)
    external_failures: int = 0
    external_failure_details: List[ToolErrorInfo] = field(default_factory=list)


@dataclass
class AgentStaticMetrics:
    """Static metrics for Agent LLM (computed from trajectory, no LLM cost)."""
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    tool_success_rate: float = 0.0
    unique_servers_used: int = 0
    servers_used: List[str] = field(default_factory=list)
    meta_mcp_calls: int = 0  # Tool discovery/registry calls via meta-mcp server
    has_meaningful_response: bool = False
    total_reasoning_steps: int = 0
    avg_reasoning_steps_per_turn: float = 0.0
    # New: detailed tool failure analysis
    tool_failure_breakdown: ToolFailureBreakdown = field(default_factory=ToolFailureBreakdown)


@dataclass
class ConstraintVerification:
    """Static verification result for a single constraint."""
    constraint_type: str = ""
    description: str = ""
    satisfied: bool = False
    details: str = ""  # Explanation of why satisfied/violated
    # For quantifiable constraints
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None


@dataclass
class ConstraintVerificationResult:
    """Aggregate result of all constraint verifications."""
    total_constraints: int = 0
    static_verifiable: int = 0  # Constraints we can verify with code
    llm_required: int = 0       # Constraints that need LLM judgment
    static_satisfied: int = 0
    static_violated: int = 0
    static_satisfaction_rate: float = 0.0
    # Detailed results
    verifications: List[ConstraintVerification] = field(default_factory=list)


@dataclass
class PerTurnGroundTruth:
    """
    Ground truth signals from User LLM's own evaluation during the conversation.
    These can be used to validate LLM-as-judge scores.
    """
    turn_number: int = 0
    goal_progress: float = 0.0  # 0-1: Progress toward goal
    constraint_satisfaction_rate: float = 0.0  # 0-1: Per-turn constraint satisfaction
    satisfaction_level: float = 0.0  # 0-1: User's satisfaction with agent response
    tool_calls_count: int = 0
    meta_mcp_calls_count: int = 0  # Number of meta-mcp (tool discovery) calls this turn
    completed_sub_goals_count: int = 0
    constraints_violated_count: int = 0
    user_decision: str = ""  # CONTINUE/TERMINATE


@dataclass
class GroundTruthComparison:
    """
    Comparison between ground truth (User LLM signals) and LLM-as-judge scores.
    Useful for validating judge reliability.
    """
    per_turn_ground_truth: List[PerTurnGroundTruth] = field(default_factory=list)

    # Aggregate ground truth metrics
    avg_satisfaction_level: float = 0.0
    avg_goal_progress: float = 0.0
    avg_constraint_satisfaction: float = 0.0
    total_constraints_violated: int = 0

    # Correlation analysis (computed after LLM evaluation)
    # These compare User LLM's satisfaction with Judge's step scores
    satisfaction_vs_step_score_correlation: Optional[float] = None
    goal_progress_vs_final_score_correlation: Optional[float] = None


@dataclass
class TrajectoryMetrics:
    """Static metrics for the trajectory overall (from metadata)."""
    goal_completion_rate: float = 0.0
    overall_constraint_satisfaction_rate: float = 0.0
    total_turns: int = 0
    total_sub_goals: int = 0
    completed_sub_goals: int = 0
    final_termination_reason: str = ""
    final_satisfaction_level: float = 0.0
    goal_achieved: bool = False
    # Constraint verification
    constraint_verification: ConstraintVerificationResult = field(default_factory=ConstraintVerificationResult)
    # Ground truth comparison
    ground_truth: GroundTruthComparison = field(default_factory=GroundTruthComparison)


@dataclass
class UserLLMQualityScores:
    """LLM-as-Judge scores for User LLM quality."""
    subgoal_decomposition_quality: float = 0.0  # 0-10: How well are sub-goals defined?
    goal_tracking_coherence: float = 0.0        # 0-10: How well does user track progress?
    follow_up_intent_quality: float = 0.0       # 0-10: How relevant are follow-up queries?
    overall_user_quality: float = 0.0           # 0-10: Overall user LLM quality
    reasoning: str = ""


@dataclass
class AgentStepEvaluation:
    """Evaluation of a single turn/step."""
    turn_number: int = 0
    thinking_quality: float = 0.0       # 0-10: Quality of reasoning/thinking
    tool_selection_quality: float = 0.0 # 0-10: Appropriateness of tool choices
    tool_execution_quality: float = 0.0 # 0-10: How well tools were used
    response_quality: float = 0.0       # 0-10: Quality of final response for this turn
    grounding: float = 0.0              # 0-10: Are claims in the response grounded in tool results?
    step_overall: float = 0.0           # 0-10: Overall step quality
    reasoning: str = ""


@dataclass
class AgentFinalAnswerEvaluation:
    """Evaluation of concatenated final answers across all turns."""
    completeness: float = 0.0           # 0-10: Did agent address all parts of the goal?
    coherence: float = 0.0              # 0-10: Do answers across turns form coherent narrative?
    actionability: float = 0.0          # 0-10: Are responses actionable/useful?
    constraint_adherence: float = 0.0   # 0-10: Did agent respect stated constraints?
    overall_final_answer: float = 0.0   # 0-10: Overall quality of combined answers
    reasoning: str = ""


@dataclass
class FullEvaluationResult:
    """Complete evaluation result for a trajectory."""
    trajectory_file: str = ""
    trajectory_uuid: str = ""

    # Static metrics
    agent_metrics: AgentStaticMetrics = field(default_factory=AgentStaticMetrics)
    trajectory_metrics: TrajectoryMetrics = field(default_factory=TrajectoryMetrics)

    # LLM-as-Judge scores
    user_llm_quality: Optional[UserLLMQualityScores] = None
    agent_step_evaluations: List[AgentStepEvaluation] = field(default_factory=list)
    agent_final_answer: Optional[AgentFinalAnswerEvaluation] = None

    # Summary scores
    agent_step_avg_score: float = 0.0
    overall_score: float = 0.0

    # Metadata
    evaluation_timestamp: str = ""
    evaluation_model: str = ""
    static_only: bool = False


# =============================================================================
# STATIC VERIFICATION MODULE
# =============================================================================

# Patterns for EXTERNAL FAILURES (clearly not agent's fault)
# These are deterministic - no LLM needed
EXTERNAL_FAILURE_PATTERNS = [
    # Authentication/Authorization (API misconfiguration, not agent issue)
    "access_token is required",
    "api_key is required",
    "api key is required",
    "fmp_access_token is required",
    "unauthorized",
    "401",
    "403 forbidden",
    "invalid api key",
    "expired token",
    # Rate limits (external constraint)
    "rate limit",
    "too many requests",
    "quota exceeded",
    "429",
    # Server errors (external service issues)
    "500",
    "502",
    "503",
    "504",
    "service unavailable",
    "internal server error",
    "connection refused",
    "connection timeout",
    "network error",
]


def is_external_failure(result_content: str) -> Tuple[bool, str]:
    """
    Check if a tool result indicates an EXTERNAL failure (not agent's fault).

    Returns:
        Tuple of (is_external_failure, matched_pattern)
    """
    if not result_content:
        return (False, "")

    content_lower = result_content.lower()

    for pattern in EXTERNAL_FAILURE_PATTERNS:
        if pattern in content_lower:
            return (True, pattern)

    return (False, "")


def compute_agent_static_metrics(trajectory: Dict[str, Any]) -> AgentStaticMetrics:
    """
    Compute static metrics for Agent LLM from trajectory data.

    For tool error analysis:
    - STATIC: Detect external failures (auth, rate limits, server errors) - not agent's fault
    - LLM: Agent misuse is evaluated in step-by-step LLM evaluation (tool_selection/execution quality)
    """
    metrics = AgentStaticMetrics()

    turns = trajectory.get("turns", [])
    if not turns:
        return metrics

    servers_used = set()
    breakdown = ToolFailureBreakdown()

    for turn in turns:
        # Count tool calls
        tool_calls = turn.get("tool_calls", [])
        reasoning_trace = turn.get("reasoning_trace", [])

        metrics.total_tool_calls += len(tool_calls)
        breakdown.total_calls += len(tool_calls)

        # Extract results from reasoning trace to match with tool calls
        results = []
        for item in reasoning_trace:
            if item.get("type", "").lower() in ("result", "tool result", "tool_result"):
                results.append(str(item.get("content", "")))

        for i, tc in enumerate(tool_calls):
            status = tc.get("status", "")

            if status == "success":
                metrics.successful_tool_calls += 1
                breakdown.calls_with_status_success += 1
            else:
                metrics.failed_tool_calls += 1
                breakdown.calls_with_status_error += 1

            # Track servers
            server = tc.get("server", "")
            tool_name = tc.get("tool", "")
            if server:
                servers_used.add(server)

            # Count meta-mcp calls (tool discovery/registry)
            if server.lower() == "meta-mcp":
                metrics.meta_mcp_calls += 1

            # Check result for hidden errors (status=success but result contains error)
            result_content = results[i] if i < len(results) else ""
            if result_content:
                is_ext_fail, pattern = is_external_failure(result_content)

                if is_ext_fail:
                    breakdown.external_failures += 1
                    breakdown.external_failure_details.append(ToolErrorInfo(
                        server=server,
                        tool=tool_name,
                        arguments=str(tc.get("arguments", {})),  # No truncation
                        error_message=pattern,
                        is_external_failure=True,
                    ))
                    # Count as hidden error if status was "success"
                    if status == "success":
                        breakdown.hidden_errors += 1

        # Count reasoning steps
        step_count = sum(1 for item in reasoning_trace if item.get("type", "").lower() == "thought")
        metrics.total_reasoning_steps += step_count

        # Check for meaningful response
        agent_response = turn.get("agent_response", "")
        if isinstance(agent_response, dict):
            agent_response = json.dumps(agent_response)
        if agent_response and len(str(agent_response)) > 100:
            if not any(p in str(agent_response).lower() for p in ['timed out', 'error occurred', 'failed to']):
                metrics.has_meaningful_response = True

    # Store breakdown
    metrics.tool_failure_breakdown = breakdown

    # Compute derived metrics
    metrics.servers_used = sorted(list(servers_used))
    metrics.unique_servers_used = len(servers_used)

    if metrics.total_tool_calls > 0:
        metrics.tool_success_rate = metrics.successful_tool_calls / metrics.total_tool_calls

    if len(turns) > 0:
        metrics.avg_reasoning_steps_per_turn = metrics.total_reasoning_steps / len(turns)

    return metrics


def verify_constraints_static(trajectory: Dict[str, Any]) -> ConstraintVerificationResult:
    """
    Verify constraints using STATIC analysis (code-based, no LLM).

    Constraint types we can verify statically:
    - SERVER_DIVERSITY: min_servers requirement
    - NO_REDUNDANCY: duplicate tool calls detection
    - SEQUENCE_ORDER: check tool call ordering patterns
    - DATA_COVERAGE: check if required entities appear in tool args
    - TOOL_COUNT: min_calls, max_calls

    Constraint types requiring LLM judgment:
    - RESPONSE_CONTENT: must_include keywords, recommendations
    - TRADEOFF: explicit reasoning about tradeoffs
    """
    result = ConstraintVerificationResult()

    metadata = trajectory.get("metadata", {})
    constraints = metadata.get("constraints", [])
    turns = trajectory.get("turns", [])

    result.total_constraints = len(constraints)

    # Gather all tool calls across turns
    all_tool_calls = []
    all_tool_args_str = ""
    servers_used = set()

    for turn in turns:
        for tc in turn.get("tool_calls", []):
            all_tool_calls.append(tc)
            server = tc.get("server", "")
            if server:
                servers_used.add(server)
            # Collect all arguments as string for entity search
            args = tc.get("arguments", {})
            all_tool_args_str += " " + json.dumps(args).lower()

    total_calls = len(all_tool_calls)

    for constraint in constraints:
        c_type = constraint.get("type", "")
        c_desc = constraint.get("description", "")
        verification = constraint.get("verification", {})

        v = ConstraintVerification(
            constraint_type=c_type,
            description=c_desc,
        )

        # =====================================================================
        # SERVER_DIVERSITY: Check min_servers requirement
        # =====================================================================
        if c_type == "SERVER_DIVERSITY":
            min_servers = verification.get("min_servers", 1)
            actual_servers = len(servers_used)
            v.expected_value = f"min {min_servers} servers"
            v.actual_value = f"{actual_servers} servers"
            v.satisfied = actual_servers >= min_servers
            v.details = f"Used {actual_servers} unique servers: {sorted(servers_used)}"
            result.static_verifiable += 1
            if v.satisfied:
                result.static_satisfied += 1
            else:
                result.static_violated += 1

        # =====================================================================
        # NO_REDUNDANCY: Detect duplicate tool calls
        # =====================================================================
        elif c_type == "NO_REDUNDANCY":
            # Check for duplicate (server, tool, args) combinations
            call_signatures = []
            duplicates = []
            for tc in all_tool_calls:
                sig = (tc.get("server", ""), tc.get("tool", ""), json.dumps(tc.get("arguments", {}), sort_keys=True))
                if sig in call_signatures:
                    duplicates.append(f"{sig[0]}/{sig[1]}")
                else:
                    call_signatures.append(sig)

            v.expected_value = "no duplicate calls"
            v.actual_value = f"{len(duplicates)} duplicates"
            v.satisfied = len(duplicates) == 0
            if duplicates:
                v.details = f"Duplicate calls found: {duplicates[:5]}"
            else:
                v.details = "No duplicate tool calls detected"
            result.static_verifiable += 1
            if v.satisfied:
                result.static_satisfied += 1
            else:
                result.static_violated += 1

        # =====================================================================
        # SEQUENCE_ORDER: Check if discovery tools come before detail tools
        # =====================================================================
        elif c_type == "SEQUENCE_ORDER":
            required_sequences = verification.get("required_sequence", [])
            # e.g., [["search", "fetch"], ["list", "get"]]
            violations = []

            for seq in required_sequences:
                if len(seq) >= 2:
                    first_pattern = seq[0].lower()
                    second_pattern = seq[1].lower()

                    # Find positions of tools matching patterns
                    first_positions = []
                    second_positions = []

                    for i, tc in enumerate(all_tool_calls):
                        tool_name = tc.get("tool", "").lower()
                        if first_pattern in tool_name:
                            first_positions.append(i)
                        if second_pattern in tool_name:
                            second_positions.append(i)

                    # Check if any "second" appears before all "first"
                    if second_positions and first_positions:
                        min_first = min(first_positions)
                        for pos in second_positions:
                            if pos < min_first:
                                violations.append(f"{second_pattern} before {first_pattern}")
                                break

            v.expected_value = f"sequences: {required_sequences}"
            v.actual_value = f"{len(violations)} violations"
            v.satisfied = len(violations) == 0
            v.details = f"Sequence violations: {violations}" if violations else "Tool ordering follows required sequences"
            result.static_verifiable += 1
            if v.satisfied:
                result.static_satisfied += 1
            else:
                result.static_violated += 1

        # =====================================================================
        # DATA_COVERAGE: Check if required entities appear in tool arguments
        # =====================================================================
        elif c_type == "DATA_COVERAGE":
            entities = verification.get("entities", [])
            entity_type = verification.get("entity_type", "entities")

            found = []
            missing = []
            for entity in entities:
                if entity.lower() in all_tool_args_str:
                    found.append(entity)
                else:
                    missing.append(entity)

            v.expected_value = f"all {len(entities)} {entity_type}"
            v.actual_value = f"{len(found)}/{len(entities)} found"
            v.satisfied = len(missing) == 0
            if missing:
                v.details = f"Missing entities: {missing}"
            else:
                v.details = f"All required {entity_type} covered: {entities}"
            result.static_verifiable += 1
            if v.satisfied:
                result.static_satisfied += 1
            else:
                result.static_violated += 1

        # =====================================================================
        # TOOL_COUNT: Check min/max tool calls
        # =====================================================================
        elif c_type == "TOOL_COUNT":
            min_calls = verification.get("min_calls", 0)
            max_calls = verification.get("max_calls", float('inf'))

            v.expected_value = f"{min_calls}-{max_calls} calls"
            v.actual_value = f"{total_calls} calls"
            v.satisfied = min_calls <= total_calls <= max_calls
            if total_calls < min_calls:
                v.details = f"Too few tool calls ({total_calls} < {min_calls})"
            elif total_calls > max_calls:
                v.details = f"Too many tool calls ({total_calls} > {max_calls})"
            else:
                v.details = f"Tool count {total_calls} within range [{min_calls}, {max_calls}]"
            result.static_verifiable += 1
            if v.satisfied:
                result.static_satisfied += 1
            else:
                result.static_violated += 1

        # =====================================================================
        # LLM-REQUIRED CONSTRAINTS: Mark for later evaluation
        # =====================================================================
        elif c_type in ("RESPONSE_CONTENT", "TRADEOFF"):
            v.details = "Requires LLM judgment - evaluated in step/final answer evaluation"
            result.llm_required += 1

        else:
            # Unknown constraint type - mark as needing LLM
            v.details = f"Unknown constraint type '{c_type}' - requires LLM judgment"
            result.llm_required += 1

        result.verifications.append(v)

    # Compute static satisfaction rate
    if result.static_verifiable > 0:
        result.static_satisfaction_rate = result.static_satisfied / result.static_verifiable

    return result


def compute_trajectory_metrics(trajectory: Dict[str, Any]) -> TrajectoryMetrics:
    """Compute static trajectory-level metrics from metadata and turns."""
    metrics = TrajectoryMetrics()

    metadata = trajectory.get("metadata", {})
    turns = trajectory.get("turns", [])

    # From metadata
    metrics.goal_completion_rate = metadata.get("goal_completion_rate", 0.0)
    metrics.overall_constraint_satisfaction_rate = metadata.get("overall_constraint_satisfaction_rate", 0.0)
    metrics.goal_achieved = metadata.get("goal_achieved", False)

    # Sub-goals
    sub_goals = metadata.get("sub_goals", [])
    metrics.total_sub_goals = len(sub_goals)

    # From turns
    metrics.total_turns = len(turns)

    if turns:
        last_turn = turns[-1]
        metrics.final_termination_reason = last_turn.get("termination_reason", "")
        metrics.final_satisfaction_level = last_turn.get("satisfaction_level", 0.0)

        # Count completed sub-goals from last turn
        completed = last_turn.get("completed_sub_goals", [])
        metrics.completed_sub_goals = len(completed) if isinstance(completed, list) else 0

    # Verify constraints statically
    metrics.constraint_verification = verify_constraints_static(trajectory)

    # Extract ground truth signals from User LLM evaluations
    metrics.ground_truth = extract_ground_truth(trajectory)

    return metrics


def extract_ground_truth(trajectory: Dict[str, Any]) -> GroundTruthComparison:
    """
    Extract ground truth signals from User LLM's own evaluation during the conversation.
    These serve as numerical "labels" that can validate LLM-as-judge scores.
    """
    result = GroundTruthComparison()
    turns = trajectory.get("turns", [])

    total_satisfaction = 0.0
    total_goal_progress = 0.0
    total_constraint_sat = 0.0
    total_constraints_violated = 0

    for turn in turns:
        # Count tool calls and meta-mcp calls for this turn
        tool_calls = turn.get("tool_calls", [])
        tool_calls_count = len(tool_calls)
        meta_mcp_calls_count = sum(
            1 for tc in tool_calls
            if tc.get("server", "").lower() == "meta-mcp"
        )

        gt = PerTurnGroundTruth(
            turn_number=turn.get("turn_number", 0),
            goal_progress=turn.get("goal_progress", 0.0),
            constraint_satisfaction_rate=turn.get("constraint_satisfaction_rate", 0.0),
            satisfaction_level=turn.get("satisfaction_level", 0.0),
            tool_calls_count=tool_calls_count,
            meta_mcp_calls_count=meta_mcp_calls_count,
            completed_sub_goals_count=len(turn.get("completed_sub_goals", [])),
            constraints_violated_count=len(turn.get("constraints_violated", [])),
            user_decision=turn.get("user_decision", ""),
        )
        result.per_turn_ground_truth.append(gt)

        total_satisfaction += gt.satisfaction_level
        total_goal_progress += gt.goal_progress
        total_constraint_sat += gt.constraint_satisfaction_rate
        total_constraints_violated += gt.constraints_violated_count

    # Compute averages
    if turns:
        n = len(turns)
        result.avg_satisfaction_level = total_satisfaction / n
        result.avg_goal_progress = total_goal_progress / n
        result.avg_constraint_satisfaction = total_constraint_sat / n
        result.total_constraints_violated = total_constraints_violated

    return result


def compute_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient between two lists."""
    if len(x) != len(y) or len(x) < 2:
        return None

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return None

    return numerator / (denom_x * denom_y)


# =============================================================================
# LLM-AS-JUDGE: USER LLM QUALITY
# =============================================================================

USER_LLM_QUALITY_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of a User LLM's behavior in a goal-oriented multi-turn conversation.

The User LLM is responsible for:
1. Decomposing the main goal into actionable sub-goals
2. Tracking progress toward the goal
3. Deciding when to continue or terminate
4. Generating coherent follow-up queries

You will evaluate these aspects on a 0-10 scale where:
- 0-2: Very poor quality
- 3-4: Below average
- 5-6: Average/acceptable
- 7-8: Good quality
- 9-10: Excellent quality

Respond with a JSON object containing:
{
    "subgoal_decomposition_quality": <0-10>,
    "goal_tracking_coherence": <0-10>,
    "follow_up_intent_quality": <0-10>,
    "overall_user_quality": <0-10>,
    "reasoning": "<brief explanation of scores>"
}
"""


def evaluate_user_llm_quality(trajectory: Dict[str, Any], model_name: str) -> Optional[UserLLMQualityScores]:
    """Evaluate User LLM quality using LLM-as-Judge."""
    metadata = trajectory.get("metadata", {})
    turns = trajectory.get("turns", [])

    if not turns:
        return None

    # Build context for evaluation
    user_goal = metadata.get("user_goal", metadata.get("seed_query", ""))
    sub_goals = metadata.get("sub_goals", [])

    # Collect user decisions and follow-ups across turns
    turn_summaries = []
    for turn in turns:
        turn_summary = {
            "turn": turn.get("turn_number", 0),
            "query": turn.get("query", ""),  # No truncation
            "user_decision": turn.get("user_decision", ""),
            "satisfaction_level": turn.get("satisfaction_level", 0),
            "user_reasoning": turn.get("user_reasoning", ""),
            "follow_up_intent": turn.get("follow_up_intent", ""),
            "completed_sub_goals": turn.get("completed_sub_goals", []),
            "remaining_sub_goals": turn.get("remaining_sub_goals", []),
            "goal_progress": turn.get("goal_progress", 0),
        }
        turn_summaries.append(turn_summary)

    prompt = f"""Evaluate the User LLM's quality in this goal-oriented conversation.

## Original User Goal:
{user_goal}

## Sub-Goals Defined:
{json.dumps(sub_goals, indent=2)}

## Turn-by-Turn User Behavior:
{json.dumps(turn_summaries, indent=2)}

## Final Metrics:
- Goal Completion Rate: {metadata.get('goal_completion_rate', 0)}
- Constraint Satisfaction Rate: {metadata.get('overall_constraint_satisfaction_rate', 0)}
- Goal Achieved: {metadata.get('goal_achieved', False)}

Please evaluate the User LLM's quality and respond with a JSON object."""

    response = _call_llm(
        prompt=prompt,
        system_prompt=USER_LLM_QUALITY_SYSTEM_PROMPT,
        model_name=model_name,
    )

    parsed = _safe_parse_json(response)
    if not parsed:
        return None

    return UserLLMQualityScores(
        subgoal_decomposition_quality=float(parsed.get("subgoal_decomposition_quality", 0)),
        goal_tracking_coherence=float(parsed.get("goal_tracking_coherence", 0)),
        follow_up_intent_quality=float(parsed.get("follow_up_intent_quality", 0)),
        overall_user_quality=float(parsed.get("overall_user_quality", 0)),
        reasoning=parsed.get("reasoning", ""),
    )


# =============================================================================
# LLM-AS-JUDGE: AGENT STEP-BY-STEP EVALUATION
# =============================================================================

AGENT_STEP_SYSTEM_PROMPT = """You are an expert evaluator assessing an AI agent's performance on a single turn of a multi-turn conversation.

For this turn, evaluate:
1. Thinking Quality: How well did the agent reason about the task?
2. Tool Selection Quality: Did the agent choose appropriate tools?
3. Tool Execution Quality: Did the agent use tools correctly with proper arguments?
4. Response Quality: How well did the agent respond to the user's query?
5. Grounding: Are the claims in the agent's response grounded in the tool results? Does the response reference and use the actual data returned by tools, or does it hallucinate/make unsupported claims?

IMPORTANT - Tool Definition:
- Search tools (search_tools server) are NOT considered actual tool usage
- If the agent made NO tool calls OR only used search_tools, score Tool Selection Quality, Tool Execution Quality, AND Grounding as 0
- Only tools from other servers (non-search_tools) count as actual tool usage

Score each on 0-10 scale:
- 0-2: Very poor
- 3-4: Below average
- 5-6: Average
- 7-8: Good
- 9-10: Excellent

Respond with a JSON object:
{
    "thinking_quality": <0-10>,
    "tool_selection_quality": <0-10>,
    "tool_execution_quality": <0-10>,
    "response_quality": <0-10>,
    "grounding": <0-10>,
    "step_overall": <0-10>,
    "reasoning": "<brief explanation>"
}
"""


def evaluate_agent_step(
    turn: Dict[str, Any],
    user_goal: str,
    constraints: List[Dict[str, Any]],
    model_name: str,
) -> Optional[AgentStepEvaluation]:
    """Evaluate a single turn/step of the agent."""
    turn_number = turn.get("turn_number", 0)
    query = turn.get("query", "")
    agent_response = turn.get("agent_response", "")
    if isinstance(agent_response, dict):
        agent_response = json.dumps(agent_response)

    tool_calls = turn.get("tool_calls", [])
    reasoning_trace = turn.get("reasoning_trace", [])

    # Build reasoning trace summary - NO TRUNCATION
    trace_summary = []
    for item in reasoning_trace:  # Include all items
        trace_type = item.get("type", "")
        content = str(item.get("content", ""))
        trace_summary.append({"type": trace_type, "content": content})

    # Build tool calls summary - NO TRUNCATION
    tool_summary = []
    for tc in tool_calls:  # Include all tool calls
        tool_summary.append({
            "server": tc.get("server", ""),
            "tool": tc.get("tool", ""),
            "status": tc.get("status", ""),
            "arguments": str(tc.get("arguments", {})),  # No truncation
        })

    prompt = f"""Evaluate the agent's performance on Turn {turn_number}.

## User Goal (Context):
{user_goal}

## This Turn's Query:
{query}

## Agent's Reasoning Trace:
{json.dumps(trace_summary, indent=2)}

## Tool Calls Made:
{json.dumps(tool_summary, indent=2)}

## Agent's Response:
{str(agent_response)}

## Constraints to Consider:
{json.dumps(constraints, indent=2)}

Please evaluate this turn and respond with a JSON object."""

    response = _call_llm(
        prompt=prompt,
        system_prompt=AGENT_STEP_SYSTEM_PROMPT,
        model_name=model_name,
    )

    parsed = _safe_parse_json(response)
    if not parsed:
        return None

    return AgentStepEvaluation(
        turn_number=turn_number,
        thinking_quality=float(parsed.get("thinking_quality", 0)),
        tool_selection_quality=float(parsed.get("tool_selection_quality", 0)),
        tool_execution_quality=float(parsed.get("tool_execution_quality", 0)),
        response_quality=float(parsed.get("response_quality", 0)),
        grounding=float(parsed.get("grounding", 0)),
        step_overall=float(parsed.get("step_overall", 0)),
        reasoning=parsed.get("reasoning", ""),
    )


# =============================================================================
# LLM-AS-JUDGE: AGENT FINAL ANSWER (CONCATENATED)
# =============================================================================

AGENT_FINAL_ANSWER_SYSTEM_PROMPT = """You are an expert evaluator assessing the overall quality of an AI agent's responses across a multi-turn conversation.

You will see the CONCATENATED final answers from ALL turns of the conversation. Evaluate:
1. Completeness: Did the agent address all parts of the user's goal across all turns?
2. Coherence: Do the responses across turns form a coherent, non-contradictory narrative?
3. Actionability: Are the responses useful and actionable for the user?
4. Constraint Adherence: Did the agent respect the stated constraints throughout?

Score each on 0-10 scale:
- 0-2: Very poor
- 3-4: Below average
- 5-6: Average
- 7-8: Good
- 9-10: Excellent

Respond with a JSON object:
{
    "completeness": <0-10>,
    "coherence": <0-10>,
    "actionability": <0-10>,
    "constraint_adherence": <0-10>,
    "overall_final_answer": <0-10>,
    "reasoning": "<comprehensive explanation of the overall assessment>"
}
"""


def evaluate_agent_final_answer(
    trajectory: Dict[str, Any],
    model_name: str,
) -> Optional[AgentFinalAnswerEvaluation]:
    """
    Evaluate agent's final answer by CONCATENATING all turns' agent_responses.
    This provides ONE holistic evaluation of the complete conversation output.
    """
    metadata = trajectory.get("metadata", {})
    turns = trajectory.get("turns", [])

    if not turns:
        return None

    user_goal = metadata.get("user_goal", metadata.get("seed_query", ""))
    sub_goals = metadata.get("sub_goals", [])
    constraints = metadata.get("constraints", [])

    # CONCATENATE all agent responses
    concatenated_responses = []
    for turn in turns:
        turn_number = turn.get("turn_number", 0)
        agent_response = turn.get("agent_response", "")
        if isinstance(agent_response, dict):
            agent_response = json.dumps(agent_response)

        # Add turn separator and response
        concatenated_responses.append(f"\n{'='*50}")
        concatenated_responses.append(f"TURN {turn_number} RESPONSE:")
        concatenated_responses.append(f"{'='*50}")
        concatenated_responses.append(str(agent_response))

    all_responses = "\n".join(concatenated_responses)

    # No truncation - include all responses
    prompt = f"""Evaluate the agent's OVERALL performance by examining the concatenated responses from ALL turns.

## User Goal:
{user_goal}

## Sub-Goals to Achieve:
{json.dumps(sub_goals, indent=2)}

## Constraints:
{json.dumps(constraints, indent=2)}

## CONCATENATED AGENT RESPONSES (ALL TURNS):
{all_responses}

## Trajectory Outcome:
- Goal Completion Rate: {metadata.get('goal_completion_rate', 0)}
- Goal Achieved: {metadata.get('goal_achieved', False)}
- Constraint Satisfaction Rate: {metadata.get('overall_constraint_satisfaction_rate', 0)}

Please provide a comprehensive evaluation of the agent's overall performance and respond with a JSON object."""

    response = _call_llm(
        prompt=prompt,
        system_prompt=AGENT_FINAL_ANSWER_SYSTEM_PROMPT,
        model_name=model_name,
        max_tokens=4000,
    )

    parsed = _safe_parse_json(response)
    if not parsed:
        return None

    return AgentFinalAnswerEvaluation(
        completeness=float(parsed.get("completeness", 0)),
        coherence=float(parsed.get("coherence", 0)),
        actionability=float(parsed.get("actionability", 0)),
        constraint_adherence=float(parsed.get("constraint_adherence", 0)),
        overall_final_answer=float(parsed.get("overall_final_answer", 0)),
        reasoning=parsed.get("reasoning", ""),
    )


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_trajectory(
    trajectory: Dict[str, Any],
    model_name: str = "openai/gpt-4o-mini",
    static_only: bool = False,
    evaluate_all_steps: bool = True,
) -> FullEvaluationResult:
    """
    Evaluate a single goal-oriented trajectory.

    Args:
        trajectory: The trajectory data
        model_name: LLM model to use for evaluation
        static_only: If True, only compute static metrics (no LLM calls)
        evaluate_all_steps: If True, evaluate each turn individually

    Returns:
        FullEvaluationResult with all metrics and scores
    """
    result = FullEvaluationResult()
    result.trajectory_file = trajectory.get("_filename", "")
    result.trajectory_uuid = trajectory.get("metadata", {}).get("uuid", "")
    result.evaluation_timestamp = datetime.now().isoformat()
    result.evaluation_model = model_name
    result.static_only = static_only

    # 1. Compute static metrics (always)
    result.agent_metrics = compute_agent_static_metrics(trajectory)
    result.trajectory_metrics = compute_trajectory_metrics(trajectory)

    if static_only:
        return result

    # 2. LLM-as-Judge: User LLM Quality
    result.user_llm_quality = evaluate_user_llm_quality(trajectory, model_name)

    # 3. LLM-as-Judge: Agent Step-by-Step Evaluation
    if evaluate_all_steps:
        metadata = trajectory.get("metadata", {})
        user_goal = metadata.get("user_goal", metadata.get("seed_query", ""))
        constraints = metadata.get("constraints", [])
        turns = trajectory.get("turns", [])

        for turn in turns:
            step_eval = evaluate_agent_step(turn, user_goal, constraints, model_name)
            if step_eval:
                result.agent_step_evaluations.append(step_eval)

        # Compute average step score
        if result.agent_step_evaluations:
            total_step_scores = sum(e.step_overall for e in result.agent_step_evaluations)
            result.agent_step_avg_score = total_step_scores / len(result.agent_step_evaluations)

    # 4. LLM-as-Judge: Agent Final Answer (Concatenated)
    result.agent_final_answer = evaluate_agent_final_answer(trajectory, model_name)

    # 5. Compute overall score
    scores = []
    if result.user_llm_quality:
        scores.append(result.user_llm_quality.overall_user_quality)
    if result.agent_step_evaluations:
        scores.append(result.agent_step_avg_score)
    if result.agent_final_answer:
        scores.append(result.agent_final_answer.overall_final_answer)

    if scores:
        result.overall_score = sum(scores) / len(scores)

    # 6. Compute correlation between ground truth and LLM-as-judge scores
    # This validates whether judge agrees with User LLM's own evaluations
    gt = result.trajectory_metrics.ground_truth
    if result.agent_step_evaluations and len(gt.per_turn_ground_truth) >= 2:
        # Compare User LLM's satisfaction_level with Judge's step_overall score
        user_satisfaction = [t.satisfaction_level for t in gt.per_turn_ground_truth]
        judge_step_scores = [e.step_overall / 10.0 for e in result.agent_step_evaluations]  # Normalize to 0-1

        if len(user_satisfaction) == len(judge_step_scores):
            gt.satisfaction_vs_step_score_correlation = compute_correlation(
                user_satisfaction, judge_step_scores
            )

    return result


# =============================================================================
# FILE I/O AND BATCH PROCESSING
# =============================================================================

def load_trajectory(filepath: str) -> Optional[Dict[str, Any]]:
    """Load a single trajectory file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # Validate it's a goal-oriented trajectory
        if "turns" not in obj:
            print(f"[WARN] {filepath} doesn't have 'turns' field, skipping", file=sys.stderr)
            return None

        obj["_filename"] = os.path.basename(filepath)
        obj["_filepath"] = filepath
        return obj
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}", file=sys.stderr)
        return None


def load_trajectories_from_dir(dirpath: str, recursive: bool = False) -> List[Dict[str, Any]]:
    """Load all goal-oriented trajectories from a directory."""
    trajs = []

    if recursive:
        patterns = [
            os.path.join(dirpath, "**", "trajectory_*.json"),
            os.path.join(dirpath, "**", "goal_*.json"),
        ]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(pattern, recursive=True))
    else:
        patterns = [
            os.path.join(dirpath, "trajectory_*.json"),
            os.path.join(dirpath, "goal_*.json"),
        ]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(pattern))

    paths = list(set(paths))  # Deduplicate

    for p in paths:
        traj = load_trajectory(p)
        if traj:
            trajs.append(traj)

    return trajs


def save_evaluation_result(result: FullEvaluationResult, output_dir: str):
    """Save evaluation result to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename from trajectory file
    base_name = os.path.splitext(result.trajectory_file)[0]
    output_file = os.path.join(output_dir, f"eval_{base_name}.json")

    # Convert to dict
    result_dict = asdict(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    return output_file


def generate_summary_report(results: List[FullEvaluationResult], output_dir: str):
    """Generate a summary report of all evaluations."""
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "total_trajectories": len(results),
        "evaluation_timestamp": datetime.now().isoformat(),
        "aggregate_metrics": {},
        "score_distribution": {},
        "trajectories": [],
    }

    # Aggregate metrics
    if results:
        # Static metrics averages
        total_tool_calls = sum(r.agent_metrics.total_tool_calls for r in results)
        avg_tool_success = sum(r.agent_metrics.tool_success_rate for r in results) / len(results)
        avg_servers = sum(r.agent_metrics.unique_servers_used for r in results) / len(results)

        # Trajectory metrics averages
        avg_goal_completion = sum(r.trajectory_metrics.goal_completion_rate for r in results) / len(results)
        avg_constraint_sat = sum(r.trajectory_metrics.overall_constraint_satisfaction_rate for r in results) / len(results)
        goals_achieved = sum(1 for r in results if r.trajectory_metrics.goal_achieved)

        summary["aggregate_metrics"] = {
            "avg_tool_calls_per_trajectory": total_tool_calls / len(results),
            "avg_tool_success_rate": avg_tool_success,
            "avg_servers_used": avg_servers,
            "avg_goal_completion_rate": avg_goal_completion,
            "avg_constraint_satisfaction_rate": avg_constraint_sat,
            "goals_achieved_rate": goals_achieved / len(results),
        }

        # LLM scores (if available)
        llm_results = [r for r in results if not r.static_only]
        if llm_results:
            user_scores = [r.user_llm_quality.overall_user_quality for r in llm_results if r.user_llm_quality]
            step_scores = [r.agent_step_avg_score for r in llm_results if r.agent_step_evaluations]
            final_scores = [r.agent_final_answer.overall_final_answer for r in llm_results if r.agent_final_answer]
            overall_scores = [r.overall_score for r in llm_results if r.overall_score > 0]

            if user_scores:
                summary["aggregate_metrics"]["avg_user_llm_quality"] = sum(user_scores) / len(user_scores)
            if step_scores:
                summary["aggregate_metrics"]["avg_agent_step_score"] = sum(step_scores) / len(step_scores)
            if final_scores:
                summary["aggregate_metrics"]["avg_agent_final_answer_score"] = sum(final_scores) / len(final_scores)
            if overall_scores:
                summary["aggregate_metrics"]["avg_overall_score"] = sum(overall_scores) / len(overall_scores)

        # Individual trajectory summaries
        for r in results:
            traj_summary = {
                "file": r.trajectory_file,
                "uuid": r.trajectory_uuid,
                "goal_completion_rate": r.trajectory_metrics.goal_completion_rate,
                "goal_achieved": r.trajectory_metrics.goal_achieved,
                "total_turns": r.trajectory_metrics.total_turns,
                "tool_calls": r.agent_metrics.total_tool_calls,
                "tool_success_rate": r.agent_metrics.tool_success_rate,
            }
            if not r.static_only:
                traj_summary["overall_score"] = r.overall_score
                if r.user_llm_quality:
                    traj_summary["user_llm_score"] = r.user_llm_quality.overall_user_quality
                if r.agent_final_answer:
                    traj_summary["agent_final_score"] = r.agent_final_answer.overall_final_answer
            summary["trajectories"].append(traj_summary)

    # Save summary
    output_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return output_file


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Goal-Oriented Multi-Turn Trajectory Evaluator"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--trajectory", "-t",
        help="Path to a single trajectory file"
    )
    input_group.add_argument(
        "--traj_dir", "-d",
        help="Directory containing trajectory files"
    )

    # Evaluation options
    parser.add_argument(
        "--model", "-m",
        default="openai/gpt-4o-mini",
        help="Model to use for LLM-as-Judge evaluation"
    )
    parser.add_argument(
        "--static-only", "-s",
        action="store_true",
        help="Only compute static metrics (no LLM calls, no cost)"
    )
    parser.add_argument(
        "--skip-step-eval",
        action="store_true",
        help="Skip per-turn step evaluation (faster, still does final answer eval)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search subdirectories"
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        default="evaluation/goaloriented",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers for batch evaluation"
    )

    args = parser.parse_args()

    # Load trajectories
    if args.trajectory:
        traj = load_trajectory(args.trajectory)
        if not traj:
            print(f"Failed to load trajectory: {args.trajectory}", file=sys.stderr)
            sys.exit(1)
        trajectories = [traj]
    else:
        trajectories = load_trajectories_from_dir(args.traj_dir, args.recursive)
        if not trajectories:
            print(f"No trajectories found in: {args.traj_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Loaded {len(trajectories)} trajectories")
    print(f"Mode: {'Static only' if args.static_only else 'Full evaluation with LLM-as-Judge'}")

    # Evaluate
    results = []

    def eval_one(traj):
        return evaluate_trajectory(
            traj,
            model_name=args.model,
            static_only=args.static_only,
            evaluate_all_steps=not args.skip_step_eval,
        )

    if args.parallel > 1 and len(trajectories) > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            results = list(executor.map(eval_one, trajectories))
    else:
        for i, traj in enumerate(trajectories):
            print(f"Evaluating [{i+1}/{len(trajectories)}]: {traj.get('_filename', 'unknown')}")
            result = eval_one(traj)
            results.append(result)

    # Save results
    for result in results:
        output_file = save_evaluation_result(result, args.output_dir)
        print(f"Saved: {output_file}")

    # Generate summary
    summary_file = generate_summary_report(results, args.output_dir)
    print(f"\nSummary saved: {summary_file}")

    # Print quick summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total trajectories: {len(results)}")

    if results:
        avg_goal = sum(r.trajectory_metrics.goal_completion_rate for r in results) / len(results)
        avg_tools = sum(r.agent_metrics.total_tool_calls for r in results) / len(results)
        print(f"Avg goal completion rate: {avg_goal:.2%}")
        print(f"Avg tool calls per trajectory: {avg_tools:.1f}")

        if not args.static_only:
            scores = [r.overall_score for r in results if r.overall_score > 0]
            if scores:
                print(f"Avg overall score: {sum(scores)/len(scores):.2f}/10")


if __name__ == "__main__":
    main()
