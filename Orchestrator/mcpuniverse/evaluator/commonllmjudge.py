from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

from mcpuniverse.common.context import Context
from mcpuniverse.evaluator.functions import eval_func, compare_func, FunctionResult

load_dotenv()

# =====================================================
# internal utils
# =====================================================

def _safe_parse_json(s: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Best-effort parse of judge output.

    We assume ONLY the new multi-dimension rubric format, e.g.:

    {
      "task_fulfillment": <int 0-10>,
      "grounding": <int 0-10>,
      "tool_choice": <int 0-10>,
      "tool_execution": <int 0-10>,
      "requirement_satisfaction": <int 0-10>,
      "overall_score": <float 0-1>,
      "explanation": "<short reason referencing HISTORY>",
      "binary": "success" or "failure"
    }

    We accept fenced code blocks like ```json ... ``` or ``` ... ```.
    Return dict or None.
    """
    if not s:
        return None

    s = str(s).strip()

    # strip ``` fences and optional ```json prefix
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


# =====================================================
# LLM judge prompt template
# =====================================================

LLM_JUDGE_TEMPLATE = """You are a **strict and conservative** evaluator of multi-step, tool-using ReAct agents.
Your role is to score the quality of reasoning and tool use across multiple dimensions.

🔒 Important:
- **Perfect 10s should be rare (<10%)**.  
- Start from 8 and deduct points for any uncertainty, redundancy, or lack of verification.  
- Be conservative: if unsure between two scores, choose the lower.

------------------------------------------------------------
You will receive:
1. META: task_id, category, and optional correct_answer.
2. TASK: question and expected output_format.
3. HISTORY: the agent's reasoning trace with tool "Result:" outputs.
4. FINAL_ANSWER: the final response of the agent.

------------------------------------------------------------
EVALUATION RUBRIC (each 0–10, integers only):

1) Task Fulfillment (0–10)
   - 10 = completely and unambiguously answers the question in correct format.
   - 8–9 = mostly correct but small detail/format missing.
   - 6–7 = partial completion or unclear.
   - ≤5 = wrong type, empty, or irrelevant.

2) Grounding (0–10)
   - 10 = every claim is directly supported by tool results; no speculation.
   - 8–9 = mostly supported, but 1–2 claims lack explicit evidence.
   - 6–7 = several claims not grounded.
   - ≤5 = hallucinated or contradicts evidence.

3) Tool Choice (0–10)
   - 10 = used only appropriate tools that directly address the task.
   - 8–9 = some redundancy or missed a helpful tool.
   - 6–7 = several irrelevant or missing calls.
   - ≤5 = poor tool selection or none used when needed.

4) Tool Execution (0–10)
   - 10 = all tool calls succeeded and outputs used effectively.
   - 8–9 = minor misuse or ignored part of result.
   - 6–7 = several tool calls unused or failed.
   - ≤5 = major execution errors or ignored outputs.

5) Requirement Satisfaction / Constraint Coverage (0–10)
   - 10 = explicitly checks and meets all constraints or conditions in TASK (e.g., distance/time limits, correct number of results, etc.).
   - 8–9 = likely meets most constraints but without explicit verification.
   - 6–7 = partial or uncertain constraint satisfaction.
   - ≤5 = violates or ignores explicit constraints.

------------------------------------------------------------
OVERALL SCORE
overall_score = (task_fulfillment + grounding + tool_choice + tool_execution + requirement_satisfaction) / 50.
Round to two decimals. Must be in [0,1].

------------------------------------------------------------
BINARY DECISION
"success" if overall_score >= {pass_threshold:.2f}, else "failure".

------------------------------------------------------------
OUTPUT FORMAT
Return STRICT JSON:
{{
  "task_fulfillment": <int 0-10>,
  "grounding": <int 0-10>,
  "tool_choice": <int 0-10>,
  "tool_execution": <int 0-10>,
  "requirement_satisfaction": <int 0-10>,
  "overall_score": <float 0-1>,
  "explanation": "<brief reason>",
  "binary": "success" or "failure"
}}

Return ONLY the JSON object — no prose or commentary.

------------------------------------------------------------
DATA:
{payload}
"""



# =====================================================
# low-level LLM call
# =====================================================

def _call_llm_judge(
    prompt: str,
    *,
    context: Optional[Context] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    prompt_char_limit: int = 15000,
) -> Optional[str]:
    """
    Call the judge model via OpenAI-compatible API.
    We assume the caller provides:
      - valid api key via env or context
      - model name via kwargs or env; if missing we raise
    """
    ctx = context or Context()

    api_key = ctx.get_env("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = ctx.get_env("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    # hard requirement: specify judge model
    judge_model = model or os.getenv("JUDGE_MODEL")
    if not judge_model:
        raise RuntimeError("No judge model specified (missing judge_model / JUDGE_MODEL).")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # naive truncate to avoid overlong prompts
    safe_prompt = prompt[:prompt_char_limit]

    tries = 3
    while tries > 0:
        tries -= 1
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict grading judge. "
                            "Always return ONLY the JSON object, "
                            "no prose before or after."
                        ),
                    },
                    {"role": "user", "content": safe_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[commonllmjudge] LLM judge call failed: {e}")

    return None


# =====================================================
# payload builder
# =====================================================

def _build_inputs(
    llm_response: Any,
    values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Runner is expected to put these in `values`:
        - "question": original user query (string)
        - "history": agent full ReAct trace (string)
        - "final_answer": final answer string
        - "output_format": task schema (dict or string)
        - optional: "task_id", "category", "correct_answer"

    We'll normalize that into:
        task:    {task_id, category, question, output_format}
        payload: {META, TASK, HISTORY, FINAL_ANSWER}
    """

    question = values.get("question", "")
    history = values.get("history", "")
    final_answer = values.get("final_answer", "")

    task = {
        "task_id": values.get("task_id", "unknown_task"),
        "category": values.get("category", "general"),
        "question": question,
        "output_format": values.get("output_format", {}),
    }

    correct_answer = values.get("correct_answer", "")

    payload = {
        "META": {
            "task_id": task["task_id"],
            "category": task["category"],
            "correct_answer": correct_answer,
        },
        "TASK": {
            "question": task["question"],
            "output_format": task["output_format"],
        },
        "HISTORY": history,
        "FINAL_ANSWER": final_answer,
    }

    return {
        "task": task,
        "history": history,
        "final_answer": final_answer,
        "payload": payload,
    }


# =====================================================
# core scoring logic (new rubric only)
# =====================================================

def llm_as_judge_score(
    *,
    task: Dict[str, Any],
    history: str,
    final_answer: str,
    payload: Dict[str, Any],
    judge_model: Optional[str] = None,
    temperature: float = 0.0,
    pass_threshold: float = 0.85,
    context: Optional[Context] = None,
    max_completion_tokens: int = 512,
    prompt_char_limit: int = 15000,
) -> Dict[str, Any]:
    """
    Call the judge LLM with the new rubric-only prompt, parse it,
    and convert to our internal normalized result:

    {
        "score": float in [0,1],        # final overall score
        "binary": "success"|"failure",  # pass/fail vs threshold
        "explanation": str,

        "task_fulfillment": float|None,
        "grounding": float|None,
        "tool_choice": float|None,
        "tool_execution": float|None,
        "requirement_satisfaction": float|None,

        "raw_judge_output": original_text
    }

    If anything is invalid/unparsable, we return score=0.0,failure.
    """

    # build judge prompt from template
    prompt = LLM_JUDGE_TEMPLATE.format(
        pass_threshold=pass_threshold,
        payload=json.dumps(payload, ensure_ascii=False),
    )

    # call judge model
    raw = _call_llm_judge(
        prompt,
        context=context,
        model=judge_model,
        temperature=temperature,
        max_tokens=max_completion_tokens,
        prompt_char_limit=prompt_char_limit,
    )

    # parse JSON
    parsed = _safe_parse_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return {
            "score": 0.0,
            "binary": "failure",
            "explanation": "Judge model did not return valid JSON.",
            "task_fulfillment": None,
            "grounding": None,
            "tool_choice": None,
            "tool_execution": None,
            "requirement_satisfaction": None,
            "raw_judge_output": raw,
        }

    # Extract subscores
    def _coerce_0_10(x: Any) -> Optional[float]:
        try:
            val = float(x)
        except Exception:
            return None
        if val < 0:
            val = 0
        if val > 10:
            val = 10
        return val

    subs = {
        "task_fulfillment": _coerce_0_10(parsed.get("task_fulfillment")),
        "grounding": _coerce_0_10(parsed.get("grounding")),
        "tool_choice": _coerce_0_10(parsed.get("tool_choice")),
        "tool_execution": _coerce_0_10(parsed.get("tool_execution")),
        "requirement_satisfaction": _coerce_0_10(parsed.get("requirement_satisfaction")),
    }

    # compute / validate overall score
    def _avg_subscores_to_overall(subvals: Dict[str, Optional[float]]) -> Optional[float]:
        vals = [v for v in subvals.values() if v is not None]
        if not vals:
            return None
        # average 0-10 subscores, then /10 to scale to [0,1]
        return (sum(vals) / len(vals)) / 10.0

    try:
        overall = float(parsed.get("overall_score", 0.0))
    except Exception:
        overall = None

    if overall is None or overall < 0 or overall > 1:
        overall = _avg_subscores_to_overall(subs)
        if overall is None:
            overall = 0.0

    # clamp final
    if overall < 0.0:
        overall = 0.0
    if overall > 1.0:
        overall = 1.0

    # derive binary if missing / malformed
    binary = parsed.get("binary")
    if binary not in ("success", "failure"):
        binary = "success" if overall >= pass_threshold else "failure"

    explanation = parsed.get("explanation", "")

    return {
        "score": overall,  # canonical score in [0,1]
        "binary": binary,
        "explanation": explanation,
        "task_fulfillment": subs["task_fulfillment"],
        "grounding": subs["grounding"],
        "tool_choice": subs["tool_choice"],
        "tool_execution": subs["tool_execution"],
        "requirement_satisfaction": subs["requirement_satisfaction"],
        "raw_judge_output": raw,
    }


# =====================================================
# shared evaluation helper
# =====================================================

def _evaluate_once(
    agent_output: Any,
    values_dict: Dict[str, Any],
    ctx: Context,
    judge_model: Optional[str],
    temperature: float,
    pass_threshold: float,
    max_completion_tokens: int,
    prompt_char_limit: int,
) -> Dict[str, Any]:
    """
    Common logic used by both commonllmjudge_score and commonllmjudge_pass.
    Builds payload, recovers final_answer, calls llm_as_judge_score.
    """
    built = _build_inputs(agent_output, values_dict)

    # Recover final answer:
    final_answer = built["final_answer"]
    if not final_answer:
        # fallback to agent_output.response if available
        if hasattr(agent_output, "response"):
            final_answer = getattr(agent_output, "response")
        else:
            final_answer = str(agent_output)

    # Sync FINAL_ANSWER back into payload copy:
    payload = built["payload"].copy()
    payload["FINAL_ANSWER"] = final_answer

    return llm_as_judge_score(
        task=built["task"],
        history=built["history"],
        final_answer=final_answer,
        payload=payload,
        judge_model=judge_model,
        temperature=temperature,
        pass_threshold=pass_threshold,
        context=ctx,
        max_completion_tokens=max_completion_tokens,
        prompt_char_limit=prompt_char_limit,
    )


# =====================================================
# adapter funcs exposed to the Benchmark/Evaluator layer
# =====================================================

def _extract_values_arg(args) -> Dict[str, Any]:
    """
    Evaluator calls often pass extra metadata dict as the first (or sometimes
    second) positional arg after the agent output. We try both for safety.
    """
    if len(args) >= 1 and isinstance(args[0], dict):
        return args[0]

    if len(args) >= 2 and isinstance(args[1], dict):
        return args[1]

    return {}


@eval_func(name="commonllmjudge.score")
async def commonllmjudge_score(llm_response: Any, *args, **kwargs) -> FunctionResult:
    """
    Return the full structured judge result as FunctionResult(result=...).
    """
    values = _extract_values_arg(args)
    ctx: Context = kwargs.get("context", Context())

    obj = _evaluate_once(
        agent_output=llm_response,
        values_dict=values,
        ctx=ctx,
        judge_model=kwargs.get("judge_model"),
        temperature=float(kwargs.get("temperature", 0.0)),
        pass_threshold=float(kwargs.get("pass_threshold", 0.85)),
        max_completion_tokens=int(kwargs.get("max_completion_tokens", 512)),
        prompt_char_limit=int(kwargs.get("prompt_char_limit", 15000)),
    )

    return FunctionResult(result=obj)


@compare_func(name="commonllmjudge.pass")
async def commonllmjudge_pass(a: Any, *args, **kwargs) -> (bool, str):
    """
    Return (ok, reason) where ok tells you if binary=="success".
    Also logs a human-readable breakdown to stderr for benchmark output.
    """
    import sys

    values = _extract_values_arg(args)
    ctx: Context = kwargs.get("context", Context())

    obj = _evaluate_once(
        agent_output=a,
        values_dict=values,
        ctx=ctx,
        judge_model=kwargs.get("judge_model"),
        temperature=float(kwargs.get("temperature", 0.0)),
        pass_threshold=float(kwargs.get("pass_threshold", 0.85)),
        max_completion_tokens=int(kwargs.get("max_completion_tokens", 512)),
        prompt_char_limit=int(kwargs.get("prompt_char_limit", 15000)),
    )

    ok = (obj.get("binary") == "success")
    score_val = obj.get("score", None)
    reason = obj.get("explanation", "")

    # print breakdown to stderr for nice benchmark logs
    print(
        "[LLM-JUDGE SCORE] "
        f"overall={score_val} "
        f"binary={obj.get('binary')} "
        f"task_fulfillment={obj.get('task_fulfillment')} "
        f"grounding={obj.get('grounding')} "
        f"tool_choice={obj.get('tool_choice')} "
        f"tool_execution={obj.get('tool_execution')} "
        f"requirement_satisfaction={obj.get('requirement_satisfaction')} "
        f"reason={reason}",
        file=sys.stderr
    )

    return ok, reason


@compare_func(name="score>=")
async def score_ge(a: Any, b: Any, *args, **kwargs) -> (bool, str):
    """
    Utility comparator for benchmark YAMLs:
    - left side is typically commonllmjudge.score result, or a dict with "score".
    - right side is threshold, or kwargs["threshold"] fallback.
    """
    if isinstance(a, FunctionResult):
        a = a.result
    if isinstance(a, dict):
        a = a.get("score", 0.0)

    try:
        a_val = float(a)
    except Exception:
        return False, f"invalid score: {a}"

    if isinstance(b, FunctionResult):
        b = b.result
    try:
        b_val = float(b)
    except Exception:
        b_val = float(kwargs.get("threshold", 0.85))

    ok = (a_val >= b_val)
    return (ok, "" if ok else f"score {a_val:.3f} < threshold {b_val:.3f}")
