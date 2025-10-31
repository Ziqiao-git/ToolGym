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
    Try to parse model output as JSON.
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
# LLM helper: generic chat call (always gpt-4o-mini)
# =====================================================

def _call_llm(
    *,
    prompt: str,
    system_prompt: str,
    context: Optional[Context] = None,
    temperature: float = 0.0,
    max_tokens: int = 8000,
    prompt_char_limit: int = 100000,
) -> Optional[str]:
    """
    Thin wrapper around OpenAI-compatible chat.completions.create
    used both for trajectory summarization and judging.

    The model is hard-coded to openai/gpt-4o-mini.
    """
    ctx = context or Context()

    api_key = ctx.get_env("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = ctx.get_env("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    MODEL_NAME = "openai/gpt-4o-mini"

    client = OpenAI(api_key=api_key, base_url=base_url)

    safe_prompt = prompt[:prompt_char_limit]

    tries = 3
    while tries > 0:
        tries -= 1
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": safe_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[commonllmjudge] LLM call failed: {e}")

    return None


# =====================================================
# Trajectory summarization prompt template
# =====================================================

# 关键更新：
# - Thought / Action 必须逐字/逐步保留语义，不要省略信息，不要“概括到看不出原意”。
# - Tool Result 只抽“推理真正用到的字段”，例如:
#   * geocode: 只要坐标

# - 最后一步（通常是选winner的那一步）一定要点名哪家被选中 + 为什么它赢
#   (平衡性/是cafe/满足约束等)
# - 不要重复最终答案的JSON Schema，但可以提店名

TRAJECTORY_SUMMARY_TEMPLATE = """You are compressing an agent's multi-step ReAct trace
to create a compact trajectory summary for an evaluation judge.

GOALS
- Preserve reasoning faithfulness.
- Give the judge enough evidence to verify constraint satisfaction and grounding.
- Keep it short enough to not blow up the context.
- DO NOT hide or skip any step that is important for how the final cafe / answer was chosen.

CRITICAL GUARANTEES (DO NOT VIOLATE):
- You MUST include EVERY step from the original trace, in order.
- You MUST include the FINAL step's Thought (the agent's final reasoning / choice rationale).
- You MUST NOT output the final structured answer JSON (e.g. {{'stops': ...}}).

INSTRUCTIONS

1. You MUST keep the step structure and numbering exactly:
   Step 1:
   Thought: ...
   Action: server=<server>, tool=<tool>
   Action Input: ...
   Tool Result: ...
   Step 2:
   ...
   (Continue for ALL steps in the original trace. Do not skip the last step.)

2. THOUGHT:
   - You MUST copy the agent's Thought **exactly as written**, character-for-character.
   - DO NOT paraphrase, summarize, shorten, reorder, or modify any wording, punctuation, or formatting.
   - The Thought text must be a **verbatim copy** from the original trace.
   - If a Thought contains reasoning, constraint checking, or justification, you must include it in full.

3. ACTION:
   - You MUST include both server and tool exactly
     (e.g. "server=google-maps, tool=maps_distance_matrix").
   - You MUST ALSO include the agent's stated reason / motivation for calling the tool
     if it exists (e.g. "to compare travel times across all friends").
     Put that reason directly under Action.
   - Action Input must include all essential arguments actually sent to the tool.
     Essential arguments include:
       - origins / source addresses
       - destinations / candidate places (entire list, not truncated)
       - query text
       - radius / mode
     You are NOT allowed to silently drop destinations from the list.
     If there is a long list (e.g. many cafes), you MUST keep all names in that list,
     but you MAY omit unneeded address suffixes like ", Singapore" if they repeat.
   - Non-essential: internal flags, formatting clutter, debug metadata, etc.

   FORMAT EXAMPLE:
   Action: server=google-maps, tool=maps_distance_matrix
   Reason: Using this tool lets me compare drive times from all 3 homes.
   Action Input:
     origins: [...]
     destinations: [...]
     mode: driving

4. TOOL RESULT:
   - You MUST preserve the **entire tool output** as returned by the environment or API.
   - You may reformat it for readability (e.g., convert JSON to bullet lists or inline summaries),
     but you MUST NOT drop, condense, or selectively omit any entries, keys, or values.
   - The ONLY exception is to remove clearly irrelevant low-value noise such as:
       • extremely long coordinate arrays,
       • large 3D geometry meshes or bounding boxes,
       • base64-encoded image blobs.
   - You MUST keep all fields, even if they do not appear to be explicitly referenced later.
     Do NOT assume which parts were “used for reasoning” — preserve everything except obvious noise.
   - If the tool returned multiple items (e.g., distance matrix, route list, address list, result set),
     keep **all rows, columns, and entries**, not just samples or partial previews.
   - When you rewrite for readability, ensure that all numeric values and key–value pairs remain intact and unaltered.

   Examples of acceptable Tool Result style:
     Tool Result:
       origin_addresses: [...]
       destination_addresses: [...]
       durations (mins):
         - Cafe A: [20, 22, 21]
         - Cafe B: [18, 24, 19]
       Observation: spread ~3 min for Cafe A (most balanced)

   Examples of what NOT to do:
     - Do NOT shorten destination list to first 5 items if there were 20.
     - Do NOT convert a full distance matrix into only one cafe unless the agent ALSO
       only used that cafe later.

5. DO NOT add any extra "Winner Justification" section.
   The reasoning in the agent's Thought (especially the last step) IS the justification.

6. DO NOT include giant raw JSON dumps or long coordinate lists.
   DO NOT include multi-dozen-line arrays of lat/lng pairs.
   DO NOT include the final structured answer JSON schema or the final structured answer.

7. Output must be plain text EXACTLY in this format, no markdown fences.

Now here is the full original trace you must compress:

{raw_history}
"""



def summarize_trajectory_for_judge(
    raw_history: str,
    *,
    context: Optional[Context] = None,
    temperature: float = 0.0,
    max_tokens: int = 8000,
    prompt_char_limit: int = 100000,
) -> str:
    """
    Use gpt-4o-mini to convert the full verbose trace into a compact,
    judge-friendly trajectory:
      - Step-by-step.
      - Thought and Action preserved with intent intact.
      - Tool Result distilled to only decision-relevant fields.
      - Final step MUST justify why the chosen cafe was selected
        (balance of travel times, is cafe, etc).
      - DOES NOT echo the final JSON answer block.

    If summarization fails or returns empty, fallback to last ~8000 chars of raw_history.
    """

    system_prompt = (
        "You are a faithful summarizer that keeps agent reasoning steps intact "
        "while pruning unneeded raw JSON. You MUST follow the requested output "
        "format exactly."
    )

    user_prompt = TRAJECTORY_SUMMARY_TEMPLATE.format(raw_history=raw_history)

    summary = _call_llm(
        prompt=user_prompt,
        system_prompt=system_prompt,
        context=context,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_char_limit=prompt_char_limit,
    )

    if not summary:
        MAX_FALLBACK = 8000
        return raw_history[-MAX_FALLBACK:]

    summary = summary.strip()
    if not summary:
        MAX_FALLBACK = 8000
        return raw_history[-MAX_FALLBACK:]

    return summary


# =====================================================
# Judge prompt template
# =====================================================

LLM_JUDGE_TEMPLATE = """You are a **strict and conservative** evaluator of multi-step, tool-using ReAct agents.
Your role is to score the quality of reasoning and tool use across multiple dimensions.

🔒 Important:
- **Perfect 10s should be rare (<10%)**.
- Start from 8 and deduct points for any uncertainty, redundancy, or lack of verification.
- Be conservative: if unsure between two scores, choose the lower.

------------------------------------------------------------
META:
{meta_json}

TASK:
{task_json}

HISTORY:
{history_text}

FINAL_ANSWER:
{final_answer_text}

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
   - 10 = explicitly checks and meets all constraints or conditions in TASK
         (e.g., correct number of items, distance/time limits, etc.).
   - 8–9 = likely meets most constraints but without explicit verification.
   - 6–7 = partial or uncertain constraint satisfaction.
   - ≤5 = violates or ignores explicit constraints.

------------------------------------------------------------
OVERALL SCORE
overall_score = (task_fulfillment + grounding + tool_choice
                 + tool_execution + requirement_satisfaction) / 50.
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
"""


# =====================================================
# judge call
# =====================================================

def llm_as_judge_score(
    *,
    meta: Dict[str, Any],
    task: Dict[str, Any],
    history: str,
    final_answer: str,
    context: Optional[Context] = None,
    temperature: float = 0.0,
    pass_threshold: float = 0.85,
    max_completion_tokens: int = 8000,
    prompt_char_limit: int = 100000,
) -> Dict[str, Any]:
    """
    Call the judge LLM with the rubric, parse output, and normalize it.
    Always uses openai/gpt-4o-mini via _call_llm().

    Returns a dict with overall score, subscores, and explanation.
    """

    # stringify
    meta_json = json.dumps(meta, ensure_ascii=False)
    task_json = json.dumps(task, ensure_ascii=False)
    history_text = str(history).strip()
    final_answer_text = str(final_answer).strip()

    prompt = LLM_JUDGE_TEMPLATE.format(
        pass_threshold=pass_threshold,
        meta_json=meta_json,
        task_json=task_json,
        history_text=history_text,
        final_answer_text=final_answer_text,
    )

    # Debug to stderr
    import sys
    print(
        "========== [DEBUG] FULL prompt SENT TO JUDGE ==========\n"
        f"{prompt}\n"
        "=====================================================\n",
        file=sys.stderr,
        flush=True,
    )

    raw = _call_llm(
        prompt=prompt,
        system_prompt=(
            "You are an impartial evaluator judging the quality of an AI agent’s multi-server, tool-based task execution. "
            "Evaluate the reasoning, grounding, and adherence to requirements objectively and strictly. "
            "Return ONLY a valid JSON object containing your scores and explanation — no extra text, commentary, or formatting outside JSON."
        ),
        context=context,
        temperature=temperature,
        max_tokens=max_completion_tokens,
        prompt_char_limit=prompt_char_limit,
    )

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

    if overall is None or not (0.0 <= overall <= 1.0):
        overall = _avg_subscores_to_overall(subs)
        if overall is None:
            overall = 0.0

    overall = max(0.0, min(1.0, overall))

    binary = parsed.get("binary")
    if binary not in ("success", "failure"):
        binary = "success" if overall >= pass_threshold else "failure"

    explanation = parsed.get("explanation", "")

    return {
        "score": overall,
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
# history cleanup + evaluation orchestration
# =====================================================

def _clean_history_before_summary(raw_history: str) -> str:
    """
    Remove trailing "Answer:" block so FINAL_ANSWER stays only in FINAL_ANSWER,
    not leaked into HISTORY. We split on the first "Answer:".
    """
    if not raw_history:
        return ""
    parts = raw_history.split("Answer:", 1)
    return parts[0].strip()


def _evaluate_once(
    agent_output: Any,
    values_dict: Dict[str, Any],
    ctx: Context,
    temperature: float,
    pass_threshold: float,
    max_completion_tokens: int,
    prompt_char_limit: int,
) -> Dict[str, Any]:
    """
    Shared logic for score/pass.

    Expected values_dict keys:
        - task_id
        - category
        - correct_answer (optional)
        - question
        - output_format
        - history  (full raw chain-of-thought/tool log)
        - final_answer
    """

    question = values_dict.get("question", "")
    raw_history = values_dict.get("history", "") or ""

    final_answer = (
        values_dict.get("final_answer")
        or getattr(agent_output, "response", None)
        or str(agent_output)
    )

    meta = {
        "task_id": values_dict.get("task_id", "unknown_task"),
        "category": values_dict.get("category", "general"),
        "correct_answer": values_dict.get("correct_answer", ""),
    }

    task = {
        "question": question,
        "output_format": values_dict.get("output_format", {}),
    }

    # 1) strip final Answer block from history
    cleaned_history = _clean_history_before_summary(raw_history)

    # 2) summarize (this will now keep all Thought/Action intent,
    #    and will keep only useful Tool Result fields, PLUS winner justification)
    summarized_history = summarize_trajectory_for_judge(
        cleaned_history,
        context=ctx,
        temperature=0.0,
        max_tokens=8000,
        prompt_char_limit=prompt_char_limit,
    )

    # 3) send to judge
    return llm_as_judge_score(
        meta=meta,
        task=task,
        history=summarized_history,
        final_answer=final_answer,
        context=ctx,
        temperature=temperature,
        pass_threshold=pass_threshold,
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
        temperature=float(kwargs.get("temperature", 0.0)),
        pass_threshold=float(kwargs.get("pass_threshold", 0.85)),
        max_completion_tokens=int(kwargs.get("max_completion_tokens", 8000)),
        prompt_char_limit=int(kwargs.get("prompt_char_limit", 100000)),
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
        temperature=float(kwargs.get("temperature", 0.0)),
        pass_threshold=float(kwargs.get("pass_threshold", 0.85)),
        max_completion_tokens=int(kwargs.get("max_completion_tokens", 8000)),
        prompt_char_limit=int(kwargs.get("prompt_char_limit", 100000)),
    )

    ok = (obj.get("binary") == "success")
    score_val = obj.get("score", None)
    reason = obj.get("explanation", "")

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
    Comparator for benchmark YAMLs:
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
        # if parse fails, fallback to kwarg
    except Exception:
        b_val = float(kwargs.get("threshold", 0.85))

    ok = (a_val >= b_val)
    return (ok, "" if ok else f"score {a_val:.3f} < threshold {b_val:.3f}")