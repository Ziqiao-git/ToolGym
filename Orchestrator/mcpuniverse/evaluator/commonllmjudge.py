# commonllmjudge.py
# Judge that reads prompt.json (NEW format only) + trajectories/trajectory_*.json
# HISTORY is built ONLY from reasoning_trace (Thought-aware).
# No fallback to execution.tool_calls. No LLM summarization step.

from __future__ import annotations
import os, json, glob, argparse, sys
from typing import Any, Dict, List, Optional
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
    prompt_char_limit: int = 100000,
    model_name: str = "openai/gpt-4o-mini",
) -> Optional[str]:
    """
    Thin wrapper around OpenAI-compatible chat.completions.create.
    Requires env:
      - OPENAI_API_KEY
      - OPENAI_BASE_URL  (e.g., https://openrouter.ai/api/v1)
    """
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
            print(f"[commonllmjudge] LLM call failed: {e}", file=sys.stderr)
    return None


def _safe_parse_json(s: Optional[str]) -> Optional[Dict[str, Any]]:
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


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# =========================
# File loading & matching
# =========================

def load_prompts(prompt_path: str) -> List[Dict[str, Any]]:
    """
    NEW FORMAT ONLY:
    {
      "metadata": {...},
      "items": [
        {"query": "...", "reference_tools": [{"server":"...", "tool":"...", "why":"..."}]},
        ...
      ]
    }
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "items" not in data or not isinstance(data["items"], list):
        raise ValueError("prompt.json must use the NEW format with top-level 'items' list.")

    out: List[Dict[str, Any]] = []
    for it in data["items"]:
        q = (it or {}).get("query", "")
        ref = (it or {}).get("reference_tools", [])
        if not isinstance(ref, list):
            ref = []
        norm_ref = []
        for r in ref:
            if isinstance(r, dict):
                norm_ref.append({
                    "server": r.get("server", ""),
                    "tool": r.get("tool", ""),
                    "why": r.get("why", "")
                })
        out.append({"query": q, "reference_tools": norm_ref})
    return out


def load_all_trajectories(dirpath: str) -> List[Dict[str, Any]]:
    trajs: List[Dict[str, Any]] = []
    for p in glob.glob(os.path.join(dirpath, "trajectory_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            obj["_filename"] = os.path.basename(p)
            trajs.append(obj)
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}", file=sys.stderr)
    return trajs


def match_traj_by_query(trajs: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    q_strip = (query or "").strip()
    for t in trajs:
        q = (t.get("metadata") or {}).get("query", "")
        if q.strip() == q_strip:
            return t
    return None


# =========================
# HISTORY construction (reasoning_trace only)
# =========================

def build_history_from_reasoning_trace(traj_obj: Dict[str, Any]) -> str:
    """
    Only trajectory['reasoning_trace'] to build HISTORY (keep Thought).
    Ignore execution.tool_calls (no fallback, no completion).
    """
    import ast

    rt = (traj_obj.get("reasoning_trace") or [])
    if not rt:
        return "\n".join([
            "Step 1:",
            "Thought: (not recorded in trajectory file)",
            "Action: (none)",
            "Action Input: {}",
            "Tool Result: (none)",
        ]).strip()

    def _new_step(idx: int) -> Dict[str, Any]:
        return {"step": idx, "thought": None, "action": None, "action_input": None, "tool_result": None}

    def _is_placeholder_thought(t: Optional[str]) -> bool:
        if not t: return True
        return str(t).strip() == "(not recorded in trajectory file)"

    def _jsonish_parse(s: Any) -> Any:
        if isinstance(s, (dict, list)):
            return s
        if isinstance(s, str):
            txt = s.strip()
            try:
                return json.loads(txt)
            except Exception:
                pass
            try:
                import ast as _ast
                v = _ast.literal_eval(txt)
                return v
            except Exception:
                return s
        return s

    steps: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    idx = 0

    for item in rt:
        ttype = (item.get("type") or "").strip()
        content = (item.get("content") or "")

        if ttype.lower() == "error":
            continue

        if ttype.lower().startswith("step"):
            if current:
                steps.append(current)
            idx += 1
            current = _new_step(idx)
            continue

        if current is None:
            idx += 1
            current = _new_step(idx)

        low = ttype.lower()
        if low == "thought":
            current["thought"] = content
        elif low == "action":
            current["action"] = content or current.get("action")
        elif low in ("action input", "action_input"):
            current["action_input"] = _jsonish_parse(content)
        elif low in ("result", "tool result", "tool_result"):
            current["tool_result"] = _jsonish_parse(content)
        elif low in ("answer",):
            pass
        else:
            pass

    if current:
        steps.append(current)

    def _is_meaningful(s: Dict[str, Any]) -> bool:
        if s.get("action"):
            return True
        ai = s.get("action_input")
        tr = s.get("tool_result")
        if ai not in (None, {}, "", "{}", "[]"):
            return True
        if tr not in (None, {}, "", "(none)", "{}", "[]"):
            return True
        if s.get("thought") and not _is_placeholder_thought(s["thought"]):
            return True
        return False

    steps = [s for s in steps if _is_meaningful(s)]

    def _norm_ai(x: Any) -> Any:
        return _jsonish_parse(x)

    deduped: List[Dict[str, Any]] = []
    for s in steps:
        if deduped:
            prev = deduped[-1]
            same_action = (str(prev.get("action")).strip() == str(s.get("action")).strip())
            same_input = (_norm_ai(prev.get("action_input")) == _norm_ai(s.get("action_input")))
            prev_rich = (not _is_placeholder_thought(prev.get("thought"))) or bool(prev.get("tool_result"))
            curr_rich = (not _is_placeholder_thought(s.get("thought"))) or bool(s.get("tool_result"))
            if same_action and same_input:
                if curr_rich and not prev_rich:
                    deduped[-1] = s
                continue
        deduped.append(s)

    for s in deduped:
        if not s.get("thought"):
            s["thought"] = "(not recorded in trajectory file)"

    lines: List[str] = []
    for i, s in enumerate(deduped, 1):
        lines.append(f"Step {i}:")
        lines.append(f"Thought: {s.get('thought')}")
        lines.append(f"Action: {s.get('action') or '(none)'}")

        ai = s.get("action_input")
        ai_str = json.dumps(ai, ensure_ascii=False) if isinstance(ai, (dict, list)) else (str(ai) if ai is not None else "{}")
        lines.append("Action Input: " + ai_str)

        tr = s.get("tool_result")
        tr_str = json.dumps(tr, ensure_ascii=False) if isinstance(tr, (dict, list)) else (str(tr) if tr is not None else "(none)")
        lines.append("Tool Result: " + tr_str)

    return "\n".join(lines).strip()


def _clean_history_before_summary(raw_history: str) -> str:
    if not raw_history:
        return ""
    parts = raw_history.split("Answer:", 1)
    return parts[0].strip()


# =========================
# Rubric (FOUR dimensions + reasons)
# =========================

LLM_JUDGE_TEMPLATE = """You are a strict and conservative evaluator of multi-step, tool-using ReAct agents.
Score across FOUR dimensions that diagnose where to improve and help query generation.

🔒 Policy
- Perfect 10s are rare (<10%).
- Start from 8, deduct for uncertainty, missing verification, redundancy.
- If unsure between two scores, choose the lower.

------------------------------------------------------------
META:
{meta_json}

TASK:
{task_json}

REFERENCE_TOOLS (expected/allowed):
{reference_tools_json}

HISTORY:
{history_text}

FINAL_ANSWER:
{final_answer_text}

------------------------------------------------------------
EVALUATION RUBRIC (each 0–10, integers only)

1) Task Alignment (0–10)
   Does the final answer satisfy the task intent/constraints and requested format?
   10 = fully satisfies task & constraints with correct format;
   8–9 = minor omission/format gap;
   6–7 = partially meets task; unclear/incomplete;
   ≤5 = misses task intent or violates explicit constraints.

2) Grounding & Evidence (0–10)
   Are claims supported by tool outputs (verbatim fields/values/quotes)?
   10 = all key claims grounded; no speculation;
   8–9 = mostly grounded, 1–2 claims not explicit;
   6–7 = several ungrounded claims;
   ≤5 = hallucinated or contradicts tool evidence.

3) Tool Planning & Selection (0–10)
   Did the agent select the right tool(s) and order, aligned with REFERENCE_TOOLS unless a clearly better justified alternative is used?
   10 = correct tools/sequence; justified deviations;
   8–9 = minor redundancy/omission with little impact;
   6–7 = noticeable mismatch (wrong/extra/missing tools);
   ≤5 = ignored necessary tools or planned poorly.

4) Execution & Recovery (0–10)
   Did calls succeed, parameters match schema, errors get diagnosed/retried/fallback correctly, and outputs get used?
   10 = clean execution; robust error handling; outputs used effectively;
   8–9 = small misuse or ignored part of outputs;
   6–7 = several failed/unused calls or weak recovery;
   ≤5 = major execution errors; no recovery.

------------------------------------------------------------
OVERALL SCORE
overall_score = (task_alignment + grounding + tool_planning + execution_recovery) / 40.
Round to two decimals. Must be in [0,1].

BINARY DECISION
"success" if overall_score >= {pass_threshold:.2f}, else "failure".

------------------------------------------------------------
OUTPUT FORMAT
Return STRICT JSON ONLY:
{{
  "task_alignment": <int 0-10>,
  "grounding": <int 0-10>,
  "tool_planning": <int 0-10>,
  "execution_recovery": <int 0-10>,
  "overall_score": <float 0-1>,
  "binary": "success" or "failure",
  "reasons": {{
    "task_alignment": "<1-3 concise sentences>",
    "grounding": "<1-3 concise sentences>",
    "tool_planning": "<1-3 concise sentences>",
    "execution_recovery": "<1-3 concise sentences>"
  }}
}}
"""


def llm_as_judge_score(
    *,
    meta: Dict[str, Any],
    task: Dict[str, Any],
    history: str,
    final_answer: str,
    reference_tools: List[Dict[str, Any]],
    temperature: float = 0.0,
    pass_threshold: float = 0.85,
    max_completion_tokens: int = 8000,
    prompt_char_limit: int = 100000,
    model_name: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    meta_json = _json(meta)
    task_json = _json(task)
    ref_tools_json = _json(reference_tools or [])

    prompt = LLM_JUDGE_TEMPLATE.format(
        pass_threshold=pass_threshold,
        meta_json=meta_json,
        task_json=task_json,
        reference_tools_json=ref_tools_json,
        history_text=history,
        final_answer_text=final_answer,
    )

    print(
        "========== [DEBUG] FULL prompt SENT TO JUDGE ==========\n"
        f"{prompt}\n"
        "===================================================== \n",
        file=sys.stderr, flush=True
    )

    raw = _call_llm(
        prompt=prompt,
        system_prompt=(
            "You are an impartial evaluator judging the quality of an AI agent's multi-server, tool-based task execution. "
            "Return ONLY a valid JSON object with your scores, reasons, and binary decision."
        ),
        temperature=temperature,
        max_tokens=max_completion_tokens,
        prompt_char_limit=prompt_char_limit,
        model_name=model_name,
    )

    parsed = _safe_parse_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return {
            "score": 0.0,
            "binary": "failure",
            "reasons": {},
            "explanation": "Judge model did not return valid JSON.",
            "task_alignment": None,
            "grounding": None,
            "tool_planning": None,
            "execution_recovery": None,
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
        "task_alignment": _coerce_0_10(parsed.get("task_alignment")),
        "grounding": _coerce_0_10(parsed.get("grounding")),
        "tool_planning": _coerce_0_10(parsed.get("tool_planning")),
        "execution_recovery": _coerce_0_10(parsed.get("execution_recovery")),
    }

    def _avg_subscores_to_overall(d: Dict[str, Optional[float]]) -> Optional[float]:
        vals = [v for v in d.values() if v is not None]
        if not vals:
            return None
        # Four dimensions, each 0-10; overall in [0,1]
        return (sum(vals) / len(vals)) / 10.0

    try:
        overall = float(parsed.get("overall_score", 0.0))
    except Exception:
        overall = None

    if overall is None or not (0.0 <= overall <= 1.0):
        overall = _avg_subscores_to_overall(subs) or 0.0
    overall = max(0.0, min(1.0, overall))

    binary = parsed.get("binary")
    if binary not in ("success", "failure"):
        binary = "success" if overall >= pass_threshold else "failure"

    reasons = parsed.get("reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}

    # Keep 'explanation' for backward compatibility with any callers
    explanation = ""
    # Optionally synthesize a short explanation from reasons if needed
    if not explanation and reasons:
        try:
            explanation = " | ".join(
                f"{k}: {str(v)}" for k, v in reasons.items() if v
            )[:1000]
        except Exception:
            explanation = ""

    return {
        "score": overall,
        "binary": binary,
        "reasons": reasons,
        "explanation": explanation,
        "task_alignment": subs["task_alignment"],
        "grounding": subs["grounding"],
        "tool_planning": subs["tool_planning"],
        "execution_recovery": subs["execution_recovery"],
        "raw_judge_output": raw,
    }


# =========================
# Pipeline from files only
# =========================

def extract_actual_tools_from_trajectory(traj_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract the actual tools used from the trajectory.
    Returns list of {"server": str, "tool": str}
    """
    actual_tools = []
    rt = (traj_obj.get("reasoning_trace") or [])

    for item in rt:
        ttype = (item.get("type") or "").strip().lower()
        content = (item.get("content") or "")

        if ttype == "action" and content:
            # Extract server and tool from action format like "server/tool"
            if "/" in content:
                parts = content.split("/", 1)
                if len(parts) == 2:
                    actual_tools.append({
                        "server": parts[0].strip(),
                        "tool": parts[1].strip()
                    })
            else:
                # Just tool name without server
                actual_tools.append({
                    "server": "",
                    "tool": content.strip()
                })

    return actual_tools


def _values_from_prompt_and_traj(query: str, traj_obj: Dict[str, Any], task_id: str, reference_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    exe = traj_obj.get("execution", {}) or {}
    final_resp = exe.get("final_response", "")

    history_text = build_history_from_reasoning_trace(traj_obj)
    cleaned_history = _clean_history_before_summary(history_text)
    summarized_history = cleaned_history

    # Extract actual tools used
    actual_tools = extract_actual_tools_from_trajectory(traj_obj)

    meta = {
        "task_id": task_id,
        "category": (traj_obj.get("metadata") or {}).get("model", "unknown_model"),
        "correct_answer": "",
    }
    task = {"question": query, "output_format": {}}

    return {
        "meta": meta,
        "task": task,
        "history": summarized_history,
        "final_answer": final_resp,
        "reference_tools": reference_tools or [],
        "actual_tools": actual_tools,
    }


def run_judge_from_files(
    prompt_path: str = "prompt.json",
    trajectories_dir: str = "trajectories",
    *,
    pass_threshold: float = 0.85,
    temperature: float = 0.0,
    max_completion_tokens: int = 8000,
    prompt_char_limit: int = 100000,
    model_name: str = "openai/gpt-4o-mini",
) -> List[Dict[str, Any]]:
    items = load_prompts(prompt_path)  # NEW format only
    trajs = load_all_trajectories(trajectories_dir)

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, 1):
        q = item.get("query", "")
        ref_tools = item.get("reference_tools", []) or []
        matched = match_traj_by_query(trajs, q)
        if not matched:
            results.append({
                "task_id": f"prompt_idx_{idx}",
                "query": q,
                "error": "No trajectory matched for this query.",
            })
            continue

        task_id = matched.get("_filename", f"task_{idx}")
        pack = _values_from_prompt_and_traj(q, matched, task_id, ref_tools)

        obj = llm_as_judge_score(
            meta=pack["meta"],
            task=pack["task"],
            history=pack["history"],
            final_answer=pack["final_answer"],
            reference_tools=pack["reference_tools"],
            temperature=temperature,
            pass_threshold=pass_threshold,
            max_completion_tokens=max_completion_tokens,
            prompt_char_limit=prompt_char_limit,
            model_name=model_name,
        )

        results.append({
            "task_id": task_id,
            "query": q,
            "reference_tools": ref_tools,
            "actual_tools": pack.get("actual_tools", []),
            "binary": obj.get("binary"),
            "score": obj.get("score"),
            "task_alignment": obj.get("task_alignment"),
            "grounding": obj.get("grounding"),
            "tool_planning": obj.get("tool_planning"),
            "execution_recovery": obj.get("execution_recovery"),
            "reasons": obj.get("reasons", {}),
            "explanation": obj.get("explanation"),
        })
    return results


# =========================
# Step-by-Step Evaluation
# =========================

STEP_EVALUATION_TEMPLATE = """You are evaluating a SINGLE STEP in a multi-step ReAct agent trajectory.

ULTIMATE GOAL (User's Query):
{query}

STEP NUMBER: {step_num}

PREVIOUS CONTEXT:
{previous_context}

CURRENT STEP:
Thought: {thought}
Action: {action}
Action Input: {action_input}
Tool Result: {tool_result}

---
Evaluate this single step on FOUR dimensions (0-10, integers only):

1) Thought Quality (0-10)
   - Is the thought logical given the query and previous context?
   - Does it show clear reasoning about what to do next?
   10 = excellent reasoning, clear plan
   7-9 = good reasoning, minor gaps
   4-6 = weak reasoning, unclear plan
   0-3 = poor/missing reasoning

2) Action Appropriateness (0-10)
   - Is the chosen action/tool appropriate for the thought and query?
   - Are the action inputs correct and complete?
   10 = perfect tool choice, correct inputs
   7-9 = good choice, minor input issues
   4-6 = questionable choice or significant input issues
   0-3 = wrong tool or invalid inputs

3) Result Utilization (0-10)
   - Is the tool result valid and useful?
   - Does it provide information needed for the query?
   10 = excellent result, directly useful
   7-9 = good result, mostly useful
   4-6 = partial result or limited usefulness
   0-3 = error, irrelevant result, or not utilized

4) Progress Toward Goal (0-10)
   - CRITICAL: Does this step move the agent CLOSER to answering the ultimate query?
   - Does it gather necessary information or eliminate uncertainty?
   - Is this step essential for the final answer, or is it redundant/tangential?
   10 = essential step, directly advances toward goal
   8-9 = helpful step, clearly contributes to progress
   6-7 = marginal progress, somewhat relevant
   4-5 = minimal progress, mostly redundant or tangential
   0-3 = no progress, wrong direction, or completely irrelevant

---
OUTPUT FORMAT (strict JSON only):
{{
  "thought_quality": <int 0-10>,
  "action_appropriateness": <int 0-10>,
  "result_utilization": <int 0-10>,
  "progress_toward_goal": <int 0-10>,
  "step_score": <float 0-1>,
  "issues": "<brief description of any problems, or 'none'>",
  "suggestions": "<brief suggestion for improvement, or 'none'>",
  "progress_explanation": "<1-2 sentences: HOW does this step contribute to answering the query?>"
}}

where step_score = (thought_quality + action_appropriateness + result_utilization + progress_toward_goal) / 40.0

IMPORTANT: The "progress_toward_goal" dimension is the MOST CRITICAL. Always ask: "Is this step necessary for answering '{query}'?"
"""


def extract_steps_from_reasoning_trace(rt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract individual steps from reasoning_trace."""
    steps: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    step_num = 0

    def _new_step(idx: int) -> Dict[str, Any]:
        return {
            "step_num": idx,
            "thought": None,
            "action": None,
            "action_input": None,
            "tool_result": None
        }

    for item in rt:
        ttype = (item.get("type") or "").strip().lower()
        content = item.get("content", "")

        if ttype == "error":
            continue

        if ttype.startswith("step"):
            if current:
                steps.append(current)
            step_num += 1
            current = _new_step(step_num)
            continue

        if current is None:
            step_num += 1
            current = _new_step(step_num)

        if ttype == "thought":
            current["thought"] = content
        elif ttype == "action":
            current["action"] = content
        elif ttype in ("action input", "action_input"):
            try:
                current["action_input"] = json.loads(content) if isinstance(content, str) else content
            except:
                current["action_input"] = content
        elif ttype in ("result", "tool result", "tool_result"):
            try:
                current["tool_result"] = json.loads(content) if isinstance(content, str) else content
            except:
                current["tool_result"] = content
        elif ttype == "answer":
            if current.get("thought") is None:
                current["thought"] = "Formulating final answer"
            current["answer"] = content

    if current:
        steps.append(current)

    return steps


def evaluate_single_step(
    *,
    query: str,
    step: Dict[str, Any],
    previous_steps: List[Dict[str, Any]],
    model_name: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate a single step."""
    # Build previous context
    prev_context = []
    for i, ps in enumerate(previous_steps, 1):
        prev_context.append(f"Step {i}:")
        prev_context.append(f"  Thought: {ps.get('thought', 'N/A')}")
        prev_context.append(f"  Action: {ps.get('action', 'N/A')}")
        prev_context.append(f"  Result: {str(ps.get('tool_result', 'N/A'))[:200]}...")
    previous_context_text = "\n".join(prev_context) if prev_context else "(This is the first step)"

    thought = step.get("thought") or "(not recorded)"
    action = step.get("action") or "(none)"
    action_input = _json(step.get("action_input")) if step.get("action_input") else "{}"
    tool_result = str(step.get("tool_result") or "(none)")[:500]

    prompt = STEP_EVALUATION_TEMPLATE.format(
        query=query,
        step_num=step.get("step_num", "?"),
        thought=thought,
        action=action,
        action_input=action_input,
        tool_result=tool_result,
        previous_context=previous_context_text,
    )

    raw = _call_llm(
        prompt=prompt,
        system_prompt="You are an expert evaluator of AI agent reasoning steps. Return only valid JSON.",
        temperature=temperature,
        max_tokens=2000,
        model_name=model_name,
    )

    parsed = _safe_parse_json(raw)
    if not parsed:
        return {
            "thought_quality": 0,
            "action_appropriateness": 0,
            "result_utilization": 0,
            "progress_toward_goal": 0,
            "step_score": 0.0,
            "issues": "Failed to parse judge response",
            "suggestions": "N/A",
            "progress_explanation": "N/A",
        }

    return {
        "thought_quality": parsed.get("thought_quality", 0),
        "action_appropriateness": parsed.get("action_appropriateness", 0),
        "result_utilization": parsed.get("result_utilization", 0),
        "progress_toward_goal": parsed.get("progress_toward_goal", 0),
        "step_score": parsed.get("step_score", 0.0),
        "issues": parsed.get("issues", ""),
        "suggestions": parsed.get("suggestions", ""),
        "progress_explanation": parsed.get("progress_explanation", ""),
    }


def evaluate_trajectory_with_steps(
    traj_obj: Dict[str, Any],
    query: str,
    reference_tools: List[Dict[str, Any]],
    *,
    model_name: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
    pass_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Evaluate trajectory with both step-by-step and holistic evaluation.

    Returns combined result with both evaluations.
    """
    # Extract steps from reasoning trace
    rt = traj_obj.get("reasoning_trace", [])
    steps = extract_steps_from_reasoning_trace(rt)

    if not steps:
        print("[WARNING] No steps found in reasoning trace for step-by-step evaluation", file=sys.stderr)
        step_evaluations = []
        avg_step_score = 0.0
    else:
        # Evaluate each step
        step_evaluations = []
        for i, step in enumerate(steps):
            previous_steps = steps[:i]
            eval_result = evaluate_single_step(
                query=query,
                step=step,
                previous_steps=previous_steps,
                model_name=model_name,
                temperature=temperature,
            )
            step_evaluations.append({
                "step_num": step["step_num"],
                "thought": step.get("thought"),
                "action": step.get("action"),
                "evaluation": eval_result,
            })

        avg_step_score = sum(e["evaluation"].get("step_score", 0) for e in step_evaluations) / len(step_evaluations)

    # Get holistic evaluation (existing function)
    task_id = traj_obj.get("_filename", "unknown")
    pack = _values_from_prompt_and_traj(query, traj_obj, task_id, reference_tools)

    holistic_eval = llm_as_judge_score(
        meta=pack["meta"],
        task=pack["task"],
        history=pack["history"],
        final_answer=pack["final_answer"],
        reference_tools=pack["reference_tools"],
        temperature=temperature,
        pass_threshold=pass_threshold,
        model_name=model_name,
    )

    # Combine results
    return {
        "query": query,
        "reference_tools": reference_tools,
        "actual_tools": pack.get("actual_tools", []),
        "holistic_evaluation": holistic_eval,
        "step_by_step_evaluation": {
            "total_steps": len(steps),
            "average_step_score": avg_step_score,
            "steps": step_evaluations,
        },
        "task_id": task_id,
    }


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Judge from NEW prompt.json format + trajectories/*.json (reasoning_trace only; supports reference_tools; FOUR-dimension rubric)")
    parser.add_argument("--prompt", default="prompt.json", help="path to NEW-format prompt.json")
    parser.add_argument("--traj_dir", default="trajectories", help="directory of trajectory_*.json")
    parser.add_argument("--trajectory", default="", help="evaluate single trajectory file (auto-extracts query)")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="LLM model to use for evaluation (default: openai/gpt-4o-mini)")
    parser.add_argument("--save_json", default="", help="optional: path to save JSON results")
    parser.add_argument("--step-by-step", action="store_true", help="Enable step-by-step evaluation in addition to holistic evaluation")
    args = parser.parse_args()

    # Single trajectory mode → create NEW-format temp prompt with empty reference_tools
    if args.trajectory:
        import tempfile
        import shutil
        from pathlib import Path

        traj_path = Path(args.trajectory)
        if not traj_path.exists():
            print(f"[ERROR] Trajectory file not found: {traj_path}", file=sys.stderr)
            sys.exit(1)

        with open(traj_path, 'r', encoding='utf-8') as f:
            traj_data = json.load(f)

        query = traj_data.get("metadata", {}).get("query", "")
        if not query:
            print("[ERROR] No query found in trajectory metadata", file=sys.stderr)
            sys.exit(1)

        # Step-by-step evaluation mode
        if args.step_by_step:
            print(f"\n{'='*70}")
            print(f"Step-by-Step Evaluation: {traj_path.name}")
            print(f"{'='*70}")
            print(f"Query: {query}")
            print(f"Model: {args.model}")
            print(f"{'='*70}\n")

            results = [evaluate_trajectory_with_steps(
                traj_obj=traj_data,
                query=query,
                reference_tools=[],
                model_name=args.model,
                temperature=args.temperature,
                pass_threshold=args.threshold,
            )]
        else:
            # Original holistic-only evaluation
            temp_prompt = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
            json.dump({
                "metadata": {},
                "items": [{"query": query, "reference_tools": []}]
            }, temp_prompt, ensure_ascii=False)
            temp_prompt.close()

            temp_dir = tempfile.mkdtemp()
            shutil.copy(traj_path, Path(temp_dir) / traj_path.name)

            try:
                results = run_judge_from_files(
                    prompt_path=temp_prompt.name,
                    trajectories_dir=temp_dir,
                    pass_threshold=args.threshold,
                    temperature=args.temperature,
                    model_name=args.model,
                )
            finally:
                os.unlink(temp_prompt.name)
                shutil.rmtree(temp_dir)
    else:
        # Batch evaluation mode (not supported for step-by-step yet)
        if args.step_by_step:
            print("[WARNING] --step-by-step is only supported with --trajectory flag", file=sys.stderr)
            print("[WARNING] Falling back to holistic evaluation only", file=sys.stderr)

        results = run_judge_from_files(
            prompt_path=args.prompt,
            trajectories_dir=args.traj_dir,
            pass_threshold=args.threshold,
            temperature=args.temperature,
            model_name=args.model,
        )

    out = json.dumps(results, ensure_ascii=False, indent=2)
    print(out)

    # Auto-generate filename for step-by-step mode if not specified
    save_path = args.save_json
    if args.step_by_step and args.trajectory and not save_path:
        import re
        from pathlib import Path
        # Extract timestamp from trajectory filename (e.g., trajectory_20251114_140303.json)
        traj_name = Path(args.trajectory).stem
        match = re.search(r'(\d{8}_\d{6})', traj_name)
        if match:
            timestamp = match.group(1)
            save_path = f"evaluation/step_by_step_{timestamp}.json"
            # Create evaluation directory if it doesn't exist
            Path("evaluation").mkdir(exist_ok=True)
        else:
            save_path = "evaluation/step_by_step_results.json"

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[OK] Saved results to {save_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
