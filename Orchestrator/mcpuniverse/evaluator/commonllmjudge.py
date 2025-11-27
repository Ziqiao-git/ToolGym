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
    prompt_char_limit: int = 200000,
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
        {
          "query": "...",
          "reference_tools": [{"server":"...", "tool":"...", "why":"..."}],
          "hard_constraints": [{"type": "...", "description": "..."}],
          "soft_constraints": [{"type": "...", "description": "..."}]
        },
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
        hard_constraints = (it or {}).get("hard_constraints", [])
        soft_constraints = (it or {}).get("soft_constraints", [])

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

        if not isinstance(hard_constraints, list):
            hard_constraints = []
        if not isinstance(soft_constraints, list):
            soft_constraints = []

        out.append({
            "query": q,
            "reference_tools": norm_ref,
            "hard_constraints": hard_constraints,
            "soft_constraints": soft_constraints
        })
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
Score across FIVE dimensions that diagnose where to improve and help query generation.

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

CONSTRAINTS (must follow):
{constraints_json}

HISTORY:
{history_text}

FINAL_ANSWER:
{final_answer_text}

------------------------------------------------------------
EVALUATION RUBRIC (each 0–10, integers only)

1) Answer Reasonableness (0–10)
   Does the final answer reasonably solve the problem? Is it logical and complete?
   10 = fully reasonable, complete solution to the problem;
   8–9 = mostly reasonable, minor gaps or unclear parts;
   6–7 = partially reasonable, some logic issues or incompleteness;
   ≤5 = unreasonable, illogical, or fails to address the problem.

2) Tool Correctness (0–10)
   Based on the HISTORY above, were the tools called with correct parameters and did they execute successfully?
   Evaluate the actual tool calls made in HISTORY, NOT the REFERENCE_TOOLS.
   10 = all tools executed successfully with correct parameters;
   8–9 = mostly correct, 1-2 minor parameter errors but tools still executed;
   6–7 = several parameter errors or failed tool executions;
   ≤5 = major parameter errors, invalid calls, or most tools failed to execute.

3) Tool Relevance (0–10)
   Based on the HISTORY above, were the tools actually used relevant and useful for solving the task?
   Evaluate the actual tool calls made in HISTORY, NOT the REFERENCE_TOOLS.
   REFERENCE_TOOLS are provided as a guide but not mandatory—what matters is usefulness.
   10 = all tools used are highly relevant and contribute to solving the task;
   8–9 = most tools are relevant, with minor unnecessary or missing useful tools;
   6–7 = some irrelevant tools used, or missed several useful tools;
   ≤5 = mostly irrelevant tools, or ignored many necessary tools for the task.

4) Grounding & Evidence (0–10)
   Does the answer reference tool results? Are claims supported by tool outputs (not hallucinated)?
   10 = all claims grounded in tool results; no hallucination;
   8–9 = mostly grounded, 1–2 claims not explicitly from tools;
   6–7 = several ungrounded claims or partial hallucination;
   ≤5 = answer is mostly hallucinated or contradicts tool evidence.

5) Constraint Adherence (0–10)
   Did the agent follow all specified CONSTRAINTS?
   10 = all constraints perfectly followed;
   8–9 = minor constraint deviation with little impact;
   6–7 = noticeable constraint violations;
   ≤5 = major constraint violations or completely ignored constraints.
   N/A if no constraints provided (use 10 as default).

------------------------------------------------------------
OVERALL SCORE
overall_score = (answer_reasonableness + tool_correctness + tool_relevance + grounding + constraint_adherence) / 50.
Round to two decimals. Must be in [0,1].

BINARY DECISION
"success" if overall_score >= {pass_threshold:.2f}, else "failure".

------------------------------------------------------------
OUTPUT FORMAT
Return STRICT JSON ONLY:
{{
  "answer_reasonableness": <int 0-10>,
  "tool_correctness": <int 0-10>,
  "tool_relevance": <int 0-10>,
  "grounding": <int 0-10>,
  "constraint_adherence": <int 0-10>,
  "overall_score": <float 0-1>,
  "binary": "success" or "failure",
  "reasons": {{
    "answer_reasonableness": "<1-3 concise sentences>",
    "tool_correctness": "<1-3 concise sentences>",
    "tool_relevance": "<1-3 concise sentences>",
    "grounding": "<1-3 concise sentences>",
    "constraint_adherence": "<1-3 concise sentences>"
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
    hard_constraints: Optional[List[Dict[str, Any]]] = None,
    soft_constraints: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    pass_threshold: float = 0.85,
    max_completion_tokens: int = 8000,
    prompt_char_limit: int = 200000,
    model_name: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    meta_json = _json(meta)
    task_json = _json(task)
    ref_tools_json = _json(reference_tools or [])

    # Combine hard and soft constraints
    all_constraints = {
        "hard_constraints": hard_constraints or [],
        "soft_constraints": soft_constraints or []
    }
    constraints_json = _json(all_constraints)

    prompt = LLM_JUDGE_TEMPLATE.format(
        pass_threshold=pass_threshold,
        meta_json=meta_json,
        task_json=task_json,
        reference_tools_json=ref_tools_json,
        constraints_json=constraints_json,
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
            "answer_reasonableness": None,
            "tool_correctness": None,
            "tool_relevance": None,
            "grounding": None,
            "constraint_adherence": None,
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
        "answer_reasonableness": _coerce_0_10(parsed.get("answer_reasonableness")),
        "tool_correctness": _coerce_0_10(parsed.get("tool_correctness")),
        "tool_relevance": _coerce_0_10(parsed.get("tool_relevance")),
        "grounding": _coerce_0_10(parsed.get("grounding")),
        "constraint_adherence": _coerce_0_10(parsed.get("constraint_adherence")),
    }

    def _avg_subscores_to_overall(d: Dict[str, Optional[float]]) -> Optional[float]:
        vals = [v for v in d.values() if v is not None]
        if not vals:
            return None
        # Five dimensions, each 0-10; overall in [0,1]
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
        "answer_reasonableness": subs["answer_reasonableness"],
        "tool_correctness": subs["tool_correctness"],
        "tool_relevance": subs["tool_relevance"],
        "grounding": subs["grounding"],
        "constraint_adherence": subs["constraint_adherence"],
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


def _values_from_prompt_and_traj(
    query: str,
    traj_obj: Dict[str, Any],
    task_id: str,
    reference_tools: List[Dict[str, Any]],
    hard_constraints: Optional[List[Dict[str, Any]]] = None,
    soft_constraints: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
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
        "hard_constraints": hard_constraints or [],
        "soft_constraints": soft_constraints or [],
        "actual_tools": actual_tools,
    }


def run_judge_from_files(
    prompt_path: str = "prompt.json",
    trajectories_dir: str = "trajectories",
    *,
    pass_threshold: float = 0.85,
    temperature: float = 0.0,
    max_completion_tokens: int = 8000,
    prompt_char_limit: int = 200000,
    model_name: str = "openai/gpt-4o-mini",
) -> List[Dict[str, Any]]:
    items = load_prompts(prompt_path)  # NEW format only
    trajs = load_all_trajectories(trajectories_dir)

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, 1):
        q = item.get("query", "")
        ref_tools = item.get("reference_tools", []) or []
        hard_constraints = item.get("hard_constraints", []) or []
        soft_constraints = item.get("soft_constraints", []) or []
        matched = match_traj_by_query(trajs, q)
        if not matched:
            results.append({
                "task_id": f"prompt_idx_{idx}",
                "query": q,
                "error": "No trajectory matched for this query.",
            })
            continue

        task_id = matched.get("_filename", f"task_{idx}")
        pack = _values_from_prompt_and_traj(q, matched, task_id, ref_tools, hard_constraints, soft_constraints)

        obj = llm_as_judge_score(
            meta=pack["meta"],
            task=pack["task"],
            history=pack["history"],
            final_answer=pack["final_answer"],
            reference_tools=pack["reference_tools"],
            hard_constraints=pack["hard_constraints"],
            soft_constraints=pack["soft_constraints"],
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
            "hard_constraints": hard_constraints,
            "soft_constraints": soft_constraints,
            "actual_tools": pack.get("actual_tools", []),
            "binary": obj.get("binary"),
            "score": obj.get("score"),
            "answer_reasonableness": obj.get("answer_reasonableness"),
            "tool_correctness": obj.get("tool_correctness"),
            "tool_relevance": obj.get("tool_relevance"),
            "grounding": obj.get("grounding"),
            "constraint_adherence": obj.get("constraint_adherence"),
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
Evaluate this single step on TWO dimensions (0-10, integers only):

1) Tool Correctness (0-10)
   - Is the tool called correctly with valid parameters?
   - Does the tool execution succeed without errors?
   - Are the tool inputs properly formatted and complete?
   10 = perfect tool call, correct parameters, successful execution
   7-9 = correct tool, minor parameter issues but succeeds
   4-6 = tool called with issues (wrong params, partial success)
   0-3 = tool failed, wrong tool called, or major execution errors

2) Tool Relevance (0-10)
   - Is the chosen tool relevant to the current thought and query?
   - Does this tool choice make sense given what the agent is trying to accomplish in this step?
   - Does this tool contribute meaningful information toward solving the query?
   10 = highly relevant to thought and query, excellent choice
   7-9 = relevant and reasonable choice for this step
   4-6 = marginally relevant, could work but not ideal
   0-3 = irrelevant to thought/query, poor choice

---
OUTPUT FORMAT (strict JSON only):
{{
  "tool_correctness": <int 0-10>,
  "tool_relevance": <int 0-10>,
  "step_score": <float 0-1>,
  "issues": "<brief description of any problems, or 'none'>",
  "suggestions": "<brief suggestion for improvement, or 'none'>"
}}

where step_score = (tool_correctness + tool_relevance) / 20.0
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
            "tool_correctness": 0,
            "tool_relevance": 0,
            "step_score": 0.0,
            "issues": "Failed to parse judge response",
            "suggestions": "N/A",
        }

    return {
        "tool_correctness": parsed.get("tool_correctness", 0),
        "tool_relevance": parsed.get("tool_relevance", 0),
        "step_score": parsed.get("step_score", 0.0),
        "issues": parsed.get("issues", ""),
        "suggestions": parsed.get("suggestions", ""),
    }


def evaluate_trajectory_with_steps(
    traj_obj: Dict[str, Any],
    query: str,
    reference_tools: List[Dict[str, Any]],
    hard_constraints: Optional[List[Dict[str, Any]]] = None,
    soft_constraints: Optional[List[Dict[str, Any]]] = None,
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
    pack = _values_from_prompt_and_traj(query, traj_obj, task_id, reference_tools, hard_constraints, soft_constraints)

    holistic_eval = llm_as_judge_score(
        meta=pack["meta"],
        task=pack["task"],
        history=pack["history"],
        final_answer=pack["final_answer"],
        reference_tools=pack["reference_tools"],
        hard_constraints=pack["hard_constraints"],
        soft_constraints=pack["soft_constraints"],
        temperature=temperature,
        pass_threshold=pass_threshold,
        model_name=model_name,
    )

    # Combine results
    return {
        "query": query,
        "reference_tools": reference_tools,
        "hard_constraints": hard_constraints or [],
        "soft_constraints": soft_constraints or [],
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
    parser = argparse.ArgumentParser(description="Judge from NEW prompt.json format + trajectories/*.json (reasoning_trace only; supports reference_tools, hard/soft constraints; FIVE-dimension rubric)")
    parser.add_argument("--prompt", default="prompt.json", help="path to NEW-format prompt.json (with hard_constraints and soft_constraints)")
    parser.add_argument("--traj_dir", default="trajectories", help="directory of trajectory_*.json")
    parser.add_argument("--trajectory", default="", help="evaluate single trajectory file (auto-extracts query)")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="LLM model to use for evaluation (default: openai/gpt-4o-mini)")
    parser.add_argument("--save_json", default="", help="optional: path to save JSON results")
    parser.add_argument("--step-by-step", action="store_true", help="Enable step-by-step evaluation in addition to holistic evaluation (supports batch mode)")
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
        # Batch evaluation mode
        if args.step_by_step:
            # Batch step-by-step evaluation mode
            print(f"\n{'='*70}")
            print(f"Batch Step-by-Step Evaluation")
            print(f"{'='*70}")
            print(f"Prompt file: {args.prompt}")
            print(f"Trajectories dir: {args.traj_dir}")
            print(f"Model: {args.model}")
            print(f"{'='*70}\n")

            items = load_prompts(args.prompt)
            trajs = load_all_trajectories(args.traj_dir)

            results = []
            for idx, item in enumerate(items, 1):
                q = item.get("query", "")
                ref_tools = item.get("reference_tools", []) or []
                hard_constraints = item.get("hard_constraints", []) or []
                soft_constraints = item.get("soft_constraints", []) or []

                matched = match_traj_by_query(trajs, q)
                if not matched:
                    results.append({
                        "task_id": f"prompt_idx_{idx}",
                        "query": q,
                        "error": "No trajectory matched for this query.",
                    })
                    print(f"[WARNING] No trajectory found for query {idx}: {q[:100]}...", file=sys.stderr)
                    continue

                print(f"\n[{idx}/{len(items)}] Evaluating: {q[:100]}...", file=sys.stderr)

                eval_result = evaluate_trajectory_with_steps(
                    traj_obj=matched,
                    query=q,
                    reference_tools=ref_tools,
                    hard_constraints=hard_constraints,
                    soft_constraints=soft_constraints,
                    model_name=args.model,
                    temperature=args.temperature,
                    pass_threshold=args.threshold,
                )
                results.append(eval_result)

            print(f"\n✅ Batch step-by-step evaluation complete!", file=sys.stderr)
        else:
            # Batch holistic-only evaluation
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
