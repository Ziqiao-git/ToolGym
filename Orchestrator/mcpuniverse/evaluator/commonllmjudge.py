# commonllmjudge.py
# Minimal judge that ONLY reads prompt.json + trajectories/trajectory_*.json
# Rubric preserved exactly; builds HISTORY from tool_calls if no raw history exists.

from __future__ import annotations
import os, json, glob, argparse, sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# If you don't use python-dotenv, you can remove the next line safely.
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

    # strip code fences if present
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

def load_prompts(prompt_path: str) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])


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
# HISTORY construction
# =========================

def _build_history_from_tool_calls(traj_obj: Dict[str, Any]) -> str:
    """
    Convert trajectory.execution.tool_calls into a HISTORY text for judge.
    Current trajectories don't include Thought; we mark it explicitly.
    """
    exe = traj_obj.get("execution", {}) or {}
    tool_calls = exe.get("tool_calls", []) or []

    lines: List[str] = []
    if not tool_calls:
        lines += [
            "Step 1:",
            "Thought: (not recorded in trajectory file)",
            "Action: (no tools called)",
            "Action Input: {}",
            "Tool Result: (none)",
        ]
        return "\n".join(lines).strip()

    for i, call in enumerate(tool_calls, 1):
        server = call.get("server", "unknown")
        tool = call.get("tool", "unknown")
        args = call.get("arguments", {})
        status = call.get("status", "")
        preview = call.get("result_preview", "")
        duration = call.get("duration_seconds", None)
        dyn = call.get("dynamically_loaded", None)

        lines.append(f"Step {i}:")
        lines.append("Thought: (not recorded in trajectory file)")
        lines.append(f"Action: server={server}, tool={tool}")
        lines.append("Action Input: " + _json(args))
        tr = {
            "status": status,
            "duration_seconds": duration,
            "dynamically_loaded": dyn,
            "result_preview": preview,
        }
        lines.append("Tool Result: " + _json(tr))

    return "\n".join(lines).strip()


def _clean_history_before_summary(raw_history: str) -> str:
    """
    Remove trailing 'Answer:' block if any.
    """
    if not raw_history:
        return ""
    parts = raw_history.split("Answer:", 1)
    return parts[0].strip()


# =========================
# Summarizer (Rubric unchanged; only summary rules adapted)
# =========================

TRAJECTORY_SUMMARY_TEMPLATE = """You are compressing an agent's multi-step ReAct trace
to create a compact trajectory summary for an evaluation judge.

GOALS
- Preserve reasoning faithfulness when available.
- Provide enough evidence to verify constraint satisfaction and grounding.
- Keep it concise.
- If the original trace has no explicit Thought lines, use exactly:
  "Thought: (not recorded in trajectory file)".

CRITICAL GUARANTEES:
- Include EVERY step from the original trace, in order.
- Do NOT fabricate tool calls or results.
- Do NOT output the final structured answer JSON.

INSTRUCTIONS

Use this exact format:

Step 1:
Thought: ...
Action: server=<server>, tool=<tool>
Action Input: ...
Tool Result: ...
Step 2:
...

Here is the trace to compress:

{raw_history}
"""

def summarize_trajectory_for_judge(
    raw_history: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 8000,
    prompt_char_limit: int = 100000,
) -> str:
    system_prompt = (
        "You are a faithful summarizer that keeps agent steps intact while removing noise. "
        "Follow the requested output format exactly."
    )
    user_prompt = TRAJECTORY_SUMMARY_TEMPLATE.format(raw_history=raw_history)

    summary = _call_llm(
        prompt=user_prompt,
        system_prompt=system_prompt,
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


# =========================
# Rubric (unchanged)
# =========================

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


def llm_as_judge_score(
    *,
    meta: Dict[str, Any],
    task: Dict[str, Any],
    history: str,
    final_answer: str,
    temperature: float = 0.0,
    pass_threshold: float = 0.85,
    max_completion_tokens: int = 8000,
    prompt_char_limit: int = 100000,
) -> Dict[str, Any]:
    meta_json = _json(meta)
    task_json = _json(task)

    prompt = LLM_JUDGE_TEMPLATE.format(
        pass_threshold=pass_threshold,
        meta_json=meta_json,
        task_json=task_json,
        history_text=history,
        final_answer_text=final_answer,
    )

    # Debug
    print(
        "========== [DEBUG] FULL prompt SENT TO JUDGE ==========\n"
        f"{prompt}\n"
        "=====================================================\n",
        file=sys.stderr, flush=True
    )

    raw = _call_llm(
        prompt=prompt,
        system_prompt=(
            "You are an impartial evaluator judging the quality of an AI agent’s multi-server, tool-based task execution. "
            "Return ONLY a valid JSON object with your scores and explanation."
        ),
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

    def _avg_subscores_to_overall(d: Dict[str, Optional[float]]) -> Optional[float]:
        vals = [v for v in d.values() if v is not None]
        if not vals:
            return None
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


# =========================
# Pipeline from files only
# =========================

def _values_from_prompt_and_traj(query: str, traj_obj: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    exe = traj_obj.get("execution", {}) or {}
    final_resp = exe.get("final_response", "")

    # Prefer raw history if you later add it; currently we build from tool_calls
    history_text = _build_history_from_tool_calls(traj_obj)

    cleaned_history = _clean_history_before_summary(history_text)
    summarized_history = summarize_trajectory_for_judge(
        cleaned_history,
        temperature=0.0,
        max_tokens=8000,
        prompt_char_limit=100000,
    )

    meta = {
        "task_id": task_id,
        "category": (traj_obj.get("metadata") or {}).get("model", "unknown_model"),
        "correct_answer": "",
    }
    task = {
        "question": query,
        "output_format": {},
    }

    return {
        "meta": meta,
        "task": task,
        "history": summarized_history,
        "final_answer": final_resp,
    }


def run_judge_from_files(
    prompt_path: str = "prompt.json",
    trajectories_dir: str = "trajectories",
    *,
    pass_threshold: float = 0.85,
    temperature: float = 0.0,
    max_completion_tokens: int = 8000,
    prompt_char_limit: int = 100000,
) -> List[Dict[str, Any]]:
    prompts = load_prompts(prompt_path)
    trajs = load_all_trajectories(trajectories_dir)

    results: List[Dict[str, Any]] = []
    for idx, q in enumerate(prompts, 1):
        matched = match_traj_by_query(trajs, q)
        if not matched:
            results.append({
                "task_id": f"prompt_idx_{idx}",
                "query": q,
                "error": "No trajectory matched for this query.",
            })
            continue

        task_id = matched.get("_filename", f"task_{idx}")
        pack = _values_from_prompt_and_traj(q, matched, task_id)

        obj = llm_as_judge_score(
            meta=pack["meta"],
            task=pack["task"],
            history=pack["history"],
            final_answer=pack["final_answer"],
            temperature=temperature,
            pass_threshold=pass_threshold,
            max_completion_tokens=max_completion_tokens,
            prompt_char_limit=prompt_char_limit,
        )

        results.append({
            "task_id": task_id,
            "query": q,
            "binary": obj.get("binary"),
            "score": obj.get("score"),
            "task_fulfillment": obj.get("task_fulfillment"),
            "grounding": obj.get("grounding"),
            "tool_choice": obj.get("tool_choice"),
            "tool_execution": obj.get("tool_execution"),
            "requirement_satisfaction": obj.get("requirement_satisfaction"),
            "explanation": obj.get("explanation"),
        })
    return results


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Judge from prompt.json + trajectories/*.json")
    parser.add_argument("--prompt", default="prompt.json", help="path to prompt.json")
    parser.add_argument("--traj_dir", default="trajectories", help="directory of trajectory_*.json")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save_json", default="", help="optional: path to save JSON results")
    args = parser.parse_args()

    results = run_judge_from_files(
        prompt_path=args.prompt,
        trajectories_dir=args.traj_dir,
        pass_threshold=args.threshold,
        temperature=args.temperature,
    )

    out = _json(results)
    print(out)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[OK] Saved results to {args.save_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
