# commonllmjudge.py
# Judge that reads prompt + trajectories/trajectory_*.json
# HISTORY is built ONLY from reasoning_trace (Thought-aware).
# No fallback to execution.tool_calls. No LLM summarization step.

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
# HISTORY construction (reasoning_trace only)
# =========================

def build_history_from_reasoning_trace(traj_obj: Dict[str, Any]) -> str:
    """
    仅使用 trajectory['reasoning_trace'] 构建 HISTORY（保留 Thought），
    忽略 execution.tool_calls（不回退、不补齐）。
    - 跳过 type=="error"、空 content 的占位项和空 Step 标记
    - 允许 trace 中缺少显式 'Step k' 标记：自动按顺序合并成步骤
    - 过滤“全空步”；去重相邻 action+input 等价且后者信息更完整的情况
    """
    import re, ast

    rt = (traj_obj.get("reasoning_trace") or [])
    if not rt:
        # 没有 trace：给一个最小兜底（不回退 tool_calls）
        return "\n".join([
            "Step 1:",
            "Thought: (not recorded in trajectory file)",
            "Action: (none)",
            "Action Input: {}",
            "Tool Result: (none)",
        ]).strip()

    def _new_step(idx: int) -> Dict[str, Any]:
        return {
            "step": idx,
            "thought": None,
            "action": None,
            "action_input": None,
            "tool_result": None,
        }

    def _is_placeholder_thought(t: Optional[str]) -> bool:
        if not t: return True
        return str(t).strip() == "(not recorded in trajectory file)"

    def _jsonish_parse(s: Any) -> Any:
        """尽量把字符串解析成 dict/list；先 json，再 ast.literal_eval；失败则原样返回。"""
        if isinstance(s, (dict, list)):
            return s
        if isinstance(s, str):
            txt = s.strip()
            try:
                return json.loads(txt)
            except Exception:
                pass
            try:
                v = ast.literal_eval(txt)
                return v
            except Exception:
                return s
        return s

    # 1) 解析为步骤
    steps: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    idx = 0

    for item in rt:
        ttype = (item.get("type") or "").strip()
        content = (item.get("content") or "")

        # 跳过错误项与纯空项
        if ttype.lower() == "error":
            continue

        # 显式 Step 标记
        if ttype.lower().startswith("step"):
            if current:
                steps.append(current)
            idx += 1
            current = _new_step(idx)
            # 不要求这里必须有内容；后续字段会填充
            continue

        # 若没有显式 Step，从 1 开始自动累计
        if current is None:
            idx += 1
            current = _new_step(idx)

        low = ttype.lower()
        if low == "thought":
            current["thought"] = content

        elif low == "action":
            # 原样保留 action 文本（不再从文本里抽取 server/tool）
            current["action"] = content or current.get("action")

        elif low in ("action input", "action_input"):
            current["action_input"] = _jsonish_parse(content)

        elif low in ("result", "tool result", "tool_result"):
            current["tool_result"] = _jsonish_parse(content)

        elif low in ("answer",):
            # 保留宽松：answer 后不强制截断
            pass

        else:
            # 其他类型忽略
            pass

    if current:
        steps.append(current)

    # 2) 过滤“全空步”
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

    # 3) 去重：相邻 action+input 等价时，优先保留信息更丰富的那一条
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
                # 若两者同样丰富，保留先到的
                continue
        deduped.append(s)

    # 4) Thought 兜底
    for s in deduped:
        if not s.get("thought"):
            s["thought"] = "(not recorded in trajectory file)"

    # 5) 序列化输出
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
    """
    Remove trailing 'Answer:' block if any.
    """
    if not raw_history:
        return ""
    parts = raw_history.split("Answer:", 1)
    return parts[0].strip()


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

    # Build HISTORY strictly from reasoning_trace (no summarization)
    history_text = build_history_from_reasoning_trace(traj_obj)
    cleaned_history = _clean_history_before_summary(history_text)
    summarized_history = cleaned_history  # no LLM compression

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
    parser = argparse.ArgumentParser(description="Judge from prompt.json + trajectories/*.json (reasoning_trace only)")
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
