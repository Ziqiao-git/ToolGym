#!/usr/bin/env python3
"""
concurrent_judge.py
并发评测多个trajectory文件

支持两种模式:
1. 批量模式: 从prompt.json读取queries，匹配traj_dir中的所有trajectories
2. 单trajectory模式: 评测单个trajectory文件

使用方法:
  # 批量并发评测
  python runtime/concurrent_judge.py \
    --prompt prompt.json \
    --step-by-step \
    --traj_dir trajectories/anthropic-claude-3.5-sonnet/pass@1 \
    --model openai/gpt-4o-mini \
    --save_json evaluation/results.json \
    --workers 5

  # 单个trajectory评测
  python runtime/concurrent_judge.py \
    --trajectory trajectories/trajectory_20251114_140303.json \
    --model openai/gpt-4o-mini \
    --workers 1
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import from commonllmjudge
sys.path.insert(0, str(Path(__file__).parent.parent / "Orchestrator"))
from mcpuniverse.evaluator.commonllmjudge import (
    load_prompts,
    load_all_trajectories,
    match_traj_by_query,
    match_traj_by_uuid,
    evaluate_trajectory_with_steps,
    llm_as_judge_score,
    _values_from_prompt_and_traj,
)


def evaluate_single_item(
    idx: int,
    total: int,
    query: str,
    query_uuid: Optional[str],
    traj_obj: Dict[str, Any],
    reference_tools: List[Dict[str, Any]],
    hard_constraints: List[Dict[str, Any]],
    soft_constraints: List[Dict[str, Any]],
    model_name: str,
    temperature: float,
    pass_threshold: float,
    step_by_step: bool,
) -> Dict[str, Any]:
    """
    评测单个query-trajectory对

    Args:
        idx: 当前任务索引
        total: 总任务数
        query: 查询文本
        query_uuid: 查询UUID
        traj_obj: trajectory对象
        reference_tools: 参考工具列表
        hard_constraints: 硬约束列表
        soft_constraints: 软约束列表
        model_name: LLM模型名称
        temperature: 温度参数
        pass_threshold: 通过阈值
        step_by_step: 是否启用逐步评测

    Returns:
        评测结果字典
    """
    try:
        print(f"[{idx}/{total}] 🔄 Evaluating: {query[:80]}...", file=sys.stderr, flush=True)

        if step_by_step:
            # 逐步评测模式
            result = evaluate_trajectory_with_steps(
                traj_obj=traj_obj,
                query=query,
                reference_tools=reference_tools,
                hard_constraints=hard_constraints,
                soft_constraints=soft_constraints,
                query_uuid=query_uuid,
                model_name=model_name,
                temperature=temperature,
                pass_threshold=pass_threshold,
            )
        else:
            # 整体评测模式
            task_id = traj_obj.get("_filename", f"task_{idx}")
            pack = _values_from_prompt_and_traj(
                query, traj_obj, task_id, reference_tools,
                hard_constraints, soft_constraints
            )

            eval_obj = llm_as_judge_score(
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

            result = {
                "task_id": task_id,
                "uuid": query_uuid,
                "query": query,
                "reference_tools": reference_tools,
                "hard_constraints": hard_constraints,
                "soft_constraints": soft_constraints,
                "actual_tools": pack.get("actual_tools", []),
                "binary": eval_obj.get("binary"),
                "score": eval_obj.get("score"),
                "answer_reasonableness": eval_obj.get("answer_reasonableness"),
                "tool_correctness": eval_obj.get("tool_correctness"),
                "tool_relevance": eval_obj.get("tool_relevance"),
                "grounding": eval_obj.get("grounding"),
                "constraint_adherence": eval_obj.get("constraint_adherence"),
                "reasons": eval_obj.get("reasons", {}),
                "explanation": eval_obj.get("explanation"),
            }

        status = "✅" if result.get("binary") == "success" or \
                       result.get("holistic_evaluation", {}).get("binary") == "success" else "❌"
        print(f"[{idx}/{total}] {status} Completed: {query[:80]}...", file=sys.stderr, flush=True)

        return result

    except Exception as e:
        print(f"[{idx}/{total}] ❌ Error evaluating query: {str(e)}", file=sys.stderr, flush=True)
        return {
            "task_id": f"prompt_idx_{idx}",
            "uuid": query_uuid,
            "query": query,
            "error": str(e),
        }


def concurrent_batch_evaluation(
    prompt_path: str,
    traj_dir: str,
    model_name: str,
    temperature: float,
    pass_threshold: float,
    step_by_step: bool,
    max_workers: int = 5,
) -> List[Dict[str, Any]]:
    """
    并发批量评测

    Args:
        prompt_path: prompt.json文件路径
        traj_dir: trajectories目录路径
        model_name: LLM模型名称
        temperature: 温度参数
        pass_threshold: 通过阈值
        step_by_step: 是否启用逐步评测
        max_workers: 最大并发worker数量

    Returns:
        评测结果列表
    """
    print(f"\n{'='*70}")
    print(f"🚀 Concurrent Batch Evaluation")
    print(f"{'='*70}")
    print(f"Prompt file: {prompt_path}")
    print(f"Trajectories dir: {traj_dir}")
    print(f"Model: {model_name}")
    print(f"Max workers: {max_workers}")
    print(f"Step-by-step: {step_by_step}")
    print(f"{'='*70}\n")

    # 加载数据
    print("📂 Loading prompts and trajectories...", file=sys.stderr)
    items = load_prompts(prompt_path)
    trajs = load_all_trajectories(traj_dir)
    print(f"   Found {len(items)} queries and {len(trajs)} trajectories", file=sys.stderr)

    # 匹配query和trajectory
    tasks = []
    for idx, item in enumerate(items, 1):
        query_uuid = item.get("uuid")
        query = item.get("query", "")
        ref_tools = item.get("reference_tools", []) or []
        hard_constraints = item.get("hard_constraints", []) or []
        soft_constraints = item.get("soft_constraints", []) or []

        # 优先使用UUID匹配，fallback到query文本匹配
        matched = None
        if query_uuid:
            matched = match_traj_by_uuid(trajs, query_uuid)
        if not matched:
            matched = match_traj_by_query(trajs, query)

        if not matched:
            print(f"⚠️  [{idx}/{len(items)}] No trajectory found for: {query[:80]}...", file=sys.stderr)
            tasks.append({
                "idx": idx,
                "result": {
                    "task_id": f"prompt_idx_{idx}",
                    "uuid": query_uuid,
                    "query": query,
                    "error": "No trajectory matched for this query.",
                },
                "matched": False,
            })
        else:
            tasks.append({
                "idx": idx,
                "query": query,
                "query_uuid": query_uuid,
                "traj_obj": matched,
                "reference_tools": ref_tools,
                "hard_constraints": hard_constraints,
                "soft_constraints": soft_constraints,
                "matched": True,
            })

    # 并发执行评测
    results = []
    total_tasks = len(items)
    matched_tasks = [t for t in tasks if t.get("matched", False)]

    print(f"\n⚡ Starting concurrent evaluation of {len(matched_tasks)} matched trajectories...\n", file=sys.stderr)
    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {}
        for task in tasks:
            if task.get("matched"):
                future = executor.submit(
                    evaluate_single_item,
                    idx=task["idx"],
                    total=total_tasks,
                    query=task["query"],
                    query_uuid=task.get("query_uuid"),
                    traj_obj=task["traj_obj"],
                    reference_tools=task["reference_tools"],
                    hard_constraints=task["hard_constraints"],
                    soft_constraints=task["soft_constraints"],
                    model_name=model_name,
                    temperature=temperature,
                    pass_threshold=pass_threshold,
                    step_by_step=step_by_step,
                )
                future_to_task[future] = task
            else:
                # 未匹配的任务直接添加结果
                results.append(task["result"])

        # 收集结果
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = future_to_task[future]
                print(f"❌ Task {task['idx']} raised exception: {e}", file=sys.stderr)
                results.append({
                    "task_id": f"prompt_idx_{task['idx']}",
                    "uuid": task.get("query_uuid"),
                    "query": task.get("query", ""),
                    "error": f"Exception during evaluation: {str(e)}",
                })

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*70}")
    print(f"✅ Concurrent evaluation complete!")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Matched: {len(matched_tasks)}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Average: {duration/len(matched_tasks):.2f}s per task")
    print(f"{'='*70}\n")

    # 按照原始顺序排序结果
    results.sort(key=lambda x: int(x.get("task_id", "prompt_idx_0").split("_")[-1]) if "prompt_idx" in x.get("task_id", "") else 0)

    return results


def evaluate_single_trajectory(
    trajectory_path: str,
    model_name: str,
    temperature: float,
    pass_threshold: float,
    step_by_step: bool,
) -> List[Dict[str, Any]]:
    """
    评测单个trajectory文件

    Args:
        trajectory_path: trajectory文件路径
        model_name: LLM模型名称
        temperature: 温度参数
        pass_threshold: 通过阈值
        step_by_step: 是否启用逐步评测

    Returns:
        评测结果列表（单个元素）
    """
    print(f"\n{'='*70}")
    print(f"📝 Single Trajectory Evaluation")
    print(f"{'='*70}")
    print(f"Trajectory: {trajectory_path}")
    print(f"Model: {model_name}")
    print(f"Step-by-step: {step_by_step}")
    print(f"{'='*70}\n")

    traj_path = Path(trajectory_path)
    if not traj_path.exists():
        print(f"❌ Trajectory file not found: {traj_path}", file=sys.stderr)
        sys.exit(1)

    with open(traj_path, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)

    query = traj_data.get("metadata", {}).get("query", "")
    query_uuid = traj_data.get("metadata", {}).get("query_uuid")

    if not query:
        print("❌ No query found in trajectory metadata", file=sys.stderr)
        sys.exit(1)

    print(f"Query: {query}\n", file=sys.stderr)

    result = evaluate_single_item(
        idx=1,
        total=1,
        query=query,
        query_uuid=query_uuid,
        traj_obj=traj_data,
        reference_tools=[],
        hard_constraints=[],
        soft_constraints=[],
        model_name=model_name,
        temperature=temperature,
        pass_threshold=pass_threshold,
        step_by_step=step_by_step,
    )

    return [result]


def main():
    parser = argparse.ArgumentParser(
        description="并发评测多个trajectory文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 批量并发评测（5个worker）
  python runtime/concurrent_judge.py \\
    --prompt prompt.json \\
    --traj_dir trajectories/anthropic-claude-3.5-sonnet/pass@1 \\
    --model openai/gpt-4o-mini \\
    --workers 5 \\
    --save_json evaluation/results.json

  # 单个trajectory评测
  python runtime/concurrent_judge.py \\
    --trajectory trajectories/trajectory_20251114_140303.json \\
    --model openai/gpt-4o-mini \\
    --step-by-step
        """
    )

    # 输入参数
    parser.add_argument("--prompt", default="prompt.json",
                       help="路径到NEW格式的prompt.json文件 (批量模式)")
    parser.add_argument("--traj_dir", default="trajectories",
                       help="trajectories目录路径 (批量模式)")
    parser.add_argument("--trajectory", default="",
                       help="单个trajectory文件路径 (单文件模式)")

    # 评测参数
    parser.add_argument("--threshold", type=float, default=0.85,
                       help="通过阈值 (默认: 0.85)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="LLM温度参数 (默认: 0.0)")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                       help="LLM模型名称 (默认: openai/gpt-4o-mini)")
    parser.add_argument("--step-by-step", action="store_true",
                       help="启用逐步评测模式")

    # 并发参数
    parser.add_argument("--workers", type=int, default=5,
                       help="最大并发worker数量 (默认: 5)")

    # 输出参数
    parser.add_argument("--save_json", default="",
                       help="保存JSON结果的路径")

    args = parser.parse_args()

    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        print("❌ Error: Please set OPENAI_API_KEY and OPENAI_BASE_URL environment variables.",
              file=sys.stderr)
        sys.exit(1)

    # 选择模式
    if args.trajectory:
        # 单文件模式
        results = evaluate_single_trajectory(
            trajectory_path=args.trajectory,
            model_name=args.model,
            temperature=args.temperature,
            pass_threshold=args.threshold,
            step_by_step=args.step_by_step,
        )
    else:
        # 批量模式
        results = concurrent_batch_evaluation(
            prompt_path=args.prompt,
            traj_dir=args.traj_dir,
            model_name=args.model,
            temperature=args.temperature,
            pass_threshold=args.threshold,
            step_by_step=args.step_by_step,
            max_workers=args.workers,
        )

    # 输出结果
    out = json.dumps(results, ensure_ascii=False, indent=2)
    print(out)

    # 保存结果
    save_path = args.save_json
    if not save_path and args.trajectory and args.step_by_step:
        # 自动生成文件名
        import re
        traj_name = Path(args.trajectory).stem
        match = re.search(r'(\d{8}_\d{6})', traj_name)
        if match:
            timestamp = match.group(1)
            save_path = f"evaluation/concurrent_step_{timestamp}.json"
        else:
            save_path = "evaluation/concurrent_results.json"
        Path("evaluation").mkdir(exist_ok=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\n💾 Saved results to {save_path}", file=sys.stderr)

    # 统计摘要
    if results:
        total = len(results)
        errors = sum(1 for r in results if "error" in r)

        # 计算成功率（处理不同的结果格式）
        successes = 0
        for r in results:
            if "error" not in r:
                # 检查holistic_evaluation（step-by-step模式）
                if "holistic_evaluation" in r:
                    if r.get("holistic_evaluation", {}).get("binary") == "success":
                        successes += 1
                # 检查binary（普通模式）
                elif r.get("binary") == "success":
                    successes += 1

        print(f"\n📊 Summary:", file=sys.stderr)
        print(f"   Total: {total}", file=sys.stderr)
        print(f"   Success: {successes}/{total-errors} ({successes/(total-errors)*100:.1f}%)" if total > errors else "   Success: 0/0 (N/A)", file=sys.stderr)
        print(f"   Errors: {errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
