#!/usr/bin/env python3
"""
Budget Constraint Test - Tool Call Efficiency vs Quality Tradeoff

This script runs the ReAct agent with budget-constrained queries to analyze
the tradeoff between tool call efficiency and answer quality.

Budget Levels:
- budget_3:  300 credits / 100 per call = 3 max non-search calls (tight)
- budget_5:  500 credits / 100 per call = 5 max non-search calls (medium)
- budget_7:  700 credits / 100 per call = 7 max non-search calls (loose)
- baseline:  No constraint (comparison baseline)

Usage:
    # Run budget_3 experiment
    python runtime/budget_constraint_test.py \
        --query-file task_creation_engine/queries_budget_3.json \
        --budget-level budget_3 \
        --model anthropic/claude-3.5-sonnet \
        --pass-number 1

    # Run all budget levels in sequence
    for budget in budget_3 budget_5 budget_7; do
        python runtime/budget_constraint_test.py \
            --query-file task_creation_engine/queries_${budget}.json \
            --budget-level ${budget} \
            --model anthropic/claude-3.5-sonnet \
            --pass-number 1
    done

Output Structure:
    trajectories/Budget_test/{model}/pass@{N}/{budget_level}/trajectory_{uuid}.json

Metadata Tracked:
    - budget_config: {total_budget, cost_per_call, max_allowed_calls}
    - budget_usage: {actual_calls, remaining_budget, budget_exhausted, etc.}
    - tool_call_breakdown: {search_tools, other_tools, total_calls}
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def run_single_query(
    query_file: str,
    query_index: int,
    query_uuid: str,
    max_iterations: int,
    model: str,
    pass_number: int,
    budget_level: str,
    batch_id: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """Run a single budget-constrained query."""
    async with semaphore:
        # Build output directory: trajectories/Budget_test/{model}/{budget_level}/
        # Note: run_react_agent.py will automatically add pass@{N} subdirectory
        output_dir = (
            PROJECT_ROOT / "trajectories" / "Budget_test" /
            model.replace("/", "-") / budget_level
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "runtime" / "run_react_agent.py"),
            "--query-file", query_file,
            "--query-index", str(query_index),
            "--max-iterations", str(max_iterations),
            "--model", model,
            "--pass-number", str(pass_number),
            "--batch-id", batch_id,
            "--save-trajectory",
            "--output-dir", str(output_dir),
            "--disable-compression",
        ]

        print(f"[{query_index + 1}] Starting query {query_uuid} (budget: {budget_level})...")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            _, stderr = await process.communicate()

            if process.returncode == 0:
                print(f"[{query_index + 1}] ✓ Query {query_uuid} completed successfully")
                return {
                    "index": query_index + 1,
                    "uuid": query_uuid,
                    "budget_level": budget_level,
                    "status": "success"
                }
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                print(f"[{query_index + 1}] ✗ Query {query_uuid} failed:")
                print(error_msg)  # Print full error message
                return {
                    "index": query_index + 1,
                    "uuid": query_uuid,
                    "budget_level": budget_level,
                    "status": "failed",
                    "error": error_msg  # Store full error message
                }
        except Exception as e:
            print(f"[{query_index + 1}] ✗ Query {query_uuid} failed with exception: {e}")
            return {
                "index": query_index + 1,
                "uuid": query_uuid,
                "budget_level": budget_level,
                "status": "failed",
                "error": str(e)
            }


async def generate_all_trajectories(
    query_file: str,
    budget_level: str,
    max_iterations: int = 20,
    model: str = "anthropic/claude-3.5-sonnet",
    pass_number: int = 1,
    max_concurrent: int = 5,
):
    """Generate trajectories for all queries in parallel."""
    import uuid as uuid_module

    # Generate batch ID
    batch_id = str(uuid_module.uuid4())[:8]

    # Load queries
    with open(query_file, 'r') as f:
        data = json.load(f)

    queries = data.get("items", [])
    total = len(queries)

    # Extract budget config from metadata
    budget_config = data.get("metadata", {}).get("budget_constraint", {})

    print(f"\n{'='*70}")
    print(f"Budget Constraint Test - Parallel Generation")
    print(f"{'='*70}")
    print(f"Batch ID:         {batch_id}")
    print(f"Budget level:     {budget_level}")
    print(f"Budget config:    {budget_config.get('total_budget', 'N/A')} credits, "
          f"{budget_config.get('cost_per_call', 'N/A')} per call, "
          f"max {budget_config.get('max_allowed_calls', 'N/A')} calls")
    print(f"Total queries:    {total}")
    print(f"Model:            {model}")
    print(f"Max iterations:   {max_iterations}")
    print(f"Pass number:      {pass_number}")
    print(f"Max concurrent:   {max_concurrent}")
    print(f"Output:           trajectories/Budget_test/{model.replace('/', '-')}/{budget_level}/pass@{pass_number}/")
    print(f"{'='*70}\n")

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all queries
    tasks = []
    for idx, query_item in enumerate(queries):
        query_uuid = query_item.get("uuid")
        task = run_single_query(
            query_file=query_file,
            query_index=idx,
            query_uuid=query_uuid,
            max_iterations=max_iterations,
            model=model,
            pass_number=pass_number,
            budget_level=budget_level,
            batch_id=batch_id,
            semaphore=semaphore
        )
        tasks.append(task)

    # Run all tasks in parallel
    print(f"Starting {total} queries with max concurrency of {max_concurrent}...\n")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results (handle any exceptions)
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "index": i + 1,
                "uuid": queries[i].get("uuid"),
                "budget_level": budget_level,
                "status": "failed",
                "error": str(result)
            })
        else:
            processed_results.append(result)

    results = processed_results

    # Print summary
    print(f"\n{'='*70}")
    print(f"Budget Test Generation Summary ({budget_level})")
    print(f"{'='*70}")
    print(f"Total:       {total}")
    print(f"Successful:  {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed:      {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"{'='*70}\n")

    # Save summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"budget_test_{budget_level}_pass{pass_number}_{batch_id}_{timestamp}.json"

    summary_dir = PROJECT_ROOT / "trajectories" / "Budget_test"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / summary_filename

    with open(summary_path, 'w') as f:
        json.dump({
            "metadata": {
                "batch_id": batch_id,
                "experiment": "Budget_test",
                "budget_level": budget_level,
                "budget_config": budget_config,
                "timestamp": datetime.now().isoformat(),
                "query_file": query_file,
                "total_queries": total,
                "successful": sum(1 for r in results if r['status'] == 'success'),
                "failed": sum(1 for r in results if r['status'] == 'failed'),
                "model": model,
                "max_iterations": max_iterations,
                "pass_number": pass_number,
                "max_concurrent": max_concurrent,
            },
            "results": results
        }, f, indent=2)

    print(f"✓ Batch ID:      {batch_id}")
    print(f"✓ Summary saved: {summary_path}\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run budget constraint test to analyze tool call efficiency vs quality tradeoff"
    )
    parser.add_argument(
        "--query-file",
        required=True,
        help="Path to JSON file containing budget-constrained queries"
    )
    parser.add_argument(
        "--budget-level",
        required=True,
        choices=["budget_3", "budget_5", "budget_7", "baseline"],
        help="Budget level identifier (budget_3, budget_5, budget_7, or baseline)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum iterations per query (default: 20)"
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="Model to use (default: anthropic/claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--pass-number",
        type=int,
        default=1,
        help="Pass number for multiple attempts (default: 1)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent queries (default: 5)"
    )

    args = parser.parse_args()

    await generate_all_trajectories(
        query_file=args.query_file,
        budget_level=args.budget_level,
        max_iterations=args.max_iterations,
        model=args.model,
        pass_number=args.pass_number,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())
