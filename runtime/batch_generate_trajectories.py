#!/usr/bin/env python3
"""
Batch Generate Trajectories for All Queries

This script runs the ReAct agent on all queries from a JSON file and generates
trajectories with UUIDs for each one.

Usage:
    python runtime/batch_generate_trajectories.py \
        --query-file mcp_generate/generated_queries.json \
        --max-iterations 10 \
        --model anthropic/claude-3.5-sonnet
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import the main function from run_react_agent
sys.path.insert(0, str(PROJECT_ROOT / "runtime"))
from run_react_agent import main as run_agent_main


async def run_single_query(
    query_file: str,
    query_index: int,
    query_uuid: str,
    max_iterations: int,
    model: str,
    pass_number: int,
    batch_id: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """Run a single query with subprocess to avoid shared state issues."""
    import subprocess

    async with semaphore:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "runtime" / "run_react_agent.py"),
            "--query-file", query_file,
            "--query-index", str(query_index),
            "--max-iterations", str(max_iterations),
            "--model", model,
            "--pass-number", str(pass_number),
            "--batch-id", batch_id,
            "--save-trajectory"
        ]

        print(f"[{query_index + 1}] Starting query {query_uuid}...")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print(f"[{query_index + 1}] ✓ Query {query_uuid} completed successfully")
                return {
                    "index": query_index + 1,
                    "uuid": query_uuid,
                    "status": "success"
                }
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                print(f"[{query_index + 1}] ✗ Query {query_uuid} failed: {error_msg[:200]}")
                return {
                    "index": query_index + 1,
                    "uuid": query_uuid,
                    "status": "failed",
                    "error": error_msg[:500]
                }
        except Exception as e:
            print(f"[{query_index + 1}] ✗ Query {query_uuid} failed with exception: {e}")
            return {
                "index": query_index + 1,
                "uuid": query_uuid,
                "status": "failed",
                "error": str(e)
            }


async def generate_all_trajectories(
    query_file: str,
    max_iterations: int = 10,
    model: str = "anthropic/claude-3.5-sonnet",
    pass_number: int = 1,
    max_concurrent: int = 5,
    batch_name: str = None,
):
    """Generate trajectories for all queries in parallel."""
    import uuid

    # Generate batch ID
    batch_id = str(uuid.uuid4())[:8]

    # Get batch name from query filename if not provided
    if batch_name is None:
        batch_name = Path(query_file).stem

    # Load queries
    with open(query_file, 'r') as f:
        data = json.load(f)

    queries = data.get("items", [])
    total = len(queries)

    print(f"\n{'='*70}")
    print(f"Batch Trajectory Generation (Parallel)")
    print(f"{'='*70}")
    print(f"Batch ID: {batch_id}")
    print(f"Batch name: {batch_name}")
    print(f"Total queries: {total}")
    print(f"Model: {model}")
    print(f"Max iterations per query: {max_iterations}")
    print(f"Pass number: {pass_number}")
    print(f"Max concurrent: {max_concurrent}")
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
                "status": "failed",
                "error": str(result)
            })
        else:
            processed_results.append(result)

    results = processed_results

    # Print summary
    print(f"\n{'='*70}")
    print(f"Batch Generation Summary")
    print(f"{'='*70}")
    print(f"Total: {total}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"{'='*70}\n")

    # Save summary with batch name and ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"batch_{batch_name}_pass{pass_number}_{batch_id}_{timestamp}.json"
    summary_path = PROJECT_ROOT / "trajectories" / summary_filename
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump({
            "metadata": {
                "batch_id": batch_id,
                "batch_name": batch_name,
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

    print(f"✓ Batch ID: {batch_id}")
    print(f"✓ Summary saved to: {summary_path}\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch generate trajectories for all queries in a JSON file"
    )
    parser.add_argument(
        "--query-file",
        required=True,
        help="Path to JSON file containing queries with UUIDs"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations per query (default: 10)"
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
        max_iterations=args.max_iterations,
        model=args.model,
        pass_number=args.pass_number,
        max_concurrent=args.max_concurrent
    )


if __name__ == "__main__":
    asyncio.run(main())
