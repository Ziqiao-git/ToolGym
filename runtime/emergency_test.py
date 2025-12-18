#!/usr/bin/env python3
"""
Emergency Test - Batch Generate Trajectories with Tool Interception

This script runs the ReAct agent with emergency interception to test agent
robustness when tools fail unexpectedly.

Usage:
    # Control group: No interception
    python runtime/emergency_test.py \
        --query-file mcp_generate/generated_queries.json \
        --strategy no_interception \
        --max-iterations 20 \
        --model anthropic/claude-3.5-sonnet \
        --max-concurrent 3

    # Strategy 1: Intercept first non-search tool
    python runtime/emergency_test.py \
        --query-file mcp_generate/generated_queries.json \
        --strategy first_non_search \
        --max-iterations 20 \
        --model anthropic/claude-3.5-sonnet \
        --max-concurrent 3

    # Strategy 2: Random 20% probability interception
    python runtime/emergency_test.py \
        --query-file mcp_generate/generated_queries.json \
        --strategy random_20 \
        --max-iterations 20 \
        --model anthropic/claude-3.5-sonnet \
        --max-concurrent 3 \
        --random-seed 42

    # Run all strategies (recommended)
    python runtime/emergency_test.py \
        --query-file mcp_generate/queries10.json \
        --strategy all \
        --max-iterations 20 \
        --model anthropic/claude-3.5-sonnet \
        --max-concurrent 3
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import uuid as uuid_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import the main function from run_react_agent
sys.path.insert(0, str(PROJECT_ROOT / "runtime"))


async def run_single_query_with_interception(
    query_file: str,
    query_index: int,
    query_uuid: str,
    max_iterations: int,
    model: str,
    strategy: str,
    pass_number: int,
    batch_id: str,
    semaphore: asyncio.Semaphore,
    error_message: str = "Error: Tool temporarily unavailable (503 Service Unavailable)",
    random_seed: int = None,
) -> dict:
    """
    Run a single query with emergency interception using subprocess.

    Similar to batch_generate_trajectories.py, uses subprocess to isolate
    stdout/stderr and avoid logging conflicts.
    """
    async with semaphore:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "runtime" / "_emergency_single_run.py"),
            "--query-file", query_file,
            "--query-index", str(query_index),
            "--max-iterations", str(max_iterations),
            "--model", model,
            "--strategy", strategy,
            "--pass-number", str(pass_number),
            "--batch-id", batch_id,
            "--error-message", error_message,
        ]

        if random_seed is not None:
            cmd.extend(["--random-seed", str(random_seed)])

        print(f"[{query_index + 1}] Starting query {query_uuid}...")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print(f"[{query_index + 1}] ✓ Query {query_uuid} completed")

                # Read the saved trajectory to get interception stats
                model_name = model.split("/")[-1] if "/" in model else model
                model_safe = model_name.replace(":", "-")
                pass_folder = f"pass@{pass_number}"
                trajectory_dir = (
                    PROJECT_ROOT / "trajectories" / "Emergency_test" /
                    model_safe / pass_folder / strategy
                )

                # Find the most recent trajectory file for this query
                import glob
                pattern = str(trajectory_dir / f"trajectory_{query_uuid}_*.json")
                files = glob.glob(pattern)
                if files:
                    latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        trajectory_data = json.load(f)
                    intercepted = trajectory_data["metadata"]["interception_stats"]["intercepted"]
                else:
                    intercepted = False

                return {
                    "index": query_index + 1,
                    "uuid": query_uuid,
                    "status": "success",
                    "strategy": strategy,
                    "intercepted": intercepted,
                    "trajectory_path": latest_file if files else None,
                }
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                print(f"[{query_index + 1}] ✗ Query {query_uuid} failed: {error_msg[:200]}")
                return {
                    "index": query_index + 1,
                    "uuid": query_uuid,
                    "status": "failed",
                    "error": error_msg[:500],
                    "strategy": strategy,
                }
        except Exception as e:
            print(f"[{query_index + 1}] ✗ Query {query_uuid} failed with exception: {e}")
            return {
                "index": query_index + 1,
                "uuid": query_uuid,
                "status": "failed",
                "error": str(e),
                "strategy": strategy,
            }


async def generate_emergency_trajectories(
    query_file: str,
    max_iterations: int = 10,
    model: str = "anthropic/claude-3.5-sonnet",
    strategy: str = "first_non_search",
    pass_number: int = 1,
    max_concurrent: int = 5,
    error_message: str = "Error: Tool temporarily unavailable (503 Service Unavailable)",
    random_seed: int = 42,
):
    """Generate trajectories with emergency interception for all queries."""

    # Generate batch ID
    batch_id = str(uuid_module.uuid4())[:8]

    # Load queries
    with open(query_file, 'r') as f:
        data = json.load(f)

    queries = data.get("items", [])
    total = len(queries)

    print(f"\n{'='*70}")
    print(f"Emergency Test - Batch Trajectory Generation")
    print(f"{'='*70}")
    print(f"Batch ID: {batch_id}")
    print(f"Total queries: {total}")
    print(f"Model: {model}")
    print(f"Max iterations per query: {max_iterations}")
    print(f"Interception strategy: {strategy}")
    print(f"Pass number: {pass_number}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"Random seed: {random_seed}")
    print(f"Error message: {error_message}")
    print(f"{'='*70}\n")

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # Handle "all" strategy - run all three strategies
    if strategy == "all":
        strategies_to_run = [
            "no_interception",
            "first_non_search",
            "random_20",
        ]
    else:
        strategies_to_run = [strategy]

    all_results = []

    for strat in strategies_to_run:
        print(f"\n{'─'*70}")
        print(f"Running strategy: {strat}")
        print(f"{'─'*70}")
        print(f"Starting {total} queries with max concurrency of {max_concurrent}...\n")

        # Create tasks for all queries
        tasks = []
        for idx, query_item in enumerate(queries):
            query_uuid = query_item.get("uuid")
            # Use different random seed for each query to ensure reproducibility
            query_random_seed = random_seed + idx if random_seed is not None else None
            task = run_single_query_with_interception(
                query_file=query_file,
                query_index=idx,
                query_uuid=query_uuid,
                max_iterations=max_iterations,
                model=model,
                strategy=strat,
                pass_number=pass_number,
                batch_id=batch_id,
                semaphore=semaphore,
                error_message=error_message,
                random_seed=query_random_seed,
            )
            tasks.append(task)

        # Run all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "index": i + 1,
                    "uuid": queries[i].get("uuid"),
                    "status": "failed",
                    "error": str(result),
                    "strategy": strat,
                })
            else:
                processed_results.append(result)

        all_results.extend(processed_results)

        # Print summary for this strategy
        successful = [r for r in processed_results if r['status'] == 'success']
        intercepted = [r for r in successful if r.get('intercepted')]

        print(f"\n{'─'*70}")
        print(f"Strategy '{strat}' Summary")
        print(f"{'─'*70}")
        print(f"Total: {total}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {sum(1 for r in processed_results if r['status'] == 'failed')}")
        print(f"Intercepted: {len(intercepted)} / {len(successful)} successful runs")
        print(f"{'─'*70}\n")

    # Print overall summary
    total_successful = [r for r in all_results if r['status'] == 'success']
    total_intercepted = [r for r in total_successful if r.get('intercepted')]

    print(f"\n{'='*70}")
    print(f"Emergency Test - Overall Summary")
    print(f"{'='*70}")
    print(f"Total runs: {len(all_results)}")
    print(f"Successful: {len(total_successful)}")
    print(f"Failed: {sum(1 for r in all_results if r['status'] == 'failed')}")
    print(f"Successfully intercepted: {len(total_intercepted)} / {len(total_successful)}")
    print(f"Interception success rate: {len(total_intercepted)/len(total_successful)*100:.1f}%" if total_successful else "N/A")
    print(f"{'='*70}\n")

    # Save summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"emergency_test_pass{pass_number}_{batch_id}_{timestamp}.json"
    summary_path = PROJECT_ROOT / "trajectories" / "Emergency_test" / summary_filename
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump({
            "metadata": {
                "batch_id": batch_id,
                "test_type": "emergency_interception",
                "timestamp": datetime.now().isoformat(),
                "query_file": query_file,
                "total_queries": total,
                "total_runs": len(all_results),
                "successful": len(total_successful),
                "failed": sum(1 for r in all_results if r['status'] == 'failed'),
                "successfully_intercepted": len(total_intercepted),
                "interception_success_rate": len(total_intercepted)/len(total_successful) if total_successful else 0,
                "model": model,
                "max_iterations": max_iterations,
                "pass_number": pass_number,
                "max_concurrent": max_concurrent,
                "strategies": strategies_to_run,
                "random_seed": random_seed,
                "error_message": error_message,
            },
            "results": all_results
        }, f, indent=2)

    print(f"✓ Batch ID: {batch_id}")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"✓ Trajectories saved to: trajectories/Emergency_test/\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Emergency test - batch generate trajectories with tool interception"
    )
    parser.add_argument(
        "--query-file",
        required=True,
        help="Path to JSON file containing queries with UUIDs"
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
        "--strategy",
        choices=["no_interception", "first_non_search", "random_20", "all"],
        default="first_non_search",
        help="Interception strategy: no_interception (control), first_non_search, random_20 (20%% probability), or all (run all strategies)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible random interception (default: 42)"
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
    parser.add_argument(
        "--error-message",
        default="Error: Tool temporarily unavailable (503 Service Unavailable)",
        help="Error message to return when intercepting"
    )

    args = parser.parse_args()

    await generate_emergency_trajectories(
        query_file=args.query_file,
        max_iterations=args.max_iterations,
        model=args.model,
        strategy=args.strategy,
        pass_number=args.pass_number,
        max_concurrent=args.max_concurrent,
        error_message=args.error_message,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    asyncio.run(main())
