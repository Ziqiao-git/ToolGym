#!/usr/bin/env python3
"""
Retry Failed Queries from Batch Summary

This script reads a batch summary JSON file and retries only the queries that failed.

Usage:
    python runtime/retry_failed_queries.py \
        --batch-summary trajectories/batch_generated_queries_clean_pass1_c0d1ab7c_20251203_122600.json \
        --max-iterations 20 \
        --max-concurrent 3
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import from batch_generate_trajectories
sys.path.insert(0, str(PROJECT_ROOT / "runtime"))
from batch_generate_trajectories import run_single_query


async def retry_failed_queries(
    batch_summary_path: str,
    max_iterations: int = 10,
    max_concurrent: int = 3,
):
    """Retry only the failed queries from a batch summary file."""

    # Load batch summary
    with open(batch_summary_path, 'r') as f:
        batch_summary = json.load(f)

    metadata = batch_summary.get("metadata", {})
    results = batch_summary.get("results", [])

    # Extract original parameters
    query_file = metadata.get("query_file")
    model = metadata.get("model")
    pass_number = metadata.get("pass_number", 1)
    original_batch_id = metadata.get("batch_id", "unknown")

    if not query_file or not model:
        print("Error: Batch summary is missing required metadata (query_file or model)")
        return

    # Find failed queries
    failed_queries = [r for r in results if r.get("status") == "failed"]

    if not failed_queries:
        print("No failed queries found in batch summary. All queries succeeded!")
        return

    # Load original query file to get query details
    query_file_path = PROJECT_ROOT / query_file
    with open(query_file_path, 'r') as f:
        query_data = json.load(f)

    queries = query_data.get("items", [])

    print(f"\n{'='*70}")
    print(f"Retrying Failed Queries")
    print(f"{'='*70}")
    print(f"Original Batch ID: {original_batch_id}")
    print(f"Query file: {query_file}")
    print(f"Model: {model}")
    print(f"Pass number: {pass_number}")
    print(f"Max iterations: {max_iterations}")
    print(f"Failed queries to retry: {len(failed_queries)}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"{'='*70}\n")

    # Print list of failed queries
    print("Failed queries:")
    for fq in failed_queries:
        idx = fq.get("index", 0)
        uuid = fq.get("uuid", "unknown")
        error = fq.get("error", "Unknown error")[:100]
        print(f"  [{idx}] {uuid}: {error}...")
    print()

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for failed queries only
    tasks = []
    for failed_query in failed_queries:
        query_index = failed_query.get("index") - 1  # Convert 1-based to 0-based
        query_uuid = failed_query.get("uuid")

        task = run_single_query(
            query_file=str(query_file_path),
            query_index=query_index,
            query_uuid=query_uuid,
            max_iterations=max_iterations,
            model=model,
            pass_number=pass_number,
            batch_id=original_batch_id,  # Use original batch ID
            semaphore=semaphore
        )
        tasks.append(task)

    # Run all retry tasks in parallel
    print(f"Starting retry of {len(failed_queries)} queries with max concurrency of {max_concurrent}...\n")
    retry_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results (handle any exceptions)
    processed_results = []
    for i, result in enumerate(retry_results):
        if isinstance(result, Exception):
            processed_results.append({
                "index": failed_queries[i].get("index"),
                "uuid": failed_queries[i].get("uuid"),
                "status": "failed",
                "error": str(result)
            })
        else:
            processed_results.append(result)

    retry_results = processed_results

    # Print summary
    print(f"\n{'='*70}")
    print(f"Retry Summary")
    print(f"{'='*70}")
    print(f"Total retried: {len(failed_queries)}")
    print(f"Now successful: {sum(1 for r in retry_results if r['status'] == 'success')}")
    print(f"Still failed: {sum(1 for r in retry_results if r['status'] == 'failed')}")
    print(f"{'='*70}\n")

    # Save retry summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_name = Path(query_file).stem
    retry_summary_filename = f"retry_{batch_name}_pass{pass_number}_{original_batch_id}_{timestamp}.json"
    retry_summary_path = PROJECT_ROOT / "trajectories" / retry_summary_filename

    with open(retry_summary_path, 'w') as f:
        json.dump({
            "metadata": {
                "retry_timestamp": datetime.now().isoformat(),
                "original_batch_id": original_batch_id,
                "original_batch_summary": batch_summary_path,
                "query_file": query_file,
                "model": model,
                "max_iterations": max_iterations,
                "pass_number": pass_number,
                "max_concurrent": max_concurrent,
                "total_retried": len(failed_queries),
                "successful": sum(1 for r in retry_results if r['status'] == 'success'),
                "failed": sum(1 for r in retry_results if r['status'] == 'failed'),
            },
            "retry_results": retry_results
        }, f, indent=2)

    print(f"✓ Retry summary saved to: {retry_summary_path}\n")

    # Show which queries are still failing
    still_failed = [r for r in retry_results if r['status'] == 'failed']
    if still_failed:
        print("Queries still failing after retry:")
        for fq in still_failed:
            idx = fq.get("index", 0)
            uuid = fq.get("uuid", "unknown")
            error = fq.get("error", "Unknown error")[:100]
            print(f"  [{idx}] {uuid}: {error}...")
    else:
        print("All queries succeeded after retry!")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Retry failed queries from a batch summary JSON file"
    )
    parser.add_argument(
        "--batch-summary",
        required=True,
        help="Path to the batch summary JSON file containing failed queries"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum iterations per query (default: 20)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum number of concurrent retry queries (default: 3, lower to reduce resource contention)"
    )

    args = parser.parse_args()

    await retry_failed_queries(
        batch_summary_path=args.batch_summary,
        max_iterations=args.max_iterations,
        max_concurrent=args.max_concurrent
    )


if __name__ == "__main__":
    asyncio.run(main())
