#!/usr/bin/env python3
"""
Batch Generate Trajectories for All Queries

This script runs the ReAct agent on all queries from a JSON file and generates
trajectories with UUIDs for each one.

Usage:
    # Generate all trajectories
    python runtime/batch_generate_trajectories.py \
        --query-file mcp_generate/generated_queries.json \
        --max-iterations 10 \
        --model anthropic/claude-3.5-sonnet

    # Regenerate only missing trajectories (scans existing and fills gaps)
    python runtime/batch_generate_trajectories.py \
        --query-file mcp_generate/queries_verification.json \
        --trajectory-dir trajectories/4omini-pass3 \
        --regenerate-missing \
        --model openai/gpt-4o-mini \
        --pass-number 1
"""
import asyncio
import json
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import the main function from run_react_agent
sys.path.insert(0, str(PROJECT_ROOT / "runtime"))
from run_react_agent import main as run_agent_main


def find_existing_trajectories(trajectory_dir: str, pass_number: int = None) -> Dict[str, Set[str]]:
    """
    Scan trajectory directory and find existing trajectories by UUID and pass number.

    Args:
        trajectory_dir: Directory to scan for trajectory files
        pass_number: If specified, only look for this pass number. If None, find all passes.

    Returns:
        Dict mapping pass_number -> set of UUIDs that have trajectories
    """
    existing: Dict[int, Set[str]] = {}

    # Find all trajectory files recursively
    pattern = str(Path(trajectory_dir) / "**" / "trajectory_*.json")
    trajectory_files = glob.glob(pattern, recursive=True)

    for traj_path in trajectory_files:
        try:
            with open(traj_path, 'r') as f:
                traj = json.load(f)

            metadata = traj.get("metadata", {})
            uuid = metadata.get("query_uuid")
            traj_pass = metadata.get("pass_number", 1)

            if uuid:
                # Filter by pass_number if specified
                if pass_number is not None and traj_pass != pass_number:
                    continue

                if traj_pass not in existing:
                    existing[traj_pass] = set()
                existing[traj_pass].add(uuid)

        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Failed to read {traj_path}: {e}", file=sys.stderr)
            continue

    return existing


def find_missing_trajectories(
    query_file: str,
    trajectory_dir: str,
    pass_numbers: List[int] = None
) -> Dict[int, List[dict]]:
    """
    Find queries that are missing trajectories for specified passes.

    Args:
        query_file: Path to JSON file with all queries
        trajectory_dir: Directory containing existing trajectories
        pass_numbers: List of pass numbers to check (default: [1, 2, 3])

    Returns:
        Dict mapping pass_number -> list of missing query items
    """
    if pass_numbers is None:
        pass_numbers = [1, 2, 3]

    # Load all queries
    with open(query_file, 'r') as f:
        data = json.load(f)

    queries = data.get("items", [])
    all_uuids = {q["uuid"] for q in queries if q.get("uuid")}
    query_by_uuid = {q["uuid"]: q for q in queries if q.get("uuid")}

    print(f"Total queries in file: {len(queries)}")
    print(f"Unique UUIDs: {len(all_uuids)}")

    # Find existing trajectories
    existing = find_existing_trajectories(trajectory_dir)

    # Find missing for each pass
    missing: Dict[int, List[dict]] = {}

    for pass_num in pass_numbers:
        existing_for_pass = existing.get(pass_num, set())
        missing_uuids = all_uuids - existing_for_pass

        missing[pass_num] = [query_by_uuid[uuid] for uuid in missing_uuids]

        print(f"Pass {pass_num}: {len(existing_for_pass)} existing, {len(missing_uuids)} missing")

    return missing


async def run_single_query_direct(
    query_item: dict,
    query_index: int,
    max_iterations: int,
    model: str,
    pass_number: int,
    batch_id: str,
    semaphore: asyncio.Semaphore,
    temp_query_file: str
) -> dict:
    """Run a single query directly using a temp query file with just this query."""
    async with semaphore:
        query_uuid = query_item.get("uuid", "unknown")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "runtime" / "run_react_agent.py"),
            "--query-file", temp_query_file,
            "--query-index", "0",  # Always index 0 in temp file
            "--max-iterations", str(max_iterations),
            "--model", model,
            "--pass-number", str(pass_number),
            "--batch-id", batch_id,
            "--save-trajectory"
        ]

        print(f"[{query_index + 1}] Starting query {query_uuid} (pass {pass_number})...")

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
                    "pass_number": pass_number,
                    "status": "success"
                }
            else:
                # Capture both stdout and stderr for better error diagnosis
                stderr_text = stderr.decode() if stderr else ""
                stdout_text = stdout.decode() if stdout else ""

                # Try to extract the actual error from the output
                # Look for common error patterns
                error_msg = ""

                # Check for Python exceptions in stderr
                if "Traceback" in stderr_text:
                    # Extract the last exception
                    lines = stderr_text.split('\n')
                    traceback_start = -1
                    for i, line in enumerate(lines):
                        if "Traceback" in line:
                            traceback_start = i
                    if traceback_start >= 0:
                        error_msg = '\n'.join(lines[traceback_start:])[-1000:]
                elif "Error" in stderr_text or "Exception" in stderr_text:
                    error_msg = stderr_text[-1000:]
                elif stderr_text:
                    error_msg = stderr_text[-1000:]

                # Also check stdout for errors (some errors go to stdout)
                if not error_msg and stdout_text:
                    if "Error" in stdout_text or "Exception" in stdout_text or "Traceback" in stdout_text:
                        error_msg = stdout_text[-1000:]

                if not error_msg:
                    error_msg = f"Process exited with code {process.returncode}"

                # Print a more informative error message
                print(f"[{query_index + 1}] ✗ Query {query_uuid} failed (exit code {process.returncode})")
                print(f"    Error: {error_msg[:300]}...")

                return {
                    "index": query_index + 1,
                    "uuid": query_uuid,
                    "pass_number": pass_number,
                    "status": "failed",
                    "error": error_msg,
                    "exit_code": process.returncode
                }
        except Exception as e:
            print(f"[{query_index + 1}] ✗ Query {query_uuid} failed with exception: {e}")
            return {
                "index": query_index + 1,
                "uuid": query_uuid,
                "pass_number": pass_number,
                "status": "failed",
                "error": str(e)
            }


async def regenerate_missing_trajectories(
    query_file: str,
    trajectory_dir: str,
    max_iterations: int = 10,
    model: str = "anthropic/claude-3.5-sonnet",
    pass_numbers: List[int] = None,
    max_concurrent: int = 5,
    batch_name: str = None,
) -> Dict[int, List[dict]]:
    """
    Scan for missing trajectories and regenerate them.

    Args:
        query_file: Path to JSON file with all queries
        trajectory_dir: Directory containing existing trajectories
        max_iterations: Max iterations per query
        model: Model to use
        pass_numbers: List of pass numbers to regenerate (default: [1, 2, 3])
        max_concurrent: Max concurrent queries
        batch_name: Name for the batch

    Returns:
        Dict mapping pass_number -> list of results
    """
    import uuid as uuid_module
    import tempfile
    import os

    if pass_numbers is None:
        pass_numbers = [1, 2, 3]

    # Find missing trajectories
    print(f"\n{'='*70}")
    print(f"Scanning for Missing Trajectories")
    print(f"{'='*70}")
    print(f"Query file: {query_file}")
    print(f"Trajectory dir: {trajectory_dir}")
    print(f"Pass numbers: {pass_numbers}")
    print(f"{'='*70}\n")

    missing = find_missing_trajectories(query_file, trajectory_dir, pass_numbers)

    total_missing = sum(len(queries) for queries in missing.values())

    if total_missing == 0:
        print("\n✓ No missing trajectories found! All queries have trajectories.\n")
        return {}

    print(f"\n{'='*70}")
    print(f"Regenerating {total_missing} Missing Trajectories")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Max iterations: {max_iterations}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"{'='*70}\n")

    # Generate batch ID
    batch_id = str(uuid_module.uuid4())[:8]

    if batch_name is None:
        batch_name = Path(query_file).stem

    # Create semaphore
    semaphore = asyncio.Semaphore(max_concurrent)

    all_results: Dict[int, List[dict]] = {}

    # Process each pass
    for pass_num in pass_numbers:
        missing_queries = missing.get(pass_num, [])

        if not missing_queries:
            print(f"Pass {pass_num}: No missing queries\n")
            continue

        print(f"\n--- Pass {pass_num}: {len(missing_queries)} missing queries ---\n")

        # Create temp directory for individual query files
        temp_dir = Path(tempfile.mkdtemp(prefix=f"regen_pass{pass_num}_"))

        try:
            tasks = []
            for idx, query_item in enumerate(missing_queries):
                # Create temp file with single query
                temp_file = temp_dir / f"query_{idx}.json"
                with open(temp_file, 'w') as f:
                    json.dump({"items": [query_item]}, f)

                task = run_single_query_direct(
                    query_item=query_item,
                    query_index=idx,
                    max_iterations=max_iterations,
                    model=model,
                    pass_number=pass_num,
                    batch_id=batch_id,
                    semaphore=semaphore,
                    temp_query_file=str(temp_file)
                )
                tasks.append(task)

            # Run all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "index": i + 1,
                        "uuid": missing_queries[i].get("uuid"),
                        "pass_number": pass_num,
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)

            all_results[pass_num] = processed_results

            successful = sum(1 for r in processed_results if r["status"] == "success")
            failed = sum(1 for r in processed_results if r["status"] == "failed")
            print(f"Pass {pass_num}: {successful} successful, {failed} failed\n")

        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Regeneration Summary")
    print(f"{'='*70}")

    total_success = 0
    total_failed = 0

    for pass_num, results in sorted(all_results.items()):
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        total_success += successful
        total_failed += failed
        print(f"Pass {pass_num}: {successful}/{len(results)} successful")

    print(f"{'='*70}")
    print(f"Total: {total_success} successful, {total_failed} failed")
    print(f"{'='*70}\n")

    # Save regeneration summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"regen_{batch_name}_{batch_id}_{timestamp}.json"
    summary_path = PROJECT_ROOT / "trajectories" / summary_filename

    with open(summary_path, 'w') as f:
        json.dump({
            "metadata": {
                "batch_id": batch_id,
                "batch_name": batch_name,
                "timestamp": datetime.now().isoformat(),
                "query_file": query_file,
                "trajectory_dir": trajectory_dir,
                "total_missing": total_missing,
                "total_regenerated": total_success,
                "total_failed": total_failed,
                "model": model,
                "max_iterations": max_iterations,
                "pass_numbers": pass_numbers,
            },
            "results_by_pass": {str(k): v for k, v in all_results.items()}
        }, f, indent=2)

    print(f"✓ Summary saved to: {summary_path}\n")

    return all_results


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
    import uuid as uuid_module

    # Generate batch ID
    batch_id = str(uuid_module.uuid4())[:8]

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
    parser.add_argument(
        "--batch-name",
        default=None,
        help="Name for the batch (default: derived from query file name)"
    )

    # Regenerate missing mode
    parser.add_argument(
        "--regenerate-missing",
        action="store_true",
        help="Scan for missing trajectories and regenerate only those"
    )
    parser.add_argument(
        "--trajectory-dir",
        default=None,
        help="Directory containing existing trajectories (required with --regenerate-missing)"
    )
    parser.add_argument(
        "--pass-numbers",
        type=str,
        default=None,
        help="Comma-separated list of pass numbers to check/regenerate (e.g., '1,2,3'). Default: all passes found or specified --pass-number"
    )

    args = parser.parse_args()

    if args.regenerate_missing:
        # Regenerate missing mode
        if not args.trajectory_dir:
            print("Error: --trajectory-dir is required when using --regenerate-missing")
            sys.exit(1)

        # Parse pass numbers
        if args.pass_numbers:
            pass_numbers = [int(p.strip()) for p in args.pass_numbers.split(",")]
        else:
            # Default to the single pass number specified
            pass_numbers = [args.pass_number]

        await regenerate_missing_trajectories(
            query_file=args.query_file,
            trajectory_dir=args.trajectory_dir,
            max_iterations=args.max_iterations,
            model=args.model,
            pass_numbers=pass_numbers,
            max_concurrent=args.max_concurrent,
            batch_name=args.batch_name,
        )
    else:
        # Normal batch generation mode
        await generate_all_trajectories(
            query_file=args.query_file,
            max_iterations=args.max_iterations,
            model=args.model,
            pass_number=args.pass_number,
            max_concurrent=args.max_concurrent,
            batch_name=args.batch_name,
        )


if __name__ == "__main__":
    asyncio.run(main())
