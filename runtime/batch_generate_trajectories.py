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


async def generate_all_trajectories(
    query_file: str,
    max_iterations: int = 10,
    model: str = "anthropic/claude-3.5-sonnet",
):
    """Generate trajectories for all queries in the file."""

    # Load queries
    with open(query_file, 'r') as f:
        data = json.load(f)

    queries = data.get("items", [])
    total = len(queries)

    print(f"\n{'='*70}")
    print(f"Batch Trajectory Generation")
    print(f"{'='*70}")
    print(f"Total queries: {total}")
    print(f"Model: {model}")
    print(f"Max iterations per query: {max_iterations}")
    print(f"{'='*70}\n")

    results = []

    for idx, query_item in enumerate(queries, 1):
        query_uuid = query_item.get("uuid")
        query_text = query_item.get("query", "")[:100] + "..."

        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] Processing Query")
        print(f"UUID: {query_uuid}")
        print(f"Query preview: {query_text}")
        print(f"{'='*70}\n")

        # Modify sys.argv to pass arguments to run_react_agent
        original_argv = sys.argv.copy()
        sys.argv = [
            'run_react_agent.py',
            '--query-file', query_file,
            '--query-index', str(idx - 1),  # 0-based index
            '--max-iterations', str(max_iterations),
            '--model', model,
            '--save-trajectory'
        ]

        try:
            # Run the agent
            await run_agent_main()
            results.append({
                "index": idx,
                "uuid": query_uuid,
                "status": "success"
            })
            print(f"\n✓ Query {idx}/{total} completed successfully\n")
        except Exception as e:
            print(f"\n✗ Query {idx}/{total} failed: {e}\n")
            results.append({
                "index": idx,
                "uuid": query_uuid,
                "status": "failed",
                "error": str(e)
            })
        finally:
            # Restore original argv
            sys.argv = original_argv

        # Small delay between queries to avoid rate limits
        if idx < total:
            print(f"Waiting 5 seconds before next query...")
            await asyncio.sleep(5)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Batch Generation Summary")
    print(f"{'='*70}")
    print(f"Total: {total}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"{'='*70}\n")

    # Save summary
    summary_path = PROJECT_ROOT / "trajectories" / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": total,
                "successful": sum(1 for r in results if r['status'] == 'success'),
                "failed": sum(1 for r in results if r['status'] == 'failed'),
                "model": model,
                "max_iterations": max_iterations,
            },
            "results": results
        }, f, indent=2)

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

    args = parser.parse_args()

    await generate_all_trajectories(
        query_file=args.query_file,
        max_iterations=args.max_iterations,
        model=args.model
    )


if __name__ == "__main__":
    asyncio.run(main())
