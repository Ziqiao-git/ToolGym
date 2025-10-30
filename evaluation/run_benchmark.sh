#!/bin/bash

# run_benchmark.sh
# Run ReAct agent on all benchmark tasks
#
# Usage: bash evaluation/run_benchmark.sh mcp_generate/prompt/benchmark_tasks.json

TASKS_FILE=$1
OUTPUT_DIR="evaluation/results"

if [ -z "$TASKS_FILE" ]; then
    echo "Usage: bash evaluation/run_benchmark.sh <path_to_benchmark_tasks.json>"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract queries from JSON and run agent on each
echo "Loading queries from $TASKS_FILE..."

# Read queries using Python
python3 << EOF
import json
import sys

with open("$TASKS_FILE") as f:
    data = json.load(f)
    queries = data.get("queries", [])

print(f"Found {len(queries)} queries")

for i, query in enumerate(queries):
    print(f"\n{'='*60}")
    print(f"Query {i+1}/{len(queries)}")
    print(f"{'='*60}")
    print(query)
    print()

    # Run agent
    import subprocess
    result = subprocess.run([
        "python", "runtime/run_react_agent.py",
        query,
        "--save-trajectory",
        "--max-iterations", "10"
    ])

    if result.returncode == 0:
        print(f"✅ Query {i+1} completed successfully")
    else:
        print(f"❌ Query {i+1} failed with code {result.returncode}")
EOF

echo ""
echo "Benchmark complete! Trajectories saved to: trajectories/"
