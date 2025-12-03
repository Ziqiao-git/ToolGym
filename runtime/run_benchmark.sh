#!/bin/bash

# run_benchmark.sh
# Run ReAct agent on all benchmark tasks
#
# Usage: bash runtime/run_benchmark.sh <path_to_benchmark_tasks.json> [--skip-eval]

TASKS_FILE=$1
SKIP_EVAL=false
OUTPUT_DIR="evaluation/results"
TRAJ_DIR="trajectories/anthropic-claude-3.5-sonnet/pass@1"
RESULT_JSON="$OUTPUT_DIR/judge_results.json"

# Parse arguments
if [ -z "$TASKS_FILE" ]; then
    echo "Usage: bash evaluation/run_benchmark.sh <path_to_benchmark_tasks.json> [--skip-eval]"
    echo ""
    echo "Options:"
    echo "  --skip-eval    Skip automatic evaluation after running agent"
    exit 1
fi

# Check for --skip-eval flag
if [ "$2" = "--skip-eval" ]; then
    SKIP_EVAL=true
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
    if "queries" in data:
        queries = data["queries"]
    elif "items" in data:
        queries = [item["query"] for item in data["items"] if "query" in item]
    else:
        queries = []

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
echo "✅ Benchmark complete! Trajectories saved to: trajectories/"

# Run evaluation unless --skip-eval flag is set
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "🔎 Running commonllmjudge on generated trajectories..."

    # 运行评测
    export PYTHONPATH="/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:$PYTHONPATH"

    python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
      --prompt "$TASKS_FILE" \
      --traj_dir "$TRAJ_DIR" \
      --step-by-step \
      --model openai/gpt-5.1 \
      --save_json "$RESULT_JSON"

    echo ""
    echo "✅ Evaluation complete! Results saved to:"
    echo "   $RESULT_JSON"
else
    echo ""
    echo "⏭️  Skipping evaluation (use --skip-eval flag to control this)"
    echo ""
    echo "To evaluate later, run:"
    echo "  export PYTHONPATH=\"/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:\$PYTHONPATH\""
    echo "  python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \\"
    echo "    --prompt $TASKS_FILE \\"
    echo "    --traj_dir $TRAJ_DIR \\"
    echo "    --step-by-step \\"
    echo "    --model openai/gpt-4o-mini \\"
    echo "    --save_json $RESULT_JSON"
fi

echo ""