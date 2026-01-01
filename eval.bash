#!/bin/bash

# Fixed evaluation model
EVALUATOR="openai/gpt-4o"

# Data paths to evaluate
DATA_PATHS=(
  "trajectories/goaloriented/claude-opus-4.5"
  "trajectories/goaloriented/gpt-oss-120b"
  "trajectories/goaloriented/qwen3-235b-a22b-2507"
)

# Number of parallel processes
PARALLEL=10

# Extract short name of evaluator (remove prefix)
EVAL_SHORT_NAME=$(echo "$EVALUATOR" | awk -F'/' '{print $NF}')

echo "Evaluator: $EVALUATOR"
echo "Number of data paths: ${#DATA_PATHS[@]}"
echo "========================================"

# Iterate through all data paths
for DATA_PATH in "${DATA_PATHS[@]}"; do
  # Extract model name from data path (last part)
  MODEL_NAME=$(basename "$DATA_PATH")

  # Construct output path
  OUTPUT_PATH="evaluation/goaloriented/${MODEL_NAME}_by_${EVAL_SHORT_NAME}"

  echo "Evaluating: $MODEL_NAME"
  echo "Data path: $DATA_PATH"
  echo "Output path: $OUTPUT_PATH"

  python Orchestrator/mcpuniverse/evaluator/goaloriented_evaluator.py \
    -d "$DATA_PATH" \
    -m "$EVALUATOR" \
    -p "$PARALLEL" \
    -r \
    -o "$OUTPUT_PATH"

  echo "Completed: $MODEL_NAME"
  echo "----------------------------------------"
done

echo "All evaluations completed!"
