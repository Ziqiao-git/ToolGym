#!/bin/bash
# Goal-Oriented Agent - Pass@3 Runner
# ============================================

# ============================================
# CONFIGURATION - Edit these variables
# ============================================

# Model settings
MODEL="deepseek/deepseek-v3.2"
USER_MODEL="deepseek/deepseek-v3.2"

# Input/Output
SEEDS_FILE="mcp_generate/requests/multitool_50.json"
OUTPUT_DIR="trajectories/goaloriented"

# Execution settings
PERSONA="curious_researcher"
MAX_TURNS=100
MAX_CONCURRENT=10
MAX_ITERATIONS=60

# Pass settings
PASSES=(1 2 3)

# Optional flags (set to "true" to enable)
SAVE_TRAJECTORY="true"
ENABLE_BONUS_QUESTIONS="false"

# ============================================
# SCRIPT EXECUTION - Do not edit below
# ============================================

# Extract model name for display (e.g., "gpt-4o-mini" from "openai/gpt-4o-mini")
MODEL_NAME="${MODEL##*/}"

echo "============================================"
echo "Goal-Oriented Agent - Multi-Pass Runner"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Model:          ${MODEL}"
echo "  User Model:     ${USER_MODEL}"
echo "  Seeds File:     ${SEEDS_FILE}"
echo "  Persona:        ${PERSONA}"
echo "  Max Turns:      ${MAX_TURNS}"
echo "  Max Concurrent: ${MAX_CONCURRENT}"
echo "  Passes:         ${PASSES[*]}"
echo "  Output Dir:     ${OUTPUT_DIR}/${MODEL_NAME}/pass@{${PASSES[*]}}"
echo ""
echo "============================================"
echo ""

# Build command flags
FLAGS=""
if [ "${SAVE_TRAJECTORY}" = "true" ]; then
  FLAGS="${FLAGS} --save-trajectory"
fi
if [ "${ENABLE_BONUS_QUESTIONS}" = "true" ]; then
  FLAGS="${FLAGS} --enable-bonus-questions"
fi

# Run each pass
for pass in "${PASSES[@]}"; do
  echo "======================================"
  echo "Running Pass @${pass}"
  echo "======================================"
  echo ""

  python runtime/run_goaloriented_agent.py \
    --seeds "${SEEDS_FILE}" \
    --model "${MODEL}" \
    --user-model "${USER_MODEL}" \
    --persona "${PERSONA}" \
    --max-turns "${MAX_TURNS}" \
    --max-concurrent "${MAX_CONCURRENT}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --pass-number "${pass}" \
    ${FLAGS}

  EXIT_CODE=$?

  if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Pass @${pass} completed successfully!"
  else
    echo ""
    echo "Pass @${pass} failed with exit code ${EXIT_CODE}"
  fi
  echo ""
done

echo "============================================"
echo "All ${#PASSES[@]} passes completed!"
echo "Results: ${OUTPUT_DIR}/${MODEL_NAME}/pass@{${PASSES[*]}}"
echo "============================================"
