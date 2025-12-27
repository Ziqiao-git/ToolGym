#!/bin/bash
# Goal-Oriented Agent - Pass@3 Runner

for pass in 1 2 3; do
  echo "======================================"
  echo "Running Pass @${pass}"
  echo "======================================"

  python runtime/run_goaloriented_agent.py \
    --seeds mcp_generate/requests/multitool_50_100.json \
    --model google/gemini-3-pro-preview \
    --user-model google/gemini-3-pro-preview \
    --persona curious_researcher \
    --max-turns 60 \
    --max-concurrent 5 \
    --pass-number $pass \
    --save-trajectory

  echo "Pass @${pass} completed!"
  echo ""
done

echo "All 3 passes completed!"
echo "Results: trajectories/goaloriented/deepseek-v3.2/pass@{1,2,3}/"