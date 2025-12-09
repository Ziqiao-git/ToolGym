python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --traj_dir trajectories/claude-3.5 \
  --model openai/gpt-5.1-chat \
  --step-by-step \
  --recursive \
  --parallel 30 \
  --output-dir evaluation/claude-3.5