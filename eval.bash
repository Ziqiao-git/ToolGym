python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --traj_dir trajectories/deepseek-v3.2 \
  --model openai/gpt-4o-mini \
  --step-by-step \
  --recursive \
  --parallel 30 \
  --output-dir evaluation/deepseek-v3.2