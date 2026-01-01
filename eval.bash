# =============================================================================
# Single-turn trajectory evaluation (commonllmjudge)
# =============================================================================
# python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
#   --traj_dir trajectories/iter20/deepseek-v3.2 \
#   --model openai/gpt-5.1-chat \
#   --step-by-step \
#   --recursive \
#   --parallel 30 \
#   --output-dir evaluation/iter20/deepseek-v3.2

# =============================================================================
# Goal-oriented multi-turn trajectory evaluation (goaloriented_evaluator)
# =============================================================================

# Full evaluation with LLM-as-Judge (per-turn + final answer)
# python Orchestrator/mcpuniverse/evaluator/goaloriented_evaluator.py \
#   -d trajectories/goaloriented/claude-opus-4.5 \
#   -m openai/gpt-4o-mini \
#   -r \
#   -p 10 \
#   -o evaluation/goaloriented/claude-opus-4.5



# Evaluate specific model trajectories
# Output format: evaluation/goaloriented/{agent_model}_by_{eval_model}

# Configuration - just change these two variables
AGENT_MODEL="qwen3-235b-a22b-2507"         
EVAL_MODEL="openai/gpt-4o"

# Extract eval model name (remove provider prefix)
EVAL_MODEL_NAME="${EVAL_MODEL##*/}"

python Orchestrator/mcpuniverse/evaluator/goaloriented_evaluator.py \
  -d "trajectories/goaloriented/${AGENT_MODEL}" \
  -m "${EVAL_MODEL}" \
  -r \
  -p 10 \
  -o "evaluation/goaloriented/${AGENT_MODEL}_by_${EVAL_MODEL_NAME}"