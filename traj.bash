# cd /Users/xiziqiao/Documents/MCP-Research/MCP-R && for pass in 1; do
#   python runtime/batch_generate_trajectories.py \
#     --query-file /Users/xiziqiao/Documents/MCP-Research/MCP-R/mcp_generate/requests/multitool_50_60.json \
#     --model openai/chatgpt-4o-latest \
#     --pass-number $pass \
#     --max-iterations 50 \
#     --max-concurrent 5
# done

python runtime/run_goaloriented_agent.py \
  --seeds mcp_generate/requests/multitool_test.json \
  --model deepseek/deepseek-v3.2 \
  --user-model deepseek/deepseek-v3.2 \
  --max-turns 100 \
  --save-trajectory