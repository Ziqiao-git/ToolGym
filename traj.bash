cd /Users/xiziqiao/Documents/MCP-Research/MCP-R && for pass in 1 2 3; do
  python runtime/batch_generate_trajectories.py \
    --query-file mcp_generate/queries_verification.json \
    --model z-ai/glm-4.6v \
    --pass-number $pass \
    --max-iterations 20 \
    --max-concurrent 5
done
