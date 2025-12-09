for pass in 1 2 3; do
  python runtime/batch_generate_trajectories.py \
    --query-file mcp_generate/queries_verification.json \
    --model  deepseek/deepseek-v3.2 \
    --pass-number $pass \
    --max-concurrent 5
done
