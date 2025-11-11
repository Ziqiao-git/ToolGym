# MCP Research Project - Command Reference

This document contains all the Python commands and scripts for the MCP Research project.

## Table of Contents
1. [Agent Execution](#agent-execution)
   - [Single-Turn Agent](#run-react-agent-with-dynamic-tool-discovery)
   - [Multi-Turn Agent](#run-multi-turn-conversational-agent)
   - [Batch Execution](#run-batch-queries-from-json-file)
2. [Tool Testing](#tool-testing)
3. [Query Generation](#query-generation)
4. [Trajectory Evaluation](#trajectory-evaluation)
5. [Data Filtering](#data-filtering)
6. [FAISS Index Building](#faiss-index-building)
7. [Analysis Scripts](#analysis-scripts)

---

## Agent Execution

### Run ReAct Agent with Dynamic Tool Discovery

Run an agent that can dynamically discover and use MCP tools based on your query:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Basic usage
python runtime/run_react_agent.py "Your query here"

# With specific model
python runtime/run_react_agent.py \
  "Find GitHub repositories about machine learning" \
  --model "anthropic/claude-3.5-sonnet"

# With custom max iterations
python runtime/run_react_agent.py \
  "What's the weather in Seattle and any tech events there?" \
  --model "anthropic/claude-3.5-sonnet" \
  --max-iterations 10

# Save execution trajectory
python runtime/run_react_agent.py \
  "Search for clinical trials about obesity drugs" \
  --model "anthropic/claude-3.5-sonnet" \
  --save-trajectory
```

**Arguments:**
- `query`: Your question or task (required)
- `--model`: LLM model (default: `anthropic/claude-3.5-sonnet`)
- `--max-iterations`: Maximum reasoning steps (default: 5)
- `--save-trajectory`: Save execution log to `trajectories/trajectory_*.json`

**Output:**
- Agent's reasoning and actions in console
- Final response
- Trajectory JSON file (if `--save-trajectory` used)

---

### Run Multi-Turn Conversational Agent

Run an interactive multi-turn conversation with a simulated user that evaluates responses and asks follow-up questions:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Basic usage with default persona
python runtime/run_multiturn_agent.py "What are the upcoming events in Bodrum?"

# With specific persona
python runtime/run_multiturn_agent.py \
  "Find research papers about small language models" \
  --persona curious_researcher \
  --max-turns 5

# Save trajectory
python runtime/run_multiturn_agent.py \
  "What's the weather in Seattle?" \
  --persona thorough_analyst \
  --save-trajectory

# Custom model and settings
python runtime/run_multiturn_agent.py \
  "Search for information about D&D spells" \
  --persona skeptical_user \
  --model "anthropic/claude-3.5-sonnet" \
  --user-model "anthropic/claude-3.5-sonnet" \
  --max-turns 8 \
  --max-iterations 10 \
  --save-trajectory

# Run multiple seed queries from file
python runtime/run_multiturn_agent.py \
  --seeds mcp_generate/requests/multiturn_seeds_test.json \
  --persona curious_researcher \
  --max-turns 5 \
  --save-trajectory
```

**Arguments:**
- `query`: Initial query (required if not using --seeds)
- `--seeds`: JSON file with multiple seed queries to run
- `--persona`: Simulated user persona (default: `curious_researcher`)
  - `curious_researcher`: Asks detailed follow-up questions, explores deeply
  - `impatient_user`: Gets frustrated quickly, expects concise answers
  - `thorough_analyst`: Methodical, wants complete information
  - `casual_user`: Relaxed conversation style, moderate depth
  - `skeptical_user`: Questions results, wants verification
- `--max-turns`: Maximum conversation turns (default: 5)
- `--model`: LLM model for agent (default: `anthropic/claude-3.5-sonnet`)
- `--user-model`: LLM model for simulated user (default: `anthropic/claude-3.5-sonnet`)
- `--max-iterations`: Maximum agent reasoning steps per turn (default: 10)
- `--save-trajectory`: Save conversation trajectory to `trajectories/multiturn/`

**Output:**
- Console: Full conversation with agent responses and user feedback
- Trajectory files (if `--save-trajectory`): `trajectories/multiturn/multiturn_XXX_TIMESTAMP.json`

**Multi-Turn Trajectory Structure:**
```json
{
  "metadata": {
    "initial_query": "What are D&D spells for sorcerers?",
    "persona": "curious_researcher",
    "max_turns": 5,
    "timestamp": "2025-11-10T15:34:27.123456"
  },
  "turns": [
    {
      "turn_number": 1,
      "query": "What are D&D spells for sorcerers?",
      "agent_response": "...",
      "tool_calls": [...],
      "tool_results": [...],
      "available_servers": ["meta-mcp", "dnd-tools"],
      "available_tool_count": 15,
      "user_decision": "continue",
      "satisfaction_level": 0.75,
      "user_reasoning": "Good start but need more details...",
      "follow_up_intent": "Ask about specific spell levels"
    }
  ],
  "summary": {
    "total_turns": 3,
    "termination_reason": "satisfied",
    "final_satisfaction": 0.85,
    "servers_used": ["meta-mcp", "dnd-tools", "spell-database"],
    "total_tool_calls": 12
  }
}
```

**Key Multi-Turn Features:**
- **Tool Exploration**: Simulated user encourages agent to discover NEW tools each turn via meta-mcp
- **Satisfaction Tracking**: User evaluates each response and assigns satisfaction score (0.0-1.0)
- **Critical Evaluation**: User checks if agent used tools vs internal knowledge
  - No tools used → satisfaction ≤ 0.4
  - Mixed tool results with internal knowledge → satisfaction ≤ 0.4
  - Proper tool usage → normal satisfaction (0.6-1.0)
- **Dynamic Termination**: Conversation ends when user is satisfied or frustrated
- **Fresh Agent per Conversation**: Each conversation gets a new agent instance to avoid async issues

**See Also:**
- Full specification: `docs/MULTI_TURN_AGENT_SPEC.md`
- Multi-turn query generation: [Generate Multi-Turn Seed Queries](#generate-multi-turn-seed-queries)

---

### Run Batch Queries from JSON File

Run multiple queries sequentially from a JSON file (useful for benchmarking):

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Run all queries in the file
bash runtime/run_benchmark.sh mcp_generate/requests/generated_queries.json

# Example with different query file
bash runtime/run_benchmark.sh mcp_generate/requests/my_queries.json
```

**Input JSON Format:**
```json
{
  "metadata": {
    "total_queries": 2,
    "servers_count": 307
  },
  "queries": [
    "Find recent machine learning projects on GitHub",
    "What's the weather in Seattle and any tech events?"
  ]
}
```

**What it does:**
- Reads all queries from the JSON file
- Runs each query through `runtime/run_react_agent.py` sequentially
- Saves trajectory for each query to `trajectories/`
- Reports success/failure for each query

**Output:**
- Console: Progress updates (Query 1/20, Query 2/20, etc.)
- Files: Individual `trajectories/trajectory_YYYYMMDD_HHMMSS.json` for each query
- Summary: Total successes and failures

**Example Output:**
```
Loading queries from mcp_generate/requests/generated_queries.json...
Found 2 queries

============================================================
Query 1/2
============================================================
Find recent machine learning projects on GitHub

✅ Query 1 completed successfully

============================================================
Query 2/2
============================================================
What's the weather in Seattle and any tech events?

✅ Query 2 completed successfully

Benchmark complete! Trajectories saved to: trajectories/
```

---

## Tool Testing

### Test All MCP Tools

Test all tools to verify they work with realistic data:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R/MCP_INFO_MGR

# Simple test (fast, uses dummy arguments)
python test_all_tools.py \
  --output tool_probe_result.json

# LLM-based test (realistic arguments)
python test_all_tools.py \
  --use-llm \
  --model "openai/gpt-4o-mini" \
  --output tool_probe_result_llm.json

# Full LLM evaluation (realistic args + quality check)
python test_all_tools.py \
  --evaluate-with-llm \
  --model "openai/gpt-4o-mini" \
  --judge-model "openai/gpt-4o-mini" \
  --output tool_probe_result_llm_full.json

# Test specific server
python test_all_tools.py \
  --server "exa" \
  --use-llm \
  --output test_exa.json

# Test first N servers (trial run)
python test_all_tools.py \
  --use-llm \
  --limit 10 \
  --model "openai/gpt-4o-mini" \
  --output trial.json
```

**Arguments:**
- `--use-llm`: Use LLM to generate realistic test arguments
- `--evaluate-with-llm`: Use LLM to evaluate response quality
- `--evaluate-existing FILE`: Evaluate existing test results (no re-testing)
- `--model`: LLM model for argument generation
- `--judge-model`: LLM model for evaluation
- `--server`: Test only specific server
- `--limit N`: Test only first N servers
- `--output`: Output JSON file

### Evaluate Existing Test Results

Re-evaluate existing test results with LLM to find false positives:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R/MCP_INFO_MGR

python test_all_tools.py \
  --evaluate-existing tool_probe_result_llm.json \
  --output tool_probe_result_evaluated.json \
  --model "openai/gpt-4o-mini"
```

**This is much faster than re-testing!** It checks if "successful" responses actually contain errors.

---

## Query Generation

### Generate Natural Language Queries

Generate test queries for single-turn benchmarking:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

python mcp_generate/query_generate.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/requests/generated_queries.json \
  --num-queries 20

# Generate more queries
python mcp_generate/query_generate.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/requests/queries_100.json \
  --num-queries 100
```

**Arguments:**
- `--in`: Input NDJSON file with tool descriptions
- `--out`: Output JSON file for generated queries
- `--num-queries`: Number of queries to generate (default: 20)

**Output:** JSON file with generated queries:
```json
{
  "metadata": {
    "total_queries": 20,
    "servers_count": 307
  },
  "queries": ["query1", "query2", ...]
}
```

---

### Generate Multi-Turn Seed Queries

Generate seed queries optimized for multi-turn conversations:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Generate multi-turn seed queries
python mcp_generate/query_generate_multiturn.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/requests/multiturn_seeds.json \
  --num-queries 20

# Generate with custom model
python mcp_generate/query_generate_multiturn.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/requests/multiturn_seeds_100.json \
  --num-queries 100 \
  --model "anthropic/claude-3.5-sonnet"
```

**Arguments:**
- `--in`: Input NDJSON file with tool descriptions
- `--out`: Output JSON file for seed queries
- `--num-queries`: Number of seed queries to generate (default: 20)
- `--model`: LLM model for generation (default: `anthropic/claude-3.5-sonnet`)

**Output:** JSON file with seed queries (same format as single-turn):
```json
{
  "metadata": {
    "total_items": 20,
    "servers_count": 307,
    "generation_method": "async_multiturn_seeds"
  },
  "items": [
    {
      "query": "What are the upcoming events in Bodrum?",
      "reference_tools": [
        {
          "server": "@ebartan/bodrum-mcp-2025",
          "tool": "getBodrumEvents",
          "why": "To find events in Bodrum"
        }
      ]
    }
  ]
}
```

**Multi-Turn Query Characteristics:**
- Open-ended questions that naturally lead to follow-ups
- Designed to encourage tool exploration across turns
- Avoid queries that can be fully answered in one turn
- Focus on domains with multiple related MCP servers

**Use with Multi-Turn Agent:**
```bash
python runtime/run_multiturn_agent.py \
  --seeds mcp_generate/requests/multiturn_seeds.json \
  --persona curious_researcher \
  --max-turns 5 \
  --save-trajectory
```

---

## Trajectory Evaluation

### Evaluate Agent Trajectories with LLM Judge

Evaluate agent execution quality using an LLM-as-judge approach with a 5-dimension rubric:

#### Evaluate Single Trajectory

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Set Python path
export PYTHONPATH="/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:$PYTHONPATH"

# Evaluate single trajectory (uses gpt-4o-mini by default)
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --trajectory trajectories/trajectory_20251101_143915.json \
  --save_json evaluation/result.json

# Use different LLM model for evaluation
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --trajectory trajectories/trajectory_20251101_143915.json \
  --model anthropic/claude-3.5-sonnet \
  --save_json evaluation/result_claude.json

# Use GPT-4o for more thorough evaluation
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --trajectory trajectories/trajectory_20251101_143915.json \
  --model openai/gpt-4o \
  --save_json evaluation/result_gpt4o.json
```

#### Evaluate Multiple Trajectories (Batch Mode)

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

export PYTHONPATH="/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:$PYTHONPATH"

# Evaluate all trajectories matching queries in prompt file
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --prompt mcp_generate/prompt/benchmark_tasks.json \
  --traj_dir trajectories \
  --save_json evaluation/batch_results.json

# With custom model and threshold
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --prompt mcp_generate/prompt/benchmark_tasks.json \
  --traj_dir trajectories \
  --model anthropic/claude-3.5-sonnet \
  --threshold 0.9 \
  --save_json evaluation/batch_results_strict.json
```

**Arguments:**
- `--trajectory`: Path to single trajectory file (auto-extracts query from metadata)
  - **Note**: Single trajectory mode will have **empty reference_tools** in output
- `--prompt`: Path to prompt JSON file with queries (for batch mode)
  - **Required for reference_tools**: Only batch mode includes reference_tools from the prompt file
- `--traj_dir`: Directory containing trajectory files (default: `trajectories`)
- `--model`: LLM model for evaluation (default: `openai/gpt-4o-mini`)
  - Available: `openai/gpt-4o-mini`, `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, etc.
- `--threshold`: Pass/fail threshold (default: 0.85)
- `--temperature`: LLM temperature (default: 0.0)
- `--save_json`: Output JSON file path

**Evaluation Dimensions:**
1. **Task Fulfillment** (0.0-1.0): Did the agent accomplish the user's goal?
2. **Grounding** (0.0-1.0): Are responses based on actual tool outputs?
3. **Tool Choice** (0.0-1.0): Were the right tools selected?
4. **Tool Execution** (0.0-1.0): Were tools used correctly?
5. **Requirement Satisfaction** (0.0-1.0): Were all sub-requirements met?

**Output Format:**
```json
[
  {
    "task_id": "trajectory_20251101_143915",
    "query": "Find research papers about small LLMs...",
    "reference_tools": [
      {"server": "arxiv", "tool": "search_papers", "why": "To search for papers"},
      {"server": "semantic-scholar", "tool": "get_paper_details", "why": "To get details"}
    ],
    "actual_tools": [
      {"server": "arxiv", "tool": "search_papers"},
      {"server": "semantic-scholar", "tool": "get_paper_details"}
    ],
    "binary": true,
    "score": 0.82,
    "task_fulfillment": 0.85,
    "grounding": 0.90,
    "tool_choice": 0.80,
    "tool_execution": 0.75,
    "requirement_satisfaction": 0.80,
    "explanation": "The agent successfully found relevant papers..."
  }
]
```

**Key Fields:**
- `reference_tools`: Ground truth tools from query generation (what tools *should* be used)
  - Only populated in batch mode with `--prompt` flag
  - Empty `[]` in single trajectory mode
- `actual_tools`: Tools the agent actually called during execution
  - Extracted from trajectory's `reasoning_trace`
- `tool_choice` score: Evaluates alignment between reference_tools and actual_tools

**Environment Variables Required:**
```bash
export OPENAI_API_KEY="your-openrouter-api-key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

---

## Trajectory Structure and Debugging

### Understanding Trajectory Files

Trajectory files (`trajectories/trajectory_*.json`) capture the complete execution trace of the agent, including all reasoning steps, tool calls, and results.

#### Key Trajectory Sections

**1. Metadata**
```json
{
  "metadata": {
    "timestamp": "2025-11-05T12:13:56.778120",
    "query": "What's the weather in New York?",
    "model": "anthropic/claude-3.5-sonnet",
    "max_iterations": 10
  }
}
```

**2. Reasoning Trace**
Contains the agent's step-by-step thought process:
```json
{
  "reasoning_trace": [
    {"type": "thought", "content": "I need to search for weather tools..."},
    {"type": "action", "content": "Using tool `search_tools` in server `meta-mcp`"},
    {"type": "action input", "content": "{'query': 'weather', 'top_k': 5}"},
    {"type": "result", "content": "Found 5 relevant tools..."}
  ]
}
```

**3. Execution Summary**
Contains tool call details and statistics:
```json
{
  "execution": {
    "final_response": "The current weather in New York is...",
    "tool_calls": [
      {
        "type": "tool_call",
        "server": "meta-mcp",
        "tool": "search_tools",
        "arguments": {"query": "weather", "top_k": 5},
        "dynamically_loaded": false,
        "duration_seconds": 3.2,
        "status": "success"
      }
    ],
    "total_tool_calls": 4,
    "tool_calls_with_dynamic_load": 1
  }
}
```

**4. Server Loading Statistics**
```json
{
  "servers": {
    "initially_loaded": ["meta-mcp"],
    "dynamically_loaded": ["@smithery-ai/national-weather-service"],
    "total_servers_used": 2,
    "dynamically_loaded_count": 1
  }
}
```

#### Server Load Failure Tracking

**New in November 2025:** Trajectories now include detailed error categorization when MCP servers fail to load dynamically.

**Failure Categories:**
- `server_not_in_configs`: Server exists in tool index but not in `remote_server_configs.json`
- `http_403_forbidden`: Server blocked by authentication/firewall (e.g., Cloudflare protection)
- `http_404_not_found`: Server URL returns 404 Not Found
- `connection_timeout`: Server connection timed out
- `connection_error`: Network/connection error
- `[ExceptionType]`: Other errors with exception type

**Example of Failed Tool Call:**
```json
{
  "type": "tool_call",
  "server": "com.tableall/mcp",
  "tool": "search_restaurants",
  "arguments": {"prefecture": "Tokyo"},
  "status": "server_load_failed",
  "load_error": "server_not_in_configs",
  "result_preview": "Error: Server 'com.tableall/mcp' could not be loaded. Reason: server_not_in_configs",
  "dynamically_loaded": false,
  "duration_seconds": 0.15
}
```

**Debugging Common Issues:**

1. **`server_not_in_configs`**:
   - Server was validated and indexed but not added to `remote_server_configs.json`
   - Fix: Add server config to `MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json`

2. **`http_403_forbidden`**:
   - Server blocked by Cloudflare or requires authentication
   - Example: `com.windowsforum/mcp-server` (Cloudflare protected)
   - Usually cannot be fixed without server owner intervention

3. **`http_404_not_found`**:
   - Server URL is incorrect or server no longer exists
   - Fix: Update server URL or remove from index

4. **`connection_timeout`/`connection_error`**:
   - Network issues or server temporarily down
   - Retry or investigate network connectivity

**Analyzing Trajectories for Failures:**
```bash
# Find all trajectories with server load failures
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R
grep -l "server_load_failed" trajectories/*.json

# Count failure types
grep -h "load_error" trajectories/*.json | sort | uniq -c

# Extract specific failure reasons
python3 << 'EOF'
import json
import glob
from collections import Counter

failures = Counter()
for traj_file in glob.glob("trajectories/*.json"):
    with open(traj_file) as f:
        traj = json.load(f)
        for call in traj.get("execution", {}).get("tool_calls", []):
            if call.get("status") == "server_load_failed":
                error = call.get("load_error", "unknown")
                error_type = error.split(":")[0]
                failures[error_type] += 1

print("\nServer Load Failure Summary:")
for error_type, count in failures.most_common():
    print(f"  {error_type}: {count}")
EOF
```

---

## Data Filtering

### Filter Working Tools

Filter out broken tools based on test results:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R/MCP_INFO_MGR

# Filter based on simple test results
python filter_working_tools.py \
  --tool-descriptions mcp_data/indexed/tool_descriptions.ndjson \
  --probe tool_probe_result.json \
  --output-dir mcp_data/indexed

# Filter based on LLM-evaluated results (recommended)
python filter_working_tools.py \
  --tool-descriptions mcp_data/indexed/tool_descriptions.ndjson \
  --probe tool_probe_result_llm_evaluated.json \
  --output-dir mcp_data/indexed
```

**Output Files:**
- `tool_descriptions.ndjson`: Only working tools
- `tool_descriptions_failed.ndjson`: Failed tools with reasons
- `removed_tools/*.json`: Categorized failures
- `filtering_stats.json`: Statistics

---

## FAISS Index Building

### Build Semantic Search Index

Build FAISS index for semantic tool search:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

python -m MCP_INFO_MGR.semantic_search.build_search_index \
  --input MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --output MCP_INFO_MGR/semantic_search/
```

**Output Files:**
- `index.faiss`: FAISS vector index
- `metadata.json`: Tool metadata
- `config.json`: Index configuration
- `model_info.json`: Embedding model info

**Time:** ~5-10 minutes for 2,324 tools

---

## Analysis Scripts

### Analyze Failure Patterns

Analyze why tools failed:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R/MCP_INFO_MGR

# Analyze server failure patterns (all-or-nothing vs partial)
python analyze_server_failure_patterns.py

# Analyze failure root causes
python analyze_failure_reasons.py

# Recalculate true success rate (treating validation errors as working)
python recalculate_true_success_rate.py
```

**Output:** Console statistics and analysis

---

## Common Workflows

### Complete Testing and Filtering Workflow

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R/MCP_INFO_MGR

# 1. Test all tools with LLM-generated arguments
python test_all_tools.py \
  --use-llm \
  --model "openai/gpt-4o-mini" \
  --output tool_probe_result_llm.json

# 2. Evaluate results to find false positives
python test_all_tools.py \
  --evaluate-existing tool_probe_result_llm.json \
  --output tool_probe_result_llm_evaluated.json \
  --model "openai/gpt-4o-mini"

# 3. Filter working tools
python filter_working_tools.py \
  --probe tool_probe_result_llm_evaluated.json \
  --output-dir mcp_data/indexed

# 4. Rebuild FAISS index
cd ..
python -m MCP_INFO_MGR.semantic_search.build_search_index \
  --input MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --output MCP_INFO_MGR/semantic_search/
```

### Complete Evaluation Pipeline Workflow

End-to-end workflow for benchmarking and evaluating MCP agents:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Step 1: Generate benchmark queries
python mcp_generate/query_generate.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/prompt/benchmark_tasks.json \
  --num-queries 50

# Step 2: Run agent on all queries (saves trajectories)
bash runtime/run_benchmark.sh mcp_generate/prompt/benchmark_tasks.json

# Step 3: Evaluate all trajectories with LLM judge
export PYTHONPATH="/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:$PYTHONPATH"
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --prompt mcp_generate/prompt/benchmark_tasks.json \
  --traj_dir trajectories \
  --model openai/gpt-4o-mini \
  --save_json evaluation/benchmark_results.json

# Step 4: Analyze results (optional - custom analysis script)
python -c "
import json
with open('evaluation/benchmark_results.json') as f:
    results = json.load(f)
    scores = [r['score'] for r in results if 'score' in r]
    print(f'Average Score: {sum(scores)/len(scores):.2f}')
    print(f'Pass Rate: {sum(1 for r in results if r.get(\"binary\"))/len(results):.1%}')
"
```

### Quick Single Query Test

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# 1. Run agent with trajectory saving
python runtime/run_react_agent.py \
  "Find recent machine learning papers on arXiv" \
  --save-trajectory

# 2. Evaluate the trajectory (use the most recent one)
export PYTHONPATH="/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:$PYTHONPATH"
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --trajectory trajectories/trajectory_$(ls -t trajectories/ | head -1 | cut -d'_' -f2-3).json \
  --save_json evaluation/single_result.json
```

---

## Environment Setup

### Required Environment Variables

Create `.env` files with these variables:

**MCP_INFO_MGR/.env:**
```bash
SMITHERY_API_KEY=your-smithery-key
SMITHERY_REGISTRY_BASE=https://registry.smithery.ai
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

**Orchestrator/.env:**
```bash
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

---

## File Locations

### Important Files

- **Tool Descriptions:**
  - Working tools: `MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson`
  - Failed tools: `MCP_INFO_MGR/mcp_data/indexed/tool_descriptions_failed.ndjson`

- **Test Results:**
  - Simple test: `MCP_INFO_MGR/tool_probe_result.json`
  - LLM test: `MCP_INFO_MGR/tool_probe_result_llm.json`
  - Evaluated: `MCP_INFO_MGR/tool_probe_result_llm_evaluated.json`

- **FAISS Index:**
  - `MCP_INFO_MGR/semantic_search/index.faiss`
  - `MCP_INFO_MGR/semantic_search/metadata.json`

- **Trajectories:**
  - `trajectories/trajectory_*.json`

- **Generated Queries:**
  - `mcp_generate/prompt/benchmark_tasks.json`

- **Evaluation Results:**
  - `evaluation/benchmark_results.json`
  - `evaluation/single_result.json`

---

## Quick Reference

### Most Common Commands

```bash
# Test tools with LLM
cd MCP_INFO_MGR
python test_all_tools.py --use-llm --model "openai/gpt-4o-mini" --output results.json

# Run agent
cd ..
python runtime/run_react_agent.py "Your query" --save-trajectory

# Generate queries
python mcp_generate/query_generate.py --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson --out mcp_generate/prompt/benchmark_tasks.json --num-queries 20

# Evaluate trajectory
export PYTHONPATH="/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:$PYTHONPATH"
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py --trajectory trajectories/trajectory_YYYYMMDD_HHMMSS.json --save_json evaluation/result.json

# Rebuild index
python -m MCP_INFO_MGR.semantic_search.build_search_index --input MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson --output MCP_INFO_MGR/semantic_search/
```

---

## Notes

- All commands assume you're running from `/Users/xiziqiao/Documents/MCP-Research/MCP-R` unless otherwise specified
- LLM-based testing is slower but much more accurate (~15-25 min for 1435 tools)
- Use `--limit 10` for quick trial runs before full testing
- FAISS index rebuilding takes ~5-10 minutes for 2,324 tools
- Always backup important files before running filtering operations

---

## Current Statistics (As of 2025-10-29)

- **Total Tools Originally**: 5,000 tools from 364 servers
- **Truly Working Tools**: 2,324 tools (46.5%) from 307 servers
- **Failed Tools**: 2,676 tools (53.5%)
  - False positives: 2,508 (93.7%)
  - Timeouts: 99 (3.7%)
  - Execution failures: 44 (1.6%)
  - Connection failures: 25 (0.9%)

---

*Last updated: 2025-11-11*
