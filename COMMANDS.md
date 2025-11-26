# MCP Research Project - Command Reference

This document contains all the Python commands and scripts for the MCP Research project.

## Table of Contents
1. [Agent Execution](#agent-execution)
   - [Single-Turn Agent](#run-react-agent-with-dynamic-tool-discovery)
   - [Multi-Turn Agent](#run-multi-turn-conversational-agent)
   - [Goal-Oriented Agent](#run-goal-oriented-conversational-agent)
   - [Batch Execution](#run-batch-queries-from-json-file)
2. [Tool Testing](#tool-testing)
3. [Query Generation](#query-generation)
   - [Single-Turn Queries](#generate-natural-language-queries)
   - [Multi-Turn Seeds](#generate-multi-turn-seed-queries)
   - [Goal-Oriented Seeds](#generate-goal-oriented-seed-queries)
4. [Trajectory Evaluation](#trajectory-evaluation)
5. [Data Collection and Pipeline](#data-collection-and-pipeline)
   - [Complete Recrawl Workflow](#complete-recrawl-workflow-automated)
6. [Data Filtering](#data-filtering)
   - [Filter Remote MCP Servers](#filter-remote-mcp-servers)
   - [Filter Working Tools](#filter-working-tools)
7. [FAISS Index Building](#faiss-index-building)
8. [Analysis Scripts](#analysis-scripts)

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

### Run Goal-Oriented Conversational Agent

Run a goal-driven multi-turn conversation where the simulated user has a hidden goal that guides follow-up questions:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Basic usage with query + goal
python runtime/run_goaloriented_agent.py \
  "What are the upcoming events in Bodrum?" \
  --goal "I'm planning a weekend trip to Bodrum and need to know what events are happening, what the weather will be like, where to eat, and where to stay." \
  --persona curious_researcher

# From seeds file (recommended)
python runtime/run_goaloriented_agent.py \
  --seeds mcp_generate/requests/goaloriented_seeds_test.json \
  --persona curious_researcher \
  --max-turns 8 \
  --save-trajectory

# Custom models and settings
python runtime/run_goaloriented_agent.py \
  --seeds mcp_generate/requests/goaloriented_seeds.json \
  --persona thorough_analyst \
  --model "anthropic/claude-3.5-sonnet" \
  --user-model "anthropic/claude-3.5-sonnet" \
  --max-iterations 10 \
  --save-trajectory
```

**Arguments:**
- `query`: Initial seed query (required if not using --seeds)
- `--goal`: User's hidden goal (required if not using --seeds)
- `--seeds`: JSON file with seed queries and goals
- `--persona`: Simulated user persona (default: `curious_researcher`)
- `--max-turns`: Maximum conversation turns (overrides persona default)
- `--model`: LLM model for agent (default: `anthropic/claude-3.5-sonnet`)
- `--user-model`: LLM model for simulated user (default: `anthropic/claude-3.5-sonnet`)
- `--max-iterations`: Maximum agent reasoning steps per turn (default: 10)
- `--save-trajectory`: Save conversation trajectory to `trajectories/goaloriented/`

**Output:**
- Console: Full conversation with goal progress tracking
- Trajectory files (if `--save-trajectory`): `trajectories/goaloriented/goal_TIMESTAMP.json`

**Goal-Oriented Trajectory Structure:**
```json
{
  "metadata": {
    "seed_query": "What events are in Bodrum?",
    "user_goal": "I'm planning a trip and need events, weather, restaurants, hotels.",
    "sub_goals": [
      "Find upcoming events in Bodrum",
      "Check weather forecast",
      "Get restaurant recommendations",
      "Find hotel options"
    ],
    "goal_completion_rate": 0.75,
    "goal_achieved": false,
    "persona": "curious_researcher",
    "timestamp": "2025-11-11T20:00:00.000000"
  },
  "turns": [
    {
      "turn_number": 1,
      "query": "What events are in Bodrum?",
      "agent_response": "...",
      "tool_calls": [...],
      "completed_sub_goals": ["Find upcoming events in Bodrum"],
      "remaining_sub_goals": ["Check weather forecast", "Get restaurant recommendations", "Find hotel options"],
      "goal_progress": 0.25,
      "satisfaction_level": 0.7,
      "user_decision": "continue"
    }
  ],
  "summary": {
    "total_turns": 4,
    "final_satisfaction": 0.85,
    "goal_completion_rate": 0.75
  }
}
```

**Key Differences from Multi-Turn Agent:**
- **Hidden Goal**: User has a specific goal driving the conversation
- **Sub-Goal Tracking**: Goal is decomposed into measurable sub-goals
- **Progress-Based Evaluation**: Satisfaction considers goal progress + tool usage + quality
- **Goal-Driven Follow-Ups**: Questions work toward completing remaining sub-goals
- **Goal-Based Termination**: Conversation ends when goal achieved (even if early)

**Example Workflow:**

User Goal: "I'm planning a trip to Bodrum and need events, weather, restaurants, hotels"

Sub-goals automatically decomposed:
1. Find upcoming events in Bodrum
2. Check weather forecast for Bodrum
3. Get restaurant recommendations
4. Find hotel options

Conversation:
- **Turn 1**: "What events are in Bodrum?" → ✅ Sub-goal 1 complete (25% progress)
- **Turn 2**: "What's the weather like?" → ✅ Sub-goal 2 complete (50% progress)
- **Turn 3**: "Can you recommend restaurants?" → ✅ Sub-goal 3 complete (75% progress)
- **Turn 4**: "Where can I find hotels?" → ✅ Sub-goal 4 complete (100% progress)
- **Result**: Goal achieved, conversation terminates with high satisfaction

**See Also:**
- Goal-oriented query generation: [Generate Goal-Oriented Seed Queries](#generate-goal-oriented-seed-queries)

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

### Tool Testing Pipeline (Complete Methodology)

Testing MCP tools requires a **4-stage pipeline** because tools need active server connections and proper authentication. This section explains why testing is complex and how to run the full pipeline.

#### Why Tool Testing Is Complex

Unlike static code testing, MCP tools require:
1. **Live server connections** - Servers must be reachable via HTTPS/SSE
2. **Authentication** - Most servers require Smithery API keys
3. **Dynamic schemas** - Tool arguments vary per server, must be generated
4. **Timeout handling** - Servers can be slow or unresponsive
5. **Quality evaluation** - Technical success ≠ useful results

**Time Estimate:** 4-6 hours for ~740 remote servers (30-60s per server for fetching + testing)

#### Stage 1: Generate Server Configurations

**Purpose:** Create MCP-Universe configuration entries for Smithery remote servers.

**Why needed:** MCPManager requires specific config structure with authentication. Since all Smithery remote servers follow a standard URL pattern (`https://server.smithery.ai/{qualifiedName}/mcp`), we can generate configs directly.

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

python MCP_INFO_MGR/generate_smithery_configs.py \
  --input mcp_data/raw/smithery_remote_servers.ndjson \
  --output MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json
```

**Arguments:**
- `--input`: NDJSON file with remote servers (from filter_remote_servers.py)
- `--output`: Output JSON file with server configs
- `--api-key-variable`: Environment variable for API key (default: SMITHERY_API_KEY)

**What it does:**
- Reads remote servers from filtered list
- Generates standard Smithery URLs for each server
- Injects `{{SMITHERY_API_KEY}}` placeholders into URLs
- Creates `streamable_http` config entries for MCPManager

**Output format:**
```json
{
  "exa": {
    "streamable_http": {
      "url": "https://server.smithery.ai/exa/mcp?api_key={{SMITHERY_API_KEY}}",
      "headers": {}
    },
    "env": {}
  }
}
```

**Time:** ~1-2 seconds (fast, just JSON processing)

**Note:** For Smithery servers, we skip the reachability check because:
1. All servers use the same URL pattern
2. Unreachable servers will fail during tool fetching (next stage)
3. Saves ~2 hours compared to pre-checking each server

#### Stage 2: Fetch Tool Descriptions

**Purpose:** Connect to each server and retrieve full tool schemas (name, description, inputSchema).

**Why needed:** Tool testing requires schemas to generate valid test arguments. This stage also serves as a reachability check - servers that are offline will fail here.

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

python MCP_INFO_MGR/fetch_tool_descriptions.py \
  --input mcp_data/raw/smithery_remote_servers.ndjson \
  --config MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json \
  --output MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --retry-failed 2 \
  --retry-timeout 60
```

**Arguments:**
- `--input`: Remote servers NDJSON (used to get server names)
- `--config`: Server configs from stage 1
- `--output`: Output NDJSON file with tool descriptions
- `--retry-failed`: Number of retry attempts for failed servers (default: 2)
- `--retry-timeout`: Timeout for retries in seconds (default: 60)
- `--no-auto-retry`: Disable automatic retry
- `--retry-only`: Only retry failed servers from existing output file

**What it does:**
- Connects to each server using MCPManager
- Calls `client.list_tools()` to fetch tool metadata
- Retries failed servers with increasing timeouts
- Writes results incrementally (one server per line)
- **Time:** ~30-60 seconds per server × 740 servers = ~4-6 hours

**Output format:**
```json
{
  "qualifiedName": "exa",
  "status": "ok",
  "toolCount": 3,
  "tools": [
    {
      "name": "search",
      "description": "Search the web with Exa",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
      }
    }
  ],
  "timestamp": "2025-11-11T20:00:00.000000"
}
```

**Retry-only mode** (faster when you have existing results):
```bash
# Only retry failed servers, skip re-processing successful ones
python MCP_INFO_MGR/fetch_tool_descriptions.py \
  --retry-only \
  --retry-failed 3 \
  --retry-timeout 90
```

#### Stage 3: Test All Tools

**Purpose:** Actually invoke each tool with test arguments to verify functionality.

**Why needed:** Having a tool schema doesn't mean the tool works. This stage tests execution.

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R/MCP_INFO_MGR

# Fast mode: Simple heuristic arguments (no LLM)
python test_all_tools.py \
  --output tool_test_results_simple.json

# LLM mode: Realistic arguments generated by LLM
python test_all_tools.py \
  --use-llm \
  --model "openai/gpt-4o-mini" \
  --output tool_test_results_llm.json

# Full evaluation mode: LLM arguments + quality evaluation
python test_all_tools.py \
  --evaluate-with-llm \
  --model "openai/gpt-4o-mini" \
  --judge-model "openai/gpt-4o-mini" \
  --output tool_test_results_evaluated.json

# Trial run (first 10 servers)
python test_all_tools.py \
  --use-llm \
  --limit 10 \
  --output trial_results.json
```

**Testing Modes:**

1. **Simple Mode (default)**: Fast technical testing
   - Generates basic arguments from schema (e.g., "test query" for string params)
   - Tests technical execution only (does tool run without errors?)
   - **Time:** ~3 seconds per tool
   - **Use case:** Quick verification that tools execute

2. **LLM Mode (`--use-llm`)**: Realistic arguments
   - Uses LLM to generate meaningful test arguments
   - Still only checks technical success (no quality evaluation)
   - **Time:** ~5 seconds per tool (includes LLM API call)
   - **Use case:** Testing with realistic data

3. **LLM Evaluation Mode (`--evaluate-with-llm`)**: Quality assessment
   - Generates realistic arguments (like LLM mode)
   - Uses LLM judge to evaluate response quality
   - Detects false positives (tool executes but returns errors)
   - **Time:** ~10 seconds per tool (2 LLM calls: generation + evaluation)
   - **Use case:** Finding truly working tools

**What it does:**
- Loads tool descriptions from stage 2
- For each tool:
  1. Generates test arguments (simple or LLM-based)
  2. Connects to server and invokes tool
  3. Records status (success/failed/timeout/connection_failed)
  4. Optionally evaluates response quality with LLM
- Writes results to JSON file

**Output format:**
```json
{
  "timestamp": "2025-11-11T20:00:00.000000",
  "total_tests": 1500,
  "results": [
    {
      "server": "exa",
      "tool": "search",
      "description": "Search the web with Exa",
      "status": "success",
      "test_arguments": {"query": "latest AI news"},
      "response": {"results": [...]},
      "test_mode": "llm",
      "llm_evaluation": {
        "success": true,
        "overall_score": 0.85,
        "explanation": "Tool executed successfully and returned relevant results"
      }
    }
  ]
}
```

**Rate Limiting:**
- Built-in rate limiter: 30 requests/minute to avoid API throttling
- Automatic retries with exponential backoff

**LLM Evaluation Criteria:**

Mark as **FAILURE** (tool is broken) only if:
- HTTP errors (404, 500, etc.) in response
- Exceptions or error messages
- "Tool not configured" or "setup required"
- Completely empty response when data is expected
- Connection/timeout errors

Mark as **SUCCESS** (tool is usable) if:
- Tool executed and returned any data (even if suboptimal)
- Search returned results (even if not perfectly relevant)
- Tool returned "no results found" (tool works, just no data)
- Response has valid structure with empty/null values

**Focus:** Technical errors, NOT result quality or relevance.

#### Complete Pipeline Example (Simplified for Smithery)

```bash
#!/bin/bash
# Complete tool testing pipeline for Smithery servers

cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Stage 1: Generate configs (<1 minute)
echo "Stage 1/3: Generating server configs..."
python MCP_INFO_MGR/generate_smithery_configs.py \
  --input mcp_data/raw/smithery_remote_servers.ndjson \
  --output MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json

# Stage 2: Fetch tool descriptions (~4-6 hours with retries)
echo "Stage 2/3: Fetching tool descriptions..."
python MCP_INFO_MGR/fetch_tool_descriptions.py \
  --input mcp_data/raw/smithery_remote_servers.ndjson \
  --config MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json \
  --output MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson

# Stage 3: Test tools (~2-3 hours with LLM mode)
echo "Stage 3/3: Testing all tools..."
cd MCP_INFO_MGR
python test_all_tools.py \
  --use-llm \
  --model "openai/gpt-4o-mini" \
  --output tool_test_results.json

echo "Pipeline complete!"
```

**Total Time:** ~4-6 hours for full pipeline with 740 remote servers

**Key Simplifications for Smithery:**
- ✅ Skip reachability check - saves ~2 hours
- ✅ Direct config generation using standard URL pattern
- ✅ Reachability verified during tool fetching (servers that are offline fail there)

**Optimization Tips:**

1. **Use `--limit` for testing:** Test first 10-20 servers to verify setup
2. **Stages 1-2 only run once:** Results can be reused for multiple test runs
3. **Use `--retry-only`:** If stage 2 fails partway, retry only failed servers
4. **Simple mode first:** Run fast test, then use `--evaluate-existing` for quality check
5. **Parallel execution:** Split servers into batches and run in parallel (advanced)

**Required Environment Variables:**
- `SMITHERY_API_KEY`: Required for all stages
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY`: Required for LLM modes
- `OPENAI_BASE_URL` or `OPENROUTER_BASE_URL`: Optional, for custom endpoints

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

### Generate Goal-Oriented Seed Queries

Generate seed queries WITH GOALS for goal-oriented conversations:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Generate goal-oriented seed queries
python mcp_generate/query_generate_goaloriented.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/requests/goaloriented_seeds.json \
  --num-queries 20

# Generate with custom model
python mcp_generate/query_generate_goaloriented.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/requests/goaloriented_seeds_100.json \
  --num-queries 100 \
  --model "anthropic/claude-3.5-sonnet"
```

**Arguments:**
- `--in`: Input NDJSON file with tool descriptions
- `--out`: Output JSON file for seed queries with goals
- `--num-queries`: Number of goal-oriented seeds to generate (default: 50)
- `--tools-per-prompt`: Number of tools to sample per prompt (default: 50)
- `--model`: LLM model for generation (default: `openai/gpt-4o-mini`)

**Output:** JSON file with seed queries and goals:
```json
{
  "metadata": {
    "total_items": 20,
    "servers_count": 15,
    "generation_method": "async_goaloriented_seeds",
    "model": "openai/gpt-4o-mini"
  },
  "items": [
    {
      "query": "What events are happening in Bodrum in 2025?",
      "goal": "I'm planning a trip to Bodrum in 2025 and need to find events, check the weather, explore local restaurants, and identify suitable hotels.",
      "reference_tools": [
        {
          "server": "@ebartan/bodrum-mcp-2025",
          "tool": "getBodrumEvents",
          "why": "To find upcoming events, festivals, and activities in Bodrum."
        },
        {
          "server": "@ebartan/bodrum-mcp-2025",
          "tool": "getBodrumWeather",
          "why": "To check typical weather conditions for Bodrum by season."
        }
      ],
      "goal_category": "trip_planning",
      "complexity": "medium"
    }
  ]
}
```

**Goal Categories:**
- `trip_planning`: Events + weather + restaurants + hotels
- `research`: Papers + methods + researchers + trends
- `investment_decision`: Price + news + financials + competitors
- `product_comparison`: Reviews + specs + prices + availability
- `problem_diagnosis`: Causes + solutions + similar issues + verification
- `event_discovery`: Events + topics + tickets + speakers

**Goal Characteristics:**
- Broader than the seed query (requires multiple steps)
- Measurable (clear when complete)
- Achievable through available MCP tools
- Decomposes into 3-6 sub-goals

**Use with Goal-Oriented Agent:**
```bash
python runtime/run_goaloriented_agent.py \
  --seeds mcp_generate/requests/goaloriented_seeds.json \
  --persona curious_researcher \
  --max-turns 8 \
  --save-trajectory
```

---

## Query Verification with Reference Tools

Verify that generated queries can be solved using ONLY their reference tools (validates query quality):

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Verify single query
python runtime/verify_query_with_reference_tools.py \
  --query-file mcp_generate/requests/multi_tool_queries_117servers.json \
  --query-index 0 \
  --max-iterations 15

# Verify all queries in file
python runtime/verify_query_with_reference_tools.py \
  --query-file mcp_generate/requests/multi_tool_queries_117servers.json \
  --all \
  --max-iterations 15

# Verify AND refine insufficient queries
python runtime/verify_query_with_reference_tools.py \
  --query-file mcp_generate/requests/multi_tool_queries_117servers.json \
  --all \
  --max-iterations 15 \
  --refine-output mcp_generate/requests/multi_tool_queries_refined.json
```

**Arguments:**
- `--query-file`: JSON file with queries and reference_tools (required)
- `--query-index N`: Verify query at index N (use with single query)
- `--all`: Verify all queries in the file
- `--max-iterations`: Max agent reasoning steps (default: 15)
- `--model`: LLM model (default: `anthropic/claude-3.5-sonnet`)
- `--refine-output`: Auto-refine insufficient queries and save to this file

**What it does:**
1. Loads ONLY reference servers (restricts tool access)
2. Runs agent on query with limited tool set
3. Tracks tools used vs reference tools
4. LLM-based tool quality assessment
5. Saves to `trajectories/verification/verify_q{N}_{timestamp}.json`
6. Creates `trajectories/verification/summary.json` (batch mode)
7. **If `--refine-output`**: Rewrites insufficient queries to be solvable with reference tools

**Key Output Fields:**
- `tools_used`: Tools actually called (`["exa/search", "github/list_repos"]`)
- `tools_matched`: Boolean - did agent use correct tools?
- `self_evaluation`: Agent's assessment (`"SUFFICIENT"` or `"INSUFFICIENT"`)
- `tool_quality_assessment`: LLM judge score (0.0-1.0)
- `tools_with_missing_descriptions`: Tools with None/empty descriptions

**Interpreting Results:**

✅ Good: `tools_matched=true`, `self_evaluation="SUFFICIENT"`, `quality_score>=0.7`
⚠️ Investigate: `tools_matched=false`, `quality_score=0.4-0.7`
❌ Bad: `tools_used=0`, `self_evaluation="INSUFFICIENT"`, `quality_score<0.4`

**Common Issues:**
- Context overflow (298K+ tokens): Some tools return huge responses
- Timeout errors: Increase `--max-iterations`
- Missing descriptions: Tracked but handled gracefully

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

### Step-by-Step Evaluation (NEW)

Evaluate each reasoning step individually in addition to holistic evaluation. This mode evaluates whether each step contributes toward the ultimate goal.

#### Basic Step-by-Step Evaluation

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R
export PYTHONPATH="/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:$PYTHONPATH"

# Evaluate single trajectory with step-by-step analysis
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --trajectory trajectories/trajectory_20251106_140812.json \
  --step-by-step \
  --model openai/gpt-4o-mini

# Save results to file
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --trajectory trajectories/trajectory_20251106_140812.json \
  --step-by-step \
  --model openai/gpt-4o-mini \
  --save_json evaluation/step_by_step_results.json

# Use different judge model
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --trajectory trajectories/trajectory_20251106_140812.json \
  --step-by-step \
  --model anthropic/claude-3.5-sonnet \
  --save_json evaluation/step_by_step_claude.json
```

**What it does:**
- Evaluates EACH reasoning step individually (Thought → Action → Result)
- Checks if each step progresses toward answering the user's query
- Provides both step-by-step evaluation AND holistic evaluation
- Each step is evaluated with full awareness of previous steps

**Step-by-Step Evaluation Dimensions (4 dimensions per step):**

1. **Thought Quality** (0-10): Clarity and reasoning quality
2. **Action Appropriateness** (0-10): Was the action/tool choice appropriate?
3. **Result Utilization** (0-10): How well was the tool result used?
4. **Progress Toward Goal** (0-10): **MOST CRITICAL** - Does this step move closer to answering the query?
   - 10 = Essential step, directly advances toward goal
   - 8-9 = Helpful step, clearly contributes
   - 6-7 = Marginal progress
   - 4-5 = Minimal progress, mostly redundant
   - 0-3 = No progress, wrong direction

**Step Score Calculation:**
```
step_score = (thought_quality + action_appropriateness + result_utilization + progress_toward_goal) / 40.0
```

**Output Format (Step-by-Step Mode):**
```json
{
  "query": "Can you find current trending games on Steam?",
  "holistic_evaluation": {
    "score": 0.82,
    "task_fulfillment": 0.85,
    "grounding": 0.90,
    "tool_choice": 0.80,
    "tool_execution": 0.75,
    "requirement_satisfaction": 0.80,
    "explanation": "Overall trajectory evaluation..."
  },
  "step_by_step_evaluation": {
    "total_steps": 3,
    "average_step_score": 0.78,
    "steps": [
      {
        "step_number": 1,
        "thought": "I need to search for tools related to Steam games...",
        "action": "search_tools",
        "action_input": "{'query': 'steam games', 'top_k': 10}",
        "tool_result": "Found 10 relevant tools...",
        "evaluation": {
          "thought_quality": 8,
          "action_appropriateness": 9,
          "result_utilization": 7,
          "progress_toward_goal": 9,
          "step_score": 0.825,
          "progress_explanation": "Essential first step - searching for relevant tools is necessary to find Steam-related functionality."
        }
      },
      {
        "step_number": 2,
        "thought": "Now I'll use the steam_trending tool...",
        "action": "steam_trending",
        "action_input": "{'category': 'popular'}",
        "tool_result": "[List of trending games...]",
        "evaluation": {
          "thought_quality": 9,
          "action_appropriateness": 10,
          "result_utilization": 8,
          "progress_toward_goal": 10,
          "step_score": 0.925,
          "progress_explanation": "Critical step - directly retrieves trending games data, exactly what the query asked for."
        }
      }
    ]
  }
}
```

**Key Features:**

- **Context Awareness**: Each step evaluation includes:
  - ULTIMATE GOAL (user's query) prominently displayed
  - All previous steps (thought, action, result truncated to 200 chars)
  - Current step details (result truncated to 500 chars)

- **Progress-Oriented**: Primary focus on whether each step advances toward the goal
  - Detects redundant steps that don't contribute
  - Identifies tangential explorations
  - Rewards essential steps highly

- **Combined Output**: Get both perspectives
  - Step-by-step: Micro-level analysis of reasoning quality
  - Holistic: Macro-level assessment of overall success

**When to Use Step-by-Step:**

✅ **Use when:**
- Debugging agent reasoning issues
- Understanding why an agent succeeded/failed at each step
- Analyzing efficiency of multi-step trajectories
- Identifying redundant or unnecessary steps
- Training data curation for agent fine-tuning

❌ **Skip when:**
- Only need final answer quality (use holistic mode)
- Processing large batches (step-by-step is slower)
- Cost-sensitive evaluation (2x LLM calls vs holistic)

**Performance Notes:**
- **Time**: ~2x slower than holistic-only mode (evaluates each step + holistic)
- **Cost**: Higher LLM API costs (N step evaluations + 1 holistic evaluation)
- **Output Size**: Larger JSON files with per-step details

**Arguments:**
- `--step-by-step`: Enable step-by-step evaluation (default: False)
- `--trajectory`: Path to single trajectory file (required)
- `--model`: LLM judge model (default: `openai/gpt-4o-mini`)
- `--temperature`: LLM temperature (default: 0.0)
- `--threshold`: Pass/fail threshold (default: 0.85)
- `--save_json`: Output file path (optional, prints to stdout if omitted)

**Example Workflow:**

```bash
# 1. Run agent and save trajectory
python runtime/run_react_agent.py \
  "Find trending Steam games" \
  --save-trajectory

# 2. Evaluate with step-by-step analysis
export PYTHONPATH="/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:$PYTHONPATH"
python Orchestrator/mcpuniverse/evaluator/commonllmjudge.py \
  --trajectory trajectories/trajectory_20251113_HHMMSS.json \
  --step-by-step \
  --model openai/gpt-4o-mini \
  --save_json evaluation/step_analysis.json

# 3. Analyze results
python -c "
import json
with open('evaluation/step_analysis.json') as f:
    result = json.load(f)[0]
    steps = result['step_by_step_evaluation']['steps']
    print(f'Average Step Score: {result[\"step_by_step_evaluation\"][\"average_step_score\"]:.2f}')
    print(f'Holistic Score: {result[\"holistic_evaluation\"][\"score\"]:.2f}')
    print(f'\nStep-by-Step Breakdown:')
    for step in steps:
        eval_data = step['evaluation']
        print(f\"  Step {step['step_number']}: {eval_data['step_score']:.2f} (Progress: {eval_data['progress_toward_goal']}/10)\")
"
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

## Data Collection and Pipeline

### Complete Recrawl Workflow (Automated)

Run the complete pipeline from Smithery crawling to FAISS index building in one script:

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Run automated recrawl workflow
bash MCP_INFO_MGR/recrawl_workflow.sh
```

**What it does:**
1. **Crawl Smithery Registry** - Fetch all MCP servers from Smithery API
2. **Filter Remote Servers** - Keep only remote MCPs (exclude local stdio servers)
3. **Fetch Tool Descriptions** - Get tool schemas from remote servers
4. **Test Tools** (optional) - Verify tool reachability and functionality
5. **Filter Working Tools** (optional) - Remove broken tools based on tests
6. **Build FAISS Index** - Create semantic search index for tool discovery
7. **Summary Statistics** - Display counts and file sizes

**Interactive prompts:**
- You'll be asked if you want to test all tools (Step 4)
- Testing can take 30-60 minutes depending on number of servers
- You can skip testing and just rebuild indexes with existing data

**Output files:**
- `mcp_data/raw/smithery_servers.ndjson` - All servers from Smithery
- `mcp_data/raw/smithery_remote_servers.ndjson` - Remote servers only
- `mcp_data/indexed/tool_descriptions.ndjson` - Tool schemas
- `MCP_INFO_MGR/semantic_search/index.faiss` - FAISS index

**When to use:**
- After Smithery registry updates with new servers
- When you want to refresh your local tool database
- Before generating new queries for benchmarks

---

## Data Filtering

### Filter Remote MCP Servers

Filter Smithery servers to keep only remote MCPs (accessible via SSE endpoints):

```bash
cd /Users/xiziqiao/Documents/MCP-Research/MCP-R

# Basic filtering
python MCP_INFO_MGR/filter_remote_servers.py \
  --input mcp_data/raw/smithery_servers.ndjson \
  --output mcp_data/raw/smithery_remote_servers.ndjson

# With detailed statistics
python MCP_INFO_MGR/filter_remote_servers.py \
  --input mcp_data/raw/smithery_servers.ndjson \
  --output mcp_data/raw/smithery_remote_servers.ndjson \
  --stats
```

**What it does:**
- Filters servers where `"remote": true` (accessible via HTTPS/SSE)
- Excludes local stdio servers (`"remote": false`)
- Typical results: ~72% remote servers (654/903 as of 2025-11-11)

**Output:**
- `smithery_remote_servers.ndjson`: Filtered remote servers only
- Console statistics showing counts and percentages

**When to use:**
- After recrawling Smithery registry data
- Before processing tool descriptions (use filtered data as input)
- To focus on remotely accessible MCPs only

---

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

*Last updated: 2025-11-11 (Added goal-oriented agent)*
