# MCP-R: Evaluation Pipeline for MCP Tool-Using Agents

An end-to-end evaluation framework for testing AI agents' ability to dynamically discover and use Model Context Protocol (MCP) tools. Generates realistic benchmark tasks, executes them with a ReAct agent using semantic tool discovery across 4,572 tools from 301 servers, and evaluates performance using LLM-as-judge with a 5-dimension rubric.

## Overview

**MCP-R's Purpose**: Benchmark and evaluate how well AI agents can:
1. **Discover** relevant MCP tools through semantic search
2. **Load** MCP servers dynamically on-demand
3. **Execute** tools correctly to complete tasks
4. **Ground** responses in actual tool outputs (no hallucination)

The system generates diverse benchmark tasks, runs a ReAct agent with dynamic server loading, and evaluates results using an LLM judge that scores on 5 dimensions: task fulfillment, grounding, tool choice, tool execution, and requirement satisfaction.

## Quick Start

### Complete Evaluation Pipeline

```bash
# Step 1: Generate benchmark tasks (5 queries per MCP server)
python task_creation_engine/query_generate.py \
  --in MCP_INFO_MGR/mcp_data/usable/useable_remote_server_metadata.ndjson \
  --out evaluation/benchmark_tasks.json

# Step 2: Run agent on all tasks
python evaluation/run_benchmark.py \
  --tasks evaluation/benchmark_tasks.json \
  --output evaluation/results/

# Step 3: Evaluate with LLM judge
python evaluation/evaluate_results.py \
  --trajectories evaluation/results/ \
  --output evaluation/evaluation_report.json
```

### Manual Agent Testing

```bash
# Basic usage
python runtime/run_react_agent.py "Search the web for latest AI news"

# With trajectory logging for evaluation
python runtime/run_react_agent.py "Find GitHub repos about ML" --save-trajectory

# Custom model and iterations
python runtime/run_react_agent.py "Your query" \
  --model anthropic/claude-3.5-sonnet \
  --max-iterations 10 \
  --save-trajectory
```

## Architecture

### Complete Flow

```
User Query: "Search the web for latest AI news"
    ↓
┌─────────────────────────────────────────────────────┐
│ ReAct Agent (initialized with meta-mcp only)        │
└─────────────────────────────────────────────────────┘
    ↓
    Reasoning: "I need to search for tools that can search news"
    ↓
┌─────────────────────────────────────────────────────┐
│ Tool Call: meta-mcp/search_tools                    │
│ Args: {query: "search news articles", top_k: 5}    │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Meta-MCP Server (FAISS + BGE-M3)                   │
│ Searches 4,572 tools semantically                   │
└─────────────────────────────────────────────────────┘
    ↓
    Returns: [@Ymuberra/geo-news-mcp/search_news (score: 0.604), ...]
    ↓
┌─────────────────────────────────────────────────────┐
│ Dynamic Server Loader (intercepts tool call)        │
│ Detects: @Ymuberra/geo-news-mcp not loaded         │
└─────────────────────────────────────────────────────┘
    ↓
    1. Fetch server config from remote_server_configs.json
    2. Replace {{SMITHERY_API_KEY}} with actual key
    3. Connect to https://server.smithery.ai/@Ymuberra/geo-news-mcp
    4. Load 2 tools: search_news, get_headlines
    ↓
┌─────────────────────────────────────────────────────┐
│ Tool Call: @Ymuberra/geo-news-mcp/search_news      │
│ Args: {query: "artificial intelligence"}           │
└─────────────────────────────────────────────────────┘
    ↓
    Returns: Real news articles about AI
    ↓
Agent synthesizes final response with citations
```

## Core Components

### 1. Dynamic ReAct Agent

**Location:** `Orchestrator/mcpuniverse/agent/dynamic_react.py`

Extends the base ReAct agent with on-demand server loading:
- **Intercepts tool calls** before execution
- **Detects unloaded servers** from tool names
- **Dynamically loads** server configs and connects
- **Tracks trajectory** for evaluation
- **Proper cleanup** to avoid asyncio errors

**Key Methods:**
```python
async def _load_server_on_demand(server_name: str) -> bool:
    """Load MCP server config, connect, and register tools."""

async def call_tool(llm_response) -> CallToolResult:
    """Intercept tool calls and load servers if needed."""

async def cleanup():
    """Properly close all MCP client connections."""
```

### 2. Meta-MCP Server

**Location:** `tool_retrieval_index/server.py`

Provides semantic search across all indexed tools:

**Tool:** `search_tools`
```json
{
  "query": "search GitHub repositories",
  "top_k": 5,
  "min_score": 0.3
}
```

**Returns:**
```json
{
  "results": [
    {
      "server": "@smithery-ai/github",
      "tool": "search_repositories",
      "description": "Search GitHub repositories...",
      "similarity_score": 0.845,
      "parameters": ["query", "language", "per_page"]
    }
  ]
}
```

### 3. Semantic Search Backend

**Location:** `MCP_INFO_MGR/semantic_search/`

- **Vector Database**: FAISS (IndexFlatIP for cosine similarity)
- **Embeddings**: BGE-M3 (`BAAI/bge-m3`)
  - Dimensions: 1024
  - Multilingual: 100+ languages
  - Context length: 8192 tokens
- **Index Size**: 4,572 tools from 301 servers
- **Search**: Fast semantic similarity with metadata

### 4. Data Organization

**Location:** `MCP_INFO_MGR/mcp_data/`

```
mcp_data/
├── raw/                    # Original Smithery data
│   ├── smithery_servers.ndjson
│   └── smithery_metadata.ndjson
│
├── usable/                 # 306 reachable servers
│   ├── reachability_ok_servers.ndjson
│   ├── remote_server_configs.json
│   └── useable_remote_server_metadata.ndjson
│
├── unusable/               # Failed reachability tests
│   └── reachability_results_*.ndjson
│
└── indexed/                # 301 servers with 4,572 tools
    └── tool_descriptions.ndjson
```

## Project Structure

```
MCP-R/
├── README.md                           # This file
│
├── runtime/                            # Agent runtime
│   ├── run_react_agent.py             # Main CLI for running agent
│   └── README.md                       # Runtime documentation
│
├── tool_retrieval_index/                    # Semantic search server
│   └── server.py                       # MCP server with search_tools
│
├── MCP_INFO_MGR/                       # MCP data management
│   ├── mcp_data/                       # Organized MCP data
│   │   ├── raw/                       # Smithery raw data
│   │   ├── usable/                    # 306 reachable servers
│   │   ├── unusable/                  # Failed servers
│   │   └── indexed/                   # 4,572 indexed tools
│   │
│   ├── semantic_search/                # Search infrastructure
│   │   ├── embeddings.py              # BGE-M3 embedding generation
│   │   ├── faiss_backend.py           # FAISS indexing & search
│   │   ├── build_search_index.py      # Build FAISS index
│   │   ├── test_semantic_search.py    # Test search quality
│   │   ├── index.faiss                # FAISS index (gitignored)
│   │   ├── metadata.json              # Tool metadata
│   │   └── model_info.json            # BGE-M3 model config
│   │
│   ├── fetch_tool_descriptions.py      # Fetch tools from servers
│   └── check_remote_reachability.py    # Test server connectivity
│
├── Orchestrator/                       # MCP orchestration framework
│   └── mcpuniverse/
│       ├── agent/
│       │   ├── react.py               # Base ReAct agent
│       │   └── dynamic_react.py       # Dynamic loading wrapper
│       ├── mcp/
│       │   ├── client.py              # MCP client
│       │   └── manager.py             # Connection manager
│       ├── evaluator/                  # Evaluation framework
│       │   ├── commonllmjudge.py      # LLM-as-judge evaluator
│       │   └── evaluator.py           # Base evaluator
│       └── benchmark/                  # Benchmark runner
│           ├── runner.py              # Task execution
│           └── task.py                # Task definitions
│
├── task_creation_engine/                       # Query generation
│   ├── query_generate.py              # Generate benchmark queries
│   └── .env                           # API keys (gitignored)
│
├── trajectories/                       # Agent execution logs
│   └── trajectory_*.json              # Saved trajectories (gitignored)
│
└── evaluation/                         # Evaluation pipeline (planned)
    ├── generate_benchmark_tasks.py    # Create evaluation tasks
    ├── run_benchmark.py               # Execute tasks with agent
    └── evaluate_results.py            # Judge trajectories with LLM
```

## Implementation Status

### ✅ Completed

#### Core Infrastructure
- [x] MCP client implementation with stdio and HTTP support
- [x] MCP connection manager with proper cleanup
- [x] Tool description fetching from 306 servers
- [x] BGE-M3 embedding generation module
- [x] FAISS semantic search backend (4,572 tools indexed)
- [x] Meta-MCP server with search_tools function

#### Dynamic Agent System
- [x] Dynamic ReAct agent with on-demand server loading
- [x] Tool call interception and server detection
- [x] Template variable replacement ({{SMITHERY_API_KEY}})
- [x] Trajectory logging for evaluation
- [x] Asyncio cleanup fixes

#### Data & Organization
- [x] Data organization (raw/usable/unusable/indexed)
- [x] Build search index script
- [x] Test semantic search script
- [x] Comprehensive documentation

#### Evaluation Pipeline (Step 1/3)
- [x] Query generation script (`task_creation_engine/query_generate.py`)
  - Generates 5 realistic queries per MCP server
  - Uses GPT-5 with structured output
  - Validates concrete, operational questions

### 🚧 In Progress

#### Evaluation Pipeline (Steps 2-3)
- [ ] Benchmark runner (`evaluation/run_benchmark.py`)
  - Run agent on all generated tasks
  - Save trajectories with metadata
  - Handle errors and timeouts
- [ ] Evaluation script (`evaluation/evaluate_results.py`)
  - Load trajectories and parse history
  - Call LLM judge with 5-dimension rubric
  - Generate comprehensive report

#### Improvements
- [ ] Handle invalid Smithery API keys gracefully
- [ ] Add retry logic for failed server connections
- [ ] WebUI for browsing tools and trajectories

### 📋 Planned

- [ ] Hybrid search (semantic + BM25 keyword)
- [ ] Tool usage analytics
- [ ] Automatic categorization
- [ ] Rate limiting and quotas
- [ ] Caching layer for frequent tools
- [ ] Multi-agent collaboration

## Evaluation Pipeline (Design)

### Overview

```
Step 1: Generate Tasks          Step 2: Run Agent              Step 3: Evaluate
─────────────────────           ──────────────────             ────────────────
task_creation_engine/                   runtime/                       Orchestrator/
query_generate.py               run_react_agent.py             evaluator/
                                                               commonllmjudge.py
        ↓                               ↓                              ↓
benchmark_tasks.json            trajectories/*.json            evaluation_report.json
```

### Step 1: Generate Benchmark Tasks

Uses LLM to create diverse evaluation tasks covering:
- Web search
- GitHub operations
- Database queries
- API interactions
- File operations
- Data analysis

**Output:** `evaluation/benchmark_tasks.json`
```json
{
  "tasks": [
    {
      "task_id": "task_001",
      "category": "web_search",
      "question": "Search for latest AI regulation news",
      "expected_tools": ["search_news", "web_search"],
      "correct_answer_type": "news_articles",
      "evaluation_criteria": "Must find recent articles with sources"
    }
  ]
}
```

### Step 2: Run Agent on Tasks

For each task:
1. Load task from `benchmark_tasks.json`
2. Execute: `python runtime/run_react_agent.py "{question}" --save-trajectory`
3. Collect trajectory with:
   - Tool calls made
   - Servers loaded dynamically
   - Execution times
   - Final response

**Output:** `trajectories/trajectory_{timestamp}.json`
```json
{
  "metadata": {
    "query": "Search for AI news",
    "model": "anthropic/claude-3.5-sonnet"
  },
  "execution": {
    "tool_calls": [
      {
        "server": "meta-mcp",
        "tool": "search_tools",
        "dynamically_loaded": false,
        "duration_seconds": 3.16
      },
      {
        "server": "@Ymuberra/geo-news-mcp",
        "tool": "search_news",
        "dynamically_loaded": true,
        "duration_seconds": 8.24
      }
    ],
    "final_response": "...",
    "loaded_servers": ["@Ymuberra/geo-news-mcp"]
  }
}
```

### Step 3: Evaluate with LLM Judge

Uses `commonllmjudge.py` to score on 5 dimensions (0-10 each):
1. **Task Fulfillment** - Answers the question correctly
2. **Grounding** - Claims supported by tool outputs
3. **Tool Choice** - Selected appropriate tools
4. **Tool Execution** - Used tools effectively
5. **Requirement Satisfaction** - Met all constraints

**Output:** `evaluation/evaluation_report.json`
```json
{
  "overall_metrics": {
    "total_tasks": 10,
    "pass_rate": 0.85,
    "avg_score": 0.87,
    "avg_tools_per_task": 2.3,
    "avg_dynamic_loads": 1.2
  },
  "task_results": [
    {
      "task_id": "task_001",
      "overall_score": 0.92,
      "binary": "success",
      "subscores": {
        "task_fulfillment": 9,
        "grounding": 10,
        "tool_choice": 9,
        "tool_execution": 9,
        "requirement_satisfaction": 9
      },
      "explanation": "Agent correctly used search_news tool..."
    }
  ]
}
```

## Key Features

### 1. Dynamic Server Loading

- **Start lightweight**: Initialize with only Meta-MCP server
- **Load on-demand**: Connect to servers only when needed
- **Automatic discovery**: Agent finds tools through semantic search
- **Efficient**: No wasted connections to unused servers

### 2. Semantic Tool Discovery

- **Natural language**: Query "search GitHub" finds GitHub tools
- **Cross-lingual**: BGE-M3 supports 100+ languages
- **Ranked results**: Similarity scores help choose best tool
- **Fast search**: FAISS enables sub-second queries

### 3. Trajectory Logging

- **Complete history**: Every tool call, load, and response
- **Timing data**: Measure performance bottlenecks
- **Evaluation ready**: Format compatible with LLM judge
- **Debugging**: Understand agent reasoning process

### 4. Proper Resource Management

- **Async cleanup**: No more RuntimeError exceptions
- **Connection pooling**: Reuse loaded servers
- **Memory efficient**: Close unused connections
- **Production ready**: Handles errors gracefully

## Use Cases

### 1. Multi-Step Research Task

```bash
python runtime/run_react_agent.py \
  "Find popular Python ML libraries on GitHub and search for their documentation"
```

**What happens:**
1. Agent searches tools: "find GitHub repositories"
2. Loads `@smithery-ai/github` server dynamically
3. Calls `search_repositories(query="python machine learning")`
4. Agent searches tools: "search documentation"
5. Loads documentation server dynamically
6. Combines results from multiple sources

### 2. News Aggregation

```bash
python runtime/run_react_agent.py \
  "Search for latest news about artificial intelligence" \
  --save-trajectory
```

**What happens:**
1. Agent searches: "search news articles"
2. Finds and loads `@Ymuberra/geo-news-mcp`
3. Executes `search_news(query="artificial intelligence")`
4. Returns news with sources and dates
5. Saves trajectory for evaluation

### 3. Capability Discovery

```bash
python runtime/run_react_agent.py \
  "What tools are available for working with databases?"
```

**What happens:**
1. Agent calls `search_tools(query="database operations")`
2. Returns list of database-related tools across servers
3. Agent can then choose and use appropriate tools

## Technology Stack

### Core Technologies

- **Agent Framework**: ReAct (Reasoning + Acting)
- **LLM**: OpenRouter (Claude 3.5 Sonnet default)
- **Vector Search**: FAISS (Meta AI)
- **Embeddings**: BGE-M3 (BAAI, 1024 dims)
- **MCP Protocol**: `mcp==1.9.4` (stdio + HTTP)
- **Language**: Python 3.11+

### Key Dependencies

```
faiss-cpu                # Vector similarity search
sentence-transformers    # BGE-M3 embeddings
torch                    # Deep learning backend
mcp                      # Model Context Protocol
openai                   # OpenRouter API client
python-dotenv            # Environment management
```

## Configuration

### Environment Variables

Create `.env` files in appropriate directories:

**`Orchestrator/.env`:**
```bash
OPENROUTER_API_KEY=sk-or-v1-...
SMITHERY_API_KEY=your-smithery-key
```

**`task_creation_engine/.env`:**
```bash
OPENROUTER_API_KEY=sk-or-v1-...
SMITHERY_API_KEY=your-smithery-key
```

### Server Configurations

Server configs in `MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json` support template variables:

```json
{
  "@smithery-ai/github": {
    "streamable_http": {
      "url": "https://server.smithery.ai/@smithery-ai/github/mcp?api_key={{SMITHERY_API_KEY}}",
      "headers": {}
    },
    "env": {}
  }
}
```

Templates like `{{SMITHERY_API_KEY}}` are replaced with environment variables during dynamic loading.

## Testing

### Test Semantic Search

```bash
python MCP_INFO_MGR/semantic_search/test_semantic_search.py
```

Tests queries like:
- "search github repositories"
- "fetch weather data"
- "analyze stock prices"

### Test Dynamic Agent

```bash
# Simple query (no tools)
python runtime/run_react_agent.py "What is 2+2?"

# Tool discovery and use
python runtime/run_react_agent.py "Search for AI news" --save-trajectory

# Complex multi-tool task
python runtime/run_react_agent.py \
  "Find ML repos on GitHub" \
  --max-iterations 10 \
  --save-trajectory
```

## Known Issues

### Minor Issues (Non-blocking)

1. **Resource tracker warning**: `resource_tracker: There appear to be 1 leaked semaphore objects`
   - Harmless warning during shutdown
   - Does not affect functionality
   - Python multiprocessing cleanup artifact

2. **JSON-RPC validation errors**: Some servers return non-standard errors during connection
   - Gracefully handled by retry logic
   - Does not affect successful connections

### Limitations

1. **Invalid API keys**: Some Smithery servers return 401 errors
   - Need to refresh Smithery API key
   - Or test with local MCP servers

2. **Large FAISS index**: 4,572-tool index not in git
   - Need to rebuild with `build_search_index.py`
   - Gitignored due to size (~20MB)

## Future Enhancements

### Short Term

- [ ] Automated evaluation pipeline
- [ ] Benchmark task suite (10-50 tasks)
- [ ] Performance metrics dashboard
- [ ] Better error messages for failed servers

### Medium Term

- [ ] Hybrid semantic + keyword search
- [ ] Tool popularity ranking
- [ ] Usage analytics
- [ ] Multi-agent collaboration
- [ ] Streaming responses

### Long Term

- [ ] WebUI for tool discovery
- [ ] Visual trajectory viewer
- [ ] Tool marketplace integration
- [ ] Version compatibility tracking
- [ ] Automatic tool caching
- [ ] Federation across multiple registries

## Contributing

This is a research project exploring:
- Meta-level MCP orchestration
- Semantic tool discovery
- Dynamic agent architectures
- LLM-based evaluation

Contributions welcome! Areas of interest:
- Evaluation benchmark tasks
- New MCP server integrations
- Performance optimizations
- Documentation improvements

## Citation

If you use MCP-R in your research, please cite:

```
@software{mcp_r_2025,
  title = {MCP-R: Dynamic ReAct Agent with Semantic MCP Tool Discovery},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/Ziqiao-git/MCP-R}
}
```

## License

[To be determined]

---

**Status**: Active Development
**Last Updated**: October 2025
**Tools Indexed**: 4,572 from 301 servers
**Agent**: ReAct with dynamic loading
**Evaluation**: LLM-as-judge (5-dimension rubric)