# Dynamic ReAct Agent Runtime

This runtime provides a ReAct agent with dynamic MCP server loading capabilities, powered by semantic tool search.

## Features

- **Semantic Tool Discovery**: Uses Meta-MCP server to search across 4,572 tools from 301 MCP servers
- **Dynamic Server Loading**: Automatically loads MCP servers on-demand when the agent discovers relevant tools
- **Trajectory Logging**: Saves detailed execution logs to JSON files for analysis
- **Proper Cleanup**: Handles asyncio cleanup to minimize errors

## Usage

### Basic Usage

```bash
python runtime/run_react_agent.py "Your query here"
```

### With Trajectory Logging

```bash
python runtime/run_react_agent.py "Search for AI news" --save-trajectory
```

### Custom Model and Iterations

```bash
python runtime/run_react_agent.py "Your query" \
  --model anthropic/claude-3.5-sonnet \
  --max-iterations 10 \
  --save-trajectory
```

## Command Line Arguments

- `query` (required): Your question or task for the agent
- `--model`: OpenRouter model name (default: `anthropic/claude-3.5-sonnet`)
- `--max-iterations`: Maximum reasoning iterations (default: 5)
- `--save-trajectory`: Save execution trajectory to JSON file

## Example Queries

### Web Search
```bash
python runtime/run_react_agent.py "Search the web for latest news about artificial intelligence"
```

### GitHub Search
```bash
python runtime/run_react_agent.py "Find GitHub repositories about machine learning"
```

### Simple Questions (no tools needed)
```bash
python runtime/run_react_agent.py "What is 2+2?"
```

## How It Works

1. **Agent Initialization**: Starts with only the Meta-MCP server loaded
2. **Tool Discovery**: When given a query, the agent uses `search_tools` to find relevant MCP tools
3. **Dynamic Loading**: If the agent wants to use a tool from an unloaded server, it automatically:
   - Fetches the server configuration
   - Replaces template variables (e.g., `{{SMITHERY_API_KEY}}`)
   - Connects to the remote server
   - Loads available tools
4. **Tool Execution**: Executes the discovered tool and returns results
5. **Cleanup**: Properly closes all MCP connections

## Trajectory Files

When using `--save-trajectory`, execution logs are saved to `trajectories/trajectory_TIMESTAMP.json`:

```json
{
  "metadata": {
    "timestamp": "2025-10-25T20:26:01.583693",
    "query": "Search the web for latest news about artificial intelligence",
    "model": "anthropic/claude-3.5-sonnet",
    "max_iterations": 5
  },
  "execution": {
    "final_response": "...",
    "tool_calls": [
      {
        "type": "tool_call",
        "server": "meta-mcp",
        "tool": "search_tools",
        "arguments": {...},
        "dynamically_loaded": false,
        "duration_seconds": 3.16
      },
      {
        "type": "tool_call",
        "server": "@Ymuberra/geo-news-mcp",
        "tool": "search_news",
        "arguments": {...},
        "dynamically_loaded": true,
        "duration_seconds": 8.244
      }
    ],
    "loaded_servers": ["@Ymuberra/geo-news-mcp"],
    "total_tool_calls": 2,
    "dynamically_loaded_count": 1
  },
  "servers": {
    "initially_loaded": ["meta-mcp"],
    "dynamically_loaded": ["@Ymuberra/geo-news-mcp"],
    "total_servers_used": 2
  }
}
```

## Architecture

```
User Query
    ↓
ReAct Agent (initialized with meta-mcp only)
    ↓
Calls: meta-mcp/search_tools("search news")
    ↓
Returns: [@Ymuberra/geo-news-mcp/search_news, ...]
    ↓
Dynamic Loader detects server not loaded
    ↓
Loads @Ymuberra/geo-news-mcp server
    ↓
Calls: @Ymuberra/geo-news-mcp/search_news("AI")
    ↓
Returns: Real news articles
    ↓
Agent synthesizes final response
    ↓
Cleanup all connections
```

## Environment Variables

Required in `Orchestrator/.env`:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `SMITHERY_API_KEY`: Your Smithery API key for remote MCP servers

## Files

- `run_react_agent.py`: Main runtime script
- `../Orchestrator/mcpuniverse/agent/dynamic_react.py`: Dynamic agent implementation
- `../tool_retrieval_index/server.py`: Meta-MCP semantic search server
- `../trajectories/*.json`: Saved execution trajectories

## Known Issues

- Minor `resource_tracker` warnings on shutdown (does not affect functionality)
- Some MCP JSON-RPC validation errors during connection (handled gracefully)
- Invalid Smithery API keys will cause 401 errors for remote servers
