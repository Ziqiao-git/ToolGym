# Using ReAct Agent with Meta-MCP Server

This demonstrates how the existing ReAct agent from Orchestrator uses your Meta-MCP server to dynamically discover and use MCP tools.

## How It Works

```
User Query → ReAct Agent → Meta-MCP (search_tools) → Find Relevant Tools → Use Those Tools → Answer
```

### Step-by-Step Flow:

1. **User asks a question**: "Find GitHub repositories about machine learning"

2. **ReAct agent reasons**: "I need to search for GitHub-related tools"

3. **Agent calls**: `meta-mcp/search_tools` with query: "search GitHub repositories"

4. **Meta-MCP returns**: Top tools like `@smithery-ai/github/search_repositories`

5. **Agent uses the tool**: Calls `@smithery-ai/github/search_repositories` with arguments

6. **Agent gets results**: List of ML repositories

7. **Agent answers**: "Here are the top ML repositories on GitHub: ..."

## Usage

### Basic Example:

```bash
python runtime/run_react_agent.py "Find popular Python machine learning repositories on GitHub"
```

### What the Agent Will Do:

1. Search for tools: `search_tools("GitHub repository search")`
2. Find: `@smithery-ai/github/search_repositories`
3. Use it: Call with `{"query": "python machine learning", "sort": "stars"}`
4. Return results

### More Examples:

```bash
# Weather query
python runtime/run_react_agent.py "What's the weather in San Francisco?"

# Stock analysis
python runtime/run_react_agent.py "Show me Apple stock performance this week"

# Web scraping
python runtime/run_react_agent.py "Fetch the content from news.ycombinator.com"
```

## Configuration

The ReAct agent is configured with:

- **MCP Manager**: Manages all MCP servers including meta-mcp
- **Meta-MCP Server**: Provides `search_tools` for discovering 4,572 tools
- **LLM**: Claude 3.5 Sonnet (default) for reasoning
- **Max Iterations**: 5 steps to solve the task

## Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       v
┌─────────────────────────────────┐
│     ReAct Agent                 │
│  (Orchestrator/agent/react.py)  │
│                                 │
│  - Reasons about the task       │
│  - Decides which tools to use   │
│  - Executes actions             │
└────────┬────────────────────────┘
         │
         ├─── Step 1: Search Tools
         │    └──> meta-mcp/search_tools("GitHub search")
         │         Returns: [@smithery-ai/github/search_repositories]
         │
         ├─── Step 2: Use Tool
         │    └──> @smithery-ai/github/search_repositories(...)
         │         Returns: [repo1, repo2, repo3, ...]
         │
         └─── Step 3: Answer
              └──> "Here are the results: ..."
```

## Key Features

1. **Dynamic Tool Discovery**: Agent doesn't need to know tools in advance
2. **Semantic Search**: Finds tools by understanding intent, not keywords
3. **Intelligent Selection**: Agent chooses the best tool for the task
4. **Multi-step Reasoning**: Can chain multiple tools together
5. **Error Handling**: Gracefully handles tool failures

## Requirements

Make sure you have:
- ✅ Semantic search index built (`MCP_INFO_MGR/semantic_search/`)
- ✅ Server configs available (`MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json`)
- ✅ Environment variables set (ANTHROPIC_API_KEY, SMITHERY_API_KEY, etc.)
- ✅ Dependencies installed:
  ```bash
  pip install -r MCP_INFO_MGR/requirements_semantic_search.txt
  pip install mcp
  ```

## Advanced Usage

### Custom LLM:

```bash
python runtime/run_react_agent.py \
    "Your query" \
    --llm "openai:gpt-4"
```

### More Iterations:

```bash
python runtime/run_react_agent.py \
    "Complex multi-step task" \
    --max-iterations 10
```

## Example Session

```bash
$ python runtime/run_react_agent.py "Find trending JavaScript repos"

============================================================
ReAct Agent with Meta-MCP Server
============================================================
Query: Find trending JavaScript repos
LLM: anthropic:claude-3-5-sonnet-20241022
Max iterations: 5
============================================================

Initializing MCP Manager...
✓ Added Meta-MCP server

Initializing LLM: anthropic:claude-3-5-sonnet-20241022...
✓ LLM ready

Creating ReAct agent...
✓ ReAct agent ready

============================================================
Running ReAct Agent...
============================================================

Step 1:
Thought: I need to find tools for searching GitHub repositories
Action: Using tool `search_tools` in server `meta-mcp`
Result: Found 5 tools:
  1. @smithery-ai/github/search_repositories (score: 0.68)
  2. @smithery-ai/github/trending_repositories (score: 0.65)
  ...

Step 2:
Thought: I'll use the trending_repositories tool
Action: Using tool `trending_repositories` in server `@smithery-ai/github`
Result: [list of trending JS repos]

Step 3:
Thought: I have the results, I can answer now
Answer: Here are the trending JavaScript repositories:
  1. facebook/react - A declarative, efficient...
  2. vercel/next.js - The React Framework...
  ...

============================================================
Agent Response:
============================================================
Here are the trending JavaScript repositories:
  1. facebook/react - A declarative, efficient...
  2. vercel/next.js - The React Framework...
============================================================
```

## Next Steps

1. Try running the agent with your own queries
2. Monitor how it discovers and uses tools
3. Extend the instruction to guide the agent's behavior
4. Add more servers to expand tool availability
