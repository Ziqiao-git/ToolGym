# Meta-MCP Server

A simple MCP server that provides semantic search across 4,572 MCP tools from 301 servers.

## Features

- **Single Tool**: `search_tools` - Semantic search for MCP tools
- **4,572 Tools Indexed**: From 301 reachable MCP servers
- **BGE-M3 Embeddings**: High-quality semantic understanding
- **FAISS Backend**: Fast similarity search

## Installation

```bash
# Install dependencies (if not already installed)
pip install -r MCP_INFO_MGR/requirements_semantic_search.txt

# Also need MCP SDK
pip install mcp
```

## Usage

### Run as MCP Server

```bash
python meta_mcp_server/server.py
```

### Add to Claude Desktop

Add this to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "meta-mcp": {
      "command": "python",
      "args": [
        "/Users/xiziqiao/Documents/MCP-Research/MCP-R/meta_mcp_server/server.py"
      ]
    }
  }
}
```

## Tool: `search_tools`

Search for MCP tools using natural language.

**Parameters:**
- `query` (string, required): What you want to do (e.g., "search GitHub repos")
- `top_k` (integer, optional): Number of results (default: 5, max: 20)
- `min_score` (number, optional): Minimum similarity threshold (default: 0.3)

**Example Queries:**
- "search GitHub repositories"
- "fetch weather data"
- "analyze stock prices"
- "send email notifications"
- "create database queries"

**Returns:**
- Server name
- Tool name
- Description
- Similarity score
- Input parameters

## Architecture

```
User Query
    ↓
BGE-M3 Embedder (1024-dim vector)
    ↓
FAISS Index Search (4,572 tools)
    ↓
Top-K Results (filtered by score)
    ↓
Formatted Response
```

## Data Source

- Index: `MCP_INFO_MGR/semantic_search/`
- Tools: `MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson`
- 301 servers, 4,572 tools total
