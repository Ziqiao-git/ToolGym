# MCP-R: Meta MCP Server with Semantic Tool Discovery

A meta-level Model Context Protocol (MCP) server that provides semantic search and automatic discovery of MCP tools across 306+ registered servers.

## Overview

MCP-R acts as a registry and orchestrator for MCP servers, allowing LLMs to dynamically discover and execute tools from any registered MCP server through natural language queries.

## Architecture

### High-Level Flow

```
User Query (Natural Language)
    ↓
  [LLM]
    ↓
Calls: search_mcp_tools(query="search GitHub repositories")
    ↓
[Meta-MCP Server]
  ├─ FAISS semantic search
  ├─ Auto-connect to @smithery-ai/github
  ├─ Add tools dynamically to Meta-MCP's tool list
  └─ Send notification: "tools/list_changed"
    ↓
Returns search results + [LLM receives notification]
    ↓
  [LLM]
  ├─ Automatically re-queries tools/list
  └─ Now sees: ["search_mcp_tools", "github__search_repositories", "github__create_issue", ...]
    ↓
  [LLM] (sees new tools, picks one)
    ↓
Calls: github__search_repositories(query="machine learning", per_page=10)
    ↓
[Meta-MCP Server]
  └─ Proxies call to @smithery-ai/github (transparent to LLM)
    ↓
Returns: Actual results from GitHub API
    ↓
  [LLM]
```

## Core Components

### 1. Semantic Search Backend (FAISS + BGE-M3)

- **Vector Database**: FAISS (Facebook AI Similarity Search)
  - Index Type: `IndexFlatIP` (Inner Product for normalized vectors)
  - Dimensions: 1024
  - Scale: 306 servers, 1000+ tools
- **Embeddings**: BGE-M3 (BAAI General Embedding)
  - Model: `BAAI/bge-m3`
  - Languages: 100+ languages supported
  - Context Length: Up to 8192 tokens
  - Quality: State-of-the-art for multilingual retrieval
  - Special Features: Dense + sparse + multi-vector retrieval
- **Index**: `tool_descriptions.ndjson` → BGE-M3 embeddings → FAISS index
- **Search Strategy**: Semantic similarity search with metadata filtering
- **Query Enhancement**: Instruction-based prompting for better retrieval

### 2. Meta-MCP Server Tools

The Meta-MCP server uses **dynamic tool registration**. It starts with one base tool and dynamically adds tools as they are discovered.

#### Base Tool: `search_mcp_tools`

Search for MCP tools across 306+ servers using semantic search. When called, this tool:
1. Performs semantic search in FAISS index
2. Connects to discovered MCP servers
3. Dynamically adds their tools to Meta-MCP's tool list
4. Sends `notifications/tools/list_changed` to LLM
5. LLM automatically re-queries tool list and sees new tools

**Input Schema:**
```json
{
  "query": "string - Natural language description of what you need",
  "limit": "number - Max results to return (default: 5)",
  "filters": {
    "server": "string - (optional) Filter by server name",
    "tags": "array - (optional) Filter by tags"
  }
}
```

**Output:**
```json
{
  "results": [
    {
      "server": "string - Qualified server name",
      "tool": "string - Tool name",
      "description": "string - Tool description",
      "inputSchema": "object - JSON schema for tool inputs",
      "similarity_score": "number - Relevance score (0-1)"
    }
  ],
  "tools_added": ["server__tool_name", ...],
  "message": "Tools from discovered servers have been added. They are now available in your tool list."
}
```

#### Dynamically Added Tools

After `search_mcp_tools` is called, tools appear with naming format: `{server}__{tool_name}`

**Examples:**
- `github__search_repositories` - Search GitHub repositories
- `github__create_issue` - Create a GitHub issue
- `exa__web_search_exa` - Search the web using Exa AI
- `fetch__fetch_url` - Fetch and analyze web pages

**These tools have the SAME schemas as the original tools** from their respective servers.

**Example: After discovering GitHub tools**
```json
{
  "name": "github__search_repositories",
  "description": "Search for GitHub repositories using various filters",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"},
      "per_page": {"type": "number", "description": "Results per page"}
    },
    "required": ["query"]
  }
}
```

When LLM calls these tools, Meta-MCP transparently proxies the calls to the actual MCP servers.

### 3. Connection Pool Manager

- **Purpose**: Maintain persistent connections to MCP servers
- **Strategy**: Lazy initialization - connect only when needed
- **Reuse**: Keep connections alive for repeated tool calls
- **Cleanup**: Automatic cleanup of idle connections
- **Implementation**: Uses `MCPManager.build_client()` from existing codebase

### 4. Tool Indexing Pipeline

```
tool_descriptions.ndjson (Source Data)
    ↓
[BGE-M3 Embedding Generation]
  For each tool, create embedding from:
  - Tool name
  - Tool description
  - Server name
  - Input schema summary
  - Combined text with instruction prefix
    ↓
  Query instruction: "Represent this sentence for searching relevant passages: "
  Document instruction: (none for BGE-M3)
    ↓
[Normalize Embeddings]
  L2 normalization for cosine similarity
    ↓
[FAISS Index Creation]
  - Index Type: faiss.IndexFlatIP (Inner Product on normalized vectors = cosine similarity)
  - Dimensions: 1024
  - Metadata store: Separate JSON file
    ↓
Saved Artifacts:
  - tools.faiss (FAISS index)
  - tools_metadata.json (Server name, tool name, description, inputSchema)
  - embeddings_config.json (Model name, version, dimensions)
```

## Project Structure

```
MCP-R/
├── README.md                           # This file
├── MCP_INFO_MGR/                       # MCP information management
│   ├── tool_descriptions.ndjson        # All tools from 306 servers
│   ├── remote_server_configs.json     # Server connection configs
│   ├── reachability_ok_servers.ndjson  # Reachable servers list
│   │
│   ├── fetch_tool_descriptions.py      # Fetch tools from servers
│   ├── build_search_index.py           # Build FAISS index (TODO)
│   ├── mcp_registry_server.py          # Meta-MCP server (TODO)
│   │
│   └── semantic_search/                # Semantic search components
│       ├── faiss_backend.py            # FAISS indexing & search (TODO)
│       └── embeddings.py               # Embedding generation (TODO)
│
└── Orchestrator/                       # MCP orchestration framework
    └── mcpuniverse/
        └── mcp/
            ├── client.py               # MCP client implementation
            └── manager.py              # MCP manager with connection handling
```

## Data Flow

### 1. Indexing Phase (One-time / Periodic Update)

```bash
# Step 1: Fetch tool descriptions from all servers
python MCP_INFO_MGR/fetch_tool_descriptions.py

# Step 2: Build FAISS search index
python MCP_INFO_MGR/build_search_index.py \
    --input MCP_INFO_MGR/tool_descriptions.ndjson \
    --output MCP_INFO_MGR/semantic_search/
```

### 2. Runtime Phase (LLM Interaction)

1. **LLM discovers tools**: Calls `search_mcp_tools(query="search GitHub repositories")`
2. **Semantic search**: FAISS finds relevant tools by similarity
3. **Dynamic registration**:
   - Meta-MCP connects to discovered servers (@smithery-ai/github)
   - Adds tools to its tool list (github__search_repositories, github__create_issue, etc.)
   - Sends `notifications/tools/list_changed` to LLM
4. **LLM refreshes**: Automatically re-queries `tools/list`, sees new tools
5. **LLM uses tool directly**: Calls `github__search_repositories(query="machine learning", per_page=10)`
6. **Transparent proxy**: Meta-server forwards call to @smithery-ai/github
7. **LLM receives result**: Gets data from GitHub API through Meta-MCP

## Key Features

### Automatic Discovery
- LLMs can discover tools without hardcoded knowledge
- Natural language queries match to technical tool descriptions
- Reduces need for manual tool selection

### Connection Pooling
- Reuses connections across multiple tool calls
- Reduces latency for repeated operations
- Efficient resource management

### Scalability
- Currently indexes 306 servers with 1000+ tools
- FAISS can scale to billions of vectors
- Fast search even with large registries

### Extensibility
- Easy to add new MCP servers to the registry
- Support for metadata filtering (tags, categories)
- Hybrid search (semantic + keyword) possible

## Technology Stack

- **Vector Search**: FAISS (Meta AI)
  - Index Type: `IndexFlatIP` for cosine similarity
  - Optimized for fast similarity search
- **Embeddings**: BGE-M3 (`BAAI/bge-m3`)
  - Framework: Sentence Transformers
  - Dimensions: 1024
  - Multilingual: 100+ languages
  - Best-in-class for multilingual retrieval tasks
- **MCP Protocol**: `mcp==1.9.4`
- **Connection Management**: Custom MCPManager with pooling
- **Data Format**: NDJSON (Newline-delimited JSON)
- **Language**: Python 3.11+
- **Dependencies**:
  - `faiss-cpu` or `faiss-gpu`: Vector similarity search
  - `sentence-transformers`: BGE-M3 model loading and inference
  - `torch`: Deep learning backend for embeddings

## Use Cases

### 1. Dynamic Tool Selection
```
User: "I need to analyze stock prices for Tesla"
  ↓
LLM: search_mcp_tools("stock price analysis")
  → Tools from yfinance server appear in tool list
  ↓
LLM: yfinance__get_stock_data(ticker="TSLA")
  → Returns Tesla stock data
```

### 2. Multi-Tool Workflows
```
User: "Search GitHub for React repos and create an issue"
  ↓
LLM: search_mcp_tools("search github repositories")
  → GitHub tools appear: github__search_repositories, github__create_issue, etc.
  ↓
LLM: github__search_repositories(query="react", per_page=10)
  → Returns list of React repositories
  ↓
LLM: github__create_issue(owner="facebook", repo="react", title="Feature request", ...)
  → Creates issue on GitHub
```

### 3. Capability Discovery
```
User: "What can you help me with regarding weather?"
  ↓
LLM: search_mcp_tools("weather forecasting")
  → Returns available weather-related tools across all servers
  → Weather server tools appear in LLM's tool list
  ↓
LLM can now use: weather__get_forecast, weather__current_conditions, etc.
```

### 4. Cross-Server Integration
```
User: "Find Python repos on GitHub and search for documentation"
  ↓
LLM: search_mcp_tools("search github")
  → Adds github__search_repositories
LLM: github__search_repositories(query="python", language="python")
  → Gets repo list
  ↓
LLM: search_mcp_tools("search documentation")
  → Adds docfork__search_docs
LLM: docfork__search_docs(query="python best practices")
  → Gets documentation
  ↓
Result: LLM combines data from 2 different MCP servers seamlessly
```

## Implementation Status

- [x] MCP client implementation
- [x] MCP connection manager
- [x] Tool description fetching from 306 servers
- [x] Fix asyncio task isolation issues
- [x] BGE-M3 embedding generation module
- [x] FAISS semantic search backend
- [x] Build search index script
- [ ] Meta-MCP server with dynamic tool registration
- [ ] Connection pool with auto-connect on discovery
- [ ] MCP notification support (tools/list_changed)
- [ ] End-to-end testing with LLM client

## Configuration

### Environment Variables

```bash
# Required for Smithery servers
SMITHERY_API_KEY=your_api_key_here
```

### Server Configuration

Server configurations are stored in `MCP_INFO_MGR/remote_server_configs.json`:

```json
{
  "@smithery-ai/github": {
    "streamable_http": {
      "url": "https://server.smithery.ai/@smithery-ai/github/mcp",
      "headers": {
        "Authorization": "Bearer {{SMITHERY_API_KEY}}"
      }
    }
  }
}
```

## Future Enhancements

- [ ] Hybrid search (semantic + keyword BM25)
- [ ] Tool usage analytics and popularity ranking
- [ ] Automatic tool categorization/tagging
- [ ] Multi-language support for tool descriptions
- [ ] WebUI for browsing available tools
- [ ] Rate limiting and quota management
- [ ] Tool versioning and compatibility tracking
- [ ] Caching layer for frequently used tools

## Contributing

This is a research project exploring meta-level MCP orchestration and semantic tool discovery.

## License

[To be determined]