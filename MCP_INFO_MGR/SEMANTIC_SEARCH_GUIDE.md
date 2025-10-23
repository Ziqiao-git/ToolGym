## Semantic Search Implementation Guide

This guide explains how to use the BGE-M3 + FAISS semantic search system for MCP tool discovery.

## Overview

The semantic search system allows LLMs to discover MCP tools using natural language queries. It uses:
- **BGE-M3**: State-of-the-art multilingual embeddings (1024 dimensions, 100+ languages)
- **FAISS**: Fast similarity search on normalized vectors
- **Cosine Similarity**: Measures semantic relevance between queries and tools

## Installation

### 1. Install Dependencies

```bash
pip install -r MCP_INFO_MGR/requirements_semantic_search.txt
```

This installs:
- `faiss-cpu` - Vector similarity search (or `faiss-gpu` for GPU support)
- `sentence-transformers` - BGE-M3 model loading
- `torch` - Deep learning backend
- `numpy` - Numerical computing

### 2. Verify Installation

```bash
python MCP_INFO_MGR/test_semantic_search.py
```

This will test the system with sample tools.

## Building the Index

### Step 1: Fetch Tool Descriptions

If you haven't already, fetch tool descriptions from all MCP servers:

```bash
python MCP_INFO_MGR/fetch_tool_descriptions.py
```

This creates `MCP_INFO_MGR/tool_descriptions.ndjson` with tools from 306+ servers.

### Step 2: Build FAISS Index

```bash
python MCP_INFO_MGR/build_search_index.py \
    --input MCP_INFO_MGR/tool_descriptions.ndjson \
    --output MCP_INFO_MGR/semantic_search/ \
    --batch-size 32
```

**Options:**
- `--input`: Path to tool descriptions NDJSON file
- `--output`: Directory for index files (default: `MCP_INFO_MGR/semantic_search/`)
- `--batch-size`: Batch size for embedding generation (default: 32)
- `--device`: Device for inference (`cpu`, `cuda`, `mps`, or None for auto)
- `--skip-errors`: Skip tools with errors instead of failing

**Output Files:**
- `index.faiss` - FAISS vector index
- `metadata.json` - Tool metadata (server, name, description, schema)
- `config.json` - Index configuration
- `model_info.json` - BGE-M3 model information

**Expected Time:**
- ~5-10 minutes for 1000 tools on CPU
- ~1-2 minutes with GPU acceleration

## Using the Index

### Python API

```python
from pathlib import Path
from MCP_INFO_MGR.semantic_search import BGEEmbedder, FAISSIndex

# Load embedder and index
embedder = BGEEmbedder()
index = FAISSIndex.load(Path("MCP_INFO_MGR/semantic_search/"))

# Search for tools
query = "search GitHub repositories"
query_embedding = embedder.encode_query(query)
results = index.search(query_embedding, top_k=5)

# Display results
for result in results:
    print(f"{result['server']}/{result['tool']}")
    print(f"  Score: {result['similarity_score']:.4f}")
    print(f"  Desc: {result['description']}")
```

### Search with Filters

```python
# Filter by server
results = index.search(
    query_embedding,
    top_k=5,
    filters={"server": "@smithery-ai/github"}
)

# Filter by minimum score
results = index.search(
    query_embedding,
    top_k=5,
    filters={"min_score": 0.5}
)

# Combine filters
results = index.search(
    query_embedding,
    top_k=5,
    filters={
        "server": "@smithery-ai/github",
        "min_score": 0.3
    }
)
```

### Batch Search

```python
# Search multiple queries at once
queries = [
    "search GitHub repositories",
    "fetch web page content",
    "analyze stock prices"
]
query_embeddings = embedder.encode_query(queries)
batch_results = index.batch_search(query_embeddings, top_k=3)

for query, results in zip(queries, batch_results):
    print(f"Query: {query}")
    for result in results:
        print(f"  - {result['server']}/{result['tool']}")
```

## File Structure

```
MCP_INFO_MGR/
├── semantic_search/
│   ├── __init__.py              # Package initialization
│   ├── embeddings.py            # BGE-M3 embedder wrapper
│   ├── faiss_backend.py         # FAISS index implementation
│   ├── index.faiss              # Vector index (generated)
│   ├── metadata.json            # Tool metadata (generated)
│   ├── config.json              # Index config (generated)
│   └── model_info.json          # Model info (generated)
├── build_search_index.py        # Index building script
├── test_semantic_search.py      # Test/demo script
└── requirements_semantic_search.txt  # Dependencies
```

## Key Concepts

### BGE-M3 Query Instructions

BGE-M3 performs better with query prefixes:

```python
# Automatically handled by BGEEmbedder
query_instruction = "Represent this sentence for searching relevant passages: "

# Query embedding (with instruction)
embedder.encode_query("search GitHub")
# → Embeds: "Represent this sentence for searching relevant passages: search GitHub"

# Document embedding (no instruction)
embedder.encode_documents("Search for GitHub repositories")
# → Embeds: "Search for GitHub repositories"
```

### Cosine Similarity via Inner Product

FAISS `IndexFlatIP` computes inner product. Since BGE-M3 normalizes embeddings (L2 norm = 1), inner product equals cosine similarity:

```
cosine_sim(a, b) = (a · b) / (||a|| × ||b||)
                 = a · b  (when ||a|| = ||b|| = 1)
```

Scores range from -1 to 1:
- **1.0**: Identical
- **0.8-1.0**: Highly similar
- **0.5-0.8**: Moderately similar
- **< 0.5**: Less relevant

### Tool Text Creation

Tools are embedded as combined text:

```
Server: @smithery-ai/github | Tool: search_repositories | Description: Search for GitHub repositories using various filters | Parameters: query, per_page | query: Search query | per_page: Results per page
```

This captures:
- Server context
- Tool name (often contains keywords)
- Description (main searchable content)
- Parameter names and descriptions

## Performance

### Index Build Performance

| Tools | CPU Time | GPU Time | Index Size |
|-------|----------|----------|------------|
| 100   | ~30s     | ~10s     | ~400 KB    |
| 1000  | ~5 min   | ~1 min   | ~4 MB      |
| 10000 | ~50 min  | ~10 min  | ~40 MB     |

### Search Performance

- **Single query**: < 1ms for 1000 tools
- **Batch (100 queries)**: ~50ms for 1000 tools
- **Memory**: ~4 MB RAM per 1000 tools

### Model Download

First run downloads BGE-M3 (~2.2 GB):
- Cached in `~/.cache/huggingface/`
- Only downloaded once
- Loads in ~3-5 seconds after caching

## Troubleshooting

### Out of Memory

If you get OOM errors during indexing:

```bash
# Reduce batch size
python MCP_INFO_MGR/build_search_index.py --batch-size 8

# Use CPU instead of GPU
python MCP_INFO_MGR/build_search_index.py --device cpu
```

### Slow Embedding Generation

- **Use GPU**: Install `faiss-gpu` and use `--device cuda`
- **Increase batch size**: Try `--batch-size 64` (if memory allows)
- **Use MPS** (Apple Silicon): Use `--device mps`

### Poor Search Results

1. **Check normalization**: Verify embeddings are normalized (norms ~1.0)
2. **Increase top_k**: Try returning more results
3. **Lower min_score**: Reduce score threshold
4. **Rebuild index**: Index may be corrupted

## Next Steps

Now that you have the semantic search backend:

1. **Implement Meta-MCP Server** with `search_mcp_tools` tool
2. **Add connection pooling** for `execute_mcp_tool`
3. **Test with LLM** to validate end-to-end workflow

See the main [README.md](../README.md) for the complete architecture.

## Examples

### Example 1: Find Web Search Tools

```python
query = "search the internet for information"
query_embedding = embedder.encode_query(query)
results = index.search(query_embedding, top_k=3)

# Results:
# 1. exa/web_search_exa (score: 0.78)
# 2. @smithery-ai/fetch/fetch_url (score: 0.65)
# 3. @docfork/mcp/docfork_search_docs (score: 0.62)
```

### Example 2: Find GitHub Tools

```python
query = "create issue on github"
query_embedding = embedder.encode_query(query)
results = index.search(
    query_embedding,
    top_k=5,
    filters={"server": "@smithery-ai/github"}
)

# Results (filtered to GitHub server only):
# 1. @smithery-ai/github/create_issue (score: 0.89)
# 2. @smithery-ai/github/update_issue (score: 0.71)
# 3. @smithery-ai/github/list_issues (score: 0.68)
```

### Example 3: Multilingual Search

```python
# Chinese query
query = "搜索GitHub仓库"  # "Search GitHub repositories"
query_embedding = embedder.encode_query(query)
results = index.search(query_embedding, top_k=3)

# Results:
# 1. @smithery-ai/github/search_repositories (score: 0.75)
# 2. @smithery-ai/github/get_repository (score: 0.64)
# 3. @smithery-ai/github/list_repositories (score: 0.61)
```

BGE-M3 handles 100+ languages seamlessly!

## References

- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)