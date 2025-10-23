"""
Test script for semantic search components.

This script demonstrates the complete workflow:
1. Load BGE-M3 embedder
2. Generate embeddings for sample tools
3. Build FAISS index
4. Perform semantic searches

Usage:
    python MCP_INFO_MGR/test_semantic_search.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from MCP_INFO_MGR.semantic_search.embeddings import BGEEmbedder, create_tool_text
from MCP_INFO_MGR.semantic_search.faiss_backend import FAISSIndex


def main():
    print("="*60)
    print("Semantic Search Test")
    print("="*60)

    # Sample tools for testing
    sample_tools = [
        {
            "server": "@smithery-ai/github",
            "tool": "search_repositories",
            "description": "Search for GitHub repositories using various filters",
            "inputSchema": {
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "per_page": {"type": "number", "description": "Results per page"},
                }
            },
        },
        {
            "server": "@smithery-ai/fetch",
            "tool": "fetch_url",
            "description": "Fetch a URL and return basic information about the page",
            "inputSchema": {
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                }
            },
        },
        {
            "server": "exa",
            "tool": "web_search_exa",
            "description": "Search the web using Exa AI - performs real-time web searches",
            "inputSchema": {
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "numResults": {"type": "number", "description": "Number of results"},
                }
            },
        },
        {
            "server": "@smithery-ai/github",
            "tool": "create_issue",
            "description": "Create a new issue in a GitHub repository",
            "inputSchema": {
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "title": {"type": "string", "description": "Issue title"},
                }
            },
        },
        {
            "server": "yfinance",
            "tool": "get_stock_data",
            "description": "Get stock price and financial data for a given ticker symbol",
            "inputSchema": {
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "period": {"type": "string", "description": "Time period for data"},
                }
            },
        },
    ]

    # Step 1: Initialize embedder
    print("\n1. Initializing BGE-M3 embedder...")
    embedder = BGEEmbedder()
    model_info = embedder.get_model_info()
    print(f"✓ Model: {model_info['model_name']}")
    print(f"  Device: {model_info['device']}")

    # Step 2: Create tool texts and generate embeddings
    print("\n2. Generating embeddings for sample tools...")
    tool_texts = []
    for tool in sample_tools:
        text = create_tool_text(
            tool_name=tool["tool"],
            tool_description=tool["description"],
            server_name=tool["server"],
            input_schema=tool.get("inputSchema"),
        )
        tool_texts.append(text)
        print(f"  - {tool['server']}/{tool['tool']}")

    embeddings = embedder.encode_documents(tool_texts, show_progress=False)
    print(f"✓ Generated {len(embeddings)} embeddings")

    # Step 3: Build FAISS index
    print("\n3. Building FAISS index...")
    index = FAISSIndex(dimension=1024)
    index.add(embeddings, sample_tools)
    print(f"✓ Index built with {index.get_stats()['total_tools']} tools")

    # Step 4: Test searches
    print("\n4. Testing semantic searches...")
    print("="*60)

    test_queries = [
        "search GitHub repositories",
        "get web page content",
        "analyze stock prices",
        "create a GitHub issue",
        "find information on the internet",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_embedding = embedder.encode_query(query)
        results = index.search(query_embedding, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['server']}/{result['tool']}")
            print(f"     Score: {result['similarity_score']:.4f}")
            print(f"     Desc: {result['description']}")

    # Step 5: Test with filters
    print("\n" + "="*60)
    print("5. Testing with filters...")
    print("="*60)

    query = "github operations"
    print(f"\nQuery: '{query}' (filter: server='@smithery-ai/github')")
    query_embedding = embedder.encode_query(query)
    results = index.search(
        query_embedding,
        top_k=5,
        filters={"server": "@smithery-ai/github"}
    )

    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['tool']}")
        print(f"     Score: {result['similarity_score']:.4f}")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r MCP_INFO_MGR/requirements_semantic_search.txt")
    print("2. Build index: python MCP_INFO_MGR/build_search_index.py")
    print("3. Use the index in your Meta-MCP server")


if __name__ == "__main__":
    main()