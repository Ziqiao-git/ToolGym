#!/usr/bin/env python3
"""
Test script for Meta-MCP Server.

This tests the search_tools functionality directly without MCP protocol.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from MCP_INFO_MGR.semantic_search.embeddings import BGEEmbedder
from MCP_INFO_MGR.semantic_search.faiss_backend import FAISSIndex


def test_search_tools():
    """Test the search_tools functionality."""
    print("="*60)
    print("Testing Meta-MCP Server - search_tools")
    print("="*60)

    # Load index
    index_path = PROJECT_ROOT / "MCP_INFO_MGR" / "mcp_data" / "working" / "semantic_index"
    print(f"\nLoading index from {index_path}...")
    index = FAISSIndex.load(index_path)
    stats = index.get_stats()
    print(f"✓ Loaded index with {stats['total_tools']} tools")

    # Load embedder
    print("\nLoading BGE-M3 embedder...")
    embedder = BGEEmbedder()
    print(f"✓ Model loaded on {embedder.get_model_info()['device']}")

    # Test queries
    test_queries = [
        ("search GitHub repositories", 5, 0.3),
        ("fetch weather data", 3, 0.4),
        ("analyze stock prices", 5, 0.3),
        ("send email notifications", 3, 0.4),
        ("database operations", 5, 0.3),
    ]

    print("\n" + "="*60)
    print("Running test queries...")
    print("="*60)

    for query, top_k, min_score in test_queries:
        print(f"\n🔍 Query: '{query}' (top_k={top_k}, min_score={min_score})")

        # Generate embedding
        query_embedding = embedder.encode_query(query)

        # Search
        results = index.search(query_embedding, top_k=top_k)

        # Filter by score
        filtered = [r for r in results if r["similarity_score"] >= min_score]

        if not filtered:
            print(f"   No results met min_score threshold")
        else:
            for i, result in enumerate(filtered, 1):
                print(f"   {i}. {result['server']}/{result['tool']}")
                print(f"      Score: {result['similarity_score']:.3f}")
                print(f"      Desc: {result['description'][:80]}...")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the MCP server: python meta_mcp_server/server.py")
    print("2. Add to Claude Desktop config")
    print("3. Test with Claude Desktop")


if __name__ == "__main__":
    test_search_tools()
