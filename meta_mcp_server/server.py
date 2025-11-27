#!/usr/bin/env python3
"""
Meta-MCP Server - Semantic Search for MCP Tools

A simple MCP server that provides semantic search across 4,572 MCP tools
from 301 servers using FAISS and BGE-M3 embeddings.

Usage:
    python meta_mcp_server/server.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from MCP_INFO_MGR.semantic_search.embeddings import BGEEmbedder
from MCP_INFO_MGR.semantic_search.faiss_backend import FAISSIndex


class MetaMCPServer:
    """Meta-MCP server for semantic tool search."""

    def __init__(self, index_path: Path):
        """Initialize the Meta-MCP server."""
        self.server = Server("meta-mcp-server")
        self.index_path = index_path
        self.index: FAISSIndex | None = None
        self.embedder: BGEEmbedder | None = None

        # Register handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP protocol handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_tools",
                    description=(
                        "Search across available MCP tools using semantic search. "
                        "**IMPORTANT USAGE RULES**: "
                        "1. This tool searches for ONE semantic concept at a time - if you need multiple different types of tools (e.g., weather + stocks + news), you MUST call this tool multiple times with separate queries. "
                        "2. Each query should describe a single, focused capability (e.g., 'fetch weather forecasts', 'analyze financial data', 'search documentation'). "
                        "3. You can adjust search parameters: increase 'top_k' (default: 5, max: 20) to get more results, or adjust 'min_score' (default: 0.3, range: 0.0-1.0) to control relevance threshold. "
                        "4. Lower min_score (e.g., 0.2) finds more tools but may include less relevant ones; higher min_score (e.g., 0.5) is stricter but may miss useful tools. "
                        "Returns tool name, server, description, parameters, and similarity score for each match."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description of a SINGLE capability you need (e.g., 'search GitHub repos', 'fetch weather data', 'analyze stock prices'). Be specific and focused - don't combine multiple unrelated requirements in one query.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5, max: 20). Increase this if you want to see more tool options.",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "min_score": {
                                "type": "number",
                                "description": "Minimum similarity score threshold (0.0-1.0, default: 0.3). Lower values (0.2-0.3) find more tools but may be less relevant. Higher values (0.4-0.6) are stricter but may miss useful tools.",
                                "default": 0.3,
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": ["query"],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Handle tool calls."""
            if name != "search_tools":
                raise ValueError(f"Unknown tool: {name}")

            # Lazy load index and embedder
            if self.index is None:
                self.index = FAISSIndex.load(self.index_path)
            if self.embedder is None:
                self.embedder = BGEEmbedder()

            # Extract arguments
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            min_score = arguments.get("min_score", 0.3)

            # Validate
            if not query:
                raise ValueError("Query is required")

            # Generate query embedding
            query_embedding = self.embedder.encode_query(query)

            # Search
            results = self.index.search(query_embedding, top_k=top_k)

            # Filter by minimum score
            filtered_results = [r for r in results if r["similarity_score"] >= min_score]

            # Format results
            if not filtered_results:
                response = f"No tools found for query: '{query}' (tried {len(results)} tools, none met min_score={min_score})"
            else:
                response = f"Found {len(filtered_results)} relevant tools for: '{query}'\n\n"
                for i, result in enumerate(filtered_results, 1):
                    response += f"{i}. **{result['server']}** / `{result['tool']}`\n"
                    response += f"   Score: {result['similarity_score']:.3f}\n"
                    response += f"   Description: {result['description']}\n"

                    # Add input schema info if available
                    if result.get('inputSchema', {}).get('properties'):
                        params = list(result['inputSchema']['properties'].keys())
                        response += f"   Parameters: {', '.join(params[:5])}"
                        if len(params) > 5:
                            response += f" (+{len(params)-5} more)"
                        response += "\n"
                    response += "\n"

            return [TextContent(type="text", text=response)]

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main():
    """Main entry point."""
    import datetime

    # Path to the FAISS index
    index_path = PROJECT_ROOT / "MCP_INFO_MGR" / "semantic_search"

    if not index_path.exists():
        print(f"Error: Index not found at {index_path}", file=sys.stderr)
        print("Please build the index first:", file=sys.stderr)
        print("  python MCP_INFO_MGR/build_search_index.py", file=sys.stderr)
        return 1

    # Print startup info to stderr (so it doesn't interfere with MCP protocol on stdout)
    print("="*60, file=sys.stderr)
    print("Meta-MCP Server Starting", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    print(f"Mode: STDIO (Model Context Protocol)", file=sys.stderr)
    print(f"Index path: {index_path}", file=sys.stderr)
    print(f"Tools available: 1 (search_tools)", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print("Server is ready and waiting for MCP client connections...", file=sys.stderr)
    print(" This server uses stdio - no HTTP port is used.", file=sys.stderr)
    print("Add this server to Claude Desktop config to use it.", file=sys.stderr)
    print("="*60, file=sys.stderr)

    # Create and run server
    server = MetaMCPServer(index_path)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
