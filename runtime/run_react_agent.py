#!/usr/bin/env python3
"""
Run ReAct Agent with Meta-MCP Server Integration

This script demonstrates how to use the existing ReAct agent from Orchestrator
with your Meta-MCP server to dynamically discover and use MCP tools.

Usage:
    python runtime/run_react_agent.py "Find GitHub repositories about machine learning"
"""
from __future__ import annotations

import sys
import json
import asyncio
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "Orchestrator"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ORCHESTRATOR_DIR))

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.agent.dynamic_react import DynamicReActAgent
from dotenv import load_dotenv
from typing import Dict, Any, Optional


def load_query_from_file(
    query_file: str,
    query_index: Optional[int] = None,
    query_uuid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a query from a JSON file containing queries with UUIDs.

    Args:
        query_file: Path to the JSON file
        query_index: Index of query to load (0-based)
        query_uuid: UUID of specific query to load

    Returns:
        Dict with keys: query, uuid, hard_constraints, soft_constraints, reference_tools

    Raises:
        ValueError: If query cannot be found or file format is invalid
    """
    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "items" not in data:
        raise ValueError(f"Query file {query_file} must have 'items' field")

    items = data["items"]

    if not items:
        raise ValueError(f"No queries found in {query_file}")

    # Find the query
    if query_uuid:
        # Search by UUID
        for item in items:
            if item.get("uuid") == query_uuid:
                return item
        raise ValueError(f"Query with UUID {query_uuid} not found in {query_file}")
    elif query_index is not None:
        # Use index
        if query_index < 0 or query_index >= len(items):
            raise ValueError(
                f"Query index {query_index} out of range (0-{len(items)-1})"
            )
        return items[query_index]
    else:
        # Default to first query
        return items[0]


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ReAct agent with Meta-MCP server for dynamic tool discovery"
    )
    parser.add_argument("query", nargs="?", help="Your question or task (optional if using --query-file)")
    parser.add_argument(
        "--query-file",
        help="Path to JSON file containing queries with UUIDs (e.g., generated_queries.json)",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        help="Index of query to run from the JSON file (0-based)",
    )
    parser.add_argument(
        "--query-uuid",
        help="UUID of specific query to run from the JSON file",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="Model name to use with OpenRouter (e.g., anthropic/claude-3.5-sonnet, openai/gpt-4)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum reasoning iterations",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save the agent's execution trajectory to a JSON file",
    )
    parser.add_argument(
        "--pass-number",
        type=int,
        default=1,
        help="Pass number for multiple attempts (e.g., 1 for pass@1, 2 for pass@2)",
    )
    parser.add_argument(
        "--batch-id",
        help="Batch ID for grouping related trajectories together",
    )
    parser.add_argument(
        "--output-dir",
        help="Direct output directory for trajectory files (overrides batch-id folder structure)",
    )
    parser.add_argument(
        "--disable-compression",
        action="store_true",
        help="Disable two-layer context compression (not recommended for long conversations)",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=200000,
        help="Model context window size in characters (default: 200000 for Claude 3.5 Sonnet)",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv(str(ORCHESTRATOR_DIR / ".env"))

    # Determine query and UUID
    query_text = None
    query_uuid = None

    if args.query_file:
        # Load query from file
        if not args.query and args.query_index is None and not args.query_uuid:
            print("Warning: No query index or UUID specified, using first query from file")

        query_data = load_query_from_file(
            args.query_file,
            query_index=args.query_index,
            query_uuid=args.query_uuid,
        )
        query_text = query_data["query"]
        query_uuid = query_data.get("uuid")
    elif args.query:
        # Use command-line query
        query_text = args.query
    else:
        parser.error("Either 'query' or '--query-file' must be provided")

    print(f"{'='*60}")
    print(f"Dynamic ReAct Agent with Meta-MCP Server")
    print(f"{'='*60}")
    print(f"Query: {query_text}")
    if query_uuid:
        print(f"UUID: {query_uuid}")
    print(f"Model: {args.model}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"{'='*60}\n")

    # Load all server configurations
    configs_path = PROJECT_ROOT / "MCP_INFO_MGR" / "mcp_data" / "working" / "remote_server_configs.json"
    print(f"Loading server configs from {configs_path}...")
    with configs_path.open("r") as f:
        all_server_configs = json.load(f)
    print(f"✓ Loaded {len(all_server_configs)} server configurations\n")

    # Initialize MCP Manager
    print("Initializing MCP Manager...")
    mcp_manager = MCPManager()

    # Add Meta-MCP server
    meta_mcp_config = {
        "stdio": {
            "command": "python",
            "args": [
                str(PROJECT_ROOT / "meta_mcp_server" / "server.py")
            ],
        }
    }
    mcp_manager.add_server_config("meta-mcp", meta_mcp_config)
    print("✓ Added Meta-MCP server")

    # Initialize LLM with OpenRouter
    print(f"\nInitializing OpenRouter model: {args.model}...")
    model_manager = ModelManager()
    llm = model_manager.build_model("openrouter", config={"model_name": args.model})
    print("✓ LLM ready")

    # Create ReAct agent configuration
    react_config = {
        "name": "meta-mcp-react-agent",
        "instruction": """You are an intelligent agent that can discover and use MCP tools dynamically.

════════════════════════════════════════════════════════════════════════
🚨 CRITICAL: YOU MUST FOLLOW THIS COMPLETE WORKFLOW 🚨
════════════════════════════════════════════════════════════════════════

Your job has TWO phases that you MUST complete:

PHASE 1: DISCOVER TOOLS (using meta-mcp/search_tools)
PHASE 2: EXECUTE TOOLS (using the tools you discovered)

⚠️  NEVER stop after Phase 1! You must ALWAYS proceed to Phase 2! ⚠️

════════════════════════════════════════════════════════════════════════
COMPLETE WORKFLOW - FOLLOW EVERY STEP:
════════════════════════════════════════════════════════════════════════

Step 1: DISCOVER tools using search_tools
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Call: meta-mcp/search_tools
- Purpose: Find which tools can help answer the user's question
- Parameters:
  * query: Natural language description of what you need
  * top_k: Number of results (default 5, increase if needed)
  * min_score: Relevance threshold (0.0-1.0, default 0.3)

Example:
Action: search_tools
Action Input: {"query": "search GitHub repositories", "top_k": 10, "min_score": 0.3}

Step 2: READ search results carefully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- The results show: server name, tool name, description, parameters
- Example result: "**@smithery-ai/github** / `search_repositories` - Search for repositories on GitHub"
- Extract: server = "@smithery-ai/github", tool = "search_repositories"

Step 3: 🚨 EXECUTE THE DISCOVERED TOOL 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  THIS IS THE MOST CRITICAL STEP - DO NOT SKIP! ⚠️

- Take the server and tool name from search results
- Call that tool with appropriate arguments based on its parameters
- The server will be loaded automatically when you call the tool
- Example:
  Action: search_repositories
  Action Input: {"query": "machine learning", "sort": "stars"}

Step 4: READ the tool results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- The tool returns actual data (repositories, weather, papers, etc.)
- This is the information you need to answer the user's question

Step 5: ANSWER the user's question
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use the tool results to provide a complete answer
- Include specific data from the tool output
- Be helpful and informative

════════════════════════════════════════════════════════════════════════
🚨 CRITICAL RULES - MEMORIZE THESE 🚨
════════════════════════════════════════════════════════════════════════

1. search_tools is NOT a data retrieval tool - it's a tool DISCOVERY tool
   ❌ WRONG: "I found tools about GitHub, here are the results"
   ✅ RIGHT: "I found the search_repositories tool, now I'll use it"

2. You MUST execute tools after discovering them
   ❌ WRONG: Call search_tools → Return search results to user
   ✅ RIGHT: Call search_tools → Call discovered tool → Return tool results to user

3. If search_tools returns results, you MUST try to use at least one tool
   - Don't make excuses like "I can't access that tool"
   - The tools will be loaded automatically when you call them
   - Just use the server and tool name from search results

4. For complex queries requiring multiple types of data:
   - Call search_tools multiple times with different focused queries
   - Each search_tools call should focus on ONE capability
   - Execute the tools you discover from each search

════════════════════════════════════════════════════════════════════════
COMPLETE EXAMPLE WORKFLOW:
════════════════════════════════════════════════════════════════════════

User Query: "Find machine learning repositories on GitHub"

Iteration 1:
  Thought: I need to find tools that can search GitHub repositories
  Action: search_tools
  Action Input: {"query": "search GitHub repositories", "top_k": 5}
  Observation: Found 5 relevant tools for: 'search GitHub repositories'
               1. **@smithery-ai/github** / `search_repositories`
                  Score: 0.856
                  Description: Search for repositories on GitHub
                  Parameters: query, sort, order

Iteration 2:
  Thought: Great! I found the search_repositories tool. Now I'll use it to actually search for machine learning repositories.
  Action: search_repositories
  Action Input: {"query": "machine learning", "sort": "stars", "order": "desc"}
  Observation: [
    {"name": "tensorflow/tensorflow", "stars": 175000, ...},
    {"name": "pytorch/pytorch", "stars": 65000, ...},
    ...
  ]

Iteration 3:
  Thought: Perfect! I got actual repository results. Now I can answer the user.
  Action: Final Answer
  Action Input: Here are the top machine learning repositories on GitHub:
                1. tensorflow/tensorflow (175,000 stars) - ...
                2. pytorch/pytorch (65,000 stars) - ...

════════════════════════════════════════════════════════════════════════

Remember:
- Phase 1 (search_tools) = Find which tools exist
- Phase 2 (execute tools) = Actually use those tools to get data
- You must complete BOTH phases to answer the user's question!""",
        "max_iterations": args.max_iterations,
        "summarize_tool_response": "auto",  # Enable smart summarization
        "summarize_threshold": 100000,      # Summarize if response > 100k chars
    }

    # Create Dynamic ReAct agent (with auto-loading capability)
    print("\nCreating Dynamic ReAct agent...")
    enable_compression = not args.disable_compression
    agent = DynamicReActAgent(
        mcp_manager=mcp_manager,
        llm=llm,
        server_configs=all_server_configs,
        config=react_config,
        enable_compression=enable_compression,
        model_context_limit=args.context_limit,
    )

    if enable_compression:
        print("✓ Two-layer context compression enabled")
        print(f"  - Layer 1: Single result > 50K chars")
        print(f"  - Layer 2: Total context > {int(args.context_limit * 0.8)} chars (80%)")
    else:
        print("⚠ Context compression disabled")

    # Initialize the agent with only meta-mcp
    # Other servers will be loaded dynamically as needed
    print("Initializing agent with meta-mcp server...")
    await agent.initialize(mcp_servers=[{"name": "meta-mcp"}])
    print("✓ Dynamic ReAct agent ready (will auto-load servers on-demand)\n")

    # Run the agent
    print(f"{'='*60}")
    print("Running ReAct Agent...")
    print(f"{'='*60}\n")

    response = await agent.execute(query_text)

    print(f"\n{'='*60}")
    print("Agent Response:")
    print(f"{'='*60}")
    print(response.response)
    print(f"{'='*60}\n")

    # Properly cleanup MCP clients
    print("Cleaning up MCP connections...")
    await agent.cleanup()

    # Save trajectory if requested
    if args.save_trajectory:
        from datetime import datetime

        # Determine trajectory directory
        if args.output_dir:
            # Direct output directory specified - use it directly with pass folder
            pass_folder = f"pass@{args.pass_number}"
            trajectory_dir = Path(args.output_dir) / pass_folder
        else:
            # Sanitize model name for folder/filename (replace / with -)
            # Extract just the model name (e.g., "deepseek" from "deepseek/deepseek-v3.2")
            model_name = args.model.split("/")[-1] if "/" in args.model else args.model
            model_safe = model_name.replace(":", "-")

            # Create hierarchical directory: trajectories/iter{N}/{model}/pass@{N}/
            iter_folder = f"iter{args.max_iterations}"
            model_folder = model_safe
            pass_folder = f"pass@{args.pass_number}"
            trajectory_dir = PROJECT_ROOT / "trajectories" / iter_folder / model_folder / pass_folder
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include UUID in filename if available (no need for model since it's in folder)
        if query_uuid:
            filename = f"trajectory_{query_uuid}_{timestamp}.json"
        else:
            filename = f"trajectory_{timestamp}.json"
        filepath = trajectory_dir / filename

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "query": query_text,
            "model": args.model,
            "max_iterations": args.max_iterations,
            "pass_number": args.pass_number,
        }

        # Include UUID in metadata
        if query_uuid:
            metadata["query_uuid"] = query_uuid

        # Include batch_id in metadata
        if args.batch_id:
            metadata["batch_id"] = args.batch_id

        trajectory_data = {
            "metadata": metadata,
            "reasoning_trace": agent.reasoning_trace,  # Complete reasoning process
            "execution": {
                "final_response": response.response,
                "tool_calls": agent.trajectory,
                "total_tool_calls": len(agent.trajectory),
                "tool_calls_with_dynamic_load": sum(1 for t in agent.trajectory if t.get("dynamically_loaded")),
            },
            "servers": {
                "initially_loaded": ["meta-mcp"],
                "dynamically_loaded": list(agent.dynamically_loaded_servers),
                "total_servers_used": len(agent.loaded_servers),
                "dynamically_loaded_count": len(agent.dynamically_loaded_servers),
            }
        }

        # Add context compression stats if enabled
        if enable_compression and agent.context_manager:
            trajectory_data["context_compression"] = {
                "enabled": True,
                "stats": agent.context_manager.get_context_usage_stats(),
            }
        else:
            trajectory_data["context_compression"] = {"enabled": False}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Trajectory saved to: {filepath}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
