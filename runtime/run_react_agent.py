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


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ReAct agent with Meta-MCP server for dynamic tool discovery"
    )
    parser.add_argument("query", help="Your question or task")
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="Model name to use with OpenRouter (e.g., anthropic/claude-3.5-sonnet, openai/gpt-4)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum reasoning iterations",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save the agent's execution trajectory to a JSON file",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv(str(ORCHESTRATOR_DIR / ".env"))

    print(f"{'='*60}")
    print(f"Dynamic ReAct Agent with Meta-MCP Server")
    print(f"{'='*60}")
    print(f"Query: {args.query}")
    print(f"Model: {args.model}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"{'='*60}\n")

    # Load all server configurations
    configs_path = PROJECT_ROOT / "MCP_INFO_MGR" / "mcp_data" / "usable" / "remote_server_configs.json"
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

CRITICAL WORKFLOW - Follow these steps exactly:

Step 1: SEARCH for tools
- Use the 'search_tools' function from 'meta-mcp' server to find relevant MCP tools
- This searches across 4,572 tools from 301 servers
- Parameters: query (string), top_k (integer), min_score (number)

Step 2: READ the search results carefully
- The search returns: server name, tool name, description, and score
- Example result: "@smithery-ai/github / search_repositories - Search for GitHub repositories"
- This tells you BOTH the server name AND the tool name

Step 3: USE the tool you found
- Take the server and tool name from search results
- Call that tool with appropriate arguments
- Example: If search found "@smithery-ai/github/search_repositories",
  then use server="@smithery-ai/github", tool="search_repositories"

Step 4: ANSWER with the results

IMPORTANT:
- search_tools only FINDS tools, it doesn't execute them
- You MUST use the tools you find in a second step
- Don't give up - if search_tools returns results, USE those tools!

Example flow:
1. Call: meta-mcp/search_tools with query="search GitHub repositories"
2. Get result: "@smithery-ai/github / search_repositories"
3. Call: @smithery-ai/github/search_repositories with arguments={"query": "machine learning"}
4. Return the actual GitHub repositories""",
        "max_iterations": args.max_iterations,
    }

    # Create Dynamic ReAct agent (with auto-loading capability)
    print("\nCreating Dynamic ReAct agent...")
    agent = DynamicReActAgent(
        mcp_manager=mcp_manager,
        llm=llm,
        server_configs=all_server_configs,
        config=react_config,
    )

    # Initialize the agent with only meta-mcp
    # Other servers will be loaded dynamically as needed
    print("Initializing agent with meta-mcp server...")
    await agent.initialize(mcp_servers=[{"name": "meta-mcp"}])
    print("✓ Dynamic ReAct agent ready (will auto-load servers on-demand)\n")

    # Run the agent
    print(f"{'='*60}")
    print("Running ReAct Agent...")
    print(f"{'='*60}\n")

    response = await agent.execute(args.query)

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

        trajectory_dir = PROJECT_ROOT / "trajectories"
        trajectory_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{timestamp}.json"
        filepath = trajectory_dir / filename

        trajectory_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query": args.query,
                "model": args.model,
                "max_iterations": args.max_iterations,
            },
            "execution": {
                "final_response": response.response,
                "tool_calls": agent.trajectory,
                "loaded_servers": list(agent.loaded_servers),
                "total_tool_calls": len(agent.trajectory),
                "dynamically_loaded_count": sum(1 for t in agent.trajectory if t.get("dynamically_loaded")),
            },
            "servers": {
                "initially_loaded": ["meta-mcp"],
                "dynamically_loaded": list(agent.loaded_servers),
                "total_servers_used": 1 + len(agent.loaded_servers),
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Trajectory saved to: {filepath}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
