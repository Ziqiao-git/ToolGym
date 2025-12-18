#!/usr/bin/env python3
"""
Internal script to run a single query with emergency interception.
Called by emergency_test.py via subprocess.
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

# Suppress all logging except errors
logging.basicConfig(level=logging.ERROR)
for logger_name in ['DynamicReActAgent', 'MCPManager', 'ModelManager', 'MCPClient', 'ContextManager']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Orchestrator"))
sys.path.insert(0, str(PROJECT_ROOT / "runtime"))

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.agent.dynamic_react import DynamicReActAgent
from dotenv import load_dotenv
from emergency_interceptor import EmergencyInterceptor, InterceptionStrategy


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run single query with emergency interception")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", required=True, choices=["no_interception", "first_non_search", "random_20"])
    parser.add_argument("--pass-number", type=int, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--error-message", default="Error: Tool temporarily unavailable (503 Service Unavailable)")

    args = parser.parse_args()

    try:
        # Load environment
        load_dotenv(str(PROJECT_ROOT / "Orchestrator" / ".env"))

        # Load query
        with open(args.query_file, "r") as f:
            data = json.load(f)
        query_data = data["items"][args.query_index]
        query_text = query_data["query"]
        query_uuid = query_data["uuid"]

        # Load server configs
        configs_path = PROJECT_ROOT / "MCP_INFO_MGR" / "mcp_data" / "working" / "remote_server_configs.json"
        with configs_path.open("r") as f:
            all_server_configs = json.load(f)

        # Initialize MCP Manager
        mcp_manager = MCPManager()

        # Add Meta-MCP server
        meta_mcp_config = {
            "stdio": {
                "command": "python",
                "args": [str(PROJECT_ROOT / "meta_mcp_server" / "server.py")],
            }
        }
        mcp_manager.add_server_config("meta-mcp", meta_mcp_config)

        # Initialize LLM
        model_manager = ModelManager()
        llm = model_manager.build_model("openrouter", config={"model_name": args.model})

        # Create ReAct agent configuration
        react_config = {
            "name": "meta-mcp-react-agent-emergency",
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
            "summarize_tool_response": "auto",
            "summarize_threshold": 100000,
        }

        # Create Dynamic ReAct agent
        agent = DynamicReActAgent(
            mcp_manager=mcp_manager,
            llm=llm,
            server_configs=all_server_configs,
            config=react_config,
            enable_compression=False,
            model_context_limit=200000,
        )

        # Initialize agent
        await agent.initialize(mcp_servers=[{"name": "meta-mcp"}])

        # ✨ INJECT EMERGENCY INTERCEPTOR ✨
        strategy_map = {
            "no_interception": InterceptionStrategy.NO_INTERCEPTION,
            "first_non_search": InterceptionStrategy.FIRST_NON_SEARCH,
            "random_20": InterceptionStrategy.RANDOM_20,
        }

        interceptor = EmergencyInterceptor(
            strategy=strategy_map[args.strategy],
            error_message=args.error_message,
            exclude_tools=["search_tools"],
            random_seed=args.random_seed,
        )
        interceptor.inject(agent)

        # Run the agent
        response = await agent.execute(query_text)

        # Get interception stats
        interception_stats = interceptor.get_stats()

        # Cleanup
        await agent.cleanup()

        # Save trajectory to Emergency_test folder
        model_name = args.model.split("/")[-1] if "/" in args.model else args.model
        model_safe = model_name.replace(":", "-")

        # Create directory: trajectories/Emergency_test/{model}/pass@{N}/{strategy}/
        model_folder = model_safe
        pass_folder = f"pass@{args.pass_number}"
        strategy_folder = args.strategy
        trajectory_dir = (
            PROJECT_ROOT / "trajectories" / "Emergency_test" /
            model_folder / pass_folder / strategy_folder
        )
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{query_uuid}_{timestamp}.json"
        filepath = trajectory_dir / filename

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "query": query_text,
            "model": args.model,
            "max_iterations": args.max_iterations,
            "pass_number": args.pass_number,
            "query_uuid": query_uuid,
            "batch_id": args.batch_id,
            "emergency_test": True,
            "interception_strategy": args.strategy,
            "interception_stats": interception_stats,
        }

        trajectory_data = {
            "metadata": metadata,
            "reasoning_trace": agent.reasoning_trace,
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
            },
            "context_compression": {
                "enabled": True,
                "stats": agent.context_manager.get_context_usage_stats() if agent.context_manager else {},
            },
            "emergency_interception": interception_stats,
        }

        with open(filepath, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        # Success
        sys.exit(0)

    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
