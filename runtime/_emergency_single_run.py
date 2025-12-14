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

Complete workflow with emergency resilience:

Step 1: DISCOVER tools using search_tools
Step 2: READ search results carefully
Step 3: EXECUTE THE DISCOVERED TOOL
Step 4: IF TOOL FAILS - Try alternatives or different approaches
Step 5: ANSWER the user's question with available information

🚨 HANDLE TOOL FAILURES GRACEFULLY 🚨
If a tool fails (503, timeout, unavailable):
- Search for alternative tools
- Try different approaches
- Be persistent and creative
- Provide best effort answer based on what works

Your response format: {"thought": "...", "server": "...", "tool": "...", "arguments": {...}}""",
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
            enable_compression=True,
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
