#!/usr/bin/env python3
"""
Verify Query Execution with Reference Tools

This script verifies that a query can actually be solved using the reference tools
specified during query generation. Instead of letting the agent search for tools,
we pre-load the specific reference tools and see if the agent can solve the task.

Usage:
    python runtime/verify_query_with_reference_tools.py \
        --query-file mcp_generate/requests/constrained_multi_tool_queries_121servers.json \
        --query-index 0 \
        --max-iterations 10

    Or verify all queries:
    python runtime/verify_query_with_reference_tools.py \
        --query-file mcp_generate/requests/constrained_multi_tool_queries_121servers.json \
        --all
"""
from __future__ import annotations

import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "Orchestrator"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ORCHESTRATOR_DIR))

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.agent.dynamic_react import DynamicReActAgent
from dotenv import load_dotenv


# Note: We removed ToolFilteredAgent because it was breaking tool calls
# Instead, we rely on only loading reference servers into the agent
# This ensures the agent can only use tools from those servers


async def verify_single_query(
    query_item: dict,
    all_server_configs: dict,
    model: str,
    max_iterations: int,
    query_idx: int,
    save_trajectory: bool = True,
) -> dict:
    """
    Verify that a single query can be solved using its reference tools.

    Args:
        query_item: Dict with "query" and "reference_tools" fields
        all_server_configs: All available server configurations
        model: Model name to use
        max_iterations: Max reasoning iterations
        query_idx: Index of this query (for logging)
        save_trajectory: Whether to save execution trajectory

    Returns:
        Verification result dict
    """
    query = query_item["query"]
    reference_tools = query_item.get("reference_tools", [])

    print(f"\n{'='*80}")
    print(f"VERIFYING QUERY {query_idx + 1}")
    print(f"{'='*80}")
    print(f"Query: {query[:100]}...")
    print(f"Reference tools: {len(reference_tools)}")
    for tool in reference_tools:
        print(f"  - {tool['server']}/{tool['tool']}")
    print(f"{'='*80}\n")

    result = {
        "query_index": query_idx,
        "query": query,
        "reference_tools": reference_tools,
        "status": "unknown",
        "error": None,
        "response": None,
        "tools_used": [],
        "tools_matched": False,
        "execution_successful": False,
        "self_evaluation": None,  # Agent's assessment of tool sufficiency
        "tools_sufficient": None,  # Boolean: did agent think tools were sufficient?
        "insufficiency_reason": None,  # Explanation of why tools are insufficient
        "tool_quality_assessment": [],  # Per-tool quality evaluation
        "blocked_tool_calls": [],  # Tools agent tried to use but were blocked
        "tools_with_missing_descriptions": [],  # Tools that have None/empty descriptions
    }

    # Extract unique server names from reference tools
    # Handle both formats: "server" or "server/tool"
    reference_servers = set()
    for tool in reference_tools:
        server = tool["server"]
        # Remove tool name if present (e.g., "@server/tool" -> "@server")
        # Split by '/' and take all but last part if it looks like it has a tool suffix
        parts = server.split("/")
        if len(parts) > 2:  # Has namespace + name + possibly tool
            # Keep namespace/name only
            server_name = "/".join(parts[:2])
        else:
            server_name = server
        reference_servers.add(server_name)

    print(f"Reference servers to load: {reference_servers}\n")

    # Validate all reference servers exist in configs
    missing_servers = []
    for server_name in reference_servers:
        if server_name not in all_server_configs:
            missing_servers.append(server_name)

    if missing_servers:
        result["status"] = "error"
        result["error"] = f"Missing server configs: {missing_servers}"
        print(f"❌ Error: Missing configs for servers: {missing_servers}\n")
        return result

    try:
        # Initialize MCP Manager
        print("Initializing MCP Manager...")
        mcp_manager = MCPManager()

        # Add reference server configs to MCPManager
        for server_name in reference_servers:
            if server_name in all_server_configs:
                mcp_manager.add_server_config(server_name, all_server_configs[server_name])

        # Initialize LLM
        print(f"Initializing model: {model}...")
        model_manager = ModelManager()
        llm = model_manager.build_model("openrouter", config={"model_name": model})

        # Set connection timeout for MCP servers
        import asyncio
        connection_timeout = 30  # 30 seconds timeout

        # Build reference tools list for instructions
        tools_list = "\n".join([
            f"- {tool['server']}/{tool['tool']}: {tool['why']}"
            for tool in reference_tools
        ])

        # Create agent configuration with reference tools constraint
        react_config = {
            "name": f"verify-agent-q{query_idx}",
            "instruction": f"""You are an intelligent agent that needs to solve a query using the available tools.

AVAILABLE TOOLS:
{tools_list}

1. Analyze the query and determine what information is needed
2. Use the available tools to gather as much relevant information as possible
3. Combine the results to provide the best answer you can

CONSTRAINTS:
- You MUST ONLY use the tools listed above
- Try to use the tools that seem most relevant to the query
- Even if tools don't perfectly match the query, use what's available and do your best
- Provide whatever information you can gather from the available tools

Remember: You have direct access to these tools. Just call them with correct arguments.
NO tool search needed - use the tools directly!
""",
            "max_iterations": max_iterations,
        }

        # Create Dynamic ReAct agent
        # The agent will only have access to tools from reference servers
        # since we only initialize those servers
        print("Creating Dynamic ReAct agent...")
        agent = DynamicReActAgent(
            mcp_manager=mcp_manager,
            llm=llm,
            server_configs=all_server_configs,
            config=react_config,
        )

        # Initialize agent with reference servers only
        print(f"Initializing agent with {len(reference_servers)} reference servers...")
        server_list = [{"name": server} for server in reference_servers]

        try:
            # Use asyncio.wait_for to add timeout to initialization
            await asyncio.wait_for(
                agent.initialize(mcp_servers=server_list),
                timeout=connection_timeout
            )
            print("✓ Agent ready\n")
        except asyncio.TimeoutError:
            raise Exception(f"Timeout connecting to MCP servers after {connection_timeout}s")
        except Exception as e:
            raise Exception(f"Failed to initialize agent: {str(e)}")

        # Check for tools with missing descriptions
        tools_with_missing_descriptions = []
        for server_name, tools_list in agent._tools.items():
            for tool in tools_list:
                if not tool.description or tool.description.strip() == "":
                    tools_with_missing_descriptions.append({
                        "server": server_name,
                        "tool": tool.name,
                        "note": "Tool has no description - agent may not understand its purpose"
                    })

        if tools_with_missing_descriptions:
            print(f"⚠️  Warning: {len(tools_with_missing_descriptions)} tools have missing descriptions:")
            for item in tools_with_missing_descriptions:
                print(f"    - {item['server']}/{item['tool']}")
            print()

        result["tools_with_missing_descriptions"] = tools_with_missing_descriptions

        # Execute query
        print(f"{'='*80}")
        print("Executing query...")
        print(f"{'='*80}\n")

        response = await agent.execute(query)

        result["status"] = "success"
        result["execution_successful"] = True

        # Handle case where response might be None or missing response attribute
        if response is None:
            result["response"] = ""
            print("Warning: Agent returned None response")
        elif hasattr(response, 'response'):
            result["response"] = response.response or ""
        else:
            result["response"] = str(response) if response else ""

        # Analyze which tools were actually used
        tools_used = []
        tool_call_details = []
        for step in agent.trajectory:
            if "tool" in step and "server" in step:
                tool_call = f"{step['server']}/{step['tool']}"
                tools_used.append(tool_call)
                tool_call_details.append({
                    "tool": tool_call,
                    "arguments": step.get("arguments", {}),
                    "result": step.get("result", ""),
                })

        result["tools_used"] = tools_used
        result["blocked_tool_calls"] = []  # No longer tracking blocked calls

        # Post-execution tool quality assessment using LLM
        tool_quality_assessments = []
        if tool_call_details:
            # Build assessment prompt
            tools_summary = "\n\n".join([
                f"Tool: {tc['tool']}\nArguments: {json.dumps(tc['arguments'], indent=2)}\nResult: {str(tc['result'])[:500]}"
                for tc in tool_call_details
            ])

            reference_tools_summary = "\n".join([
                f"- {tool['server']}/{tool['tool']}: {tool['why']}"
                for tool in reference_tools
            ])

            final_answer = result["response"][:1000] if result["response"] else "No response generated"

            assessment_prompt = f"""You are evaluating the quality of tools that were used to solve a query.

ORIGINAL QUERY:
{query}

REFERENCE TOOLS (tools that were supposed to be useful):
{reference_tools_summary}

TOOLS ACTUALLY USED AND THEIR RESULTS:
{tools_summary}

FINAL ANSWER PRODUCED:
{final_answer}

For EACH reference tool, assess its quality:

1. RELEVANCE (0-10): How relevant was this tool to solving the query?
2. COMPLETENESS (0-10): Did this tool provide sufficient information for its purpose?
3. NECESSITY: ESSENTIAL / HELPFUL / UNNECESSARY
4. REASONING: Brief explanation

Also provide an overall assessment:
- TOOLS_SUFFICIENT: true/false - Were the reference tools sufficient to solve the query?
- INSUFFICIENCY_REASON: If insufficient, explain what was missing

FORMAT YOUR RESPONSE AS JSON:
{{
  "tool_assessments": [
    {{
      "tool": "server/tool_name",
      "relevance": 0-10,
      "completeness": 0-10,
      "necessity": "ESSENTIAL|HELPFUL|UNNECESSARY",
      "reasoning": "explanation"
    }}
  ],
  "overall_assessment": {{
    "tools_sufficient": true|false,
    "insufficiency_reason": "explanation if false"
  }}
}}"""

            try:
                # Call LLM for assessment
                assessment_messages = [{"role": "user", "content": assessment_prompt}]
                assessment_response = llm.generate(assessment_messages)
                assessment_text = assessment_response if isinstance(assessment_response, str) else str(assessment_response)

                # Parse JSON response
                # Extract JSON from markdown code blocks if present
                if "```json" in assessment_text:
                    assessment_text = assessment_text.split("```json")[1].split("```")[0].strip()
                elif "```" in assessment_text:
                    assessment_text = assessment_text.split("```")[1].split("```")[0].strip()

                assessment_data = json.loads(assessment_text)
                tool_quality_assessments = assessment_data.get("tool_assessments", [])
                overall = assessment_data.get("overall_assessment", {})

                result["tools_sufficient"] = overall.get("tools_sufficient")
                result["insufficiency_reason"] = overall.get("insufficiency_reason")
                result["self_evaluation"] = "SUFFICIENT" if overall.get("tools_sufficient") else "INSUFFICIENT"

            except Exception as e:
                print(f"Warning: Could not perform post-execution assessment: {e}")
                result["tools_sufficient"] = None
                result["self_evaluation"] = "UNCLEAR"
        else:
            # No tools were used at all
            result["tools_sufficient"] = False
            result["self_evaluation"] = "INSUFFICIENT"
            result["insufficiency_reason"] = "No tools were used during execution"

        result["tool_quality_assessment"] = tool_quality_assessments

        # Check if tools used match reference tools
        # Reference tools have format: server="@server/name/tool", tool="tool"
        # We need to extract just the tool name and server name (without the tool suffix)
        reference_tool_names = set()
        for tool in reference_tools:
            server = tool["server"]
            tool_name = tool["tool"]
            # Remove tool name from server if present
            parts = server.split("/")
            if len(parts) > 2:  # Has namespace + name + tool
                server_base = "/".join(parts[:2])
            else:
                server_base = server
            reference_tool_names.add(f"{server_base}/{tool_name}")

        tools_used_set = set(tools_used)

        # Check if agent used ONLY reference tools (subset check)
        used_non_reference_tools = tools_used_set - reference_tool_names
        result["tools_matched"] = len(used_non_reference_tools) == 0
        result["non_reference_tools_used"] = list(used_non_reference_tools)

        print(f"\n{'='*80}")
        print("Verification Results:")
        print(f"{'='*80}")
        print(f"✅ Execution Status: SUCCESS")

        # Display tool quality assessment
        if result['tool_quality_assessment']:
            print(f"\nTool Quality Assessment:")
            for assessment in result['tool_quality_assessment']:
                print(f"\n  Tool: {assessment.get('tool', 'Unknown')}")
                if assessment.get('relevance') is not None:
                    rel = assessment['relevance']
                    rel_emoji = "🟢" if rel >= 7 else "🟡" if rel >= 4 else "🔴"
                    print(f"    Relevance: {rel_emoji} {rel}/10")
                if assessment.get('completeness') is not None:
                    comp = assessment['completeness']
                    comp_emoji = "🟢" if comp >= 7 else "🟡" if comp >= 4 else "🔴"
                    print(f"    Completeness: {comp_emoji} {comp}/10")
                if assessment.get('necessity'):
                    nec = assessment['necessity']
                    nec_emoji = "⭐" if nec == "ESSENTIAL" else "✓" if nec == "HELPFUL" else "⚠️"
                    print(f"    Necessity: {nec_emoji} {nec}")
                if assessment.get('reasoning'):
                    print(f"    Reasoning: {assessment['reasoning']}")

        print(f"\nSelf-Evaluation:")
        print(f"  Agent assessed tools as: {result['self_evaluation']}")
        if result['tools_sufficient'] is not None:
            print(f"  Tools sufficient: {'✅ YES' if result['tools_sufficient'] else '❌ NO'}")

        if result['insufficiency_reason']:
            print(f"\n  Reason for Insufficiency:")
            # Indent each line of the reason
            for line in result['insufficiency_reason'].split('\n'):
                print(f"    {line}")

        print(f"\nTool Usage:")
        print(f"  Tools Used: {len(tools_used)}")
        for tool in set(tools_used):
            print(f"    - {tool}")
        print(f"  Tools Matched Reference: {'✅ YES' if result['tools_matched'] else '❌ NO'}")
        if not result["tools_matched"]:
            print(f"  Non-reference tools used: {result['non_reference_tools_used']}")

        if result["blocked_tool_calls"]:
            print(f"\n  Blocked Tool Attempts: {len(result['blocked_tool_calls'])}")
            for blocked in result["blocked_tool_calls"]:
                print(f"    ⛔ {blocked['server']}/{blocked['tool']}")

        print(f"{'='*80}\n")

        # Cleanup
        await agent.cleanup()

        # Save trajectory if requested
        if save_trajectory:
            trajectory_dir = PROJECT_ROOT / "trajectories" / "verification"
            trajectory_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"verify_q{query_idx}_{timestamp}.json"
            filepath = trajectory_dir / filename

            trajectory_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_index": query_idx,
                    "query": query,
                    "model": model,
                    "max_iterations": max_iterations,
                    "reference_tools": reference_tools,
                    "reference_servers": list(reference_servers),
                },
                "verification": result,
                "reasoning_trace": agent.reasoning_trace,
                "execution": {
                    "final_response": response.response,
                    "tool_calls": agent.trajectory,
                    "total_tool_calls": len(agent.trajectory),
                },
                "servers": {
                    "loaded": list(reference_servers),
                    "total_loaded": len(reference_servers),
                },
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Trajectory saved to: {filepath}\n")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["execution_successful"] = False
        print(f"\n❌ Error during execution: {e}\n")
        import traceback
        print(f"Traceback: {traceback.format_exc()}\n")

    return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify queries can be solved with their reference tools"
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        required=True,
        help="JSON file with queries and reference_tools",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Index of query to verify (0-based). Use --all to verify all queries.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all queries in the file",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="Model name to use with OpenRouter",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum reasoning iterations",
    )
    parser.add_argument(
        "--no-save-trajectory",
        action="store_true",
        help="Don't save execution trajectories",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv(str(ORCHESTRATOR_DIR / ".env"))

    # Load query file
    print(f"Loading queries from {args.query_file}...")
    with args.query_file.open("r") as f:
        data = json.load(f)

    queries = data.get("items", data.get("queries", []))
    print(f"✓ Loaded {len(queries)} queries\n")

    if not queries:
        print("Error: No queries found in file")
        return 1

    # Load all server configurations
    configs_path = PROJECT_ROOT / "MCP_INFO_MGR" / "mcp_data" / "working" / "remote_server_configs.json"
    print(f"Loading server configs from {configs_path}...")
    with configs_path.open("r") as f:
        all_server_configs = json.load(f)
    print(f"✓ Loaded {len(all_server_configs)} server configurations\n")

    # Verify queries
    if args.all:
        # Verify all queries
        print(f"\n{'='*80}")
        print(f"VERIFYING ALL {len(queries)} QUERIES")
        print(f"{'='*80}\n")

        results = []
        for i, query_item in enumerate(queries):
            result = await verify_single_query(
                query_item=query_item,
                all_server_configs=all_server_configs,
                model=args.model,
                max_iterations=args.max_iterations,
                query_idx=i,
                save_trajectory=not args.no_save_trajectory,
            )
            results.append(result)

        # Summary
        print(f"\n{'='*80}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*80}")

        total = len(results)
        successful = sum(1 for r in results if r["execution_successful"])
        failed = sum(1 for r in results if r["status"] == "error")
        matched = sum(1 for r in results if r["tools_matched"])
        sufficient = sum(1 for r in results if r.get("tools_sufficient") == True)
        insufficient = sum(1 for r in results if r.get("tools_sufficient") == False)

        print(f"Total queries: {total}")
        print(f"Executed successfully: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed with errors: {failed} ({failed/total*100:.1f}%)")
        print(f"Used only reference tools: {matched} ({matched/total*100:.1f}%)")
        print(f"\nAgent Self-Assessment:")
        print(f"  Tools SUFFICIENT: {sufficient} ({sufficient/total*100:.1f}%)")
        print(f"  Tools INSUFFICIENT: {insufficient} ({insufficient/total*100:.1f}%)")

        # Tool quality statistics
        all_assessments = []
        for r in results:
            all_assessments.extend(r.get("tool_quality_assessment", []))

        if all_assessments:
            avg_relevance = sum(a.get("relevance", 0) for a in all_assessments if a.get("relevance")) / len([a for a in all_assessments if a.get("relevance")])
            avg_completeness = sum(a.get("completeness", 0) for a in all_assessments if a.get("completeness")) / len([a for a in all_assessments if a.get("completeness")])

            essential_count = sum(1 for a in all_assessments if a.get("necessity") == "ESSENTIAL")
            helpful_count = sum(1 for a in all_assessments if a.get("necessity") == "HELPFUL")
            unnecessary_count = sum(1 for a in all_assessments if a.get("necessity") == "UNNECESSARY")

            print(f"\nTool Quality Statistics ({len(all_assessments)} tools assessed):")
            print(f"  Average Relevance: {avg_relevance:.1f}/10")
            print(f"  Average Completeness: {avg_completeness:.1f}/10")
            print(f"  Necessity Distribution:")
            print(f"    ESSENTIAL: {essential_count} ({essential_count/len(all_assessments)*100:.1f}%)")
            print(f"    HELPFUL: {helpful_count} ({helpful_count/len(all_assessments)*100:.1f}%)")
            print(f"    UNNECESSARY: {unnecessary_count} ({unnecessary_count/len(all_assessments)*100:.1f}%)")

        # Show failed queries
        if failed > 0:
            print(f"\nFailed Queries:")
            for i, r in enumerate(results):
                if r["status"] == "error":
                    print(f"  Query {i}: {r['error'][:200]}")

        print(f"{'='*80}\n")

        # Save summary
        summary_path = PROJECT_ROOT / "trajectories" / "verification" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "query_file": str(args.query_file),
            "model": args.model,
            "max_iterations": args.max_iterations,
            "total_queries": total,
            "successful_executions": successful,
            "tools_matched": matched,
            "results": results,
        }

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Summary saved to: {summary_path}\n")

    else:
        # Verify single query
        if args.query_index >= len(queries):
            print(f"Error: Query index {args.query_index} out of range (0-{len(queries)-1})")
            return 1

        query_item = queries[args.query_index]
        result = await verify_single_query(
            query_item=query_item,
            all_server_configs=all_server_configs,
            model=args.model,
            max_iterations=args.max_iterations,
            query_idx=args.query_index,
            save_trajectory=not args.no_save_trajectory,
        )

        if result["execution_successful"]:
            print("✅ Verification completed successfully")
            return 0 if result["tools_matched"] else 2  # Return 2 if tools didn't match
        else:
            print("❌ Verification failed")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
