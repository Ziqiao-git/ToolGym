#!/usr/bin/env python3
"""
Verify Query Execution with Reference Tools

This script verifies that a query can actually be solved using the reference tools
specified during query generation. Instead of letting the agent search for tools,
we pre-load the specific reference tools and see if the agent can solve the task.

Usage:
    # Verify a single query
    python runtime/verify_query_with_reference_tools.py \
        --query-file mcp_generate/requests/constrained_multi_tool_queries_121servers.json \
        --query-index 0 \
        --max-iterations 10

    # Verify all queries sequentially
    python runtime/verify_query_with_reference_tools.py \
        --query-file mcp_generate/requests/constrained_multi_tool_queries_121servers.json \
        --all

    # Verify all queries in parallel (recommended for speed)
    python runtime/verify_query_with_reference_tools.py \
        --query-file mcp_generate/requests/constrained_multi_tool_queries_121servers.json \
        --all \
        --parallel 4

    # Verify and generate refined queries (saved as individual files in a directory)
    python runtime/verify_query_with_reference_tools.py \
        --query-file mcp_generate/requests/constrained_multi_tool_queries_121servers.json \
        --all \
        --parallel 4 \
        --refine-output mcp_generate/requests/refined_batch_001
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
    hard_constraints = query_item.get("hard_constraints", [])
    soft_constraints = query_item.get("soft_constraints", [])
    query_uuid = query_item.get("uuid")  # Preserve UUID for tracking

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
        "uuid": query_uuid,
        "query": query,
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
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
        "failure_category": None,  # Type of failure: TOOL_QUALITY, QUERY_DESIGN, CONTEXT_OVERFLOW, MIXED
        "tool_quality_assessment": [],  # Per-tool quality evaluation
        "blocked_tool_calls": [],  # Tools agent tried to use but were blocked
        "tools_with_missing_descriptions": [],  # Tools that have None/empty descriptions
    }

    # Normalize server names: add @ prefix if missing
    # Configs use @namespace/name format, but LLM may generate namespace/name
    def normalize_server_name(name: str, configs: dict) -> str:
        """Normalize server name to match config format (with @ prefix)."""
        if name in configs:
            return name
        # Try adding @ prefix
        if not name.startswith('@'):
            with_at = f"@{name}"
            if with_at in configs:
                return with_at
        return name  # Return original if no match found

    # Fix and normalize reference_tools server names
    for tool in reference_tools:
        original_server = tool["server"]
        # Remove tool name if present (e.g., "@server/tool" -> "@server")
        parts = original_server.split("/")
        if len(parts) > 2:  # Has namespace + name + possibly tool
            # Keep namespace/name only
            tool["server"] = "/".join(parts[:2])

        # Normalize to add @ prefix if needed
        tool["server"] = normalize_server_name(tool["server"], all_server_configs)

        if tool["server"] != original_server:
            print(f"  Auto-corrected: {original_server} → {tool['server']}")

    # Extract unique normalized server names
    reference_servers = set()
    for tool in reference_tools:
        reference_servers.add(tool["server"])

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

        # Build step-by-step workflow instructions
        workflow_steps = []
        for i, tool in enumerate(reference_tools, 1):
            workflow_steps.append(
                f"Step {i}: Use {tool['server']}/{tool['tool']}\n"
                f"   Purpose: {tool['why']}\n"
                f"   What to do: Call this tool with appropriate arguments based on the query"
            )
        workflow_instructions = "\n\n".join(workflow_steps)

        # Create agent configuration with reference tools constraint
        react_config = {
            "name": f"verify-agent-q{query_idx}",
            "instruction": f"""You are an intelligent agent that needs to solve a query by following a strict step-by-step workflow using specific tools.

QUERY TO SOLVE:
{query}

REQUIRED WORKFLOW - Follow these steps in order:

{workflow_instructions}

AVAILABLE TOOLS:
{tools_list}

CRITICAL INSTRUCTIONS:
1. You MUST follow the workflow steps above sequentially
2. For each step, use the specified tool with appropriate arguments extracted from the query
3. Use the results from previous steps to inform later steps if needed
4. You MUST ONLY use the tools listed in the workflow - no other tools are allowed
5. If a tool returns an error or empty result, note it and continue to the next step
6. After completing all steps, synthesize the results into a comprehensive answer

Remember: You have direct access to these tools. Call them directly with the correct arguments.
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
                    "status": step.get("status", "unknown"),
                    "error": step.get("error", None),
                    "load_error": step.get("load_error", None),
                })

        result["tools_used"] = tools_used
        result["blocked_tool_calls"] = []  # No longer tracking blocked calls

        # Post-execution tool quality assessment using LLM
        tool_quality_assessments = []

        # Build assessment prompt (even if no tools were used)
        if tool_call_details:
            tools_summary_parts = []
            for tc in tool_call_details:
                tool_info = f"Tool: {tc['tool']}"
                tool_info += f"\nStatus: {tc['status']}"
                tool_info += f"\nArguments: {json.dumps(tc['arguments'], indent=2)}"

                # Show error information prominently
                if tc['error']:
                    tool_info += f"\n❌ ERROR: {str(tc['error'])[:1000]}"
                elif tc['load_error']:
                    tool_info += f"\n❌ LOAD ERROR: {str(tc['load_error'])[:1000]}"

                # Show result (full result, not truncated, so LLM can see what actually happened)
                result_str = str(tc['result'])
                if result_str:
                    # Limit to 2000 chars to avoid token overflow, but show more than before
                    tool_info += f"\nResult: {result_str[:2000]}"
                    if len(result_str) > 2000:
                        tool_info += f"\n... [Result truncated, total length: {len(result_str)} chars]"
                else:
                    tool_info += f"\nResult: (empty/null)"

                tools_summary_parts.append(tool_info)

            tools_summary = "\n\n".join(tools_summary_parts)
        else:
            tools_summary = "NO TOOLS WERE USED"

        reference_tools_summary = "\n".join([
            f"- {tool['server']}/{tool['tool']}: {tool['why']}"
            for tool in reference_tools
        ])

        final_answer = result["response"][:1000] if result["response"] else "No response generated"
        current_time = datetime.now().strftime("%B %d, %Y %I:%M %p %Z")

        assessment_prompt = f"""You are evaluating the quality of tools that were used to solve a query.

CURRENT DATE AND TIME: {current_time}

ORIGINAL QUERY:
{query}

REFERENCE TOOLS (tools that were supposed to be useful):
{reference_tools_summary}

TOOLS ACTUALLY USED AND THEIR RESULTS:
{tools_summary}

**PAY CLOSE ATTENTION TO:**
- Status field (success/error/server_load_failed)
- ERROR messages (these explain WHY tools failed)
- Empty/null results (tool didn't return useful data)
- Whether the query provided specific enough parameters for the tools

FINAL ANSWER PRODUCED:
{final_answer}

For EACH reference tool, assess its quality:

1. RELEVANCE (0-10): How relevant was this tool to solving the query?
2. COMPLETENESS (0-10): Did this tool provide sufficient information for its purpose?
3. NECESSITY: ESSENTIAL / HELPFUL / UNNECESSARY
4. REASONING: Brief explanation

Also provide an overall assessment:
- TOOLS_SUFFICIENT: true/false - Were the reference tools sufficient to solve the query?
- INSUFFICIENCY_REASON: If insufficient, explain what was missing. **Use the STATUS and ERROR fields to diagnose the problem**:
  * If tools returned errors (402 payment required, 401 auth, 404 not found, timeout, etc.) → Tool quality issue
  * If tools returned empty/null despite being called correctly → Tool quality issue
  * If query lacks SPECIFIC, QUERYABLE details (vague references like "our company", "the location" instead of "Tesla (TSLA)", "San Francisco, CA") → Query design issue
  * If agent didn't call the right tools or couldn't figure out which tools to use → Query design issue (query wasn't clear enough)
- FAILURE_CATEGORY: Categorize the type of failure (if tools_sufficient is false):
  * TOOL_QUALITY: Tools are broken, return errors (402/401/404/500), return empty/null, require missing API keys, timeout, or have technical issues
  * QUERY_DESIGN: Query asks for capabilities the tools fundamentally don't have, OR the query is too vague/generic (lacks specific data identifiers), OR the query wasn't clear enough about which tools to use
  * CONTEXT_OVERFLOW: Tool responses are too large, causing context window issues
  * MIXED: Multiple failure types (e.g., some tools broken AND query design issues)

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
    "insufficiency_reason": "explanation if false",
    "failure_category": "TOOL_QUALITY|QUERY_DESIGN|CONTEXT_OVERFLOW|MIXED|null"
  }}
}}"""

        try:
            # Call LLM for assessment (always, even if no tools were used)
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
            result["failure_category"] = overall.get("failure_category")
            result["self_evaluation"] = "SUFFICIENT" if overall.get("tools_sufficient") else "INSUFFICIENT"

        except Exception as e:
            print(f"Warning: Could not perform post-execution assessment: {e}")
            result["tools_sufficient"] = None
            result["self_evaluation"] = "UNCLEAR"

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
            # Include UUID in filename if available
            if query_uuid:
                filename = f"trajectory_{query_uuid}_{timestamp}.json"
            else:
                filename = f"verify_q{query_idx}_{timestamp}.json"
            filepath = trajectory_dir / filename

            trajectory_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_uuid": query_uuid,
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

        # Categorize error-based failures
        error_str = str(e).lower()
        if "context length" in error_str or "tokens" in error_str or "maximum" in error_str:
            result["failure_category"] = "CONTEXT_OVERFLOW"
        elif "timeout" in error_str or "connection" in error_str:
            result["failure_category"] = "TOOL_QUALITY"
        else:
            result["failure_category"] = "UNKNOWN_ERROR"

        print(f"\n❌ Error during execution: {e}\n")
        import traceback
        print(f"Traceback: {traceback.format_exc()}\n")

    return result


def refine_insufficient_query(
    original_query: str,
    reference_tools: list,
    insufficiency_reason: str,
    tool_assessments: list,
    tools_used: list,
    failure_category: str,
    llm,
) -> tuple:
    """
    Refine a query and update reference tools based on verification results.

    Args:
        original_query: Original insufficient query
        reference_tools: List of original reference tools
        insufficiency_reason: Why tools were insufficient
        tool_assessments: Per-tool quality assessments
        tools_used: Tools actually used by the agent
        failure_category: Type of failure (TOOL_QUALITY, QUERY_DESIGN, etc.)
        llm: LLM instance

    Returns:
        Tuple of (refined_query, updated_reference_tools)
    """
    # Build useful tools list from assessments
    useful_tools = []
    tools_to_keep = []
    tools_to_add = []

    for assessment in tool_assessments:
        tool_name = assessment.get("tool", "")  # This is just the tool name, not server/tool
        necessity = assessment.get("necessity", "")
        relevance = assessment.get("relevance", 0)
        reasoning = assessment.get("reasoning", "")

        if necessity in ["ESSENTIAL", "HELPFUL"] and relevance >= 5:
            useful_tools.append({
                "tool": tool_name,
                "necessity": necessity,
                "relevance": relevance,
                "capability": reasoning,
            })

            # Find the full server/tool name from tools_used
            full_tool_name = None
            for used_tool in tools_used:
                if used_tool.endswith(f"/{tool_name}") or used_tool == tool_name:
                    full_tool_name = used_tool
                    break

            if not full_tool_name:
                # If not in tools_used, check reference tools
                for ref_tool in reference_tools:
                    if ref_tool['tool'] == tool_name:
                        full_tool_name = f"{ref_tool['server']}/{ref_tool['tool']}"
                        break

            if not full_tool_name:
                continue  # Skip if we can't find the server

            # Check if this tool is in reference_tools
            is_reference = any(f"{t['server']}/{t['tool']}" == full_tool_name for t in reference_tools)
            if is_reference:
                # Keep this reference tool
                for ref_tool in reference_tools:
                    if f"{ref_tool['server']}/{ref_tool['tool']}" == full_tool_name:
                        tools_to_keep.append(ref_tool)
                        break
            else:
                # This is a non-reference tool that was helpful - add it
                if "/" in full_tool_name:
                    server, tool = full_tool_name.rsplit("/", 1)
                    tools_to_add.append({
                        "server": server,
                        "tool": tool,
                        "why": reasoning[:100]  # Use assessment reasoning as rationale
                    })

    # Build tool descriptions
    tool_descriptions = "\n".join([
        f"- {tool['server']}/{tool['tool']}: {tool['why']}"
        for tool in reference_tools
    ])

    capabilities = "\n".join([
        f"- {t['tool']} [{t['necessity']}]: {t['capability']}"
        for t in useful_tools
    ]) if useful_tools else "No tools provided useful results"

    # Build list of tools actually used
    tools_used_str = "\n".join([f"- {t}" for t in tools_used]) if tools_used else "No tools were called"

    current_time = datetime.now().strftime("%B %d, %Y %I:%M %p %Z")

    prompt = f"""Rewrite this query and identify which tools to use based on actual execution results.

CURRENT DATE AND TIME: {current_time}

ORIGINAL QUERY:
{original_query}

REFERENCE TOOLS (originally planned):
{tool_descriptions}

ACTUAL TOOL EXECUTION RESULTS:
{capabilities}

TOOLS ACTUALLY USED:
{tools_used_str}

WHY QUERY FAILED:
{insufficiency_reason}

FAILURE TYPE: {failure_category or 'UNKNOWN'}

CRITICAL REFINEMENT INSTRUCTIONS:

***MOST IMPORTANT: REMOVE UNSUPPORTED REQUIREMENTS FROM QUERY TEXT***

1. **ANALYZE TOOLS ACTUALLY USED**:
   - Look at "TOOLS ACTUALLY USED" section above
   - ANY tool that does NOT appear in that list was NEVER successfully called
   - ANY requirement that depends on unused tools MUST be removed from the query

2. **REWRITE QUERY TO REMOVE UNSUPPORTED REQUIREMENTS**:
   Examples of what to remove:
   - If "send_whatsapp_message" is NOT in TOOLS ACTUALLY USED → Remove "send data via WhatsApp" from query
   - If no compliance tool was used → Remove "ensure GDPR compliance", "verify regulations", etc.
   - If no Canva tool was used → Remove "create visuals using Canva"
   - If tool failed with error → Remove that requirement entirely

   BE AGGRESSIVE: If a tool wasn't successfully used, DELETE that entire requirement from the query text.

3. **KEEP ONLY WHAT WORKED**:
   - Preserve requirements where tools in "TOOLS ACTUALLY USED" successfully provided data
   - Maintain multi-step planning ONLY for capabilities that actually worked

4. **ADD SPECIFIC DETAILS** (while removing unsupported parts):
   - Company names, stock tickers (e.g., "Tesla (TSLA)")
   - Exact locations (e.g., "San Francisco, CA")
   - Specific dates (future dates based on current time: {current_time})

5. **NATURAL LANGUAGE**: Sound like a real professional request, don't mention tool names

OUTPUT FORMAT - Return a JSON object with TWO fields:
{{
  "refined_query": "<the rewritten query WITHOUT any unsupported requirements>",
  "recommended_tools": [
    {{"server": "<server_name>", "tool": "<tool_name>", "why": "<rationale>"}},
    ...
  ]
}}

***CRITICAL FOR "recommended_tools"***:
- Include ONLY tools that appear in "TOOLS ACTUALLY USED" section above
- DO NOT include tools that were never called
- DO NOT include tools that failed with errors
- If a tool is in REFERENCE TOOLS but NOT in TOOLS ACTUALLY USED → EXCLUDE IT

Tool format requirements:
- If tool is "@Ymuberra/geo-news-mcp/get_news_by_country", then:
  {{"server": "@Ymuberra/geo-news-mcp", "tool": "get_news_by_country", "why": "..."}}
- If tool is "@vijitdaroch/financial-modeling-prep-mcp-server/getESGDisclosures", then:
  {{"server": "@vijitdaroch/financial-modeling-prep-mcp-server", "tool": "getESGDisclosures", "why": "..."}}
"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm.generate(messages)
        response_text = response.strip() if isinstance(response, str) else str(response).strip()

        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(response_text)
            refined_query = result.get("refined_query", original_query)
            recommended_tools = result.get("recommended_tools", tools_to_keep)

            # Use recommended tools, or fall back to tools_to_keep if empty
            updated_tools = recommended_tools if recommended_tools else tools_to_keep

            # Clean up query prefixes
            for prefix in ["Refined query:", "Refined:", "Query:", "Here is", "Here's"]:
                if refined_query.lower().startswith(prefix.lower()):
                    refined_query = refined_query[len(prefix):].strip().lstrip(":")

            # Ensure we have at least some tools
            if not updated_tools:
                updated_tools = reference_tools

            return (refined_query, updated_tools)

        except json.JSONDecodeError:
            # Fallback: treat entire response as query text
            print(f"Warning: Could not parse JSON, using text as query")
            return (response_text, tools_to_keep if tools_to_keep else reference_tools)

    except Exception as e:
        print(f"Warning: Could not refine query: {e}")
        return (original_query, reference_tools)


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
    parser.add_argument(
        "--refine-output",
        type=Path,
        default=None,
        help="Output directory to save refined queries (one file per query)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers for verification (default: 1 for sequential)",
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
        if args.parallel > 1:
            print(f"Parallel Workers: {args.parallel}")
        print(f"{'='*80}\n")

        results = []

        if args.parallel > 1:
            # Parallel execution with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(args.parallel)

            async def verify_with_semaphore(i, query_item):
                async with semaphore:
                    print(f"[Worker] Starting query {i+1}/{len(queries)}")
                    result = await verify_single_query(
                        query_item=query_item,
                        all_server_configs=all_server_configs,
                        model=args.model,
                        max_iterations=args.max_iterations,
                        query_idx=i,
                        save_trajectory=not args.no_save_trajectory,
                    )
                    print(f"[Worker] Completed query {i+1}/{len(queries)}")
                    return result

            # Create all tasks
            tasks = [
                verify_with_semaphore(i, query_item)
                for i, query_item in enumerate(queries)
            ]

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions and ensure all results are dicts
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle cancelled tasks and other exceptions
                    exception_type = type(result).__name__
                    print(f"❌ Query {i} failed with exception: {exception_type}: {result}")
                    processed_results.append({
                        "query_index": i,
                        "uuid": queries[i].get("uuid"),
                        "query": queries[i].get("query", ""),
                        "status": "error",
                        "error": f"{exception_type}: {str(result)}",
                        "execution_successful": False,
                        "reference_tools": queries[i].get("reference_tools", []),
                    })
                else:
                    # Valid result dict
                    processed_results.append(result)

            results = processed_results
        else:
            # Sequential execution (original behavior)
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

        # Failure category breakdown
        failure_categories = {}
        for r in results:
            cat = r.get("failure_category")
            if cat:
                failure_categories[cat] = failure_categories.get(cat, 0) + 1

        if failure_categories:
            print(f"\nFailure Category Breakdown:")
            for category, count in sorted(failure_categories.items()):
                emoji = {
                    "TOOL_QUALITY": "🔧",
                    "QUERY_DESIGN": "📝",
                    "CONTEXT_OVERFLOW": "📊",
                    "MIXED": "🔀",
                    "UNKNOWN_ERROR": "❓"
                }.get(category, "•")
                print(f"  {emoji} {category}: {count} ({count/total*100:.1f}%)")

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

        # Refine insufficient queries if requested
        if args.refine_output:
            print(f"{'='*80}")
            print("REFINING INSUFFICIENT QUERIES")
            print(f"{'='*80}\n")

            # Create output directory
            output_dir = args.refine_output
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_dir}\n")

            # Initialize LLM for refinement (use gpt-4o-mini for speed/cost)
            refinement_model = "openai/gpt-4o-mini"
            print(f"Initializing refinement model: {refinement_model}...")
            model_manager = ModelManager()
            refinement_llm = model_manager.build_model("openrouter", config={"model_name": refinement_model})

            refined_count = 0
            skipped_count = 0
            kept_count = 0
            failure_category_counts = {}

            for result in results:
                query_idx = result["query_index"]
                query_uuid = result.get("uuid")  # Preserve UUID for tracking
                original_query = result["query"]
                reference_tools = result["reference_tools"]
                hard_constraints = result.get("hard_constraints", [])
                soft_constraints = result.get("soft_constraints", [])
                tools_sufficient = result.get("tools_sufficient")
                insufficiency_reason = result.get("insufficiency_reason", "")
                failure_category = result.get("failure_category")
                tool_assessments = result.get("tool_quality_assessment", [])

                # Track failure categories
                if failure_category:
                    failure_category_counts[failure_category] = failure_category_counts.get(failure_category, 0) + 1

                # Refine ALL insufficient queries, regardless of failure category
                # Also refine UNCLEAR queries (tools_sufficient is None) since we can't trust the assessment
                if tools_sufficient is False or tools_sufficient is None:
                    print(f"Query {query_idx}: REFINING (failure: {failure_category or 'UNKNOWN'})")
                    print(f"  Original: {original_query[:80]}...")

                    # Get tools actually used by the agent
                    tools_used = result.get("tools_used", [])

                    refined_query, updated_tools = refine_insufficient_query(
                        original_query=original_query,
                        reference_tools=reference_tools,
                        insufficiency_reason=insufficiency_reason,
                        tool_assessments=tool_assessments,
                        tools_used=tools_used,
                        failure_category=failure_category,
                        llm=refinement_llm,
                    )

                    print(f"  Refined: {refined_query[:80]}...")
                    print(f"  Updated tools: {len(updated_tools)} tools")

                    query_data = {
                        "uuid": query_uuid,
                        "query": refined_query,
                        "hard_constraints": hard_constraints,
                        "soft_constraints": soft_constraints,
                        "reference_tools": updated_tools,
                        "original_query": original_query,
                        "original_reference_tools": reference_tools,
                        "refinement_reason": insufficiency_reason,
                        "failure_category": failure_category,
                        "was_refined": True,
                    }

                    # Save to individual file
                    filename = f"query_{query_uuid}.json" if query_uuid else f"query_{query_idx}.json"
                    filepath = output_dir / filename
                    with filepath.open("w", encoding="utf-8") as f:
                        json.dump(query_data, f, indent=2, ensure_ascii=False)
                    print(f"  ✓ Saved to: {filename}\n")

                    refined_count += 1
                else:
                    # Keep original query (was sufficient)
                    query_data = {
                        "uuid": query_uuid,
                        "query": original_query,
                        "hard_constraints": hard_constraints,
                        "soft_constraints": soft_constraints,
                        "reference_tools": reference_tools,
                        "was_refined": False,
                    }

                    # Save to individual file
                    filename = f"query_{query_uuid}.json" if query_uuid else f"query_{query_idx}.json"
                    filepath = output_dir / filename
                    with filepath.open("w", encoding="utf-8") as f:
                        json.dump(query_data, f, indent=2, ensure_ascii=False)

                    kept_count += 1

            # Save metadata summary
            metadata = {
                "total_items": total,
                "refined_count": refined_count,
                "skipped_count": skipped_count,
                "kept_count": kept_count,
                "failure_category_distribution": failure_category_counts,
                "generation_method": "query_refinement_from_verification",
                "refinement_model": refinement_model,
                "refinement_policy": "Refine ALL insufficient queries regardless of failure category, remove unsupported requirements",
                "timestamp": datetime.now().isoformat(),
                "source_verification": str(summary_path),
            }

            metadata_path = output_dir / "_metadata.json"
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"{'='*80}")
            print(f"REFINEMENT SUMMARY")
            print(f"{'='*80}")
            print(f"✓ Refined: {refined_count} queries (ALL failure types)")
            print(f"✓ Skipped: {skipped_count} queries (execution errors)")
            print(f"✓ Kept unchanged: {kept_count} queries (sufficient)")
            print(f"\nFailure Category Distribution:")
            for category, count in sorted(failure_category_counts.items()):
                print(f"  {category}: {count}")
            print(f"\n✓ Saved {total} files to: {output_dir}")
            print(f"✓ Metadata saved to: {metadata_path}")
            print(f"{'='*80}\n")

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
