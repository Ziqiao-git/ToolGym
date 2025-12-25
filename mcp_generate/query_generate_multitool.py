#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_generate_multitool.py

Generate single-turn queries that require MULTIPLE different tools from MULTIPLE servers.
This creates complex, realistic tasks that span across different tool domains.

Usage:
  python query_generate_multitool.py \
    --config runtime/working_server_configs.json \
    --out mcp_generate/requests/multitool_10_20.json \
    --num-queries 20 \
    --min-tools 10 \
    --max-tools 20 \
    --model openai/gpt-5.1
"""

import json
import argparse
import os
import asyncio
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import random

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(str(SCRIPT_DIR / ".env"))

CLIENT = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL_NAME = "openai/gpt-4o-mini"
MAX_RETRIES = 3
MAX_CONCURRENT = 3  # Lower concurrency for larger prompts


SYSTEM_PROMPT = """You are an expert at creating realistic, complex user tasks that require multiple different tools working together.

Goal:
Given a set of available tools from different MCP servers, generate ONE comprehensive user query that would naturally require using ALL of the provided tools to complete. Then provide a brief explanation of why each tool is needed.

The query should:
1. **Be Realistic**: Represent a genuine real-world task someone would ask an AI assistant
2. **Be Coherent**: All tools should work together toward a unified goal (not a random list of unrelated tasks)
3. **Have Natural Flow**: Tools should be used in a logical sequence where outputs inform next steps
4. **Include Constraints**: Add real-world constraints (time, budget, preferences) that make the task grounded

TASK CATEGORIES that naturally combine many tools:

**1. Comprehensive Research & Analysis**
- "I'm writing a detailed report on [topic] and need to gather academic papers, news articles, social media sentiment, financial data, and expert opinions."

**2. Multi-Source Investigation**
- "I'm investigating [company/topic] and need stock data, news, SEC filings, Reddit discussions, GitHub activity, and competitor analysis."

**3. Cross-Domain Planning**
- "I'm planning [event/project] and need to research venues, weather, travel options, local events, reviews, and cost estimates."

**4. Technical Problem Solving**
- "I'm debugging/building [system] and need to search GitHub issues, Stack Overflow, documentation, code examples, and best practices."

**5. Market/Competitive Analysis**
- "I'm evaluating [market/opportunity] and need financial data, news, research papers, social trends, and regulatory information."

IMPORTANT RULES:
- Use EVERY tool provided in the input - do not skip any
- Create a SINGLE coherent query, not multiple separate tasks
- Each tool should have a clear purpose in achieving the overall goal
- The query should be detailed enough that it's obvious which tools are needed
- Include at least 4 specific constraints (budget, timeline, preferences, requirements, quality standards, etc.)

OUTPUT FORMAT (strict JSON):
{
  "query": "Detailed user query that requires all the tools...",
  "constraints": [
    "constraint 1 (specific and measurable)",
    "constraint 2 (specific and measurable)",
    "constraint 3 (specific and measurable)",
    "constraint 4 (specific and measurable)",
    ... (add as many constraints as make sense for a realistic, complex task - at least 4, but can be more)
  ],
  "tool_reasons": {
    "tool_name_1": "One sentence explaining why this specific tool is needed for this task",
    "tool_name_2": "One sentence explaining why this specific tool is needed for this task",
    ... (provide a detailed reason for EVERY tool - explain the specific value it adds)
  },
  "task_category": "research|investigation|planning|technical|market_analysis|other"
}

CRITICAL:
- The "constraints" array should contain at least 4 specific, measurable constraints, but can include more if the task naturally requires them. Don't artificially limit yourself.
- The "tool_reasons" object MUST contain an entry for EVERY tool provided in the input. Each reason should be specific to how that tool contributes to solving the query - not generic phrases like "required for task". Explain what specific information or capability each tool provides.
"""


USER_PROMPT_TEMPLATE = """Here are {tool_count} available tools from {server_count} different MCP servers that MUST ALL be used:

{tools_summary}

Generate ONE realistic, coherent user query that would require using ALL {tool_count} of these tools.

Requirements:
- The query should represent a genuine real-world task
- ALL {tool_count} tools must work together toward a single unified goal
- Include at least 4 specific real-world constraints (can be more if the task naturally requires them)
- Make sure the query naturally requires EVERY tool listed above

Return a JSON object with:
- "query": The detailed user query
- "constraints": List of at least 4 specific, measurable constraints (more is fine)
- "tool_reasons": A JSON object mapping EACH tool name to a SPECIFIC explanation of why it's needed and what value it provides (MUST have exactly {tool_count} entries)
- "task_category": One of research|investigation|planning|technical|market_analysis|other

CRITICAL:
1. Your "tool_reasons" object MUST contain exactly {tool_count} entries - one for each tool listed above
2. Each tool reason must be SPECIFIC - explain what information or capability that particular tool provides for THIS query
3. DO NOT use generic phrases like "Required for completing the task" - be specific about each tool's role
4. Example of GOOD tool_reasons:
   - "stock_indicators_a": "Provides key financial metrics (P/E, revenue growth, debt ratios) for Chinese A-share companies to evaluate investment opportunities"
   - "get_news_data": "Fetches recent news articles about selected stocks to assess sentiment and identify risks"
5. Example of BAD tool_reasons (too generic):
   - "stock_indicators_a": "Required for completing the task"
   - "get_news_data": "Needed for the analysis"
"""


def sample_diverse_tools(
    all_tools: List[Dict],
    min_tools: int,
    max_tools: int,
    min_servers: int = 3
) -> List[Dict]:
    """
    Sample tools ensuring diversity across servers.

    Returns list of {"server": str, "tool": str, "description": str}
    """
    # Flatten all tools with server info
    flat_tools = []
    for server_desc in all_tools:
        server_name = server_desc.get("qualifiedName", server_desc.get("name", "unknown"))
        for tool in server_desc.get("tools", []):
            flat_tools.append({
                "server": server_name,
                "tool": tool.get("name", ""),
                "description": tool.get("description", "No description")
            })

    if len(flat_tools) == 0:
        return []

    # Target number of tools
    target_tools = random.randint(min_tools, max_tools)
    target_tools = min(target_tools, len(flat_tools))

    # Group tools by server
    by_server = {}
    for t in flat_tools:
        server = t["server"]
        if server not in by_server:
            by_server[server] = []
        by_server[server].append(t)

    # Ensure minimum server diversity
    servers = list(by_server.keys())
    random.shuffle(servers)

    selected = []
    selected_servers = set()

    # First, pick at least one tool from min_servers different servers
    for server in servers[:min(min_servers, len(servers))]:
        tools = by_server[server]
        if tools:
            selected.append(random.choice(tools))
            selected_servers.add(server)

    # Then fill up to target with random tools (preferring diversity)
    remaining = [t for t in flat_tools if t not in selected]
    random.shuffle(remaining)

    # Prioritize tools from servers we haven't used yet
    remaining.sort(key=lambda t: (t["server"] in selected_servers, random.random()))

    for t in remaining:
        if len(selected) >= target_tools:
            break
        selected.append(t)
        selected_servers.add(t["server"])

    return selected


async def generate_multitool_query(
    sampled_tools: List[Dict],
    query_idx: int,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Generate a query that uses multiple tools."""

    # Build tools summary
    tools_summary_lines = []
    for t in sampled_tools:
        tools_summary_lines.append(f"- {t['server']}/{t['tool']}: {t['description']}")

    tools_summary = "\n".join(tools_summary_lines)

    # Count unique servers
    unique_servers = set(t["server"] for t in sampled_tools)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        tool_count=len(sampled_tools),
        server_count=len(unique_servers),
        tools_summary=tools_summary
    )

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await CLIENT.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.8,
                    max_tokens=8000,  # Larger for more tools
                    response_format={"type": "json_object"},
                )

                text = resp.choices[0].message.content
                if not text or not text.strip():
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return _make_error_result(sampled_tools, "Empty response")

                data = json.loads(text.strip())

                # Validate required fields
                if not isinstance(data.get("query"), str):
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1)
                        continue
                    return _make_error_result(sampled_tools, "Missing query field")

                # Get tool_reasons from LLM response (or empty dict)
                tool_reasons = data.get("tool_reasons", {})

                # Build reference_tools from sampled_tools (guarantees all tools are included)
                reference_tools = []
                for t in sampled_tools:
                    tool_name = t["tool"]
                    # Try to find reason from LLM, fallback to default
                    why = tool_reasons.get(tool_name, "Required for completing the task")
                    reference_tools.append({
                        "server": t["server"],
                        "tool": tool_name,
                        "why": why
                    })

                # Replace any LLM-provided reference_tools with our complete list
                data["reference_tools"] = reference_tools

                # Remove tool_reasons from output (we've merged it into reference_tools)
                if "tool_reasons" in data:
                    del data["tool_reasons"]

                # Add metadata
                data["tool_count"] = len(sampled_tools)
                data["server_count"] = len(unique_servers)
                data["complexity"] = data.get("complexity", "high")
                data["task_category"] = data.get("task_category", "other")

                # Ensure constraints exists
                if "constraints" not in data:
                    data["constraints"] = []

                return data

            except json.JSONDecodeError as e:
                print(f"⚠️  Query {query_idx}: JSON decode error: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return _make_error_result(sampled_tools, f"JSON error: {e}")

            except Exception as e:
                print(f"⚠️  Query {query_idx}: Error: {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return _make_error_result(sampled_tools, str(e))

    return _make_error_result(sampled_tools, "Max retries exceeded")


def _make_error_result(sampled_tools: List[Dict], error: str) -> Dict:
    """Create an error result with the sampled tools."""
    unique_servers = set(t["server"] for t in sampled_tools)
    return {
        "query": f"[Error: {error}]",
        "constraints": [],
        "reference_tools": [
            {"server": t["server"], "tool": t["tool"], "why": "N/A"}
            for t in sampled_tools
        ],
        "tool_count": len(sampled_tools),
        "server_count": len(unique_servers),
        "task_category": "error",
        "complexity": "high"
    }


async def generate_all_queries(
    all_tools: List[Dict],
    num_queries: int,
    min_tools: int,
    max_tools: int,
    min_servers: int,
    seed: Optional[int] = None
) -> List[Dict]:
    """Generate multiple multi-tool queries."""

    if seed is not None:
        random.seed(seed)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def worker(idx: int) -> Dict:
        # Sample tools for this query
        sampled = sample_diverse_tools(all_tools, min_tools, max_tools, min_servers)
        if not sampled:
            return _make_error_result([], "No tools available")

        print(f"Query {idx+1}: Sampled {len(sampled)} tools from {len(set(t['server'] for t in sampled))} servers")
        return await generate_multitool_query(sampled, idx, semaphore)

    tasks = [worker(i) for i in range(num_queries)]
    results = []

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating multi-tool queries"):
        result = await coro
        results.append(result)

    return results


async def main_async():
    parser = argparse.ArgumentParser(
        description="Generate single-turn queries requiring multiple tools"
    )
    parser.add_argument(
        "--tools",
        default=None,
        help="Tool descriptions NDJSON file (default: MCP_INFO_MGR/mcp_data/working/tool_descriptions.ndjson)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of queries to generate (default: 20)"
    )
    parser.add_argument(
        "--min-tools",
        type=int,
        default=10,
        help="Minimum tools per query (default: 10)"
    )
    parser.add_argument(
        "--max-tools",
        type=int,
        default=20,
        help="Maximum tools per query (default: 20)"
    )
    parser.add_argument(
        "--min-servers",
        type=int,
        default=3,
        help="Minimum different servers per query (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LLM model to use (default: openai/gpt-4o-mini)"
    )

    args = parser.parse_args()

    global MODEL_NAME
    MODEL_NAME = args.model

    # Load tool descriptions
    if args.tools:
        tool_desc_path = Path(args.tools)
    else:
        tool_desc_path = Path(__file__).parent.parent / "MCP_INFO_MGR" / "mcp_data" / "working" / "tool_descriptions.ndjson"

    print(f"Loading tool descriptions from {tool_desc_path}...")

    with open(tool_desc_path, "r", encoding="utf-8") as f:
        all_tools = [json.loads(line) for line in f if line.strip()]

    # Only include servers with status='ok' (successfully fetched tools)
    working_tools = [t for t in all_tools if t.get("status") == "ok" and t.get("tools")]

    # Count total tools
    total_tool_count = sum(len(s.get("tools", [])) for s in working_tools)
    print(f"Loaded {len(working_tools)} working servers with {total_tool_count} total tools")

    if total_tool_count < args.min_tools:
        print(f"⚠️  Warning: Only {total_tool_count} tools available, but min_tools is {args.min_tools}")
        print(f"   Adjusting min_tools to {total_tool_count}")
        args.min_tools = total_tool_count

    print(f"\nGenerating {args.num_queries} multi-tool queries...")
    print(f"  Tools per query: {args.min_tools} - {args.max_tools}")
    print(f"  Min servers per query: {args.min_servers}")
    print(f"  Model: {MODEL_NAME}")
    print()

    items = await generate_all_queries(
        all_tools=working_tools,
        num_queries=args.num_queries,
        min_tools=args.min_tools,
        max_tools=args.max_tools,
        min_servers=args.min_servers,
        seed=args.seed
    )

    # Calculate stats
    avg_tools = sum(item.get("tool_count", 0) for item in items) / len(items) if items else 0
    avg_servers = sum(item.get("server_count", 0) for item in items) / len(items) if items else 0
    error_count = sum(1 for item in items if item.get("task_category") == "error")

    # Create output
    result = {
        "metadata": {
            "total_items": len(items),
            "generation_method": "multitool_single_turn",
            "min_tools": args.min_tools,
            "max_tools": args.max_tools,
            "min_servers": args.min_servers,
            "avg_tools_per_query": round(avg_tools, 1),
            "avg_servers_per_query": round(avg_servers, 1),
            "error_count": error_count,
            "model": MODEL_NAME
        },
        "items": items
    }

    # Save output
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"Generated {len(items)} multi-tool queries")
    print(f"  Average tools per query: {avg_tools:.1f}")
    print(f"  Average servers per query: {avg_servers:.1f}")
    print(f"  Errors: {error_count}")
    print(f"  Output: {args.out}")
    print(f"{'='*70}")

    # Print sample
    if items and items[0].get("task_category") != "error":
        print("\nSample query:")
        print("-" * 70)
        sample = items[0]
        print(f"Query: {sample['query'][:300]}...")
        print(f"Tools: {sample['tool_count']} from {sample['server_count']} servers")
        print(f"Category: {sample.get('task_category', 'N/A')}")
        if sample.get("constraints"):
            print(f"Constraints: {sample['constraints']}")
        print("-" * 70)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
