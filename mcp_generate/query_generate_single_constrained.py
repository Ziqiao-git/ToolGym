#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_generate_single_constrained.py

Enhanced single-turn query generation with REAL-LIFE CONSTRAINT PLANNING.
Generates realistic queries that include real-world constraints, trade-offs, and planning requirements.
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
MAX_CONCURRENT = 5

# Enhanced SYSTEM_PROMPT with real-life constraints and planning
SYSTEM_PROMPT = """You are an expert in crafting realistic, constraint-based user tasks that involve real-world planning and decision-making.

Goal:
Given a tool and its description, generate ONE detailed, realistic user query that includes REAL-LIFE CONSTRAINTS and PLANNING REQUIREMENTS.

The query should represent a scenario where:
1. **Multiple Constraints Exist**: Time limits, budget constraints, resource limitations, competing priorities, regulatory requirements, stakeholder needs, etc.
2. **Trade-offs Required**: The user must balance conflicting requirements (e.g., speed vs. quality, cost vs. features, risk vs. reward)
3. **Planning Needed**: The task requires strategic planning, sequencing, prioritization, or optimization
4. **Real-World Context**: Sounds like something a professional would actually encounter in their work

Constraint Categories to Consider (use 2-3 per query):
- **Temporal**: Deadlines, time windows, schedules, time zones, urgency levels
- **Financial**: Budgets, cost limits, ROI requirements, pricing constraints
- **Resource**: Limited capacity, availability, team size, equipment, API rate limits
- **Quality**: Performance standards, accuracy requirements, compliance needs
- **Scope**: Feature priorities, must-haves vs. nice-to-haves, phased rollouts
- **Risk**: Security requirements, error tolerance, backup plans, compliance
- **Stakeholder**: Different user groups, approval chains, communication needs

Requirements:
- The scenario must naturally require calling the provided tool 3+ times
- Include specific, concrete constraints with realistic values
- Make trade-offs explicit and challenging
- Avoid generic/simple requests - add depth and realism
- Keep language natural and professional

Examples of Constrained Scenarios:
- "Under a $5,000 budget with a 2-week deadline, prioritizing security over features"
- "Must serve both US and EU markets with different regulatory requirements"
- "Balance real-time performance with cost (API calls capped at 1000/day)"
- "Need MVP by Q1 2025, full release by Q3, with limited 3-person team"

Output format (strict JSON):
{
  "query": string,
  "reference_tools": [
    { "server": string, "tool": string, "why": string }
  ]
}

Notes:
- Include EXACTLY ONE item in "reference_tools" (the provided tool)
- In "why", explain how repeated tool calls (3+) help navigate the constraints
- Make constraints specific and measurable when possible
"""

USER_PROMPT_TEMPLATE = """Here is the ONLY available tool from its MCP server:

{servers_summary}

Generate 1 natural, realistic user query with REAL-LIFE CONSTRAINTS and PLANNING requirements that would naturally require calling THIS SAME tool three or more times.

Include specific constraints from multiple categories (time, budget, resources, quality, risk, stakeholders, etc.).

Return a JSON object with fields:
- "query": string
- "reference_tools": [ {{ "server": string, "tool": string, "why": string }} ]
(Include exactly one reference_tools item for the tool shown above.)
"""

def _coerce_reference_tools(obj: Any, server: str, tool: str) -> List[Dict[str, str]]:
    """Ensure reference_tools contains the current tool"""
    base = [{
        "server": server,
        "tool": tool,
        "why": "This tool is used repeatedly (≥3) to handle the query's multiple constraints and planning requirements."
    }]
    try:
        if isinstance(obj, list) and len(obj) >= 1:
            first = obj[0]
            if (
                isinstance(first, dict)
                and first.get("server") == server
                and first.get("tool") == tool
                and isinstance(first.get("why"), str)
            ):
                return [{
                    "server": server,
                    "tool": tool,
                    "why": first.get("why") or base[0]["why"]
                }]
    except Exception:
        pass
    return base

async def generate_single_query(tool_description: Dict, query_idx: int) -> Dict[str, Any]:
    """Generate 1 constrained planning query for a single tool"""
    server = tool_description.get("qualifiedName", tool_description.get("name") or "unknown")
    tools_summary_lines = []

    tools = tool_description.get("tools", [])
    if len(tools) == 0:
        tools_summary_lines.append(f"- {server}: No tools available")
        tool_name = ""
    else:
        t = tools[0]
        tool_name = t.get("name", "")
        tool_desc = t.get("description", "No description")
        tools_summary_lines.append(f"- {server}/{tool_name}: {tool_desc}")

    servers_summary_text = "\n".join(tools_summary_lines)
    user_prompt = USER_PROMPT_TEMPLATE.format(servers_summary=servers_summary_text)

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "constrained_planning_query",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "reference_tools": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "server": {"type": "string"},
                                "tool": {"type": "string"},
                                "why": {"type": "string"}
                            },
                            "required": ["server", "tool", "why"]
                        }
                    }
                },
                "required": ["query", "reference_tools"]
            },
            "strict": True
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,  # Higher temperature for more creative constraints
                max_tokens=2500,
                response_format=schema,
            )
            text = resp.choices[0].message.content

            if not text or not text.strip():
                print(f"⚠️  Query {query_idx}: Empty response, retrying... (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    return {
                        "query": f"[Error: Empty response after {MAX_RETRIES} attempts]",
                        "reference_tools": _coerce_reference_tools([], server, tool_name)
                    }

            text = text.strip()

            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                print(f"⚠️  Query {query_idx}: JSON decode error, retrying... (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    return {
                        "query": f"[Error: Invalid JSON after {MAX_RETRIES} attempts]",
                        "reference_tools": _coerce_reference_tools([], server, tool_name)
                    }

            if not (isinstance(obj, dict) and isinstance(obj.get("query"), str)):
                print(f"⚠️  Query {query_idx}: Invalid format, retrying... (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    return {
                        "query": f"[Error: Invalid format after {MAX_RETRIES} attempts]",
                        "reference_tools": _coerce_reference_tools([], server, tool_name)
                    }

            ref_tools = _coerce_reference_tools(obj.get("reference_tools"), server, tool_name)

            return {
                "query": obj["query"],
                "reference_tools": ref_tools
            }

        except Exception as e:
            print(f"⚠️  Query {query_idx}: Unexpected error: {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"     Retrying... (attempt {attempt}/{MAX_RETRIES})")
                await asyncio.sleep(2)
                continue
            else:
                print(f"     Failed after {MAX_RETRIES} attempts")
                return {
                    "query": f"[Error: {type(e).__name__} after {MAX_RETRIES} attempts]",
                    "reference_tools": _coerce_reference_tools([], server, tool_name)
                }

async def generate_queries_async(all_tools: List[Dict], num_tools: int, seed: Optional[int] = None) -> List[Dict]:
    """Generate constrained planning queries for sampled tools"""
    # Flatten: each tool as a generation unit
    tool_items = []
    for server_desc in all_tools:
        for t in server_desc.get("tools", []):
            tool_items.append({"server": server_desc, "tool": t})

    total_tools = len(tool_items)
    if total_tools == 0:
        return []

    # Random sampling
    if seed is not None:
        random.seed(seed)
    k = min(num_tools, total_tools)
    sampled = random.sample(tool_items, k=k)

    print(f"Total tools: {total_tools} | Sampled: {k}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def worker(idx, server_desc, tool_obj):
        server_name = server_desc.get("qualifiedName", server_desc.get("name") or "unknown")
        tool_name = tool_obj.get("name", "")
        print(f"Generating constrained planning query for: {server_name} / {tool_name}")

        server_copy = dict(server_desc)
        server_copy["tools"] = [tool_obj]
        async with semaphore:
            return await generate_single_query(server_copy, idx)

    tasks = [worker(i, item["server"], item["tool"]) for i, item in enumerate(sampled)]
    results = []

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating constrained queries"):
        res = await coro
        results.append(res)

    return results

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Working servers config JSON file")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load working server configs
    with open(args.config, "r", encoding="utf-8") as f:
        server_configs = json.load(f)

    print(f"Loaded {len(server_configs)} working servers")

    # Load tool descriptions from indexed data
    tool_desc_path = Path(__file__).parent.parent / "MCP_INFO_MGR" / "mcp_data" / "indexed" / "tool_descriptions.ndjson"
    print(f"Loading tool descriptions from {tool_desc_path}...")

    with open(tool_desc_path, "r", encoding="utf-8") as f:
        all_tools = [json.loads(line) for line in f if line.strip()]

    # Filter to only working servers
    working_server_names = set(server_configs.keys())
    working_tools = [t for t in all_tools if t.get("qualifiedName") in working_server_names]

    print(f"Filtered to {len(working_tools)} working servers with tools")

    print(f"Generating {args.num_queries} constrained planning queries (max {MAX_CONCURRENT} concurrent)...")
    items = await generate_queries_async(working_tools, num_tools=args.num_queries, seed=args.seed)

    result = {
        "metadata": {
            "total_items": len(items),
            "servers_count": len(working_tools),
            "generation_method": "single_tool_constrained_planning",
            "constraint_types": ["temporal", "financial", "resource", "quality", "scope", "risk", "stakeholder"]
        },
        "items": items
    }

    # Create output directory
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(items)} constrained planning queries and saved to {args.out}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
