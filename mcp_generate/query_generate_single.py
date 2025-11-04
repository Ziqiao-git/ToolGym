#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_generate_single.py

使用 GPT-5 基于 MCP 工具描述生成自然语言查询问题。
用法:
  python query_generate_single.py --in tool_descriptions.ndjson --out generated_queries.json --num-queries 20
"""

import json
import argparse
import os
import time
import asyncio
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import random

# OpenAI 官方 Python SDK
# pip install --upgrade openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(str(SCRIPT_DIR / ".env"))

CLIENT = AsyncOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL_NAME = "openai/gpt-5"
MAX_RETRIES = 3
MAX_CONCURRENT = 5  # Maximum concurrent API calls

# ------Easy------
SYSTEM_PROMPT = """
You are an expert in writing natural, realistic user queries that involve everyday information tasks.

Goal:
Given a tool and its description, generate ONE natural-language user query that would realistically require using that tool exactly once.

Requirements:
- The task should be concrete and simple (e.g., ask for one piece of information or perform one straightforward action).
- It should reflect how a real person might use this tool in practice.
- Choose tools based on the core functionality they provide and how they would solve real user problems
- Keep it natural, fluent, and specific to the tool's function.
- Create questions that are clear and specific enough to warrant tool usage
- Avoid overly simple questions that could be answered without tools

# Make the query natural, diverse, and genuinely useful.
"""

# ------Median------
# SYSTEM_PROMPT = """
# ou are an expert at designing realistic user requests that require a tool to perform multiple related actions.

# Goal:
# Given a tool and its description, write ONE user query that would realistically require calling this tool at least twice to fulfill the request.

# Requirements:
# - The task should feel natural and purposeful (e.g., compare, refine, or analyze multiple inputs).
# - It should NOT be overly difficult or artificial.
# - Ensure the query connects two realistic uses of the tool.
# - Use realistic context—topics, data, or entities that people might actually care about.
# - Create questions that are clear and specific enough to warrant tool usage
# - Avoid overly simple questions that could be answered without or only use once of the tool

# # Make the query natural, diverse, and genuinely useful.
# """

# ------Hard------
# SYSTEM_PROMPT = """
# You are an expert in crafting complex, realistic multi-step user tasks that make full use of a tool’s capabilities.

# Goal:
# Given a tool and its description, generate ONE detailed, realistic user query that would naturally require calling this tool three or more times to achieve the goal.

# Requirements:
# - The query should describe a real-world scenario that genuinely benefits from repeated use of the tool (e.g., iterative analysis, batch processing, monitoring, multi-source synthesis).
# - Avoid artificial chaining; the repetition must make practical sense.
# - Keep language natural and coherent.
# - Choose tools based on the core functionality they provide and how they would solve real user problems
# - Consider each tool's description and purpose when crafting the question
# - Create questions that are clear and specific enough to warrant tool usage
# - Avoid overly simple questions that could be answered without or only use couple time of the tool
# 
# Guidelines
# - Embed the same tool usage inside a broader workflow that requires coordination across teams or systems
# - The task should sound like something a researcher, professional, or advanced user might realistically ask.
# - Increase multi-dimensional complexity through realistic constraints, competing requirements, stakeholder considerations, and interconnected dependencies

# Make the query natural, diverse, and genuinely useful.
# """


USER_PROMPT_TEMPLATE = """Here are the tool descriptions from it's MCP server:

{servers_summary}

Generate 1 natural, realistic user query that people would actually ask.
This query should naturally involve researching or combining information using this tool.

Return a JSON object with "query" field containing a single string.
"""



async def generate_single_query(tool_description: Dict, query_idx: int) -> str:
    """
    Generate a single query with freshly shuffled tools.

    Args:
        all_tools: List of all tool descriptions
        query_idx: Index of this query (for logging)

    Returns:
        Generated query string
    """
    server = tool_description.get("qualifiedName", tool_description.get("name") or "unknown")
    tools_summary_lines = []

    # Build a concise summary of the server's tools
    for tool in tool_description.get("tools", []):
        tool_name = tool.get("name", "")
        tool_desc = tool.get("description", "No description")
        tools_summary_lines.append(f"- {server}/{tool_name}: {tool_desc}")

    if not tools_summary_lines:
        servers_summary_text = f"- {server}: No tools available"
    else:
        servers_summary_text = "\n".join(tools_summary_lines)

    # Fill user prompt with the server/tools summary
    user_prompt = USER_PROMPT_TEMPLATE.format(servers_summary=servers_summary_text)

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "query_schema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
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
                temperature=0.7,
                max_tokens=2000,
                response_format=schema,
            )
            text = resp.choices[0].message.content

            if not text or not text.strip():
                print(f"⚠️  Query {query_idx}: Empty response, retrying... (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    return f"[Error: Empty response after {MAX_RETRIES} attempts]"

            text = text.strip()

            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                print(f"⚠️  Query {query_idx}: JSON decode error, retrying... (attempt {attempt}/{MAX_RETRIES})")
                print(f"     Response preview: {text[:100]}...")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    return f"[Error: Invalid JSON after {MAX_RETRIES} attempts]"

            if not (isinstance(obj, dict) and isinstance(obj.get("query"), str)):
                print(f"⚠️  Query {query_idx}: Invalid format, retrying... (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    return f"[Error: Invalid format after {MAX_RETRIES} attempts]"

            return obj["query"]

        except Exception as e:
            print(f"⚠️  Query {query_idx}: Unexpected error: {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"     Retrying... (attempt {attempt}/{MAX_RETRIES})")
                await asyncio.sleep(2)
                continue
            else:
                print(f"     Failed after {MAX_RETRIES} attempts")
                return f"[Error: {type(e).__name__} after {MAX_RETRIES} attempts]"


async def generate_queries_async(all_tools: List[Dict]) -> List[Dict]:
    """
    Generate queries for a sample of servers (or all if num_queries >= len(all_tools)).

    Returns a list of dicts: {server, tools, query}
    """
    # Flatten all server tools into individual items
    tool_items = []
    for server_desc in all_tools:
        for t in server_desc.get("tools", []):
            tool_items.append({"server": server_desc, "tool": t})

    total_tools = len(tool_items)
    # print(total_tools) # total tool number
    if total_tools == 0:
        return []

    # If num_queries is positive and less than total, sample that many tools;
    # otherwise process all tools.
    # if num_queries and num_queries > 0 and num_queries < total_tools:
    #     sample = tool_items[0:num_queries]
    # else:
    #     sample = tool_items

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def worker(idx, server_desc, tool_obj):
        # Print server and tool name before generating the query for visibility
        server_name = server_desc.get("qualifiedName", server_desc.get("name") or "unknown")
        tool_name = tool_obj.get("name", "")
        print(f"Generating query for server: {server_name}  tool: {tool_name}")

        async with semaphore:
            # Build a server copy containing only this tool so the prompt is focussed
            server_copy = dict(server_desc)
            server_copy["tools"] = [tool_obj]
            q = await generate_single_query(server_copy, idx)
            return {
                "server": server_name,
                "tool": tool_name,
                "query": q,
            }

    tasks = [worker(i, item["server"], item["tool"]) for i, item in enumerate(tool_items)] # change here to test smaller cases.
    results = []

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating queries"):
        res = await coro
        results.append(res)

    return results




async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="输入 JSON 文件路径（MCP servers 列表）")
    parser.add_argument("--out", dest="out_path", required=True, help="输出 JSON 文件路径")
    # parser.add_argument("--num-queries", type=int, default=2324, help="生成查询数量 (default: ALL)")
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"输入文件不存在: {args.in_path}")

    # Load servers (support both JSON and NDJSON)
    with open(args.in_path, "r", encoding="utf-8") as f:
        if args.in_path.endswith('.ndjson'):
            servers = [json.loads(line) for line in f if line.strip()]
        else:
            servers = json.load(f)

    if not isinstance(servers, list):
        raise ValueError("输入 JSON 须为列表，每个元素为一个 MCP server 的字典。")

    # Load all tool descriptions
    tool_descriptions_path = r"C:\Users\jessi\OneDrive\Documents\GitHub\MCP-R\MCP_INFO_MGR\mcp_data\indexed\tool_descriptions.ndjson"
    # tool_descriptions_path = "MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson"
    print(f"Loading tool descriptions from {tool_descriptions_path}...")

    with open(tool_descriptions_path, "r", encoding="utf-8") as f:
        all_tools = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_tools)} servers with tools")
    #
    # Generate queries concurrently, each with freshly shuffled tools
    print(f"Generating queries concurrently (max {MAX_CONCURRENT} at a time)...")
    queries = await generate_queries_async(all_tools)

    result = {
        "metadata": {
            "total_queries": len(queries),
            "servers_count": len(servers),
            "generation_method": "async_with_each_tool_query_separately"
        },
        "queries": queries
    }

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(queries)} queries and saved to {args.out_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
