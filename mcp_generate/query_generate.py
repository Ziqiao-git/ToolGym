#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_generate.py

使用 GPT-5 基于 MCP 工具描述生成自然语言查询问题。
用法:
  python query_generate.py --in tool_descriptions.ndjson --out generated_queries.json --num-queries 20
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


# -------- 改动 1：SYSTEM_PROMPT（新增 reference_tools 要求） --------
SYSTEM_PROMPT = """You are an expert at creating natural, realistic user queries that involve research or multi-step tasks.

Goal: Given a list of available MCP servers, generate 1 natural language query that users might realistically ask, where answering would naturally involve using information from different sources.

Additionally, return a short list of reference tools the agent would likely use to answer the query. IMPORTANT:
- Only choose tools that appear in the provided servers_summary (use the exact server/tool names).
- Pick 2–4 tools MAX that are most relevant to solve the query end-to-end.
- For each tool, include a one-sentence rationale ("why").
- Do NOT invent tools or servers.

Requirements for the query:
- Natural, fluent, genuinely useful; sounds like a real user request
- Involves research, comparison, or connecting different types of information
- Be concrete and specific, but avoid meta/instructional phrasing
- Output must be a valid JSON object with fields:
  - "query": string
  - "reference_tools": array of { "server": string, "tool": string, "why": string }

Examples of natural queries:
- "What are the latest developments in AI regulation and which companies are most affected?"
- "Find recent machine learning projects on GitHub and tell me what problems they solve"
- "What's the weather like in Seattle and are there any tech events happening there?"
- "Show me trending Python projects and explain what makes them popular"

Make the query natural, diverse, and genuinely useful; choose only the most relevant tools from the list.
"""

# SYSTEM_PROMPT = """You create concrete, realistic, English benchmark questions for MCP servers.

# Goal:
# Given an MCP server (name, summary, tools), generate 5 distinct natural-language questions that would realistically trigger its tool usage and cover its capabilities.

# Hard requirements:
# - Output MUST be a valid JSON array with exactly 5 strings.
# - Each question MUST be SPECIFIC and OPERATIONAL (include concrete entities like a URL, project name, table, metric, time range, top-k, etc. as appropriate).
# - If multiple tools exist, cover as many different tools as possible (at least one per tool). If only one tool exists, create 5 diverse scenarios.
# - Vary intents: listing, filtering, top-k, troubleshooting/health, summarization/aggregation, targeted lookup, etc.
# - DO NOT use meta/instructional phrasing. Avoid generic templates.

# Banned phrases/patterns (must NOT appear):
# - "Based on the functionality of"
# - "Show me a representative example"
# - "If this service has a listing or browsing feature"
# - "Pick a typical query scenario"
# - "Demonstrate a troubleshooting or health check"
# - "Explain what it would return"
# - Any question that does not name specific entities (e.g., a URL, project name, table, service, or metric) when such entities are typical for the tool.

# Style guidance:
# - Questions should sound like real user requests with concrete details.
# - Use plausible, public examples when needed (e.g., websites for a web extractor; project names like analytics-prod/demo-project; tables like orders, events, sensor_data; time ranges like "last 24 hours").

# Few-shot examples:

# [EXAMPLE — Server: AgentQL (extract structured data from web)]
# Input tools: ["extract-web-data(url, prompt)"]
# Good outputs:
# [
#   "From https://news.ycombinator.com, extract the top 10 posts with title, link, and points.",
#   "From https://arxiv.org/list/cs.AI/recent, pull paper titles, authors, and submission dates for the most recent 15 entries.",
#   "From https://github.com/trending, list the first 8 repositories with name, primary language, and star count.",
#   "From https://www.nytimes.com, extract the homepage headlines and their article URLs under the 'Top Stories' section.",
#   "From https://openreview.net/group?id=NeurIPS.cc/2025/Conference, collect accepted paper titles with author list and decision notes if present."
# ]

# [EXAMPLE — Server: Aiven (list_projects, list_services, get_service_details)]
# Good outputs:
# [
#   "List all projects in my Aiven account.",
#   "In the project analytics-prod, list all running services and their service types.",
#   "For the service clickhouse-main in project analytics-prod, show its connection details and current status.",
#   "Do I have any Kafka services in the project demo-project? If yes, list their service names and versions.",
#   "For the PostgreSQL service pg-orders in project demo-project, show storage size and CPU/memory utilization over the last 24 hours if available."
# ]

# [EXAMPLE — Server: AnalyticDB for MySQL (execute_sql, get_query_plan)]
# Good outputs:
# [
#   "Run: SELECT customer_id, COUNT(*) AS orders FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL 30 DAY GROUP BY customer_id ORDER BY orders DESC LIMIT 10.",
#   "What is the query plan for: SELECT * FROM events WHERE event_time >= NOW() - INTERVAL 1 HOUR AND user_id = 42 ORDER BY event_time DESC LIMIT 100?",
#   "Execute: SHOW TABLES.",
#   "Execute: DESCRIBE sales;",
#   "Run: SELECT category, SUM(amount) AS revenue FROM sales WHERE ts BETWEEN '2025-08-01' AND '2025-08-31' GROUP BY category ORDER BY revenue DESC LIMIT 5."
# ]
# """



USER_PROMPT_TEMPLATE = """Here are the available MCP servers grouped by domain:

{servers_summary}

Generate 1 natural, realistic user query that people would actually ask.
This query should naturally involve researching or combining information from different sources.

Return a JSON object with "query" field containing a single string.
"""



async def generate_single_query(all_tools: List[Dict], query_idx: int) -> Any:
    """
    Generate a single query with freshly shuffled tools.

    Args:
        all_tools: List of all tool descriptions
        query_idx: Index of this query (for logging)

    Returns:
        Generated object: {"query": str, "reference_tools": [...]}
    """
    # Shuffle tools for this specific query
    tools_copy = [tool.copy() for tool in all_tools]
    for tool in tools_copy:
        tools = tool.get("tools", [])
        random.shuffle(tools)
        tool["tools"] = tools
    random.shuffle(tools_copy)

    # Select top 100 tools
    tools_summary = []
    count = 0
    for tool in tools_copy:
        server = tool.get("qualifiedName", "unknown")
        for t in tool.get("tools", []):
            if count >= 100:
                break
            tool_name = t.get("name", "")
            tool_desc = t.get("description", "No description")
            tools_summary.append(f"- {server}/{tool_name}: {tool_desc}")
            count += 1
        if count >= 100:
            break

    tools_summary_text = "\n".join(tools_summary)

    # Generate query
    user_prompt = USER_PROMPT_TEMPLATE.format(servers_summary=tools_summary_text)

    # -------- 改动 2：response_format schema（新增 reference_tools 字段） --------
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "query_with_refs_schema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "reference_tools": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 4,
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
                    return {"query": f"[Error: Empty response after {MAX_RETRIES} attempts]", "reference_tools": []}

            text = text.strip()

            try:
                obj = json.loads(text)
            except json.JSONDecodeError as je:
                print(f"⚠️  Query {query_idx}: JSON decode error, retrying... (attempt {attempt}/{MAX_RETRIES})")
                print(f"     Response preview: {text[:100]}...")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    return {"query": f"[Error: Invalid JSON after {MAX_RETRIES} attempts]", "reference_tools": []}

            # -------- 改动 3（解析校验）：必须同时包含 query 与 reference_tools --------
            if not (isinstance(obj, dict)
                    and isinstance(obj.get("query"), str)
                    and isinstance(obj.get("reference_tools"), list)):
                print(f"⚠️  Query {query_idx}: Invalid format, retrying... (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    return {"query": f"[Error: Invalid format after {MAX_RETRIES} attempts]", "reference_tools": []}

            return {
                "query": obj["query"],
                "reference_tools": obj["reference_tools"]
            }

        except Exception as e:
            print(f"⚠️  Query {query_idx}: Unexpected error: {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"     Retrying... (attempt {attempt}/{MAX_RETRIES})")
                await asyncio.sleep(2)  # Longer sleep for unexpected errors
            else:
                print(f"     Failed after {MAX_RETRIES} attempts")
                return {"query": f"[Error: {type(e).__name__} after {MAX_RETRIES} attempts]", "reference_tools": []}


async def generate_queries_async(all_tools: List[Dict], num_queries: int) -> List[Any]:
    """
    Generate multiple queries concurrently, each with freshly shuffled tools.

    Args:
        all_tools: List of all tool descriptions
        num_queries: Number of queries to generate

    Returns:
        List of generated objects: [{"query": str, "reference_tools": [...]}, ...]
    """
    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def generate_with_semaphore(idx):
        async with semaphore:
            return await generate_single_query(all_tools, idx)

    # Generate all queries concurrently
    tasks = [generate_with_semaphore(i) for i in range(num_queries)]
    queries = []

    # Use tqdm for progress bar
    for coro in tqdm(asyncio.as_completed(tasks), total=num_queries, desc="Generating queries"):
        query = await coro
        queries.append(query)

    return queries




async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="输入 JSON 文件路径（MCP servers 列表）")
    parser.add_argument("--out", dest="out_path", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--num-queries", type=int, default=20, help="生成查询数量 (default: 20)")
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
    tool_descriptions_path = "MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson"
    print(f"Loading tool descriptions from {tool_descriptions_path}...")

    with open(tool_descriptions_path, "r", encoding="utf-8") as f:
        all_tools = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_tools)} servers with tools")

    # Generate queries concurrently, each with freshly shuffled tools
    print(f"Generating {args.num_queries} queries concurrently (max {MAX_CONCURRENT} at a time)...")
    print(f"Each query will use a different random sample of 100 tools")
    queries = await generate_queries_async(all_tools, num_queries=args.num_queries)

    # -------- 改动 3（输出结构）：由 "queries": [...] 改为 "items": [...] --------
    result = {
        "metadata": {
            "total_items": len(queries),
            "servers_count": len(servers),
            "generation_method": "async_with_per_query_shuffle"
        },
        "items": queries
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
