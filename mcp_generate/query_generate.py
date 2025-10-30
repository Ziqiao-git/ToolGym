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
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import random

# OpenAI 官方 Python SDK
# pip install --upgrade openai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(str(SCRIPT_DIR / ".env"))

CLIENT = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
) 

MODEL_NAME = "openai/gpt-5" 
MAX_RETRIES = 3


SYSTEM_PROMPT = """You are an expert at creating natural, realistic user queries that involve research or multi-step tasks.

Goal: Given a list of available MCP servers, generate {num_queries} natural language queries that users might realistically ask, where answering would naturally involve using information from different sources.

Requirements:
- Write natural, fluent questions that real users would ask
- Questions should involve research, comparison, or connecting different types of information
- Be concrete and specific (mention actual topics, technologies, companies, locations, etc.)
- Questions should sound like genuine user requests, not forced "use multiple tools" prompts
- Output must be a valid JSON object with "queries" array containing exactly {num_queries} strings

Examples of natural queries:
- "What are the latest developments in AI regulation and which companies are most affected?"
- "Find recent machine learning projects on GitHub and tell me what problems they solve"
- "What's the weather like in Seattle and are there any tech events happening there?"
- "Show me trending Python projects and explain what makes them popular"

Make queries natural, diverse, and genuinely useful.
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

Generate {num_queries} natural, realistic user queries that people would actually ask.
These queries should naturally involve researching or combining information from different sources.

Return a JSON object with "queries" array containing exactly {num_queries} strings.
"""



def call_gpt5_for_queries(tools_summary: str, num_queries: int = 20) -> List[str]:
    """
    Generate natural language queries using GPT-5 based on tool descriptions.

    Args:
        tools_summary: Text summary of available tools
        num_queries: Number of queries to generate (default: 20)

    Returns:
        List of generated query strings
    """
    system_prompt = SYSTEM_PROMPT.format(num_queries=num_queries)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        servers_summary=tools_summary,
        num_queries=num_queries
    )

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "queries_schema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "queries": {
                        "type": "array",
                        "minItems": num_queries,
                        "maxItems": num_queries,
                        "items": {"type": "string"}
                    }
                },
                "required": ["queries"]
            },
            "strict": True
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        resp = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=20000,
            response_format=schema,
        )
        text = resp.choices[0].message.content.strip()
        obj = json.loads(text)

        if not (isinstance(obj, dict) and isinstance(obj.get("queries"), list)
                and len(obj["queries"]) == num_queries
                and all(isinstance(x, str) for x in obj["queries"])):
            raise ValueError(f"Model did not return an object with 'queries': [{num_queries} strings].")

        return obj["queries"]




def main():
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

    # Load tool descriptions from indexed data
    tool_descriptions_path = "MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson"
    print(f"Loading tool descriptions from {tool_descriptions_path}...")

    tools_summary = []
    # Shuffle tools to get diverse set
    with open(tool_descriptions_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    for tool in lines:
        tools = tool.get("tools", [])
        random.shuffle(tools)
        tool["tools"] = tools
    random.shuffle(lines)

    count = 0
    servers_loaded = 0
    for tool in lines:
        server = tool.get("qualifiedName", "unknown")
        servers_loaded += 1
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

    # Generate queries
    print(f"Generating {args.num_queries} natural cross-domain queries from {servers_loaded} servers ({len(tools_summary)} tools)...")
    queries = call_gpt5_for_queries(tools_summary_text, num_queries=args.num_queries)

    result = {
        "metadata": {
            "total_queries": len(queries),
            "servers_count": len(servers)
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


if __name__ == "__main__":
    main()
