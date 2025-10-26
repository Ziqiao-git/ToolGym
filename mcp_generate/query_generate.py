#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_mcp_queries.py

用 GPT-5 为每个 MCP server 生成 5 个问题，并把结果写入新的 JSON。
用法:
  python generate_mcp_queries.py --in servers.json --out servers_with_queries.json
"""

import json
import argparse
import os
import time
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path

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


SYSTEM_PROMPT = """You are an expert at creating natural language benchmark questions for MCP servers.
Goal: For the given MCP server (with name / summary / tools), generate 5 natural language queries that would realistically trigger its tool usage and cover its capabilities.

Requirements:
- Write 5 distinct, fluent, natural English questions (not commands).
- Cover as many different tools as possible (if multiple).
- Each question should correspond to a realistic user request (summaries, lookups, troubleshooting, top-k, filtering, etc.).
- Output must be a valid JSON array containing ·exactly 5 strings.
- Do not include any explanation, only the JSON array.
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



USER_PROMPT_TEMPLATE = """Here is an MCP server. 
Generate 5 concrete, realistic ENGLISH questions as a JSON array of EXACTLY 5 strings.
Remember the hard requirements and banned phrases.
server:
{server_json}
"""



def call_gpt5_for_queries(server: Dict[str, Any]) -> List[str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(server_json=json.dumps(server, ensure_ascii=False, indent=2))

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
                        "minItems": 5,
                        "maxItems": 5,
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=20000,
            response_format=schema, 
        )
        text = resp.choices[0].message.content.strip()
        obj = json.loads(text)

        if not (isinstance(obj, dict) and isinstance(obj.get("queries"), list)
                and len(obj["queries"]) == 5
                and all(isinstance(x, str) for x in obj["queries"])):
            raise ValueError("Model did not return an object with 'queries': [5 strings].")

        return obj["queries"]




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="输入 JSON 文件路径（MCP servers 列表）")
    parser.add_argument("--out", dest="out_path", required=True, help="输出 JSON 文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"输入文件不存在: {args.in_path}")

    with open(args.in_path, "r", encoding="utf-8") as f:
        servers = json.load(f)

    if not isinstance(servers, list):
        raise ValueError("输入 JSON 须为列表，每个元素为一个 MCP server 的字典。")

    result: List[Dict[str, Any]] = []
    i = 0
    for server in tqdm(servers, desc="Generating queries for MCP servers"):
        # 仅提取必要字段传给模型，避免噪声；但输出保留完整 server 信息
        minimal_server = {
            "name": server.get("name"),
            "summary": server.get("summary") or server.get("description"),
            "tools": server.get("tools"),
        }
        queries = call_gpt5_for_queries(minimal_server)

        # 合并到新结构
        item = dict(server)  # 原样复制 server 所有字段
        item["generated_queries"] = queries
        result.append(item)

        i+=1


  


    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 已写入: {args.out_path}")


if __name__ == "__main__":
    main()
