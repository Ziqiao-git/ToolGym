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

MODEL_NAME = "openai/gpt-4o-mini"
MAX_RETRIES = 3
MAX_CONCURRENT = 5  # Maximum concurrent API calls


# -------- 改动 1：SYSTEM_PROMPT（新增 reference_tools 和 constraints 要求） --------
SYSTEM_PROMPT = """You are an expert at creating natural, realistic user queries that involve complex planning with multiple constraints.

Goal:
Given a list of available MCP servers, generate ONE realistic, multi-step natural language query that users might actually ask, where solving it would require:
1. Combining information or functions from DIFFERENT MCP servers (2-4 tools from distinct servers)
2. Navigating both HARD CONSTRAINTS (must satisfy) and SOFT CONSTRAINTS (preferred/optimized)

In servers_summary, each entry follows this format:
<server_namespace>/<server_name>/<tool_name>: <tool_description>

CONSTRAINT REQUIREMENTS:
The query MUST include realistic planning constraints that an agent needs to consider:

**HARD CONSTRAINTS** (at least 2 - must be satisfied, non-negotiable):
- Budget limits (e.g., "under $5,000", "API calls limited to 1000/day")
- Deadlines (e.g., "by end of Q1 2025", "within 2 weeks")
- Regulatory/compliance (e.g., "GDPR compliant", "must use approved vendors only")
- Technical requirements (e.g., "must support real-time updates", "99.9% uptime SLA")
- Resource limits (e.g., "team of 3 developers max", "existing infrastructure only")
- Geographic/legal (e.g., "US-based services only", "data must stay in EU")

**SOFT CONSTRAINTS** (at least 2 - preferences to optimize, can trade off):
- Performance preferences (e.g., "prefer faster response time", "optimize for throughput")
- Cost optimization (e.g., "minimize recurring costs where possible")
- Quality preferences (e.g., "prefer higher accuracy if feasible")
- Convenience (e.g., "easier onboarding preferred", "minimal training time")
- Feature priorities (e.g., "nice to have advanced analytics", "prefer modern UI")
- Risk preferences (e.g., "prefer established vendors", "minimize dependencies")

The query should present a scenario where the agent must:
- Make trade-offs between competing soft constraints
- Ensure all hard constraints are met
- Balance priorities across multiple dimensions

Additionally, return a short list of reference tools that the agent would likely use to answer the query.

IMPORTANT SELECTION RULES:
- Only use tools that appear in the provided servers_summary (use exact names).
- Pick 2–4 tools MAX.
- ✅ Each selected tool MUST come from a **different MCP server namespace**.
- The “server” field refers to the **ignore the last path components** (e.g., `plainyogurt21/clintrials-mcp`).
- ❌ You may NOT include multiple tools whose server prefix (the part before the third “/”) is the same.
- Example of NOT allowed:
  1. 
  - plainyogurt21/clintrials-mcp/search_trials_by_condition
  - plainyogurt21/clintrials-mcp/get_trial_details_batched
  (⛔ Both share the same server prefix `plainyogurt21/clintrials-mcp`, only count for one distinct server, but we need at lest 2 distinct servers)
  2. 
  - plainyogurt21/clintrials-mcp
  - plainyogurt21/clintrials-mcp
  (⛔ Both share the same server prefix `plainyogurt21/clintrials-mcp`, only count for one distinct server, but we need at lest 2 distinct servers)
  3. 
  - plainyogurt21/clintrials-mcp
  - plainyogurt21/clintrials-mcp/get_trial_details_batched
  (⛔ Both share the same server prefix `plainyogurt21/clintrials-mcp`, only count for one distinct server, but we need at lest 2 distinct servers)
- Example of allowed:
  - plainyogurt21/clintrials-mcp/search_trials_by_condition
  - smithery-ai/national-weather-service/get_weather_forecast
  (✅ Different server prefixes)
  - plainyogurt21/clintrials-mcp
  - smithery-ai/national-weather-service
  (✅ Different server prefixes)

Each selected tool must serve a distinct, complementary role in the overall workflow.

For each tool, include:
- "server": the MCP server prefix only (the first two parts, e.g., `plainyogurt21/clintrials-mcp`)
- "tool": the tool name (the third part)
- "why": a one-sentence rationale

⚠️ **CONTENT RESTRICTIONS**
- Never include personal, financial, medical, or confidential data:
  - No credit cards, bank accounts, passwords, private health info, or PII.
- Keep all queries feasible within public, safe contexts (e.g., news, research, weather, travel, tech, public APIs).
- Avoid speculative or illegal content.
- Avoid medical/clinical trial scenarios unless they are open, public, and anonymized datasets.

QUERY REQUIREMENTS:
- Must sound natural, fluent, and genuinely useful.
- Must require combining information or operations from tools in *different MCP servers* (distinct prefixes).
- Each tool must add unique, necessary information to the overall task.
- Be specific, concrete, and realistic.
- Avoid meta or instructional phrasing.
- The task should make sense as something a real person or researcher might request.
- MUST explicitly state both hard constraints (requirements) and soft constraints (preferences/optimizations).
- Constraints should create realistic trade-offs and planning challenges.

CONSTRAINT EXAMPLES IN QUERIES:

Example 1 (Project Planning):
"We need to build a customer analytics dashboard by March 15th [HARD: deadline] with a budget under $8,000 [HARD: budget]. The solution must be GDPR compliant [HARD: regulatory] and integrate with our existing Postgres database [HARD: technical]. We'd prefer a solution with minimal setup time [SOFT: convenience], prioritize data visualization quality over advanced ML features [SOFT: feature priority], and would like to minimize ongoing maintenance costs [SOFT: cost optimization]."

Example 2 (Infrastructure Selection):
"Select cloud hosting for our API service that guarantees 99.9% uptime [HARD: SLA], supports auto-scaling [HARD: technical], costs less than $500/month [HARD: budget], and uses US-based data centers only [HARD: geographic]. Prefer faster deployment over feature richness [SOFT: speed vs features], optimize for ease of DevOps integration [SOFT: convenience], and minimize vendor lock-in where possible [SOFT: risk]."

Example 3 (Research Task):
"Conduct market analysis for launching in EU by Q2 2025 [HARD: deadline] with research budget capped at $3,000 [HARD: budget]. Must cover Germany, France, and Spain [HARD: scope]. Data sources must be publicly available [HARD: compliance]. Prefer more recent data (2024+) over historical [SOFT: recency], prioritize quantitative data if available [SOFT: data type], and favor established research firms for credibility [SOFT: quality]."

OUTPUT FORMAT:
Return a valid JSON object:
{
  "query": "<the realistic user query>",
  "reference_tools": [
    { "server": "<server_namespace/server_name>", "tool": "<tool_name>", "why": "<short rationale>" },
    ...
  ]
}

❌ DO NOT repeat the same 'server' prefix across different entries.
❌ DO NOT output servers like "@smithery-ai/national-weather-service/get_weather_forecast" in the "server" field, since @smithery-ai/national-weather-service is server name and get_weather_forecast is tool name.
✅ The "server" field should stop at the server name level (e.g., "@smithery-ai/national-weather-service").

---

### ✅ SELF-CHECK BEFORE RETURN
Before producing final JSON:
1. Extract all `server` prefixes (the first two path parts).  
2. Verify that there are **at least two distinct servers** among them.  
   - If fewer than two unique servers → **rewrite the query** and **reselect tools** until this condition is met.  
3. Ensure all servers are distinct and exist in the provided list.  
4. Ensure the query avoids personal/financial/medical data.  
5. Ensure output strictly follows JSON schema above.  

Only output once all checks pass.

---

Examples of good constraint-based queries:

Example 1:
"Plan a tech conference for 200 attendees by June 2025 [HARD: deadline, capacity] within a $50,000 budget [HARD: budget]. Must find venues in San Francisco or Austin [HARD: location], with availability on weekends [HARD: scheduling]. Prefer venues with modern AV equipment [SOFT: quality], prioritize proximity to airports over downtown locations [SOFT: trade-off], and would like backup date options if possible [SOFT: flexibility]. Use weather data to check seasonal patterns and GitHub API to identify trending tech topics for the agenda."

Example 2:
"Research sustainable energy solutions for a manufacturing facility by Q1 2025 [HARD: deadline] with capital budget under $2M [HARD: budget]. Must achieve 30% carbon reduction [HARD: performance] and comply with state regulations [HARD: compliance]. Prefer solutions with faster ROI [SOFT: financial], prioritize proven technology over experimental [SOFT: risk], and favor options requiring minimal operational changes [SOFT: convenience]. Cross-reference environmental data APIs with cost calculators and regulatory databases."

Example 3:
"Select developer tools for a remote team of 15 by end of month [HARD: deadline, team size] spending max $5,000/year [HARD: budget]. Must integrate with Slack and GitHub [HARD: technical integration] and provide real-time collaboration [HARD: feature requirement]. Prefer tools with shorter learning curve [SOFT: usability], optimize for fewer total tools over feature completeness [SOFT: simplicity vs features], and favor vendors with strong security track record [SOFT: risk mitigation]. Query software comparison APIs, pricing databases, and integration catalogs."

Make sure all reference tools come from distinct MCP servers (different prefixes before the third '/').

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
servers_summary:
{servers_summary}
For each server, the format is: server_name/tool_name: tool_description. Did not include tool name in server name for output

Generate 1 natural, realistic user query with CONSTRAINT-BASED PLANNING that people would actually ask.

REQUIREMENTS:
1. Query must require 2-4 tools from DIFFERENT MCP servers
2. Include at least 2 HARD CONSTRAINTS (must satisfy - e.g., budget, deadline, compliance, technical requirements)
3. Include at least 2 SOFT CONSTRAINTS (preferences to optimize - e.g., performance, cost, quality, convenience)
4. Create realistic trade-offs where the agent must balance competing priorities
5. Make it sound like a real professional/business scenario

Use concrete, specific values for constraints (e.g., "$5,000 budget", "by March 2025", "team of 10", "99% uptime").

Return a JSON object with "query" and "reference_tools" fields.
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
                temperature=0.8,  # Higher for more diverse constraint scenarios
                max_tokens=2500,  # More tokens for detailed constraints
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
    parser.add_argument("--in", dest="in_path", required=True, help="Working servers config JSON file")
    parser.add_argument("--out", dest="out_path", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--num-queries", type=int, default=20, help="生成查询数量 (default: 20)")
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"输入文件不存在: {args.in_path}")

    # Load working server configs (JSON dict format)
    print(f"Loading working servers from {args.in_path}...")
    with open(args.in_path, "r", encoding="utf-8") as f:
        working_servers_config = json.load(f)

    if not isinstance(working_servers_config, dict):
        raise ValueError("Working servers config should be a JSON object (dict).")

    working_server_names = set(working_servers_config.keys())
    print(f"Loaded {len(working_server_names)} working servers")

    # Load all tool descriptions
    tool_descriptions_path = "MCP_INFO_MGR/mcp_data/working/tool_descriptions.ndjson"
    print(f"Loading tool descriptions from {tool_descriptions_path}...")

    with open(tool_descriptions_path, "r", encoding="utf-8") as f:
        all_tools = [json.loads(line) for line in f if line.strip()]

    # Filter to only working servers
    all_tools = [t for t in all_tools if t.get("qualifiedName") in working_server_names]

    print(f"Filtered to {len(all_tools)} working servers with tools")

    # Generate queries concurrently, each with freshly shuffled tools
    print(f"Generating {args.num_queries} queries concurrently (max {MAX_CONCURRENT} at a time)...")
    print(f"Each query will use a different random sample of 100 tools")
    queries = await generate_queries_async(all_tools, num_queries=args.num_queries)

    # -------- 改动 3（输出结构）：由 "queries": [...] 改为 "items": [...] --------
    result = {
        "metadata": {
            "total_items": len(queries),
            "servers_count": len(all_tools),
            "working_servers_count": len(working_server_names),
            "generation_method": "async_with_per_query_shuffle_constrained_planning",
            "constraint_types": {
                "hard": ["budget", "deadline", "regulatory", "technical", "resource", "geographic"],
                "soft": ["performance", "cost_optimization", "quality", "convenience", "features", "risk"]
            }
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
