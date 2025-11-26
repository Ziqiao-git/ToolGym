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
from datetime import datetime

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
SYSTEM_PROMPT_TEMPLATE = """CURRENT DATE AND TIME: {current_time}

You are an expert at creating natural, realistic user queries that involve complex planning with multiple constraints.

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
- Only use tools from the provided servers_summary (use exact names)
- Pick 2-4 tools MAX
- **CRITICAL: Each tool MUST come from a DIFFERENT server** (different server name before the last "/")
  * Example: ✅ `plainyogurt21/clintrials-mcp` and `smithery-ai/national-weather-service` are different servers
  * Example: ❌ `plainyogurt21/clintrials-mcp/search_trials` and `plainyogurt21/clintrials-mcp/get_details` are from the SAME server (NOT ALLOWED)
  * Example: ❌ `@vijitdaroch/financial-modeling-prep-mcp-server/getFundHoldings` and `@vijitdaroch/financial-modeling-prep-mcp-server/getFundInfo` are from the SAME server (NOT ALLOWED)
- Each tool should serve a distinct, complementary role in the workflow
- Tools should make sense together - avoid nonsensical combinations (e.g., Figma design + weather API for app development makes no sense)

For each tool, include:
- "server": the MCP server prefix ONLY (e.g., `plainyogurt21/clintrials-mcp`) - **DO NOT include the tool name in the server field**
- "tool": the tool name ONLY (the last part, e.g., `search_trials`)
- "why": a one-sentence rationale

**CRITICAL FORMAT CHECK**:
- ❌ WRONG: {{"server": "foo/bar-server/get_data", "tool": "get_data"}} - tool name is in server field!
- ✅ CORRECT: {{"server": "foo/bar-server", "tool": "get_data"}}

QUERY REQUIREMENTS:
- Must sound natural, fluent, and genuinely useful.
- Must require combining information or operations from tools in *different MCP servers* (distinct prefixes).
- Each tool must add unique, necessary information to the overall task.
- **CRITICAL: Include SPECIFIC, CONCRETE details that the agent can actually query**:
  * Specific company names, stock tickers (e.g., "Tesla (TSLA)" not "a tech company")
  * Exact locations with city/state (e.g., "San Francisco, CA" not "the event location")
  * Specific dates (e.g., "July 15, 2025" not "the scheduled date")
  * Concrete data identifiers the tools can use - NO vague references
- **NEVER mention tools or tell the agent which tools to use**: Don't say "use the X tool" or "call the Y API" - the agent should figure out which tools to use on its own
- Avoid meta or instructional phrasing
- The task should make sense as something a real person or researcher might request
- MUST explicitly state both hard constraints (requirements) and soft constraints (preferences/optimizations)
- Constraints should create realistic trade-offs and planning challenges

CONSTRAINT EXAMPLES:

IMPORTANT: Be creative with constraint types! Don't limit yourself to predefined categories. Use natural, descriptive names that capture real-world planning challenges.

Example 1 (Travel Planning - showing diverse constraint types):
Query: "Plan a 5-day family trip to Tokyo for 4 people departing from San Francisco on March 20, 2026. We need direct flights, family-friendly hotels in Shibuya or Shinjuku, and restaurant recommendations near major attractions. One family member has a shellfish allergy. We want to visit TeamLab Borderless and the Ghibli Museum."

Hard Constraints:
- {{"type": "time_window", "description": "Must depart March 20, 2026 and return March 25, 2026"}}
- {{"type": "party_size", "description": "Accommodation and activities must support 4 people"}}
- {{"type": "dietary_restriction", "description": "Must avoid shellfish due to allergy"}}
- {{"type": "flight_preference", "description": "Only direct flights from SFO to Tokyo"}}

Soft Constraints:
- {{"type": "neighborhood_preference", "description": "Prefer hotels in Shibuya or Shinjuku districts"}}
- {{"type": "family_friendly", "description": "Prioritize child-appropriate activities and venues"}}
- {{"type": "proximity", "description": "Prefer restaurants near major tourist attractions"}}
- {{"type": "cultural_experience", "description": "Include visits to TeamLab and Ghibli Museum if available"}}

Example 2 (Event Planning - creative constraints):
Query: "Organize a 200-person tech conference in Austin, TX for June 15-17, 2026 with keynote speakers from the AI industry. Need venue with stage, A/V equipment, and breakout rooms. Catering must include vegan options. We have $75,000 total budget."

Hard Constraints:
- {{"type": "venue_capacity", "description": "Must accommodate 200 attendees"}}
- {{"type": "date_range", "description": "June 15-17, 2026 - all 3 days required"}}
- {{"type": "equipment_requirements", "description": "Stage, professional A/V, and 3+ breakout rooms"}}
- {{"type": "budget_ceiling", "description": "Total costs cannot exceed $75,000"}}
- {{"type": "dietary_accommodation", "description": "Catering must include vegan meal options"}}

Soft Constraints:
- {{"type": "speaker_profile", "description": "Prefer keynote speakers from AI/ML industry"}}
- {{"type": "location_convenience", "description": "Venue close to airport or downtown Austin"}}
- {{"type": "vendor_reputation", "description": "Prefer established catering companies with tech event experience"}}
- {{"type": "weather_consideration", "description": "Indoor backup plan for any outdoor sessions"}}

Example 3 (Investment Analysis - domain-specific):
Query: "Evaluate cryptocurrency investments for a $50,000 portfolio by December 2025. Must include Bitcoin and Ethereum. Looking for assets with market cap over $1B and 24h trading volume above $100M. Want to understand recent price trends and news sentiment."

Hard Constraints:
- {{"type": "portfolio_size", "description": "Total investment $50,000"}}
- {{"type": "asset_inclusion", "description": "Must include BTC and ETH"}}
- {{"type": "market_cap_floor", "description": "Only assets with $1B+ market capitalization"}}
- {{"type": "liquidity_requirement", "description": "24-hour trading volume must exceed $100M"}}

Soft Constraints:
- {{"type": "trend_analysis", "description": "Prefer assets showing positive 30-day momentum"}}
- {{"type": "sentiment_consideration", "description": "Factor in recent news sentiment for decision-making"}}
- {{"type": "diversification", "description": "Spread across different crypto sectors (DeFi, L1, L2)"}}
- {{"type": "volatility_tolerance", "description": "Accept moderate volatility for higher potential returns"}}

OUTPUT FORMAT:
Return a valid JSON object with separate query and constraint fields:
{{
  "query": "<the realistic user query WITHOUT constraint labels>",
  "hard_constraints": [
    {{"type": "<constraint_category>", "description": "<what must be satisfied>"}},
    ...
  ],
  "soft_constraints": [
    {{"type": "<constraint_category>", "description": "<what is preferred/optimized>"}},
    ...
  ],
  "reference_tools": [
    {{"server": "<server_namespace/server_name>", "tool": "<tool_name>", "why": "<short rationale>"}},
    ...
  ]
}}

IMPORTANT: The "query" field should contain natural language WITHOUT [HARD:] or [SOFT:] labels - those belong in the separate constraint arrays.

CONSTRAINT TYPE CREATIVITY:
- Don't just use "budget" and "deadline" - be creative and domain-specific!
- Travel: time_conflict, layover_limits, visa_requirements, luggage_restrictions, jet_lag_consideration
- Finance: risk_tolerance, liquidity_needs, tax_efficiency, correlation_limits, drawdown_threshold
- Events: noise_restrictions, parking_capacity, ADA_compliance, insurance_requirements, cancellation_policy
- Research: citation_recency, peer_review_requirement, dataset_size, reproducibility_standard
- Real constraints people face: allergies, timezone_compatibility, language_barriers, skill_prerequisites
- Make constraint types read naturally and describe what they actually constrain!

SELF-CHECK BEFORE RETURN:
1. Verify you selected tools from at least 3 DIFFERENT servers
2. Ensure all tools exist in the provided servers_summary
3. **CRITICAL: Verify query contains SPECIFIC, CONCRETE details (company names, locations, dates, tickers, repo names, etc.) - NO VAGUE REFERENCES**
4. **CRITICAL: Verify query does NOT mention tool names or tell agent which tools to use** - the agent should figure it out
5. **CREATIVE CONSTRAINTS: Use domain-appropriate, natural constraint types - avoid generic ones where specific ones fit better**
6. Ensure the query avoids personal/financial/medical data
7. Ensure output follows the JSON schema exactly

---

Examples of SPECIFIC constraint-based queries (showing query text WITHOUT labels):

Example 1 (with specific location and dates):
Query: "Plan a tech conference for 200 attendees by June 15, 2026 within a $50,000 budget. Must find venues in San Francisco, CA or Austin, TX with availability on weekends. Need weather forecast for San Francisco for June 12-14, 2026 to assess outdoor venue viability. Also need to identify trending topics in popular open-source repositories like microsoft/vscode, facebook/react, and vercel/next.js to inform the conference agenda. Prefer venues with modern AV equipment and prioritize proximity to SFO/AUS airports over downtown locations."

Example 2 (with specific stocks and data):
Query: "Compare investment performance of Tesla (TSLA), Apple (AAPL), and Microsoft (MSFT) for Q4 2024 with budget under $50,000 per position. Must achieve minimum 8% annual return and maintain portfolio volatility below 15%. Need historical price data and recent news affecting these tickers to make informed decisions. Prefer stocks with higher dividend yield, prioritize established companies over growth stocks, and favor companies with strong ESG ratings."

Example 3 (with specific tools and integrations):
Query: "Select project management tools for a remote team of 15 developers by January 31, 2026 spending max $5,000/year. Must integrate with Slack, GitHub, and Jira and provide real-time collaboration. Looking to evaluate Asana, Monday.com, and Linear - need their feature sets, API capabilities, and pricing information. Prefer tools with shorter learning curve, optimize for fewer total tools over feature completeness, and favor vendors with SOC 2 certification."

Make sure all reference tools come from distinct MCP servers and queries contain SPECIFIC identifiable data.
Remember: Extract constraints into separate hard_constraints and soft_constraints arrays - do NOT include [HARD:] or [SOFT:] labels in the query text.

"""


USER_PROMPT_TEMPLATE = """CURRENT DATE AND TIME: {current_time}

Here are the available MCP servers grouped by domain:
servers_summary:
{servers_summary}
For each server, the format is: server_name/tool_name: tool_description. Did not include tool name in server name for output

Generate 1 natural, realistic user query with CONSTRAINT-BASED PLANNING that people would actually ask.

REQUIREMENTS:
1. Query must require 3 or more tools from DIFFERENT MCP servers
2. Include at least 2 HARD CONSTRAINTS (absolute requirements that must be met)
3. Include at least 2 SOFT CONSTRAINTS (preferences to optimize or trade-offs to balance)
4. **BE CREATIVE WITH CONSTRAINT TYPES**: Use natural, domain-specific types that fit the scenario
   - Instead of just "budget": budget_ceiling, cost_per_unit, total_expenditure_cap, burn_rate_limit
   - Instead of just "deadline": delivery_date, time_window, response_time, completion_timeline
   - Think about real constraints: allergies, accessibility, compatibility, availability, capacity, licenses
5. Create realistic trade-offs where the agent must balance competing priorities
6. Make it sound like a real professional/business scenario
7. **CRITICAL: Include SPECIFIC, QUERYABLE details**:
   - Company names, stock tickers (e.g., "Tesla (TSLA)", "Apple (AAPL)")
   - Exact locations (e.g., "San Francisco, CA", "New York, NY")
   - Specific dates - use realistic future dates based on CURRENT DATE AND TIME above (e.g., if current date is Nov 2025, use dates like "December 15, 2025" or "Q1 2026", NOT past dates)
   - Named products, repositories, APIs (e.g., "microsoft/vscode", "Slack API")
   - Concrete identifiers the tools can actually query - NO VAGUE REFERENCES like "our company", "the location", "top products"

Use concrete, specific values for constraints and natural constraint type names that describe what they constrain.

**CRITICAL OUTPUT FORMAT**:
Return a JSON object with separate fields for query (WITHOUT constraint labels), hard_constraints, soft_constraints, and reference_tools.
- The "query" field should be natural language WITHOUT [HARD:] or [SOFT:] markers
- Extract all constraints into the hard_constraints and soft_constraints arrays
- Each constraint should have "type" (category like "budget", "deadline") and "description" (specific requirement)
"""



async def generate_single_query(all_tools: List[Dict], query_idx: int) -> Any:
    """
    Generate a single query with tools from 10 random servers, all mixed together.

    Args:
        all_tools: List of all tool descriptions
        query_idx: Index of this query (for logging)

    Returns:
        Generated object: {"query": str, "reference_tools": [...]}
    """
    # Randomly select 10 servers
    selected_servers = random.sample(all_tools, min(10, len(all_tools)))

    # Flatten all tools from the selected 10 servers into a single list
    flattened_tools = []
    for server in selected_servers:
        server_name = server.get("qualifiedName", "unknown")
        for tool in server.get("tools", []):
            tool_name = tool.get("name", "")
            tool_desc = tool.get("description", "No description")
            flattened_tools.append({
                "server": server_name,
                "tool_name": tool_name,
                "description": tool_desc
            })

    # Shuffle all tools together (mixing tools from different servers)
    random.shuffle(flattened_tools)

    # Format all tools for the prompt
    tools_summary = []
    for t in flattened_tools:
        tools_summary.append(f"- {t['server']}/{t['tool_name']}: {t['description']}")

    tools_summary_text = "\n".join(tools_summary)

    # Generate query with current time context
    current_time = datetime.now().strftime("%B %d, %Y %I:%M %p %Z")

    # Format both system and user prompts with current time
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(current_time=current_time)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        current_time=current_time,
        servers_summary=tools_summary_text
    )

    # -------- 改动 2：response_format schema（新增 reference_tools 和 constraints） --------
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "query_with_refs_schema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "hard_constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["type", "description"]
                        }
                    },
                    "soft_constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["type", "description"]
                        }
                    },
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
                "required": ["query", "hard_constraints", "soft_constraints", "reference_tools"]
            },
            "strict": True
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
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
                    return {
                        "query": f"[Error: Empty response after {MAX_RETRIES} attempts]",
                        "hard_constraints": [],
                        "soft_constraints": [],
                        "reference_tools": []
                    }

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
                    return {
                        "query": f"[Error: Invalid JSON after {MAX_RETRIES} attempts]",
                        "hard_constraints": [],
                        "soft_constraints": [],
                        "reference_tools": []
                    }

            # -------- 改动 3（解析校验）：必须同时包含 query, constraints, reference_tools --------
            if not (isinstance(obj, dict)
                    and isinstance(obj.get("query"), str)
                    and isinstance(obj.get("hard_constraints"), list)
                    and isinstance(obj.get("soft_constraints"), list)
                    and isinstance(obj.get("reference_tools"), list)):
                print(f"⚠️  Query {query_idx}: Invalid format, retrying... (attempt {attempt}/{MAX_RETRIES})")
                print(f"     Missing fields: query={isinstance(obj.get('query'), str)}, hard={isinstance(obj.get('hard_constraints'), list)}, soft={isinstance(obj.get('soft_constraints'), list)}, tools={isinstance(obj.get('reference_tools'), list)}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    return {
                        "query": f"[Error: Invalid format after {MAX_RETRIES} attempts]",
                        "hard_constraints": [],
                        "soft_constraints": [],
                        "reference_tools": []
                    }

            return {
                "query": obj["query"],
                "hard_constraints": obj["hard_constraints"],
                "soft_constraints": obj["soft_constraints"],
                "reference_tools": obj["reference_tools"]
            }

        except Exception as e:
            print(f"⚠️  Query {query_idx}: Unexpected error: {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"     Retrying... (attempt {attempt}/{MAX_RETRIES})")
                await asyncio.sleep(2)  # Longer sleep for unexpected errors
            else:
                print(f"     Failed after {MAX_RETRIES} attempts")
                return {
                    "query": f"[Error: {type(e).__name__} after {MAX_RETRIES} attempts]",
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "reference_tools": []
                }


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
            "constraint_approach": "creative_flexible",
            "constraint_guidelines": {
                "hard_examples": [
                    "time_window", "budget_ceiling", "party_size", "dietary_restriction",
                    "equipment_requirements", "venue_capacity", "date_range", "asset_inclusion",
                    "market_cap_floor", "liquidity_requirement", "flight_preference", "regulatory",
                    "technical_integration", "security_requirement", "accessibility_need"
                ],
                "soft_examples": [
                    "neighborhood_preference", "family_friendly", "proximity", "cultural_experience",
                    "speaker_profile", "location_convenience", "vendor_reputation", "weather_consideration",
                    "trend_analysis", "sentiment_consideration", "diversification", "volatility_tolerance",
                    "performance_preference", "cost_optimization", "quality_priority", "user_experience"
                ],
                "note": "These are examples only - queries should use creative, domain-specific constraint types that naturally fit the scenario"
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
