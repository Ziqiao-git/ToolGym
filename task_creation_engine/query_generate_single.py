#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_generate_single.py (HARD mode, aligned output, sampled tools)

改动要点：
- 使用 Hard 模式提示词：为“同一工具”设计一个现实场景，**自然需要调用该工具 ≥3 次**。
- 输出对齐 query_generate.py：
  {
    "metadata": { "total_items": ..., "servers_count": ..., "generation_method": ... },
    "items": [ { "query": "...", "reference_tools": [ { "server": "...", "tool": "...", "why": "..." } ] }, ... ]
  }
- 支持从传入的所有 tools 中**随机抽样 num-tools（默认 5）**生成 query；可用 --seed 复现。
- 健壮性：若模型未正确给出 reference_tools 或给错，自动回填为当前唯一工具，并生成 why。
- 路径：使用相对路径 MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson（跨平台）。
"""

import json
import argparse
import os
import asyncio
from typing import List, Dict, Any, Optional
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

# 模型名称（保持与你的需求一致）
MODEL_NAME = "openai/gpt-4o-mini"
MAX_RETRIES = 3
MAX_CONCURRENT = 5  # Maximum concurrent API calls

# ------HARD mode------
SYSTEM_PROMPT = """
You are an expert in crafting complex, realistic multi-step user tasks that make full use of a tool’s capabilities.

Goal:
Given a tool and its description, generate ONE detailed, realistic user query that would naturally require calling this tool three or more times to achieve the goal.

Requirements:
- The query should describe a real-world scenario that genuinely benefits from repeated use of the SAME tool (e.g., iterative analysis, batch processing, monitoring, pagination, multi-entity lookups, refinement loops).
- Avoid artificial chaining; the repetition must make practical sense.
- Keep language natural and coherent.
- Choose tools based on the core functionality they provide and how they would solve real user problems.
- Consider the tool's description and purpose when crafting the question.
- Create questions that are clear and specific enough to warrant tool usage.
- Avoid overly simple questions that could be answered without tools or with only one or two calls.

Guidelines:
- Embed the same tool usage inside a broader workflow that may involve coordination across teams or systems, but keep the actual calls centered on THIS single tool.
- The task should sound like something a researcher, professional, or advanced user might realistically ask.
- Increase multi-dimensional complexity through realistic constraints, competing requirements, stakeholder considerations, and interconnected dependencies.

Output format (strict):
Return a JSON object:
{
  "query": string,
  "reference_tools": [
    { "server": string, "tool": string, "why": string }
  ]
}
Notes:
- Include EXACTLY ONE item in "reference_tools", referring to the SAME provided tool (use exact names).
- In "why", concisely justify why the scenario naturally requires calling this tool ≥3 times.

Make the query natural, diverse, and genuinely useful.
"""

# SYSTEM_PROMPT = """
# You are an expert in crafting complex, realistic multi-step user tasks that make full use of a tool’s capabilities.

# Goal:
# Given a tool and its description, generate ONE detailed, realistic user query that would naturally require calling this tool three or more times to achieve the goal.

# Requirements:
# - The query should describe a real-world scenario that genuinely benefits from repeated use of the SAME tool (e.g., iterative analysis, batch processing, monitoring, pagination, multi-entity lookups, or refinement loops).
# - Avoid artificial chaining; the repetition must make practical sense.
# - Keep language natural and coherent.
# - Choose tools based on the core functionality they provide and how they would solve real user problems.
# - Consider the tool's description and purpose when crafting the question.
# - Create questions that are clear and specific enough to warrant tool usage.
# - Avoid overly simple or generic questions that could be answered without tool calls.

# Hard constraints (must follow):
# - The query MUST include all required parameters from the tool’s description (for example: IDs, names, coordinates, ICAO codes, record types, etc.) with realistic, concrete values.
# - If the tool has required numeric or string parameters, include realistic examples (e.g., record_id="123456789012345678", ICAO="KJFK", page=1, limit=10).
# - The query must clearly describe a task that would need to call the SAME tool at least three times — for example, handling multiple IDs, pages, or timestamps.
# - Do NOT use placeholders like <id>, {x}, or “City A/B/C”. Always provide real, plausible values.
# - The user must ask for actual execution or results, not just instructions or general advice.
# - Keep the query natural and task-oriented, as if a professional user is talking to an assistant that can actually run the tool.

# Output format (strict):
# Return a JSON object:
# {
#   "query": string,
#   "reference_tools": [
#     { "server": string, "tool": string, "why": string }
#   ]
# }

# Notes:
# - Include exactly one item in "reference_tools", referring to the SAME provided tool (use exact names).
# - In "why", concisely justify why the task naturally requires calling this tool three or more times.
# - The "query" text should be a natural user request that already contains the concrete parameter values needed to execute the tool multiple times.
# """

# 注意：除了 {servers_summary} 外，其他花括号都已用 {{ }} 转义，避免 str.format 冲突
USER_PROMPT_TEMPLATE = """Here is the ONLY available tool from its MCP server:

{servers_summary}

Generate 1 natural, realistic user query that would naturally require calling THIS SAME tool three or more times.

Return a JSON object with fields:
- "query": string
- "reference_tools": [ {{ "server": string, "tool": string, "why": string }} ]
(Include exactly one reference_tools item for the tool shown above.)
"""

def _coerce_reference_tools(obj: Any, server: str, tool: str) -> List[Dict[str, str]]:
    """
    保底：保证 reference_tools 至少包含当前(唯一)工具的一条记录。
    若模型给了别的或为空，就回填。
    """
    base = [{
        "server": server,
        "tool": tool,
        "why": "This is the single provided tool; the task requires repeated calls (≥3) such as pagination/batch/refinement."
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
    """
    针对单个 tool 生成 1 条 Hard 模式 query，并对齐 query_generate.py 的 item 结构。
    返回: {"query": str, "reference_tools": [ {server, tool, why} ]}
    """
    server = tool_description.get("qualifiedName", tool_description.get("name") or "unknown")
    tools_summary_lines = []

    # 仅保留这个 server 的这个单一 tool（确保提示词聚焦）
    tools = tool_description.get("tools", [])
    if len(tools) == 0:
        tools_summary_lines.append(f"- {server}: No tools available")
        tool_name = ""
    else:
        # 只取第一项（调用方会传入单 tool 的 server_copy）
        t = tools[0]
        tool_name = t.get("name", "")
        tool_desc = t.get("description", "No description")
        tools_summary_lines.append(f"- {server}/{tool_name}: {tool_desc}")

    servers_summary_text = "\n".join(tools_summary_lines)
    user_prompt = USER_PROMPT_TEMPLATE.format(servers_summary=servers_summary_text)

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "query_with_refs_single_tool_hard",
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
                    return {
                        "query": f"[Error: Empty response after {MAX_RETRIES} attempts]",
                        "reference_tools": _coerce_reference_tools([], server, tool_name)
                    }

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

            # 统一校正 reference_tools（只保留当前工具且只有一条）
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
    """
    在传入的所有 server/tools 中，随机抽取 num_tools 个 tool，
    并为每个被抽中的 tool 生成 1 条 Hard 模式 query。
    返回与 query_generate.py 对齐的 item 列表：[{query, reference_tools}, ...]
    """
    # Flatten: 每个 tool 作为一个生成单元
    tool_items = []
    for server_desc in all_tools:
        for t in server_desc.get("tools", []):
            tool_items.append({"server": server_desc, "tool": t})

    total_tools = len(tool_items)
    if total_tools == 0:
        return []

    # 随机抽样
    if seed is not None:
        random.seed(seed)
    k = min(num_tools, total_tools)
    sampled = random.sample(tool_items, k=k)

    print(f"Total tools: {total_tools} | Sampled: {k}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def worker(idx, server_desc, tool_obj):
        server_name = server_desc.get("qualifiedName", server_desc.get("name") or "unknown")
        tool_name = tool_obj.get("name", "")
        print(f"Generating (HARD) for server: {server_name}  tool: {tool_name}")

        # 构造仅包含该 tool 的 server 拷贝（确保提示词聚焦）
        server_copy = dict(server_desc)
        server_copy["tools"] = [tool_obj]
        async with semaphore:
            return await generate_single_query(server_copy, idx)

    tasks = [worker(i, item["server"], item["tool"]) for i, item in enumerate(sampled)]
    results = []

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating queries (HARD, single-tool, sampled)"):
        res = await coro
        results.append(res)

    return results

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="输入 JSON 文件路径（MCP servers 列表）")
    parser.add_argument("--out", dest="out_path", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--num-tools", type=int, default=5, help="随机抽取的工具个数（默认 5）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选，便于复现）")
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

    # Load all tool descriptions（相对路径，跨平台）
    tool_descriptions_path = "MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson"
    print(f"Loading tool descriptions from {tool_descriptions_path}...")

    with open(tool_descriptions_path, "r", encoding="utf-8") as f:
        all_tools = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_tools)} servers with tools")

    print(f"Generating HARD-mode queries per sampled tool (max {MAX_CONCURRENT} concurrent)...")
    items = await generate_queries_async(all_tools, num_tools=args.num_tools, seed=args.seed)

    # —— 对齐 query_generate.py 的输出结构 —— #
    result = {
        "metadata": {
            "total_items": len(items),
            "servers_count": len(servers),
            "generation_method": "async_single_tool_HARD_mode_sampled"
        },
        "items": items
    }

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(items)} HARD-mode items and saved to {args.out_path}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
