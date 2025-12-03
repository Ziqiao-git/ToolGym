#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_fuzzy.py

为生成的查询添加噪声约束，模拟真实世界中带有额外信息的用户请求。

用法:
  python mcp_generate/query_fuzzy.py --in generated_queries.json --out fuzzy_queries.json
"""

import json
import argparse
import os
import asyncio
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import random

# OpenAI 官方 Python SDK
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(str(SCRIPT_DIR / ".env"))

CLIENT = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

MODEL_NAME = "openai/gpt-5.1"
MAX_RETRIES = 3
MAX_CONCURRENT = 5

# Fuzzy噪声生成的系统提示
FUZZY_SYSTEM_PROMPT = """You are an expert at making user queries more realistic by adding natural but slightly noisy information that real users often include.

Your task is to:
1. Take a clean, well-structured query
2. Add 2-4 pieces of NOISY INFORMATION that are:
   - Related to the context but NOT directly relevant to solving the core task
   - The kind of extra details real users often mention
   - Should NOT change the fundamental requirements or reference tools needed

Examples of good noisy information:
- User demographics: "I'm a 35-year-old male software engineer"
- Personal preferences: "I prefer modern interfaces", "I like blue color schemes"
- Background context: "This is for my startup", "My team is mostly remote"
- Soft preferences: "I'm in Pacific timezone", "We speak primarily English"
- Secondary concerns: "I prefer open-source when possible", "I care about carbon footprint"

Examples of BAD noisy information (too relevant, changes the task):
- Adding new hard requirements that change which tools are needed
- Adding constraints that fundamentally change the problem
- Information that is critical to solving the task

CRITICAL RULES:
1. DO NOT change the original query's core intent or requirements
2. DO NOT add information that would require different reference_tools
3. DO NOT modify or remove any existing hard_constraints or soft_constraints
4. The noisy information should feel natural, like a real person casually mentioning extra details
5. Insert the noisy information naturally into the query text - don't just append it at the end
6. Keep the total noisy additions to 1-3 sentences maximum
7. Make it sound conversational and realistic

Output the modified query with noisy information naturally woven in.
"""

FUZZY_USER_PROMPT_TEMPLATE = """Original query:
{original_query}

Add 2-4 pieces of realistic but noisy information to this query. The noisy information should be:
- Contextually related but not critical to solving the task
- The kind of extra details real users often mention
- Naturally integrated into the query text (not just appended)

Return ONLY the modified query text with the noisy information added. Do not include any explanation or metadata.

Examples of how to integrate noisy information:

Before: "Find me a flight from San Francisco to New York on June 15."
After: "I'm a 42-year-old business consultant and I need to find a flight from San Francisco to New York on June 15. I prefer window seats and I'm a vegetarian if meal service is included."

Before: "Compare the performance of Tesla and Apple stocks for Q4 2024."
After: "I'm a retail investor working from home in Seattle, and I'd like to compare the performance of Tesla and Apple stocks for Q4 2024. I prefer investing in companies with strong ESG scores."

Now add natural noisy information to the query above:
"""


async def add_fuzzy_to_query(query_item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    为单个查询添加噪声信息

    Args:
        query_item: 包含 uuid, query, hard_constraints, soft_constraints, reference_tools
        idx: 查询索引（用于日志）

    Returns:
        修改后的查询对象，query字段被添加了噪声
    """
    original_query = query_item.get("query", "")

    if not original_query or len(original_query) < 10:
        print(f"⚠️  Query {idx}: Skipping empty or too short query")
        return query_item

    user_prompt = FUZZY_USER_PROMPT_TEMPLATE.format(original_query=original_query)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": FUZZY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,  # Higher temperature for more diverse noisy additions
                max_tokens=8000,
            )

            message = resp.choices[0].message
            fuzzy_query = message.content

            if not fuzzy_query or not fuzzy_query.strip():
                print(f"⚠️  Query {idx}: Empty response, retrying... (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"❌ Query {idx}: Failed after {MAX_RETRIES} attempts, using original query")
                    return query_item

            fuzzy_query = fuzzy_query.strip()

            # 验证fuzzy query不是太短（确保添加了内容）
            if len(fuzzy_query) < len(original_query) * 0.9:
                print(f"⚠️  Query {idx}: Fuzzy query too short, retrying... (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"❌ Query {idx}: Failed validation, using original query")
                    return query_item

            # 创建新的查询对象，保留所有原始字段，只修改query
            fuzzy_item = {
                "uuid": query_item.get("uuid"),
                "query": fuzzy_query,  # 替换为带噪声的query
                "hard_constraints": query_item.get("hard_constraints", []),
                "soft_constraints": query_item.get("soft_constraints", []),
                "reference_tools": query_item.get("reference_tools", [])
            }

            return fuzzy_item

        except Exception as e:
            print(f"⚠️  Query {idx}: Error: {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"     Retrying... (attempt {attempt}/{MAX_RETRIES})")
                await asyncio.sleep(2)
            else:
                print(f"❌ Query {idx}: Failed after {MAX_RETRIES} attempts, using original query")
                return query_item

    return query_item


async def add_fuzzy_to_all_queries(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    并发为所有查询添加噪声，保持原始顺序

    Args:
        queries: 原始查询列表

    Returns:
        添加噪声后的查询列表（保持原始顺序）
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    completed_count = 0
    total = len(queries)

    async def add_fuzzy_with_semaphore(query_item, idx):
        nonlocal completed_count
        async with semaphore:
            result = await add_fuzzy_to_query(query_item, idx)
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total:
                print(f"Progress: {completed_count}/{total} queries processed")
            return result

    # 创建所有任务
    tasks = [add_fuzzy_with_semaphore(q, i) for i, q in enumerate(queries)]

    # 使用 gather 保持原始顺序
    print(f"Processing {len(tasks)} queries with concurrency limit of {MAX_CONCURRENT}...")
    fuzzy_queries = await asyncio.gather(*tasks)

    print(f"✓ Completed processing all {len(fuzzy_queries)} queries")
    return list(fuzzy_queries)


async def main_async():
    parser = argparse.ArgumentParser(
        description="为生成的查询添加噪声约束，模拟真实世界用户请求"
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="输入的 generated_queries.json 文件路径"
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="输出的 fuzzy_queries.json 文件路径"
    )
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"输入文件不存在: {args.in_path}")

    # 读取原始查询
    print(f"Loading queries from {args.in_path}...")
    with open(args.in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "items" not in data:
        raise ValueError("输入文件格式错误，应包含 'items' 字段")

    original_queries = data["items"]
    metadata = data.get("metadata", {})

    print(f"Loaded {len(original_queries)} queries")

    # 为所有查询添加噪声
    print(f"Adding fuzzy noise to queries (max {MAX_CONCURRENT} concurrent)...")
    fuzzy_queries = await add_fuzzy_to_all_queries(original_queries)

    # 构建输出数据
    output_data = {
        "metadata": {
            **metadata,
            "fuzzy_applied": True,
            "fuzzy_method": "gpt5.1_noisy_context_injection",
            "fuzzy_description": "Added 2-4 pieces of realistic but non-critical noisy information to each query",
            "original_total_items": metadata.get("total_items", len(original_queries)),
            "fuzzy_total_items": len(fuzzy_queries)
        },
        "items": fuzzy_queries
    }

    # 保存结果
    output_dir = os.path.dirname(args.out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully added fuzzy noise to {len(fuzzy_queries)} queries")
    print(f"✅ Saved to {args.out_path}")

    # 显示一个示例对比
    if fuzzy_queries:
        print("\n" + "="*80)
        print("Example comparison (first query):")
        print("="*80)
        print("\nORIGINAL:")
        print(original_queries[0]["query"][:300] + "...")
        print("\nFUZZY:")
        print(fuzzy_queries[0]["query"][:300] + "...")
        print("="*80)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
