#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_generate_multiturn.py

Generate seed queries designed for multi-turn conversations.
These are specific enough for agents to act on, but open-ended enough to invite follow-ups.

Outputs a single JSON file matching the format of query_generate_single.py:
{
  "metadata": {...},
  "items": [{"query": "...", "reference_tools": [...]}, ...]
}

Usage:
  python query_generate_multiturn.py --in tool_descriptions.ndjson --out output.json --num-queries 50
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


MULTITURN_SEED_SYSTEM_PROMPT = """You are an expert at creating natural, specific queries that lead to multi-turn conversations.

Goal:
Generate ONE realistic seed query that:
1. Is SPECIFIC enough for the agent to act on immediately (clear action + concrete subject)
2. Is OPEN-ENDED enough to naturally invite follow-up questions
3. Has interesting results worth exploring further
4. Naturally leads to "drill down", "clarification", or "expansion" questions

In servers_summary, each entry follows this format:
<server_namespace>/<server_name>/<tool_name>: <tool_description>

Return a seed query along with reference tools that the agent would likely use.

IMPORTANT SELECTION RULES:
- Only use tools from the provided servers_summary (exact names)
- Pick 1-3 tools MAX
- Each tool should come from a different MCP server namespace
- The "server" field is the first two path components (e.g., `@smithery-ai/github`)

For each tool, include:
- "server": MCP server prefix (e.g., `@smithery-ai/github`)
- "tool": tool name (e.g., `search_repositories`)
- "why": one-sentence rationale

⚠️ **CONTENT RESTRICTIONS**
- No personal, financial, medical, or confidential data
- Keep queries feasible within public, safe contexts
- Avoid speculative or illegal content

SEED QUERY CHARACTERISTICS:

**GOOD seed queries** (specific + invites follow-ups):
✅ "Find recent papers on arXiv about small language models"
   - Specific: clear action (find), clear subject (small language models), clear source (arXiv)
   - Follow-ups: "Tell me about the first paper", "Who are the authors?", "What methods did they use?"

✅ "Search for GitHub repositories about reinforcement learning"
   - Specific: clear action (search), clear subject (RL repos), clear source (GitHub)
   - Follow-ups: "Which one has most stars?", "Show me the README of the top one", "Are there recent commits?"

✅ "What's the weather in San Francisco this week?"
   - Specific: clear subject (SF), clear timeframe (this week), clear data (weather)
   - Follow-ups: "How does it compare to New York?", "Should I bring an umbrella?", "What about next week?"

✅ "Find news articles about artificial intelligence from the past month"
   - Specific: clear action, subject, timeframe
   - Follow-ups: "What's the most important story?", "Tell me more about that", "Are there controversies?"

**BAD seed queries** (too vague OR too specific):
❌ "Give me data about stuff"
   (Too vague - what stuff? agent can't act)

❌ "Tell me about recent developments"
   (Too vague - in what field? agent confused)

❌ "Search for papers"
   (Too vague - on what topic? where?)

❌ "Get weather for Tokyo on 2025-11-10 at 3pm with humidity and wind speed"
   (Too specific - no room for follow-ups, single answer done)

❌ "What is the exact GDP of France in Q3 2024?"
   (Too specific - single answer, no exploration)

FORMULA FOR GOOD SEED QUERIES:
[Clear Action] + [Specific Subject/Topic] + [Optional: Source/Timeframe/Context]

Examples:
- "Find" + "papers about small LLMs" + "on arXiv"
- "Search for" + "machine learning tools" + "on GitHub"
- "Get" + "weather in Tokyo" + "this week"
- "Find" + "news about AI" + "from the past month"

The agent should be able to START immediately, but the results should be interesting enough to explore further.

OUTPUT FORMAT (strict JSON):
{
  "seed_query": "...",
  "reference_tools": [
    {
      "server": "...",
      "tool": "...",
      "why": "..."
    }
  ],
  "likely_followup_types": ["drill_down", "clarification", "expansion", "verification"],
  "category": "research|news|code_discovery|information_lookup|data_analysis|...",
  "complexity": "easy|medium|hard"
}

Generate the multi-turn seed query now.
"""


async def sample_tools(tools_data: List[Dict], num_samples: int) -> List[Dict]:
    """
    Sample a subset of tools to include in the prompt.
    """
    if num_samples >= len(tools_data):
        return tools_data
    return random.sample(tools_data, num_samples)


async def generate_one_multiturn_seed(
    tools_sample: List[Dict],
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    Call LLM to generate one multi-turn seed query.
    """
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await CLIENT.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": MULTITURN_SEED_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.8,  # Higher for more variety
                    max_tokens=800,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content.strip()
                data = json.loads(content)

                # Validate required fields
                required = ["seed_query", "reference_tools", "likely_followup_types", "category", "complexity"]
                if not all(k in data for k in required):
                    raise ValueError(f"Missing required fields. Got: {list(data.keys())}")

                return data

            except json.JSONDecodeError as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    print(f"JSON decode error after {MAX_RETRIES} attempts: {e}")
                    return None

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    print(f"Error after {MAX_RETRIES} attempts: {e}")
                    return None

    return None


async def generate_multiturn_seeds(
    tools_data: List[Dict],
    num_queries: int,
    tools_per_prompt: int = 50,
) -> List[Dict[str, Any]]:
    """
    Generate multiple multi-turn seed queries concurrently.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []

    for i in range(num_queries):
        # Sample random tools for variety
        sampled = await sample_tools(tools_data, tools_per_prompt)

        # Build servers summary
        servers_summary = "\n".join(
            f"{t['server']}/{t['tool']}: {t['description']}"
            for t in sampled
        )

        user_prompt = f"""Available MCP tools (sample):

{servers_summary}

Generate a multi-turn seed query based on these tools.

REMEMBER THE FORMULA:
[Clear Action] + [Specific Subject/Topic] + [Optional: Source/Timeframe/Context]

The query must be:
- Specific enough for agent to act immediately (no vague "stuff" or "things")
- Open-ended enough to invite natural follow-ups
- Concrete subject/topic mentioned explicitly

Output strict JSON only."""

        tasks.append(
            generate_one_multiturn_seed(sampled, user_prompt, semaphore)
        )

    # Run with progress bar
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating seeds"):
        result = await coro
        if result:
            results.append(result)

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn seed queries from MCP tool descriptions"
    )
    parser.add_argument(
        "--in",
        dest="input_file",
        required=True,
        help="Input NDJSON file with tool descriptions",
    )
    parser.add_argument(
        "--out",
        dest="output_file",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help="Number of seed queries to generate (default: 50)",
    )
    parser.add_argument(
        "--tools-per-prompt",
        type=int,
        default=50,
        help="Number of tools to sample per prompt (default: 50)",
    )

    args = parser.parse_args()

    # Load tool descriptions
    print(f"\nLoading tool descriptions from {args.input_file}...")
    tools_data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            server_data = json.loads(line)
            server_name = server_data.get("qualifiedName", "unknown")

            # Extract each tool from this server
            for tool in server_data.get("tools", []):
                tools_data.append({
                    "server": server_name,
                    "tool": tool.get("name", "unknown"),
                    "description": tool.get("description", "No description"),
                })

    print(f"✓ Loaded {len(tools_data)} tools from servers")

    # Generate seed queries
    print(f"\nGenerating {args.num_queries} multi-turn seed queries...")
    print(f"Output file: {args.output_file}\n")
    start_time = time.time()

    seeds = await generate_multiturn_seeds(
        tools_data=tools_data,
        num_queries=args.num_queries,
        tools_per_prompt=args.tools_per_prompt,
    )

    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(seeds)} seed queries in {elapsed:.1f}s\n")

    # Convert seeds to items format (matching single-turn output)
    items = []
    for seed in seeds:
        items.append({
            "query": seed['seed_query'],
            "reference_tools": seed['reference_tools']
        })

    # Count unique servers
    unique_servers = set()
    for seed in seeds:
        for tool in seed['reference_tools']:
            unique_servers.add(tool['server'])

    # Create output data matching query_generate_single.py format
    output_data = {
        "metadata": {
            "total_items": len(items),
            "servers_count": len(unique_servers),
            "generation_method": "async_multiturn_seeds"
        },
        "items": items
    }

    # Save to output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(items)} queries to {args.output_file}")

    # Print sample
    if items:
        print("\n" + "="*70)
        print("Sample generated queries:")
        print("="*70)
        for i, item in enumerate(items[:3], 1):
            print(f"\n{i}. {item['query']}")
            print(f"   Tools ({len(item['reference_tools'])}): {', '.join(t['tool'] for t in item['reference_tools'])}")

    print("\n" + "="*70)
    print(f"Total queries: {len(items)}")
    print(f"Unique servers: {len(unique_servers)}")
    print(f"Output file: {args.output_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
