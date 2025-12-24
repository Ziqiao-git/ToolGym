#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_generate_goaloriented.py

Generate seed queries WITH GOALS for goal-oriented multi-turn conversations.
Each seed includes:
1. Initial query (what user asks first)
2. Hidden goal (what user wants to achieve overall)
3. Reference tools (tools needed to achieve goal)

Outputs JSON file with format:
{
  "metadata": {...},
  "items": [
    {
      "query": "...",
      "goal": "...",
      "reference_tools": [...]
    },
    ...
  ]
}

Usage:
  python query_generate_goaloriented.py --in tool_descriptions.ndjson --out output.json --num-queries 50
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


GOALORIENTED_SEED_SYSTEM_PROMPT = """You are an expert at creating realistic user scenarios with hidden goals and real-world constraints.

Goal:
Generate ONE realistic scenario with:
1. **Seed Query**: The user's first question (specific + actionable)
2. **Hidden Goal**: What the user wants to achieve overall (broader than seed query)
3. **Reference Tools**: MCP tools needed to achieve the goal
4. **Constraints**: Real-world constraints that affect how the goal should be achieved

In servers_summary, each entry follows this format:
<server_namespace>/<server_name>/<tool_name>: <tool_description>

IMPORTANT: The goal should:
- Be BROADER than the seed query
- Require MULTIPLE information-gathering steps (3-6 sub-goals)
- Be ACHIEVABLE using available MCP tools
- Be MEASURABLE (clear when complete)
- Represent a realistic real-world task
- Include REAL-WORLD CONSTRAINTS that affect planning

CONSTRAINT CATEGORIES (include 1-3 per scenario):
- **Temporal**: Deadlines, time windows, schedules, urgency (e.g., "within the next 2 weeks", "before December")
- **Financial**: Budget limits, cost considerations (e.g., "under $500", "budget-friendly options")
- **Resource**: Capacity limits, availability (e.g., "for a group of 4", "vegetarian options only")
- **Quality**: Standards, requirements (e.g., "highly-rated only", "must have parking")
- **Scope**: Priorities, must-haves (e.g., "focus on beginner-friendly", "must be open-source")
- **Geographic**: Location constraints (e.g., "within 30 miles", "in the downtown area")
- **Preference**: User preferences (e.g., "prefer outdoor activities", "avoid crowded places")

GOAL CATEGORIES (with constraint examples):

**1. Trip Planning**
- Seed: "What are the upcoming events in Bodrum?"
- Goal: "I'm planning a weekend trip to Bodrum and need to know what events are happening, what the weather will be like, where to eat, and where to stay."
- Constraints: ["budget under $1000 for the whole trip", "must be family-friendly", "prefer beachfront hotels"]
- Sub-goals: Find events, check weather, get restaurants, find hotels

**2. Research Report**
- Seed: "Find recent papers on arXiv about small language models"
- Goal: "I'm writing a research report on small language models and need to find recent papers, understand the key methods being used, identify top researchers, and analyze current trends."
- Constraints: ["papers from 2024 or later only", "focus on models under 7B parameters", "must have code available"]
- Sub-goals: Find papers, understand methods, identify researchers, analyze trends

**3. Investment Decision**
- Seed: "Get the latest stock price for Tesla"
- Goal: "I'm deciding whether to invest in Tesla and need the current stock price, recent news about the company, financial performance data, and competitor comparison."
- Constraints: ["investment amount around $5000", "time horizon of 2-3 years", "moderate risk tolerance"]
- Sub-goals: Get price, find news, get financials, compare competitors

**4. Product Comparison**
- Seed: "Search for laptop reviews on Reddit"
- Goal: "I'm buying a new laptop and need to read reviews, compare specifications, check current prices, and see availability in my area."
- Constraints: ["budget max $1500", "must have 16GB+ RAM", "prefer lightweight under 4 lbs"]
- Sub-goals: Read reviews, compare specs, check prices, verify availability

**5. Problem Diagnosis**
- Seed: "Search GitHub issues about Python memory leaks"
- Goal: "My Python application has memory leaks and I need to understand common causes, find potential solutions, see if others fixed similar issues, and verify fixes work."
- Constraints: ["using Python 3.11", "Django web application", "must not require major refactoring"]
- Sub-goals: Understand causes, find solutions, check similar issues, verify fixes

**6. Event Discovery**
- Seed: "What tech conferences are happening in 2025?"
- Goal: "I want to attend tech conferences in 2025 and need to find upcoming events, understand the topics covered, check ticket availability, and see speaker lineups."
- Constraints: ["within the US only", "budget under $2000 including travel", "focus on AI/ML topics"]
- Sub-goals: Find events, check topics, verify tickets, review speakers

FORMULA FOR SEED QUERIES:
[Clear Action] + [Specific Subject] + [Optional: Context/Constraint]

FORMULA FOR GOALS:
"I'm [doing task] and need to [step 1], [step 2], [step 3], and [step 4]."

FORMULA FOR CONSTRAINTS:
Each constraint should be:
- Specific and measurable when possible
- Realistic and commonly encountered
- Relevant to the goal category
- Affecting how tools should be used or results filtered

⚠️ **CONTENT RESTRICTIONS**
- No personal, financial, medical, or confidential data
- Keep queries feasible within public, safe contexts
- Avoid speculative or illegal content

TOOL SELECTION RULES:
- Only use tools from the provided servers_summary (exact names)
- Pick 2-5 tools that cover different aspects of the goal
- Each tool should come from a different MCP server namespace
- The "server" field is the first two path components (e.g., `@smithery-ai/github`)

For each tool, include:
- "server": MCP server prefix (e.g., `@smithery-ai/github`)
- "tool": tool name (e.g., `search_repositories`)
- "why": one-sentence rationale connecting to a sub-goal

OUTPUT FORMAT (strict JSON):
{
  "seed_query": "...",
  "goal": "I'm [doing task] and need to [sub-goal 1], [sub-goal 2], [sub-goal 3], and [sub-goal 4].",
  "constraints": [
    "constraint 1 (specific and measurable)",
    "constraint 2",
    "constraint 3 (optional)"
  ],
  "reference_tools": [
    {
      "server": "...",
      "tool": "...",
      "why": "..."
    }
  ],
  "goal_category": "trip_planning|research|investment|product_comparison|problem_diagnosis|event_discovery|...",
  "complexity": "easy|medium|hard"
}

Generate the goal-oriented seed query now.
"""


async def sample_tools(tools_data: List[Dict], num_samples: int) -> List[Dict]:
    """
    Sample a subset of tools to include in the prompt.
    """
    if num_samples >= len(tools_data):
        return tools_data
    return random.sample(tools_data, num_samples)


async def generate_one_goaloriented_seed(
    tools_sample: List[Dict],
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    Call LLM to generate one goal-oriented seed query.
    """
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await CLIENT.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": GOALORIENTED_SEED_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.8,  # Higher for more variety
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content.strip()
                data = json.loads(content)

                # Validate required fields
                required = ["seed_query", "goal", "reference_tools", "goal_category", "complexity"]
                if not all(k in data for k in required):
                    raise ValueError(f"Missing required fields. Got: {list(data.keys())}")

                # Ensure constraints field exists (default to empty list if missing)
                if "constraints" not in data:
                    data["constraints"] = []

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


async def generate_goaloriented_seeds(
    tools_data: List[Dict],
    num_queries: int,
    tools_per_prompt: int = 50,
) -> List[Dict[str, Any]]:
    """
    Generate multiple goal-oriented seed queries concurrently.
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

Generate a goal-oriented seed query based on these tools.

REMEMBER:
- Seed query: Specific first question
- Goal: Broader task requiring multiple steps
- Formula: "I'm [doing task] and need to [step 1], [step 2], [step 3], and [step 4]."

The goal should require 3-6 information-gathering steps to complete.

Output strict JSON only."""

        tasks.append(
            generate_one_goaloriented_seed(sampled, user_prompt, semaphore)
        )

    # Run with progress bar
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating goal-oriented seeds"):
        result = await coro
        if result:
            results.append(result)

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Generate goal-oriented seed queries from MCP tool descriptions"
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
        help="Number of goal-oriented seed queries to generate (default: 50)",
    )
    parser.add_argument(
        "--tools-per-prompt",
        type=int,
        default=50,
        help="Number of tools to sample per prompt (default: 50)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LLM model to use (default: openai/gpt-4o-mini)",
    )

    args = parser.parse_args()

    # Override global MODEL_NAME if specified
    global MODEL_NAME
    MODEL_NAME = args.model

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
    print(f"\nGenerating {args.num_queries} goal-oriented seed queries...")
    print(f"Model: {MODEL_NAME}")
    print(f"Output file: {args.output_file}\n")
    start_time = time.time()

    seeds = await generate_goaloriented_seeds(
        tools_data=tools_data,
        num_queries=args.num_queries,
        tools_per_prompt=args.tools_per_prompt,
    )

    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(seeds)} goal-oriented seed queries in {elapsed:.1f}s\n")

    # Convert seeds to items format
    items = []
    for seed in seeds:
        items.append({
            "query": seed['seed_query'],
            "goal": seed['goal'],
            "constraints": seed.get('constraints', []),
            "reference_tools": seed['reference_tools'],
            "goal_category": seed.get('goal_category', 'unknown'),
            "complexity": seed.get('complexity', 'medium')
        })

    # Count unique servers
    unique_servers = set()
    for seed in seeds:
        for tool in seed['reference_tools']:
            unique_servers.add(tool['server'])

    # Create output data
    output_data = {
        "metadata": {
            "total_items": len(items),
            "servers_count": len(unique_servers),
            "generation_method": "async_goaloriented_seeds",
            "model": MODEL_NAME
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
        print("Sample generated goal-oriented queries:")
        print("="*70)
        for i, item in enumerate(items[:3], 1):
            print(f"\n{i}. QUERY: {item['query']}")
            print(f"   GOAL: {item['goal']}")
            if item.get('constraints'):
                print(f"   CONSTRAINTS: {item['constraints']}")
            print(f"   TOOLS ({len(item['reference_tools'])}): {', '.join(t['tool'] for t in item['reference_tools'])}")
            print(f"   CATEGORY: {item['goal_category']} | COMPLEXITY: {item['complexity']}")

    print("\n" + "="*70)
    print(f"Total queries: {len(items)}")
    print(f"Unique servers: {len(unique_servers)}")
    print(f"Output file: {args.output_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
