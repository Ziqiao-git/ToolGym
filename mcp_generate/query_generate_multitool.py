#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_generate_multitool.py

Generate single-turn queries that require MULTIPLE different tools from MULTIPLE servers.
This creates complex, realistic tasks that span across different tool domains.

Usage:
  python mcp_generate/query_generate_multitool.py \
    --out mcp_generate/requests/multitool_10_20.json \
    --num-queries 20 \
    --min-tools 10 \
    --max-tools 20 \
    --model openai/gpt-5.1
"""

import json
import argparse
import os
import asyncio
import sys
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import random

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(str(SCRIPT_DIR / ".env"))

# Semantic search components (lazy loaded)
_embedder = None
_faiss_index = None

CLIENT = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL_NAME = "openai/gpt-4o-mini"
MAX_RETRIES = 3
MAX_CONCURRENT = 3  # Lower concurrency for larger prompts


SYSTEM_PROMPT = """You are an expert at creating realistic, complex user tasks that require multiple different tools working together.

Goal:
Given a set of available tools from different MCP servers, generate ONE comprehensive user query that would naturally require using ALL of the provided tools to complete. Then provide a brief explanation of why each tool is needed.

The query should:
1. **Be Realistic**: Represent a genuine real-world task someone would ask an AI assistant
2. **Be Coherent**: All tools should work together toward a unified goal (not a random list of unrelated tasks)
3. **Have Natural Flow**: Tools should be used in a logical sequence where outputs inform next steps
4. **Embed Constraints IMPLICITLY**: Constraints should be woven naturally into the query text, NOT stated as explicit rules

TASK CATEGORIES that naturally combine many tools:

**1. Comprehensive Research & Analysis**
- "I'm writing a detailed report on [topic] and need to gather academic papers, news articles, social media sentiment, financial data, and expert opinions."

**2. Multi-Source Investigation**
- "I'm investigating [company/topic] and need stock data, news, SEC filings, Reddit discussions, GitHub activity, and competitor analysis."

**3. Cross-Domain Planning**
- "I'm planning [event/project] and need to research venues, weather, travel options, local events, reviews, and cost estimates."

**4. Technical Problem Solving**
- "I'm debugging/building [system] and need to search GitHub issues, Stack Overflow, documentation, code examples, and best practices."

**5. Market/Competitive Analysis**
- "I'm evaluating [market/opportunity] and need financial data, news, research papers, social trends, and regulatory information."

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: IMPLICIT CONSTRAINTS (embedded in natural language)
═══════════════════════════════════════════════════════════════════════════════

Constraints must be IMPLICIT in the query text—the agent should infer good behavior from context, not follow explicit rules.

CONSTRAINT TYPES AND HOW TO EMBED THEM IMPLICITLY:

1. **NO_REDUNDANCY** - Imply efficiency without saying "don't repeat"
   ❌ EXPLICIT (bad): "Do not call the same tool twice with identical parameters"
   ✅ IMPLICIT (good):
      - "Work efficiently—each data fetch should add new information"
      - "Use batch endpoints where available instead of multiple individual calls"
      - "The report should be comprehensive but streamlined"
      - "Reuse data you've already fetched (e.g., use the same ticker for news that you analyzed for financials)"

2. **SERVER_DIVERSITY** - Imply cross-referencing without counting servers
   ❌ EXPLICIT (bad): "Must use tools from at least 5 different servers"
   ✅ IMPLICIT (good):
      - "Cross-reference data from independent sources to catch blind spots"
      - "Don't rely on a single provider—verify findings across platforms"
      - "The analysis should triangulate information from multiple providers"

3. **SEQUENCE_ORDER** - Imply logical flow without prescribing order
   ❌ EXPLICIT (bad): "Search before fetching details"
   ✅ IMPLICIT (good):
      - "First discover what's available, then dive into the most relevant items"
      - "Identify the top candidates from search results, then fetch their full details in one batch"
      - "Check current permissions before making changes, then report what changed"
      - "Gather all context before taking actions"

4. **DATA_COVERAGE** - Imply breadth through task structure
   ❌ EXPLICIT (bad): "Must analyze at least 3 companies"
   ✅ IMPLICIT (good):
      - "Compare AAPL, MSFT, and GOOGL side-by-side"
      - "Analyze the top 3 players in this space"
      - "Check conditions in at least a few major cities"

5. **RESPONSE_CONTENT** - Imply deliverables naturally
   ❌ EXPLICIT (bad): "Must provide at least 3 recommendations"
   ✅ IMPLICIT (good):
      - "End with actionable recommendations based on the data"
      - "Include a recommendations section with concrete next steps"
      - "Summarize findings in a comparison table"

6. **TOOL_COUNT** - Imply efficiency without hard limits
   ❌ EXPLICIT (bad): "Use no more than 20 tool calls"
   ✅ IMPLICIT (good):
      - "Be efficient—don't make unnecessary API calls"
      - "The workflow should be comprehensive but not wasteful"

7. **TOOL_TYPE_PRIORITY** - Imply preferences through task framing
   ❌ EXPLICIT (bad): "Prioritize search tools before fetch tools"
   ✅ IMPLICIT (good):
      - "Start by discovering what resources exist, then retrieve the most relevant ones"
      - "Prefer authoritative/official sources when available"

═══════════════════════════════════════════════════════════════════════════════
QUERY STRUCTURE GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

The query should have this natural structure:

1. **Context/Goal**: What the user is trying to accomplish
2. **Efficiency Framing**: A sentence implying the agent should work smart
   - Example: "This needs to pull data from many independent sources for credibility, but work efficiently—each call should add new information."
3. **Deliverables**: What the output should contain (implies RESPONSE_CONTENT)
4. **Workflow Sections**: Detailed requirements that imply SEQUENCE_ORDER
   - Use phrases like "first... then...", "after gathering X, use it to...", "identify candidates, then fetch details"
5. **Data Reuse Hints**: Suggest reusing fetched data (implies NO_REDUNDANCY)
   - Example: "For the tickers you're already analyzing, also check their news"

═══════════════════════════════════════════════════════════════════════════════
FORBIDDEN PATTERNS
═══════════════════════════════════════════════════════════════════════════════

DO NOT include these explicit constraint phrasings in the query:
- "Use at least N different servers/sources"
- "Do not call the same tool twice"
- "Must use no more than N tool calls"
- "Each constraint must be..."
- Any meta-language about constraints

The query should read like a natural user request, not a test specification.

IMPORTANT RULES:
- Use EVERY tool provided in the input - do not skip any
- Create a SINGLE coherent query, not multiple separate tasks
- Each tool should have a clear purpose in achieving the overall goal
- The query should be detailed enough that it's obvious which tools are needed
- Constraints are embedded IMPLICITLY in the query text (the agent must infer them)
- The "constraints" array captures what the evaluator should check, but these are NOT shown to the agent

OUTPUT FORMAT (strict JSON):
{
  "query": "Detailed user query with IMPLICIT constraints woven into natural language...",
  "constraints": [
    {
      "type": "TOOL_COUNT|SERVER_DIVERSITY|SERVER_RESTRICTION|DATA_COVERAGE|RESPONSE_CONTENT|NO_REDUNDANCY|TOOL_TYPE_PRIORITY|SEQUENCE_ORDER",
      "description": "What the evaluator checks (NOT shown to agent)",
      "implicit_phrasing": "The natural language in the query that implies this constraint",
      "verification": {
        // Type-specific verification parameters
      }
    }
  ],
  "tool_reasons": {
    "tool_name_1": "One sentence explaining why this specific tool is needed for this task",
    "tool_name_2": "One sentence explaining why this specific tool is needed for this task",
    ... (provide a detailed reason for EVERY tool - explain the specific value it adds)
  },
  "task_category": "research|investigation|planning|technical|market_analysis|other"
}

CONSTRAINT VERIFICATION SCHEMAS:

For TOOL_COUNT:
  {"max_calls": 15} or {"min_calls": 5, "max_calls": 20}

For SERVER_DIVERSITY:
  {"min_servers": 5}

For SERVER_RESTRICTION:
  {"required_servers": ["@smithery-ai/national-weather-service"]} or
  {"preferred_servers": ["@semantic-scholar", "@arxiv"], "min_preferred": 2}

For DATA_COVERAGE:
  {"min_entities": 5, "entity_type": "companies"} or
  {"required_keywords": ["machine learning", "neural networks"]}

For RESPONSE_CONTENT:
  {"must_include": ["comparison", "recommendation"]} or
  {"min_recommendations": 3}

For NO_REDUNDANCY:
  {"check_duplicate_calls": true}

For TOOL_TYPE_PRIORITY:
  {"preferred_types": ["search", "official"], "min_preferred_ratio": 0.5}

For SEQUENCE_ORDER:
  {"required_sequence": [["search", "fetch"], ["gather", "analyze"]]}

CRITICAL:
- Generate 3-6 constraints, ALL must be from the ALLOWED types
- Each constraint MUST have "type", "description", "implicit_phrasing", and "verification"
- The "implicit_phrasing" field shows WHERE in the query this constraint is implied
- The "tool_reasons" object MUST contain an entry for EVERY tool provided in the input
"""


USER_PROMPT_TEMPLATE = """Here are {tool_count} available tools from {server_count} different MCP servers that MUST ALL be used:

{tools_summary}

Generate ONE realistic, coherent user query that would require using ALL {tool_count} of these tools.

Requirements:
- The query should represent a genuine real-world task
- ALL {tool_count} tools must work together toward a single unified goal
- Constraints must be IMPLICIT in the query text (agent infers them, not reads them as rules)
- Make sure the query naturally requires EVERY tool listed above

Return a JSON object with:
- "query": The detailed user query with IMPLICIT constraints woven into natural language
- "constraints": Array of 3-6 constraint objects for the EVALUATOR (not shown to agent)
- "tool_reasons": A JSON object mapping EACH tool name to a SPECIFIC explanation of why it's needed (MUST have exactly {tool_count} entries)
- "task_category": One of research|investigation|planning|technical|market_analysis|other

CONSTRAINT FORMAT EXAMPLE (with implicit_phrasing showing WHERE in query the constraint is implied):
[
  {{
    "type": "SERVER_DIVERSITY",
    "description": "Agent should use tools from multiple independent providers",
    "implicit_phrasing": "Cross-reference data from independent sources to catch blind spots",
    "verification": {{"min_servers": 5}}
  }},
  {{
    "type": "NO_REDUNDANCY",
    "description": "Agent should not make duplicate tool calls",
    "implicit_phrasing": "Work efficiently—use batch endpoints where available instead of multiple individual calls",
    "verification": {{"check_duplicate_calls": true}}
  }},
  {{
    "type": "SEQUENCE_ORDER",
    "description": "Agent should search before fetching details",
    "implicit_phrasing": "First identify the top candidates from search results, then fetch their full details",
    "verification": {{"required_sequence": [["search", "fetch"]]}}
  }},
  {{
    "type": "RESPONSE_CONTENT",
    "description": "Final response must include recommendations",
    "implicit_phrasing": "End with actionable recommendations based on the data gathered",
    "verification": {{"must_include": ["recommendation"], "min_recommendations": 3}}
  }}
]

CRITICAL:
1. Your "tool_reasons" object MUST contain exactly {tool_count} entries - one for each tool listed above
2. Each tool reason must be SPECIFIC - explain what information or capability that particular tool provides for THIS query
3. Constraints must be IMPLIED in the query, NOT stated as explicit rules
4. The "implicit_phrasing" field must quote or paraphrase the EXACT text in the query that implies the constraint
5. DO NOT include explicit constraint language like "must use N servers" or "do not repeat calls"
"""


async def _verify_constraints(
    query: str,
    constraints: List[Dict],
    sampled_tools: List[Dict]
) -> List[Dict]:
    """
    Verify and fix constraints using a second LLM pass.
    Ensures:
    1. Each constraint has implicit_phrasing field
    2. Verification schemas use standard format
    3. Duplicate constraint types are merged
    4. Invalid constraints are removed
    """
    if not constraints:
        return []

    # Build context about available tools
    servers = set(t["server"] for t in sampled_tools)
    tool_names = [t["tool"] for t in sampled_tools]

    verification_prompt = f"""You are a constraint validator and normalizer. Your job is to:
1. Verify constraints are OBJECTIVELY VERIFIABLE
2. Ensure each constraint has an "implicit_phrasing" field that quotes the query text
3. Standardize verification schemas to the expected format
4. Merge duplicate constraint types into single entries

QUERY:
{query}

AVAILABLE TOOLS ({len(sampled_tools)} tools from {len(servers)} servers):
Servers: {', '.join(list(servers)[:10])}{'...' if len(servers) > 10 else ''}

CONSTRAINTS TO VERIFY AND NORMALIZE:
{json.dumps(constraints, indent=2)}

═══════════════════════════════════════════════════════════════════════════════
REQUIRED OUTPUT FORMAT FOR EACH CONSTRAINT:
═══════════════════════════════════════════════════════════════════════════════

Each constraint MUST have exactly these 4 fields:
{{
  "type": "CONSTRAINT_TYPE",
  "description": "What the evaluator checks",
  "implicit_phrasing": "The EXACT text from the query that implies this constraint",
  "verification": {{ /* standard schema */ }}
}}

═══════════════════════════════════════════════════════════════════════════════
STANDARD VERIFICATION SCHEMAS (use these exact formats):
═══════════════════════════════════════════════════════════════════════════════

TOOL_COUNT:
  {{"max_calls": 25}} or {{"min_calls": 10, "max_calls": 30}}

SERVER_DIVERSITY:
  {{"min_servers": 5}}

NO_REDUNDANCY:
  {{"check_duplicate_calls": true}}

SEQUENCE_ORDER:
  {{"required_sequence": [["search", "fetch"], ["list", "get"]]}}

DATA_COVERAGE:
  {{"min_entities": 3, "entity_type": "companies"}}

RESPONSE_CONTENT:
  {{"must_include": ["recommendation", "comparison"], "min_recommendations": 5}}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASKS:
═══════════════════════════════════════════════════════════════════════════════

1. For each constraint, find the text in the QUERY that implies it and put in "implicit_phrasing"
2. Convert complex verification schemas to the standard simple format above
3. If multiple constraints have the same type, MERGE them into one
4. REMOVE constraints that cannot be verified (time-based, budget-based, subjective)
5. Ensure 3-6 total constraints in output

OUTPUT FORMAT (strict JSON):
{{
  "verified_constraints": [
    {{
      "type": "SERVER_DIVERSITY",
      "description": "Agent should use tools from multiple independent providers",
      "implicit_phrasing": "cross-referencing across different platforms builds credibility",
      "verification": {{"min_servers": 10}}
    }},
    {{
      "type": "NO_REDUNDANCY",
      "description": "Agent should not make duplicate tool calls",
      "implicit_phrasing": "Work efficiently—each API call should fetch new, unique information",
      "verification": {{"check_duplicate_calls": true}}
    }}
    // ... more constraints
  ],
  "removed": [
    {{"original": "...", "reason": "..."}}
  ],
  "merges": [
    "Merged 2 RESPONSE_CONTENT constraints into one"
  ]
}}

Verify and normalize the constraints now."""

    try:
        resp = await CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a constraint validator and normalizer. Output valid JSON only."},
                {"role": "user", "content": verification_prompt},
            ],
            temperature=0.3,
            max_tokens=6000,
            response_format={"type": "json_object"},
        )

        text = resp.choices[0].message.content
        if not text or not text.strip():
            print("  ⚠️ Verifier returned empty response, keeping original constraints")
            return constraints

        result = json.loads(text.strip())
        verified = result.get("verified_constraints", [])
        removed = result.get("removed", [])
        merges = result.get("merges", [])

        if removed:
            print(f"  🔍 Verifier removed {len(removed)} invalid constraints:")
            for r in removed[:3]:
                print(f"     - {r.get('original', 'unknown')[:50]}... ({r.get('reason', 'invalid')})")

        if merges:
            print(f"  🔗 Verifier merged constraints: {len(merges)} merges")
            for m in merges[:2]:
                print(f"     - {m[:60]}...")

        # Final validation: ensure all constraints have required fields
        valid_constraints = []
        for c in verified:
            if all(k in c for k in ["type", "description", "implicit_phrasing", "verification"]):
                valid_constraints.append(c)
            else:
                missing = [k for k in ["type", "description", "implicit_phrasing", "verification"] if k not in c]
                print(f"  ⚠️ Constraint missing fields {missing}, skipping: {c.get('type', 'unknown')}")

        print(f"  ✓ Final constraint count: {len(valid_constraints)}")
        return valid_constraints

    except Exception as e:
        print(f"  ⚠️ Constraint verification failed: {e}, keeping original")
        return constraints


def _infer_constraint_type(constraint_str: str) -> Optional[Dict]:
    """
    Try to infer a structured constraint from a plain string.
    Returns None if the constraint is unverifiable.
    """
    constraint_lower = constraint_str.lower()

    # Tool count patterns
    if any(p in constraint_lower for p in ["tool call", "tool usage", "fewer than", "at most", "no more than"]):
        # Try to extract number
        match = re.search(r'(\d+)\s*(tool|call)', constraint_lower)
        if match:
            return {
                "type": "TOOL_COUNT",
                "description": constraint_str,
                "verification": {"max_calls": int(match.group(1))}
            }

    # Server diversity patterns
    if any(p in constraint_lower for p in ["different server", "multiple source", "independent source", "cross-reference"]):
        match = re.search(r'(\d+)\s*(different|server|source)', constraint_lower)
        min_servers = int(match.group(1)) if match else 3
        return {
            "type": "SERVER_DIVERSITY",
            "description": constraint_str,
            "verification": {"min_servers": min_servers}
        }

    # Data coverage patterns
    if any(p in constraint_lower for p in ["at least", "cover", "analyze", "compare"]) and \
       any(p in constraint_lower for p in ["compan", "stock", "cit", "keyword", "topic"]):
        match = re.search(r'(\d+)', constraint_lower)
        min_entities = int(match.group(1)) if match else 3
        return {
            "type": "DATA_COVERAGE",
            "description": constraint_str,
            "verification": {"min_entities": min_entities, "entity_type": "items"}
        }

    # Response content patterns
    if any(p in constraint_lower for p in ["must include", "should contain", "provide", "recommendation"]):
        return {
            "type": "RESPONSE_CONTENT",
            "description": constraint_str,
            "verification": {"must_include": ["recommendation"]}
        }

    # No redundancy patterns
    if any(p in constraint_lower for p in ["redundan", "duplicate", "same tool twice"]):
        return {
            "type": "NO_REDUNDANCY",
            "description": constraint_str,
            "verification": {"check_duplicate_calls": True}
        }

    # Cannot infer - likely unverifiable (time, budget, quality, etc.)
    return None


def sample_diverse_tools(
    all_tools: List[Dict],
    min_tools: int,
    max_tools: int,
    min_servers: int = 3
) -> List[Dict]:
    """
    Sample tools ensuring diversity across servers.

    Returns list of {"server": str, "tool": str, "description": str}
    """
    # Flatten all tools with server info
    flat_tools = []
    for server_desc in all_tools:
        server_name = server_desc.get("qualifiedName", server_desc.get("name", "unknown"))
        for tool in server_desc.get("tools", []):
            flat_tools.append({
                "server": server_name,
                "tool": tool.get("name", ""),
                "description": tool.get("description", "No description")
            })

    if len(flat_tools) == 0:
        return []

    # Target number of tools
    target_tools = random.randint(min_tools, max_tools)
    target_tools = min(target_tools, len(flat_tools))

    # Group tools by server
    by_server = {}
    for t in flat_tools:
        server = t["server"]
        if server not in by_server:
            by_server[server] = []
        by_server[server].append(t)

    # Ensure minimum server diversity
    servers = list(by_server.keys())
    random.shuffle(servers)

    selected = []
    selected_servers = set()

    # First, pick at least one tool from min_servers different servers
    for server in servers[:min(min_servers, len(servers))]:
        tools = by_server[server]
        if tools:
            selected.append(random.choice(tools))
            selected_servers.add(server)

    # Then fill up to target with random tools (preferring diversity)
    remaining = [t for t in flat_tools if t not in selected]
    random.shuffle(remaining)

    # Prioritize tools from servers we haven't used yet
    remaining.sort(key=lambda t: (t["server"] in selected_servers, random.random()))

    for t in remaining:
        if len(selected) >= target_tools:
            break
        selected.append(t)
        selected_servers.add(t["server"])

    return selected


def load_semantic_search(index_dir: Path = None) -> tuple:
    """
    Load semantic search components (lazy loading).

    Returns:
        Tuple of (embedder, faiss_index)
    """
    global _embedder, _faiss_index

    if _embedder is None:
        from MCP_INFO_MGR.semantic_search.embeddings import BGEEmbedder
        from MCP_INFO_MGR.semantic_search.faiss_backend import FAISSIndex

        print("Loading semantic search components...")
        _embedder = BGEEmbedder()

        if index_dir is None:
            index_dir = PROJECT_ROOT / "MCP_INFO_MGR" / "semantic_search"

        _faiss_index = FAISSIndex.load(index_dir)
        print(f"✓ Semantic search ready with {_faiss_index.index.ntotal} tools indexed")

    return _embedder, _faiss_index


def sample_semantic_tools(
    all_tools: List[Dict],
    min_tools: int,
    max_tools: int,
    min_servers: int = 3,
    index_dir: Path = None,
) -> List[Dict]:
    """
    Sample tools using semantic similarity to ensure coherence.

    Strategy:
    1. Randomly sample 1-3 "seed" tools to define a domain
    2. Use semantic search to find the most related tools
    3. Ensure server diversity among results

    Returns list of {"server": str, "tool": str, "description": str}
    """
    # Flatten all tools with server info
    flat_tools = []
    tool_lookup = {}  # (server, tool) -> tool_dict

    for server_desc in all_tools:
        server_name = server_desc.get("qualifiedName", server_desc.get("name", "unknown"))
        for tool in server_desc.get("tools", []):
            tool_dict = {
                "server": server_name,
                "tool": tool.get("name", ""),
                "description": tool.get("description", "No description")
            }
            flat_tools.append(tool_dict)
            tool_lookup[(server_name, tool.get("name", ""))] = tool_dict

    if len(flat_tools) == 0:
        return []

    # Load semantic search
    embedder, faiss_index = load_semantic_search(index_dir)

    # Target number of tools
    target_tools = random.randint(min_tools, max_tools)
    target_tools = min(target_tools, len(flat_tools))

    # Sample 1-3 seed tools randomly
    num_seeds = min(random.randint(1, 3), len(flat_tools))
    seed_tools = random.sample(flat_tools, num_seeds)

    # Build a query from seed tools' descriptions
    seed_descriptions = [f"{t['tool']}: {t['description']}" for t in seed_tools]
    combined_query = " | ".join(seed_descriptions)

    # Search for semantically similar tools
    query_embedding = embedder.encode_query(combined_query)

    # Search for more results than needed to allow for filtering
    search_results = faiss_index.search(query_embedding, top_k=target_tools * 3)

    # Collect results, ensuring diversity
    selected = []
    selected_keys = set()
    selected_servers = set()

    # First, add seed tools
    for t in seed_tools:
        key = (t["server"], t["tool"])
        if key not in selected_keys:
            selected.append(t)
            selected_keys.add(key)
            selected_servers.add(t["server"])

    # Group remaining results by server for diversity
    by_server = {}
    for result in search_results:
        key = (result["server"], result["tool"])
        if key in selected_keys:
            continue
        server = result["server"]
        if server not in by_server:
            by_server[server] = []
        by_server[server].append({
            "server": server,
            "tool": result["tool"],
            "description": result.get("description", "No description"),
            "score": result["similarity_score"]
        })

    # Round-robin selection to ensure server diversity
    servers = list(by_server.keys())
    random.shuffle(servers)

    # Prioritize servers we haven't used yet
    servers.sort(key=lambda s: s in selected_servers)

    while len(selected) < target_tools and any(by_server.values()):
        for server in servers:
            if len(selected) >= target_tools:
                break
            if by_server.get(server):
                tool = by_server[server].pop(0)
                key = (tool["server"], tool["tool"])
                if key not in selected_keys:
                    selected.append({
                        "server": tool["server"],
                        "tool": tool["tool"],
                        "description": tool["description"]
                    })
                    selected_keys.add(key)
                    selected_servers.add(tool["server"])

        # Remove empty servers
        servers = [s for s in servers if by_server.get(s)]

    # Check if we have minimum server diversity
    if len(selected_servers) < min_servers and len(set(t["server"] for t in flat_tools)) >= min_servers:
        # Need more diversity - add from underrepresented servers
        missing_servers = set(t["server"] for t in flat_tools) - selected_servers
        for server in list(missing_servers)[:min_servers - len(selected_servers)]:
            server_tools = [t for t in flat_tools if t["server"] == server]
            if server_tools and len(selected) < target_tools:
                tool = random.choice(server_tools)
                key = (tool["server"], tool["tool"])
                if key not in selected_keys:
                    selected.append(tool)
                    selected_keys.add(key)
                    selected_servers.add(server)

    return selected


async def generate_multitool_query(
    sampled_tools: List[Dict],
    query_idx: int,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Generate a query that uses multiple tools."""

    # Build tools summary
    tools_summary_lines = []
    for t in sampled_tools:
        tools_summary_lines.append(f"- {t['server']}/{t['tool']}: {t['description']}")

    tools_summary = "\n".join(tools_summary_lines)

    # Count unique servers
    unique_servers = set(t["server"] for t in sampled_tools)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        tool_count=len(sampled_tools),
        server_count=len(unique_servers),
        tools_summary=tools_summary
    )

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await CLIENT.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.8,
                    max_tokens=20000,  # Larger for more tools
                    response_format={"type": "json_object"},
                )

                text = resp.choices[0].message.content
                if not text or not text.strip():
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return _make_error_result(sampled_tools, "Empty response")

                data = json.loads(text.strip())

                # Validate required fields
                if not isinstance(data.get("query"), str):
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1)
                        continue
                    return _make_error_result(sampled_tools, "Missing query field")

                # Get tool_reasons from LLM response (or empty dict)
                tool_reasons = data.get("tool_reasons", {})

                # Debug: show what keys LLM returned for tool_reasons
                if tool_reasons:
                    print(f"  ✓ LLM returned {len(tool_reasons)} tool_reasons")
                    # Show first few keys as sample
                    sample_keys = list(tool_reasons.keys())[:3]
                    print(f"    Sample keys: {sample_keys}")
                else:
                    print(f"  ⚠️ LLM did not return tool_reasons (or returned empty)")

                # Build reference_tools from sampled_tools (guarantees all tools are included)
                reference_tools = []
                for t in sampled_tools:
                    tool_name = t["tool"]
                    server_name = t["server"]
                    # Try multiple key formats that LLM might use
                    why = (
                        tool_reasons.get(tool_name) or  # Just tool name
                        tool_reasons.get(f"{server_name}/{tool_name}") or  # server/tool
                        tool_reasons.get(f"{server_name}_{tool_name}") or  # server_tool
                        "Required for completing the task"  # Fallback
                    )
                    reference_tools.append({
                        "server": server_name,
                        "tool": tool_name,
                        "why": why
                    })

                # Replace any LLM-provided reference_tools with our complete list
                data["reference_tools"] = reference_tools

                # Remove tool_reasons from output (we've merged it into reference_tools)
                if "tool_reasons" in data:
                    del data["tool_reasons"]

                # Add metadata
                data["tool_count"] = len(sampled_tools)
                data["server_count"] = len(unique_servers)
                data["complexity"] = data.get("complexity", "high")
                data["task_category"] = data.get("task_category", "other")

                # Validate and normalize constraints
                raw_constraints = data.get("constraints", [])
                validated_constraints = []

                VALID_CONSTRAINT_TYPES = {
                    "TOOL_COUNT", "SERVER_DIVERSITY", "SERVER_RESTRICTION",
                    "DATA_COVERAGE", "RESPONSE_CONTENT", "NO_REDUNDANCY",
                    "TOOL_TYPE_PRIORITY", "SEQUENCE_ORDER"
                }

                for c in raw_constraints:
                    if isinstance(c, dict):
                        # New structured format
                        if c.get("type") in VALID_CONSTRAINT_TYPES:
                            validated_constraints.append({
                                "type": c["type"],
                                "description": c.get("description", ""),
                                "verification": c.get("verification", {})
                            })
                        else:
                            # Unknown type - skip with warning
                            print(f"  ⚠️ Skipping invalid constraint type: {c.get('type')}")
                    elif isinstance(c, str):
                        # Old string format - try to infer type (backwards compat)
                        inferred = _infer_constraint_type(c)
                        if inferred:
                            validated_constraints.append(inferred)
                        else:
                            print(f"  ⚠️ Skipping unverifiable constraint: {c[:50]}...")

                # Run verification pass to double-check constraints
                if validated_constraints:
                    print(f"  🔍 Running constraint verifier...")
                    verified_constraints = await _verify_constraints(
                        query=data["query"],
                        constraints=validated_constraints,
                        sampled_tools=sampled_tools
                    )
                    data["constraints"] = verified_constraints
                else:
                    data["constraints"] = []

                if len(data["constraints"]) < 3:
                    print(f"  ⚠️ Only {len(data['constraints'])} valid constraints after verification (expected 3-6)")

                return data

            except json.JSONDecodeError as e:
                print(f"⚠️  Query {query_idx}: JSON decode error: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return _make_error_result(sampled_tools, f"JSON error: {e}")

            except Exception as e:
                print(f"⚠️  Query {query_idx}: Error: {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return _make_error_result(sampled_tools, str(e))

    return _make_error_result(sampled_tools, "Max retries exceeded")


def _make_error_result(sampled_tools: List[Dict], error: str) -> Dict:
    """Create an error result with the sampled tools."""
    unique_servers = set(t["server"] for t in sampled_tools)
    return {
        "query": f"[Error: {error}]",
        "constraints": [],
        "reference_tools": [
            {"server": t["server"], "tool": t["tool"], "why": "N/A"}
            for t in sampled_tools
        ],
        "tool_count": len(sampled_tools),
        "server_count": len(unique_servers),
        "task_category": "error",
        "complexity": "high"
    }


async def generate_all_queries(
    all_tools: List[Dict],
    num_queries: int,
    min_tools: int,
    max_tools: int,
    min_servers: int,
    seed: Optional[int] = None,
    use_semantic: bool = False,
    index_dir: Path = None,
) -> List[Dict]:
    """Generate multiple multi-tool queries."""

    if seed is not None:
        random.seed(seed)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def worker(idx: int) -> Dict:
        # Sample tools for this query
        if use_semantic:
            sampled = sample_semantic_tools(all_tools, min_tools, max_tools, min_servers, index_dir)
        else:
            sampled = sample_diverse_tools(all_tools, min_tools, max_tools, min_servers)

        if not sampled:
            return _make_error_result([], "No tools available")

        print(f"Query {idx+1}: Sampled {len(sampled)} tools from {len(set(t['server'] for t in sampled))} servers")
        return await generate_multitool_query(sampled, idx, semaphore)

    tasks = [worker(i) for i in range(num_queries)]
    results = []

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating multi-tool queries"):
        result = await coro
        results.append(result)

    return results


async def main_async():
    parser = argparse.ArgumentParser(
        description="Generate single-turn queries requiring multiple tools"
    )
    parser.add_argument(
        "--tools",
        default=None,
        help="Tool descriptions NDJSON file (default: MCP_INFO_MGR/mcp_data/working/tool_descriptions.ndjson)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of queries to generate (default: 20)"
    )
    parser.add_argument(
        "--min-tools",
        type=int,
        default=10,
        help="Minimum tools per query (default: 10)"
    )
    parser.add_argument(
        "--max-tools",
        type=int,
        default=20,
        help="Maximum tools per query (default: 20)"
    )
    parser.add_argument(
        "--min-servers",
        type=int,
        default=3,
        help="Minimum different servers per query (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LLM model to use (default: openai/gpt-4o-mini)"
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use semantic search to select related tools (instead of random sampling)"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Directory containing FAISS index (default: MCP_INFO_MGR/semantic_search/)"
    )

    args = parser.parse_args()

    global MODEL_NAME
    MODEL_NAME = args.model

    # Load tool descriptions
    if args.tools:
        tool_desc_path = Path(args.tools)
    else:
        tool_desc_path = Path(__file__).parent.parent / "MCP_INFO_MGR" / "mcp_data" / "working" / "tool_descriptions.ndjson"

    print(f"Loading tool descriptions from {tool_desc_path}...")

    with open(tool_desc_path, "r", encoding="utf-8") as f:
        all_tools = [json.loads(line) for line in f if line.strip()]

    # Only include servers with status='ok' (successfully fetched tools)
    working_tools = [t for t in all_tools if t.get("status") == "ok" and t.get("tools")]

    # Count total tools
    total_tool_count = sum(len(s.get("tools", [])) for s in working_tools)
    print(f"Loaded {len(working_tools)} working servers with {total_tool_count} total tools")

    if total_tool_count < args.min_tools:
        print(f"⚠️  Warning: Only {total_tool_count} tools available, but min_tools is {args.min_tools}")
        print(f"   Adjusting min_tools to {total_tool_count}")
        args.min_tools = total_tool_count

    print(f"\nGenerating {args.num_queries} multi-tool queries...")
    print(f"  Tools per query: {args.min_tools} - {args.max_tools}")
    print(f"  Min servers per query: {args.min_servers}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Tool selection: {'semantic (coherent)' if args.semantic else 'random (diverse)'}")
    print()

    items = await generate_all_queries(
        all_tools=working_tools,
        num_queries=args.num_queries,
        min_tools=args.min_tools,
        max_tools=args.max_tools,
        min_servers=args.min_servers,
        seed=args.seed,
        use_semantic=args.semantic,
        index_dir=args.index_dir,
    )

    # Calculate stats
    avg_tools = sum(item.get("tool_count", 0) for item in items) / len(items) if items else 0
    avg_servers = sum(item.get("server_count", 0) for item in items) / len(items) if items else 0
    error_count = sum(1 for item in items if item.get("task_category") == "error")

    # Create output
    result = {
        "metadata": {
            "total_items": len(items),
            "generation_method": "multitool_single_turn",
            "tool_selection": "semantic" if args.semantic else "random",
            "min_tools": args.min_tools,
            "max_tools": args.max_tools,
            "min_servers": args.min_servers,
            "avg_tools_per_query": round(avg_tools, 1),
            "avg_servers_per_query": round(avg_servers, 1),
            "error_count": error_count,
            "model": MODEL_NAME
        },
        "items": items
    }

    # Save output
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"Generated {len(items)} multi-tool queries")
    print(f"  Average tools per query: {avg_tools:.1f}")
    print(f"  Average servers per query: {avg_servers:.1f}")
    print(f"  Errors: {error_count}")
    print(f"  Output: {args.out}")
    print(f"{'='*70}")

    # Print sample
    if items and items[0].get("task_category") != "error":
        print("\nSample query:")
        print("-" * 70)
        sample = items[0]
        print(f"Query: {sample['query'][:300]}...")
        print(f"Tools: {sample['tool_count']} from {sample['server_count']} servers")
        print(f"Category: {sample.get('task_category', 'N/A')}")
        if sample.get("constraints"):
            print(f"Constraints: {sample['constraints']}")
        print("-" * 70)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
