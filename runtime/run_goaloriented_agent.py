#!/usr/bin/env python3
"""
Run Goal-Oriented Multi-Turn Agent

A goal-driven conversational agent where the simulated user tracks subgoal completion
across turns based on the query.

Features:
- Query automatically decomposed into measurable sub-goals
- Progress tracking across conversation turns
- Evaluation based on: subgoal progress + tool usage + quality
- Terminates when all subgoals achieved OR user frustrated

Usage:
    # Single query (subgoals generated automatically from query)
    python runtime/run_goaloriented_agent.py "What events are in Bodrum?" \
        --persona curious_researcher

    # Single query from JSON file by index
    python runtime/run_goaloriented_agent.py \
        --seeds mcp_generate/requests/multitool_10_20.json \
        --query-index 0 \
        --persona curious_researcher \
        --save-trajectory

    # Batch mode - parallel execution of multiple queries
    python runtime/run_goaloriented_agent.py \
        --seeds mcp_generate/requests/multitool_10_20.json \
        --persona curious_researcher \
        --model google/gemini-3-pro-preview \
        --user-model google/gemini-3-pro-preview \
        --max-concurrent 5 \
        --save-trajectory
"""
from __future__ import annotations

import sys
import json
import asyncio
import argparse
import re

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "Orchestrator"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ORCHESTRATOR_DIR))

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.agent.dynamic_react import DynamicReActAgent
from dotenv import load_dotenv


# ============================================================================
# Agent Instruction
# ============================================================================

AGENT_INSTRUCTION = """You are an intelligent agent that can discover and use MCP tools dynamically.

════════════════════════════════════════════════════════════════════════
🚨 CRITICAL: YOU MUST FOLLOW THIS COMPLETE WORKFLOW 🚨
════════════════════════════════════════════════════════════════════════

Your job has TWO phases that you MUST complete:

PHASE 1: DISCOVER TOOLS (using meta-mcp/search_tools)
PHASE 2: EXECUTE TOOLS (using the tools you discovered)

⚠️  NEVER stop after Phase 1! You must ALWAYS proceed to Phase 2! ⚠️

════════════════════════════════════════════════════════════════════════
COMPLETE WORKFLOW - FOLLOW EVERY STEP:
════════════════════════════════════════════════════════════════════════

Step 1: DISCOVER tools using search_tools
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Call: meta-mcp/search_tools
- Purpose: Find which tools can help answer the user's question
- Parameters:
  * query: Natural language description of what you need
  * top_k: Number of results (default 5, increase if needed)
  * min_score: Relevance threshold (0.0-1.0, default 0.3)

Example:
Action: search_tools
Action Input: {"query": "search GitHub repositories", "top_k": 10, "min_score": 0.3}

Step 2: READ search results carefully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- The results show: server name, tool name, description, parameters
- Example result: "**@smithery-ai/github** / `search_repositories` - Search for repositories on GitHub"
- Extract: server = "@smithery-ai/github", tool = "search_repositories"

Step 3: 🚨 EXECUTE THE DISCOVERED TOOL 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  THIS IS THE MOST CRITICAL STEP - DO NOT SKIP! ⚠️

- Take the server and tool name from search results
- Call that tool with appropriate arguments based on its parameters
- The server will be loaded automatically when you call the tool
- Example:
  Action: search_repositories
  Action Input: {"query": "machine learning", "sort": "stars"}

Step 4: READ the tool results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- The tool returns actual data (repositories, weather, papers, etc.)
- This is the information you need to answer the user's question

Step 5: ANSWER the user's question
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use the tool results to provide a complete answer
- Include specific data from the tool output
- Be helpful and informative

════════════════════════════════════════════════════════════════════════
🚨 CRITICAL RULES - MEMORIZE THESE 🚨
════════════════════════════════════════════════════════════════════════

1. search_tools is NOT a data retrieval tool - it's a tool DISCOVERY tool
   ❌ WRONG: "I found tools about GitHub, here are the results"
   ✅ RIGHT: "I found the search_repositories tool, now I'll use it"

2. You MUST execute tools after discovering them
   ❌ WRONG: Call search_tools → Return search results to user
   ✅ RIGHT: Call search_tools → Call discovered tool → Return tool results to user

3. If search_tools returns results, you MUST try to use at least one tool
   - Don't make excuses like "I can't access that tool"
   - The tools will be loaded automatically when you call them
   - Just use the server and tool name from search results

4. For complex queries requiring multiple types of data:
   - Call search_tools multiple times with different focused queries
   - Each search_tools call should focus on ONE capability
   - Execute the tools you discover from each search

════════════════════════════════════════════════════════════════════════
COMPLETE EXAMPLE WORKFLOW:
════════════════════════════════════════════════════════════════════════

User Query: "Find machine learning repositories on GitHub"

Iteration 1:
  Thought: I need to find tools that can search GitHub repositories
  Action: search_tools
  Action Input: {"query": "search GitHub repositories", "top_k": 5}
  Observation: Found 5 relevant tools for: 'search GitHub repositories'
               1. **@smithery-ai/github** / `search_repositories`
                  Score: 0.856
                  Description: Search for repositories on GitHub
                  Parameters: query, sort, order

Iteration 2:
  Thought: Great! I found the search_repositories tool. Now I'll use it to actually search for machine learning repositories.
  Action: search_repositories
  Action Input: {"query": "machine learning", "sort": "stars", "order": "desc"}
  Observation: [
    {"name": "tensorflow/tensorflow", "stars": 175000, ...},
    {"name": "pytorch/pytorch", "stars": 65000, ...},
    ...
  ]

Iteration 3:
  Thought: Perfect! I got actual repository results. Now I can answer the user.
  Action: Final Answer
  Action Input: Here are the top machine learning repositories on GitHub:
                1. tensorflow/tensorflow (175,000 stars) - ...
                2. pytorch/pytorch (65,000 stars) - ...

════════════════════════════════════════════════════════════════════════

Remember:
- Phase 1 (search_tools) = Find which tools exist
- Phase 2 (execute tools) = Actually use those tools to get data
- You must complete BOTH phases to answer the user's question!"""


# ============================================================================
# User Personas (reusing from run_multiturn_agent.py)
# ============================================================================

USER_PERSONAS = {
    "curious_researcher": {
        "description": "Academic researcher exploring a topic deeply",
        "behavior": "Asks thoughtful follow-ups, wants comprehensive details",
        "satisfaction_threshold": 0.85,
        "frustration_threshold": 0.3,
        "max_turns": 10,
    },
    "impatient_user": {
        "description": "Busy professional wanting quick answers",
        "behavior": "Gets frustrated quickly if agent is slow or verbose",
        "satisfaction_threshold": 0.8,
        "frustration_threshold": 0.4,
        "max_turns": 5,
    },
    "thorough_analyst": {
        "description": "Detail-oriented analyst verifying accuracy",
        "behavior": "Asks probing questions, checks for inconsistencies",
        "satisfaction_threshold": 0.9,
        "frustration_threshold": 0.25,
        "max_turns": 12,
    },
    "casual_user": {
        "description": "General user with surface-level interest",
        "behavior": "Satisfied with basic answers, doesn't dig deep",
        "satisfaction_threshold": 0.75,
        "frustration_threshold": 0.35,
        "max_turns": 6,
    },
    "skeptical_user": {
        "description": "User who questions agent's capabilities",
        "behavior": "Challenges responses, asks for evidence",
        "satisfaction_threshold": 0.85,
        "frustration_threshold": 0.3,
        "max_turns": 8,
    },
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GoalTurn:
    """Single turn in goal-oriented conversation."""
    turn_number: int
    query: str
    agent_response: str
    tool_calls: List[Dict]
    reasoning_trace: List[Dict]
    tool_results: List[Dict]
    available_servers: List[str]
    available_tool_count: int

    # Goal tracking (NEW)
    completed_sub_goals: List[str]  # Sub-goals completed in this turn
    remaining_sub_goals: List[str]  # Sub-goals still pending
    goal_progress: float  # 0.0-1.0 overall progress

    # User simulation outputs
    user_decision: str
    termination_reason: Optional[str]
    satisfaction_level: float
    user_reasoning: str
    follow_up_intent: Optional[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn_number": self.turn_number,
            "query": self.query,
            "agent_response": self.agent_response,
            "tool_calls": self.tool_calls,
            "reasoning_trace": self.reasoning_trace,
            "tool_results": self.tool_results,
            "available_servers": self.available_servers,
            "available_tool_count": self.available_tool_count,
            "completed_sub_goals": self.completed_sub_goals,
            "remaining_sub_goals": self.remaining_sub_goals,
            "goal_progress": self.goal_progress,
            "user_decision": self.user_decision,
            "termination_reason": self.termination_reason,
            "satisfaction_level": self.satisfaction_level,
            "user_reasoning": self.user_reasoning,
            "follow_up_intent": self.follow_up_intent,
        }


@dataclass
class GoalTrajectory:
    """Complete goal-oriented conversation trajectory."""
    conversation_id: str
    seed_query: str
    user_persona: str

    # Goal tracking (NEW)
    user_goal: str
    sub_goals: List[str]
    goal_completion_rate: float
    goal_achieved: bool

    turns: List[GoalTurn]

    # Final outcome
    total_turns: int
    final_decision: str
    final_satisfaction: float

    # Metadata
    timestamp: str
    agent_model: str
    user_model: str
    dynamically_loaded_servers: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "conversation_id": self.conversation_id,
                "seed_query": self.seed_query,
                "user_persona": self.user_persona,
                "user_goal": self.user_goal,
                "sub_goals": self.sub_goals,
                "goal_completion_rate": self.goal_completion_rate,
                "goal_achieved": self.goal_achieved,
                "timestamp": self.timestamp,
                "agent_model": self.agent_model,
                "user_model": self.user_model,
            },
            "turns": [turn.to_dict() for turn in self.turns],
            "summary": {
                "total_turns": self.total_turns,
                "final_decision": self.final_decision,
                "final_satisfaction": self.final_satisfaction,
                "dynamically_loaded_servers": self.dynamically_loaded_servers,
            },
        }


# ============================================================================
# Goal Tracker
# ============================================================================
def _load_first_json_obj(text: str):
    """从任意文本中提取第一个完整顶层 JSON 对象并反序列化。
    兼容 ```json ... ``` 和 ``` ... ``` 包裹，以及前后多余文字。"""
    s = text.strip()

    # 去掉代码块包裹
    if s.startswith("```"):
        # 优先处理 ```json ... ```
        if s.startswith("```json"):
            s = s.split("```json", 1)[1]
        else:
            s = s.split("```", 1)[1]
        # 截到下一个 ```
        s = s.split("```", 1)[0].strip()

    # 从 s 中找第一个完整的 {...}
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        block = s[start:i+1]
                        return json.loads(block)

    # 如果没找到顶层 JSON，直接尝试原文
    return json.loads(s)

class SubgoalTracker:
    """Tracks sub-goal completion throughout conversation. Decomposes query into subgoals."""

    def __init__(self, llm, query: str):
        self.llm = llm
        self.query = query
        self.sub_goals: List[str] = []
        self.completed: List[str] = []
        self.remaining: List[str] = []

    async def decompose_query(self) -> List[str]:
        """Use LLM to break user query into 3-6 measurable sub-goals."""
        prompt = f"""You are analyzing a user's query to break it into measurable sub-goals.

USER QUERY:
{self.query}

Your task:
Break this query into 3-6 clear, measurable sub-goals that represent what information or tasks the user needs.

Each sub-goal should:
- Be specific and measurable (clear when completed)
- Represent one type of information or task needed
- Be achievable through tool usage

EXAMPLES:

Query: "What events are happening in Bodrum next week?"
Sub-goals:
1. Find upcoming events in Bodrum
2. Get event dates and details
3. Identify event locations and venues

Query: "I'm a junior investment analyst... planning a due-diligence trip to San Francisco..."
Sub-goals:
1. Find and book round-trip flights from Austin to San Francisco within budget
2. Assess flight weather risk using TAFs for departure/arrival dates
3. Verify startup's Bitcoin UTXO sat ranges and technical claims
4. Check founders' academic credentials and publication records
5. Analyze public partner company financials and valuation
6. Set up WhatsApp group coordination for the trip team

Now break down the user's query into sub-goals.

Output format (strict JSON):
{{
  "sub_goals": [
    "Sub-goal 1",
    "Sub-goal 2",
    "Sub-goal 3"
  ]
}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.generate_async(messages)

            # Try to parse JSON from response
            data = _load_first_json_obj(response)
            self.sub_goals = data.get("sub_goals", [])
            self.remaining = self.sub_goals.copy()
            return self.sub_goals

        except Exception as e:
            print(f"⚠️  Error decomposing query: {e}")
            # Fallback: create simple sub-goals from query
            self.sub_goals = [f"Answer the query: {self.query}"]
            self.remaining = self.sub_goals.copy()
            return self.sub_goals

    async def evaluate_progress(
        self,
        agent_response: str,
        tool_calls: List[Dict]
    ) -> Dict:
        """Evaluate which sub-goals were addressed in current turn."""
        if not self.remaining:
            return {
                "completed_this_turn": [],
                "remaining": [],
                "progress": 1.0
            }

        prompt = f"""You are evaluating progress toward completing a user's query.

ORIGINAL QUERY: {self.query}

REMAINING SUB-GOALS:
{chr(10).join(f"{i+1}. {sg}" for i, sg in enumerate(self.remaining))}

AGENT'S RESPONSE THIS TURN:
{agent_response}

TOOLS USED:
{chr(10).join(f"- {tc.get('server', 'unknown')}/{tc.get('tool', 'unknown')}" for tc in tool_calls) if tool_calls else "No tools used"}

Your task:
Determine which (if any) of the REMAINING sub-goals were COMPLETED by this agent response.

A sub-goal is COMPLETED if:
- Agent provided specific, actionable information addressing it
- Information came from tool usage (not agent's internal knowledge)
- User could reasonably act on this information

A sub-goal is NOT completed if:
- Agent only partially addressed it
- No tools were used (answer from internal knowledge)
- Information is too vague to be actionable

Output format (strict JSON):
{{
  "completed_this_turn": ["exact sub-goal text", "..."],
  "reasoning": "Brief explanation of what was completed and why"
}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.generate_async(messages)

            # Parse JSON
            data = _load_first_json_obj(response)
            completed_this_turn = data.get("completed_this_turn", [])
            for sg in completed_this_turn:
                if sg in self.remaining:
                    self.remaining.remove(sg)
                    self.completed.append(sg)

            return {
                "completed_this_turn": completed_this_turn,
                "remaining": self.remaining.copy(),
                "progress": self.progress_percentage,
                "reasoning": data.get("reasoning", "")
            }

        except Exception as e:
            print(f"⚠️  Error evaluating progress: {e}")
            return {
                "completed_this_turn": [],
                "remaining": self.remaining.copy(),
                "progress": self.progress_percentage,
                "reasoning": ""
            }

    def mark_completed(self, sub_goal: str):
        """Manually mark a sub-goal as completed."""
        if sub_goal in self.remaining:
            self.remaining.remove(sub_goal)
            self.completed.append(sub_goal)

    @property
    def progress_percentage(self) -> float:
        """Overall completion percentage (0.0-1.0)."""
        if not self.sub_goals:
            return 0.0
        return len(self.completed) / len(self.sub_goals)

    @property
    def is_complete(self) -> bool:
        """Check if all sub-goals completed."""
        return len(self.remaining) == 0


# ============================================================================
# Goal-Oriented User
# ============================================================================

class GoalOrientedUser:
    """Simulated user tracking subgoal progress across turns."""

    def __init__(self, llm, persona_name: str, query: str, subgoal_tracker: SubgoalTracker):
        self.llm = llm
        self.persona_name = persona_name
        self.persona = USER_PERSONAS[persona_name]
        self.query = query
        self.subgoal_tracker = subgoal_tracker

    def _format_tool_usage(self, tool_calls: List[Dict] = None) -> str:
        """Format tool usage information for display."""
        if not tool_calls:
            return "❌ NO TOOLS USED - Agent answered from internal knowledge (HIGHLY SUSPICIOUS)"

        tools_list = []
        for tc in tool_calls:
            server = tc.get('server', 'unknown')
            tool = tc.get('tool', 'unknown')
            tools_list.append(f"  ✓ {server}/{tool}")

        return f"Tools used in this turn:\n" + "\n".join(tools_list)

    def _format_goal_progress(self, completed: List[str], remaining: List[str], progress_pct: float) -> str:
        """Format goal progress information."""
        lines = [
            f"Goal Progress: {progress_pct:.0%}",
            "",
            "Completed sub-goals:"
        ]
        if completed:
            for sg in completed:
                lines.append(f"  ✅ {sg}")
        else:
            lines.append("  (none yet)")

        lines.append("")
        lines.append("Remaining sub-goals:")
        if remaining:
            for sg in remaining:
                lines.append(f"  ⏳ {sg}")
        else:
            lines.append("  (all done!)")

        return "\n".join(lines)

    async def evaluate_and_decide(
        self,
        query: str,
        agent_response: str,
        conversation_history: List[Dict],
        current_turn: int,
        tool_calls: List[Dict],
        completed_sub_goals: List[str],
        remaining_sub_goals: List[str],
        goal_progress: float,
        progress_reasoning: str = ""
    ) -> Dict:
        """Evaluate response and decide next action with goal awareness."""

        # Format conversation history
        history_text = "\n\n".join([
            f"Turn {h['turn']}: {h['query']}\nAgent: {h['response'][:200]}..."
            for h in conversation_history[-3:]  # Last 3 turns
        ]) if conversation_history else "No previous turns"

        prompt = f"""You are a simulated user evaluating an AI agent's response and tracking progress on your query.

**YOUR PERSONA:**
- Name: {self.persona_name}
- Description: {self.persona['description']}
- Behavior: {self.persona['behavior']}

**YOUR ORIGINAL QUERY:**
{self.query}

**SUBGOAL PROGRESS:**
{self._format_goal_progress(completed_sub_goals, remaining_sub_goals, goal_progress)}

{f"Progress this turn: {progress_reasoning}" if progress_reasoning else ""}

**CONVERSATION SO FAR:**
{history_text}

**CURRENT TURN ({current_turn}):**
Your Question: {query}

Agent's Response:
{agent_response}

**CURRENT TURN TOOL USAGE:**
{self._format_tool_usage(tool_calls)}

---

**YOUR TASK:**
Evaluate the agent's response and decide whether to CONTINUE or TERMINATE the conversation.

**EVALUATION CRITERIA:**

1. **Subgoal Progress** (Most Important):
   - Did this response help complete any remaining sub-goals?
   - Which sub-goals (if any) were addressed?
   - Are there blocking issues preventing progress?

2. **Tool Usage**:
   - Did agent use external tools OR answer from internal knowledge?
   - If no tools used → VERY SUSPICIOUS (subtract 0.3-0.5 from satisfaction)
   - If tools used → Good!

3. **Response Quality**:
   - Is information specific and actionable?
   - Can I use this to make progress on my query?
   - Is it grounded in tool results?

**SATISFACTION CALCULATION:**
- Base satisfaction: Quality of response (0.0-1.0)
- Tool usage penalty: No tools → -0.3 to -0.5
- Subgoal progress bonus: Completed subgoals → +0.1 to +0.2
- Subgoal progress penalty: No progress → -0.2 to -0.3
- Query completion bonus: All sub-goals done → +0.2

**PENALTY RULES:**
- No tools used → satisfaction ≤ 0.4 (frustrated)
- No subgoal progress despite tools → satisfaction ≤ 0.5 (somewhat frustrated)
- All subgoals achieved → satisfaction ≥ 0.8 (satisfied)

**TERMINATION DECISION:**
- If all sub-goals completed (progress = 100%) → TERMINATE with "subgoals_achieved"
- If satisfaction >= {self.persona['satisfaction_threshold']} → TERMINATE with "satisfied"
- If satisfaction <= {self.persona['frustration_threshold']} → TERMINATE with "frustrated"
- If turn >= {self.persona['max_turns']} → TERMINATE with "max_turns"
- Otherwise → CONTINUE with follow-up question

**FOLLOW-UP QUESTION (if continuing):**
Your follow-up should work toward completing REMAINING sub-goals.
- Pick a remaining sub-goal that hasn't been addressed yet
- Ask a natural question that helps the agent work on that sub-goal
- Be conversational and build on what the agent has already provided

**OUTPUT FORMAT (strict JSON):**
{{
  "decision": "CONTINUE" or "TERMINATE",
  "termination_reason": "goal_achieved|satisfied|frustrated|max_turns|null",
  "follow_up_query": "Your next question..." (if CONTINUE),
  "intent": "goal_progress|clarification|expansion|verification",
  "satisfaction_level": 0.0-1.0,
  "reasoning": "Explain your evaluation and decision (2-3 sentences)"
}}

Evaluate the agent's response now."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.generate_async(messages)

            # Parse JSON
            decision = _load_first_json_obj(response)

            # Validate required fields
            if "decision" not in decision:
                decision["decision"] = "TERMINATE"
                decision["termination_reason"] = "error"

            return decision

        except Exception as e:
            print(f"⚠️  Error in user evaluation: {e}")
            # Default to terminate on error
            return {
                "decision": "TERMINATE",
                "termination_reason": "error",
                "follow_up_query": None,
                "intent": None,
                "satisfaction_level": 0.5,
                "reasoning": f"Error during evaluation: {str(e)}"
            }


# ============================================================================
# Goal-Oriented Controller
# ============================================================================

class GoalOrientedController:
    """Orchestrates goal-oriented multi-turn conversations with subgoal tracking."""

    def __init__(
        self,
        agent: DynamicReActAgent,
        goal_oriented_user: GoalOrientedUser,
        subgoal_tracker: SubgoalTracker,
        max_turns: int
    ):
        self.agent = agent
        self.user = goal_oriented_user
        self.subgoal_tracker = subgoal_tracker
        self.max_turns = max_turns
        self.turns: List[GoalTurn] = []

    def _format_history(self) -> List[Dict]:
        """Format conversation history for user evaluation."""
        return [
            {
                "turn": turn.turn_number,
                "query": turn.query,
                "response": turn.agent_response,
                "satisfaction": turn.satisfaction_level
            }
            for turn in self.turns
        ]

    async def run_conversation(self, seed_query: str) -> GoalTrajectory:
        """Run a complete goal-oriented conversation."""
        print("\n" + "="*70)
        print("🎯 STARTING GOAL-ORIENTED CONVERSATION")
        print("="*70)
        print(f"Seed Query: {seed_query}")
        print(f"Original Query: {self.user.query}")
        print(f"Persona: {self.user.persona_name}")
        print("="*70 + "\n")

        # Reset turns for new conversation
        self.turns = []

        # Step 1: Decompose query into sub-goals
        print("📋 Decomposing query into sub-goals...")
        await self.subgoal_tracker.decompose_query()
        print(f"✓ Identified {len(self.subgoal_tracker.sub_goals)} sub-goals:")
        for i, sg in enumerate(self.subgoal_tracker.sub_goals, 1):
            print(f"  {i}. {sg}")
        print()

        # Step 2: Initialize conversation
        current_query = seed_query

        # Step 3: Multi-turn loop
        for turn_num in range(1, self.max_turns + 1):
            print(f"\n{'='*70}")
            print(f"TURN {turn_num}/{self.max_turns}")
            print(f"{'='*70}\n")

            # Capture tool cache state at start of turn
            available_servers = list(self.agent.loaded_servers) if hasattr(self.agent, 'loaded_servers') else []
            available_tool_count = sum(
                len(tools) for tools in self.agent._tools.values()
            ) if hasattr(self.agent, '_tools') else 0

            print(f"🔧 Available servers: {', '.join(available_servers)}")
            print(f"🔧 Total tools: {available_tool_count}\n")

            print(f"👤 User Query: {current_query}\n")

            # 3a. Agent responds
            print("🤖 Agent thinking...")

            # Reset trajectory for this turn
            self.agent.trajectory = []
            self.agent.reasoning_trace = []

            response = await self.agent.execute(current_query)
            agent_response = response.response
            print(f"\n🤖 Agent Response:\n{agent_response}\n")

            # 3b. Extract tool calls and results
            turn_tool_calls = [
                {
                    "server": entry.get("server", ""),
                    "tool": entry.get("tool", ""),
                    "arguments": entry.get("arguments", {}),
                    "status": entry.get("status", "unknown")
                }
                for entry in self.agent.trajectory
                if entry.get("type") == "tool_call"
            ]

            turn_reasoning = [
                entry for entry in self.agent.reasoning_trace
            ]

            turn_tool_results = [
                {"step": entry.get("type", ""), "result": entry.get("content", "")}
                for entry in turn_reasoning
                if entry.get("type") == "result"
            ]

            # 3c. Evaluate subgoal progress
            print("📊 Evaluating subgoal progress...")
            progress_info = await self.subgoal_tracker.evaluate_progress(
                agent_response, turn_tool_calls
            )

            completed_this_turn = progress_info["completed_this_turn"]
            if completed_this_turn:
                print(f"✅ Completed sub-goals: {', '.join(completed_this_turn)}")
            else:
                print("⏳ No sub-goals completed this turn")
            print(f"📈 Overall progress: {progress_info['progress']:.0%}\n")

            # 3d. User evaluates and decides
            print("👤 User evaluating response...")
            decision = await self.user.evaluate_and_decide(
                query=current_query,
                agent_response=agent_response,
                conversation_history=self._format_history(),
                current_turn=turn_num,
                tool_calls=turn_tool_calls,
                completed_sub_goals=self.subgoal_tracker.completed,
                remaining_sub_goals=self.subgoal_tracker.remaining,
                goal_progress=progress_info["progress"],
                progress_reasoning=progress_info.get("reasoning", "")
            )

            print(f"👤 Satisfaction: {decision['satisfaction_level']:.2f}")
            print(f"👤 Decision: {decision['decision']}")
            print(f"👤 Reasoning: {decision['reasoning']}\n")

            # 3e. Create GoalTurn record
            turn = GoalTurn(
                turn_number=turn_num,
                query=current_query,
                agent_response=agent_response,
                tool_calls=turn_tool_calls,
                reasoning_trace=turn_reasoning,
                tool_results=turn_tool_results,
                available_servers=available_servers,
                available_tool_count=available_tool_count,
                completed_sub_goals=completed_this_turn,
                remaining_sub_goals=self.subgoal_tracker.remaining.copy(),
                goal_progress=progress_info["progress"],
                user_decision=decision["decision"],
                termination_reason=decision.get("termination_reason"),
                satisfaction_level=decision["satisfaction_level"],
                user_reasoning=decision["reasoning"],
                follow_up_intent=decision.get("intent")
            )
            self.turns.append(turn)

            # 3f. Check termination
            if decision["decision"] == "TERMINATE":
                reason = decision.get("termination_reason", "unknown")
                print(f"🛑 Conversation terminated: {reason}")
                break

            # 3g. Update for next turn
            current_query = decision.get("follow_up_query", "")
            if not current_query:
                print("⚠️  No follow-up query provided, terminating")
                break

        # Step 4: Build final trajectory
        print("\n" + "="*70)
        print("📝 CONVERSATION SUMMARY")
        print("="*70)
        print(f"Total turns: {len(self.turns)}")
        print(f"Subgoal completion: {self.subgoal_tracker.progress_percentage:.0%}")
        print(f"Final satisfaction: {self.turns[-1].satisfaction_level:.2f}")

        dynamically_loaded = list(getattr(self.agent, 'dynamically_loaded_servers', set()))

        trajectory = GoalTrajectory(
            conversation_id=f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            seed_query=seed_query,
            user_persona=self.user.persona_name,
            user_goal=self.user.query,  # Store original query as "user_goal"
            sub_goals=self.subgoal_tracker.sub_goals,
            goal_completion_rate=self.subgoal_tracker.progress_percentage,
            goal_achieved=self.subgoal_tracker.is_complete,
            turns=self.turns,
            total_turns=len(self.turns),
            final_decision=self.turns[-1].user_decision if self.turns else "NONE",
            final_satisfaction=self.turns[-1].satisfaction_level if self.turns else 0.0,
            timestamp=datetime.now().isoformat(),
            agent_model=getattr(self.agent, 'model_name', 'unknown'),
            user_model='anthropic/claude-3.5-sonnet',
            dynamically_loaded_servers=dynamically_loaded
        )

        print("="*70 + "\n")

        return trajectory


# ============================================================================
# Main Entry Point
# ============================================================================

async def run_single_conversation(
    query_item: Dict,
    query_index: int,
    args,
    all_server_configs: Dict,  # Not used but kept for compatibility
    semaphore: asyncio.Semaphore
) -> Dict:
    """Run a single goal-oriented conversation with subprocess to avoid shared state issues."""
    async with semaphore:
        # Prepare temporary file for this query item
        import tempfile
        import os

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        temp_data = {
            "items": [query_item]
        }
        json.dump(temp_data, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()

        try:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "runtime" / "run_goaloriented_agent.py"),
                "--seeds", temp_file.name,
                "--query-index", "0",
                "--persona", args.persona,
                "--model", args.model,
                "--user-model", args.user_model,
                "--max-iterations", str(args.max_iterations),
            ]

            if args.max_turns:
                cmd.extend(["--max-turns", str(args.max_turns)])

            if args.save_trajectory:
                cmd.append("--save-trajectory")

            if args.batch_id:
                cmd.extend(["--batch-id", args.batch_id])

            query_uuid = query_item.get("uuid", f"query_{query_index}")
            print(f"[{query_index + 1}] Starting conversation for query {query_uuid}...")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            _, stderr = await process.communicate()

            if process.returncode == 0:
                print(f"[{query_index + 1}] ✓ Conversation {query_uuid} completed successfully")
                return {
                    "index": query_index + 1,
                    "uuid": query_uuid,
                    "status": "success"
                }
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                print(f"[{query_index + 1}] ✗ Conversation {query_uuid} failed: {error_msg[:200]}")
                return {
                    "index": query_index + 1,
                    "uuid": query_uuid,
                    "status": "failed",
                    "error": error_msg[:500]
                }
        except Exception as e:
            query_uuid = query_item.get("uuid", f"query_{query_index}")
            print(f"[{query_index + 1}] ✗ Conversation {query_uuid} failed with exception: {e}")
            return {
                "index": query_index + 1,
                "uuid": query_uuid,
                "status": "failed",
                "error": str(e)
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass


async def main():
    parser = argparse.ArgumentParser(
        description="Run Goal-Oriented Multi-Turn Agent"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Initial seed query"
    )
    parser.add_argument(
        "--seeds",
        help="JSON file with seed queries (multitool format without 'goal' field)"
    )
    parser.add_argument(
        "--query-index",
        type=int,
        help="Index of query to run from the JSON file (0-based, for single execution)"
    )
    parser.add_argument(
        "--persona",
        default="curious_researcher",
        choices=list(USER_PERSONAS.keys()),
        help="Simulated user persona"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum conversation turns (overrides persona default)"
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="Model for agent (OpenRouter format)"
    )
    parser.add_argument(
        "--user-model",
        default="anthropic/claude-3.5-sonnet",
        help="Model for simulated user (OpenRouter format)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max agent reasoning iterations per turn"
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save conversation trajectory to JSON"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent conversations (default: 5)"
    )
    parser.add_argument(
        "--batch-id",
        help="Batch ID for grouping related trajectories together"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv(str(ORCHESTRATOR_DIR / ".env"))

    # Load queries - now expecting multitool format (no 'goal' field)
    if args.seeds:
        with open(args.seeds) as f:
            data = json.load(f)
            items = data["items"]

        # If query-index specified, run single conversation
        if args.query_index is not None:
            if args.query_index < 0 or args.query_index >= len(items):
                parser.error(f"Query index {args.query_index} out of range (0-{len(items)-1})")
            items = [items[args.query_index]]
    elif args.query:
        # Single query from command line (no goal needed - will be generated by user simulator)
        items = [{"query": args.query}]
    else:
        parser.error("Must provide either 'query' or --seeds file")

    print(f"\n🎯 Goal-Oriented Multi-Turn Agent")
    print(f"{'='*70}")
    print(f"Agent model: {args.model}")
    print(f"User model: {args.user_model}")
    print(f"Persona: {args.persona}")
    print(f"Queries to run: {len(items)}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"{'='*70}\n")

    # Setup - load simple server list and convert to config format
    server_list = json.loads(
        (PROJECT_ROOT / "MCP_INFO_MGR/mcp_data/working/remote_servers.json").read_text()
    )
    all_server_configs = {
        server: {
            "streamable_http": {"url": f"https://server.smithery.ai/{server}", "headers": {}},
            "env": {}
        }
        for server in server_list
    }

    # Determine execution mode
    if args.query_index is not None or len(items) == 1:
        # Single conversation mode - run directly without subprocess
        trajectories = []
        results_summary = []

        query_idx = 0
        item = items[0]
        seed_query = item["query"]

        print(f"\n{'#'*70}")
        print(f"SINGLE CONVERSATION MODE")
        print(f"{'#'*70}\n")

        try:
            # Create fresh components for this conversation
            mcp_manager = MCPManager()
            model_manager = ModelManager()

            # Add Meta-MCP server
            meta_mcp_config = {
                "stdio": {
                    "command": "python",
                    "args": [str(PROJECT_ROOT / "meta_mcp_server" / "server.py")],
                }
            }
            mcp_manager.add_server_config("meta-mcp", meta_mcp_config)

            agent_llm = model_manager.build_model("openrouter", config={"model_name": args.model})
            user_llm = model_manager.build_model("openrouter", config={"model_name": args.user_model})

            # React config matching run_react_agent
            react_config = {
                "name": "goal-oriented-react-agent",
                "instruction": AGENT_INSTRUCTION,
                "max_iterations": args.max_iterations,
                "summarize_tool_response": "auto",  # Enable smart summarization
                "summarize_threshold": 100000,      # Summarize if response > 100k chars
            }

            # Create fresh agent
            fresh_agent = DynamicReActAgent(
                mcp_manager=mcp_manager,
                llm=agent_llm,
                server_configs=all_server_configs,
                config=react_config,
            )
            await fresh_agent.initialize(mcp_servers=[{"name": "meta-mcp"}])

            # Create subgoal tracker (will decompose query into subgoals)
            subgoal_tracker = SubgoalTracker(llm=user_llm, query=seed_query)

            # Create goal-oriented user
            max_turns = args.max_turns or USER_PERSONAS[args.persona]["max_turns"]
            goal_user = GoalOrientedUser(
                llm=user_llm,
                persona_name=args.persona,
                query=seed_query,
                subgoal_tracker=subgoal_tracker
            )

            # Create controller
            controller = GoalOrientedController(
                agent=fresh_agent,
                goal_oriented_user=goal_user,
                subgoal_tracker=subgoal_tracker,
                max_turns=max_turns
            )

            # Run conversation
            trajectory = await controller.run_conversation(seed_query)

            # Cleanup this agent
            try:
                await fresh_agent.cleanup()
            except Exception as e:
                print(f"⚠️  Warning during agent cleanup: {e}")

            # Print summary
            print(f"\n{'='*70}")
            print(f"CONVERSATION SUMMARY")
            print(f"{'='*70}")
            print(f"Total Turns: {trajectory.total_turns}")
            print(f"Goal Completion: {trajectory.goal_completion_rate:.0%}")
            print(f"Final Satisfaction: {trajectory.final_satisfaction:.2f}")
            print(f"{'='*70}\n")

            trajectories.append(trajectory)
            results_summary.append({
                "query_index": query_idx,
                "seed_query": seed_query,
                "success": trajectory.goal_achieved or trajectory.final_satisfaction >= 0.7,
                "total_turns": trajectory.total_turns,
                "goal_completion_rate": trajectory.goal_completion_rate,
                "final_satisfaction": trajectory.final_satisfaction,
            })

            # Save trajectory
            if args.save_trajectory:
                output_dir = PROJECT_ROOT / "trajectories/goaloriented"
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = output_dir / f"{trajectory.conversation_id}.json"
                with open(output_file, "w") as f:
                    json.dump(trajectory.to_dict(), f, indent=2, ensure_ascii=False)

                print(f"💾 Saved trajectory: {output_file}")

        except Exception as e:
            print(f"\n✗ Error processing query {query_idx + 1}: {e}\n")
            import traceback
            traceback.print_exc()
            results_summary.append({
                "query_index": query_idx,
                "seed_query": seed_query,
                "success": False,
                "error": str(e),
            })

    else:
        # Batch mode - run multiple conversations in parallel with subprocesses
        print(f"\n{'#'*70}")
        print(f"BATCH MODE - PARALLEL EXECUTION")
        print(f"{'#'*70}\n")

        import uuid as uuid_module
        batch_id = args.batch_id or str(uuid_module.uuid4())[:8]

        print(f"Batch ID: {batch_id}")
        print(f"Total conversations: {len(items)}")
        print(f"Max concurrent: {args.max_concurrent}\n")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(args.max_concurrent)

        # Create tasks for all conversations
        tasks = []
        for idx, query_item in enumerate(items):
            # Ensure each item has UUID
            if "uuid" not in query_item:
                query_item["uuid"] = str(uuid_module.uuid4())

            task = run_single_conversation(
                query_item=query_item,
                query_index=idx,
                args=args,
                all_server_configs=all_server_configs,
                semaphore=semaphore
            )
            tasks.append(task)

        # Run all tasks in parallel
        print(f"Starting {len(items)} conversations with max concurrency of {args.max_concurrent}...\n")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results (handle any exceptions)
        results_summary = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results_summary.append({
                    "index": i + 1,
                    "uuid": items[i].get("uuid"),
                    "status": "failed",
                    "error": str(result)
                })
            else:
                results_summary.append(result)

        trajectories = []  # Trajectories are saved individually in subprocess mode

    # Print batch summary if multiple queries
    if len(items) > 1:
        print(f"\n{'='*70}")
        print("BATCH SUMMARY")
        print(f"{'='*70}")
        print(f"Total queries: {len(items)}")
        print(f"Successful: {sum(1 for r in results_summary if r.get('success', False))}")
        print(f"Failed: {sum(1 for r in results_summary if not r.get('success', False))}")
        if results_summary:
            successful = [r for r in results_summary if r.get('success', False)]
            if successful:
                avg_completion = sum(r.get('goal_completion_rate', 0) for r in successful) / len(successful)
                avg_satisfaction = sum(r.get('final_satisfaction', 0) for r in successful) / len(successful)
                print(f"Average goal completion (successful): {avg_completion:.0%}")
                print(f"Average satisfaction (successful): {avg_satisfaction:.2f}")
        print(f"{'='*70}\n")

        # Save batch summary
        if args.save_trajectory:
            summary_file = PROJECT_ROOT / "trajectories" / "goaloriented" / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "persona": args.persona,
                    "total_queries": len(items),
                    "results": results_summary,
                }, f, indent=2, ensure_ascii=False)
            print(f"✓ Batch summary saved to: {summary_file}\n")

    print(f"{'='*70}")
    print("✅ All conversations complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
