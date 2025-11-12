#!/usr/bin/env python3
"""
Run Goal-Oriented Multi-Turn Agent

A goal-driven conversational agent where the simulated user has a hidden goal
that drives follow-up questions and evaluation.

Features:
- Hidden user goal decomposed into measurable sub-goals
- Progress tracking across conversation turns
- Evaluation based on: goal progress + tool usage + quality
- Terminates when goal achieved OR user frustrated

Usage:
    # Single query with goal
    python runtime/run_goaloriented_agent.py "What events are in Bodrum?" \
        --goal "I'm planning a trip and need events, weather, restaurants, hotels" \
        --persona curious_researcher

    # Batch from seeds file
    python runtime/run_goaloriented_agent.py \
        --seeds mcp_generate/requests/goaloriented_seeds_test.json \
        --persona curious_researcher \
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

class GoalTracker:
    """Tracks sub-goal completion throughout conversation."""

    def __init__(self, llm, goal: str):
        self.llm = llm
        self.goal = goal
        self.sub_goals: List[str] = []
        self.completed: List[str] = []
        self.remaining: List[str] = []

    async def decompose_goal(self) -> List[str]:
        """Use LLM to break goal into 3-6 measurable sub-goals."""
        prompt = f"""You are analyzing a user's goal to break it into measurable sub-goals.

USER GOAL:
{self.goal}

Your task:
Break this goal into 3-6 clear, measurable sub-goals that can be achieved through information gathering.

Each sub-goal should:
- Be specific and measurable (clear when completed)
- Represent one type of information needed
- Be achievable through tool usage

EXAMPLES:

Goal: "I'm planning a trip to Bodrum and need events, weather, restaurants, hotels"
Sub-goals:
1. Find upcoming events in Bodrum
2. Check weather forecast for Bodrum
3. Get restaurant recommendations in Bodrum
4. Find hotel options in Bodrum

Goal: "I'm researching small language models and need papers, methods, researchers, trends"
Sub-goals:
1. Find recent papers on small language models
2. Understand key methods being used
3. Identify top researchers in the field
4. Analyze current research trends

Now break down the user's goal.

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
            print(f"⚠️  Error decomposing goal: {e}")
            # Fallback: create simple sub-goals from goal text
            self.sub_goals = [f"Complete task: {self.goal}"]
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

        prompt = f"""You are evaluating progress toward a user's goal.

USER GOAL: {self.goal}

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
    """Simulated user with hidden goal driving questions."""

    def __init__(self, llm, persona_name: str, goal: str, goal_tracker: GoalTracker):
        self.llm = llm
        self.persona_name = persona_name
        self.persona = USER_PERSONAS[persona_name]
        self.goal = goal
        self.goal_tracker = goal_tracker

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

        prompt = f"""You are a simulated user with a HIDDEN GOAL evaluating an AI agent's response.

**YOUR PERSONA:**
- Name: {self.persona_name}
- Description: {self.persona['description']}
- Behavior: {self.persona['behavior']}

**YOUR HIDDEN GOAL (agent doesn't know this):**
{self.goal}

**GOAL PROGRESS:**
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

1. **Goal Progress** (Most Important):
   - Did this response move me closer to completing my goal?
   - Which sub-goals (if any) did it help complete?
   - Are there blocking issues preventing goal progress?

2. **Tool Usage**:
   - Did agent use external tools OR answer from internal knowledge?
   - If no tools used → VERY SUSPICIOUS (subtract 0.3-0.5 from satisfaction)
   - If tools used → Good!

3. **Response Quality**:
   - Is information specific and actionable?
   - Can I use this to make progress on my goal?
   - Is it grounded in tool results?

**SATISFACTION CALCULATION:**
- Base satisfaction: Quality of response (0.0-1.0)
- Tool usage penalty: No tools → -0.3 to -0.5
- Goal progress bonus: Moved toward goal → +0.1 to +0.2
- Goal progress penalty: No progress toward goal → -0.2 to -0.3
- Goal completion bonus: All sub-goals done → +0.2

**PENALTY RULES:**
- No tools used → satisfaction ≤ 0.4 (frustrated)
- No goal progress despite tools → satisfaction ≤ 0.5 (somewhat frustrated)
- Goal achieved → satisfaction ≥ 0.8 (satisfied)

**TERMINATION DECISION:**
- If all sub-goals completed (progress = 100%) → TERMINATE with "goal_achieved"
- If satisfaction >= {self.persona['satisfaction_threshold']} → TERMINATE with "satisfied"
- If satisfaction <= {self.persona['frustration_threshold']} → TERMINATE with "frustrated"
- If turn >= {self.persona['max_turns']} → TERMINATE with "max_turns"
- Otherwise → CONTINUE with follow-up question

**FOLLOW-UP QUESTION (if continuing):**
Your follow-up should work toward completing REMAINING sub-goals.
- Pick a remaining sub-goal
- Ask a question that helps agent discover tools needed for that sub-goal
- Be natural and conversational (don't reveal your hidden goal)

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
    """Orchestrates goal-oriented multi-turn conversations."""

    def __init__(
        self,
        agent: DynamicReActAgent,
        goal_oriented_user: GoalOrientedUser,
        goal_tracker: GoalTracker,
        max_turns: int
    ):
        self.agent = agent
        self.user = goal_oriented_user
        self.goal_tracker = goal_tracker
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
        print(f"Hidden Goal: {self.user.goal}")
        print(f"Persona: {self.user.persona_name}")
        print("="*70 + "\n")

        # Reset turns for new conversation
        self.turns = []

        # Step 1: Decompose goal into sub-goals
        print("📋 Decomposing goal into sub-goals...")
        await self.goal_tracker.decompose_goal()
        print(f"✓ Identified {len(self.goal_tracker.sub_goals)} sub-goals:")
        for i, sg in enumerate(self.goal_tracker.sub_goals, 1):
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

            # 3c. Evaluate goal progress
            print("📊 Evaluating goal progress...")
            progress_info = await self.goal_tracker.evaluate_progress(
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
                completed_sub_goals=self.goal_tracker.completed,
                remaining_sub_goals=self.goal_tracker.remaining,
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
                remaining_sub_goals=self.goal_tracker.remaining.copy(),
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
        print(f"Goal completion: {self.goal_tracker.progress_percentage:.0%}")
        print(f"Final satisfaction: {self.turns[-1].satisfaction_level:.2f}")

        dynamically_loaded = list(getattr(self.agent, 'dynamically_loaded_servers', set()))

        trajectory = GoalTrajectory(
            conversation_id=f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            seed_query=seed_query,
            user_persona=self.user.persona_name,
            user_goal=self.user.goal,
            sub_goals=self.goal_tracker.sub_goals,
            goal_completion_rate=self.goal_tracker.progress_percentage,
            goal_achieved=self.goal_tracker.is_complete,
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
        "--goal",
        help="User's hidden goal (if not using --seeds)"
    )
    parser.add_argument(
        "--seeds",
        help="JSON file with seed queries and goals"
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
        default=10,
        help="Max agent reasoning iterations per turn"
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save conversation trajectory to JSON"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv(str(ORCHESTRATOR_DIR / ".env"))

    # Load queries and goals
    if args.seeds:
        with open(args.seeds) as f:
            data = json.load(f)
            items = data["items"]
    elif args.query and args.goal:
        items = [{"query": args.query, "goal": args.goal}]
    else:
        parser.error("Must provide either (query + goal) or --seeds file")

    print(f"\n🎯 Goal-Oriented Multi-Turn Agent")
    print(f"{'='*70}")
    print(f"Agent model: {args.model}")
    print(f"User model: {args.user_model}")
    print(f"Persona: {args.persona}")
    print(f"Queries to run: {len(items)}")
    print(f"{'='*70}\n")

    # Setup
    all_server_configs = json.loads(
        (PROJECT_ROOT / "MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json").read_text()
    )

    # Run each conversation
    trajectories = []
    results_summary = []

    for query_idx, item in enumerate(items):
        seed_query = item["query"]
        user_goal = item["goal"]

        print(f"\n{'#'*70}")
        print(f"CONVERSATION {query_idx + 1}/{len(items)}")
        print(f"{'#'*70}\n")

        try:
            # Create fresh components for each conversation
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

            react_config = {"max_iterations": args.max_iterations}

            # Create fresh agent
            fresh_agent = DynamicReActAgent(
                mcp_manager=mcp_manager,
                llm=agent_llm,
                server_configs=all_server_configs,
                config=react_config,
            )
            await fresh_agent.initialize(mcp_servers=[{"name": "meta-mcp"}])

            # Create goal tracker
            goal_tracker = GoalTracker(llm=user_llm, goal=user_goal)

            # Create goal-oriented user
            max_turns = args.max_turns or USER_PERSONAS[args.persona]["max_turns"]
            goal_user = GoalOrientedUser(
                llm=user_llm,
                persona_name=args.persona,
                goal=user_goal,
                goal_tracker=goal_tracker
            )

            # Create controller
            controller = GoalOrientedController(
                agent=fresh_agent,
                goal_oriented_user=goal_user,
                goal_tracker=goal_tracker,
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
                "user_goal": user_goal,
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
                "user_goal": user_goal,
                "success": False,
                "error": str(e),
            })

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
