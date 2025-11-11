#!/usr/bin/env python3
"""
Run Multi-Turn Agent with Real-Time User Simulation

Uses existing DynamicReActAgent with a conversation controller that:
1. Simulates user with LLM (real-time follow-ups)
2. Maintains conversation history across turns
3. Decides termination based on user satisfaction

Usage:
    python runtime/run_multiturn_agent.py "Find GitHub repos about ML" --persona curious_researcher
"""
from __future__ import annotations

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

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
# User Personas
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
class Turn:
    """Single turn in multi-turn conversation."""
    turn_number: int
    query: str
    agent_response: str
    tool_calls: List[Dict]  # From agent.trajectory
    reasoning_trace: List[Dict]  # From agent.reasoning_trace
    tool_results: List[Dict]  # Extracted tool results for easier access

    # Tool cache state at start of turn
    available_servers: List[str]  # All loaded servers at turn start
    available_tool_count: int  # Total number of tools available

    # User simulation outputs
    user_decision: str  # "CONTINUE" | "TERMINATE"
    termination_reason: Optional[str]  # "satisfied" | "frustrated" | "natural_end"
    satisfaction_level: float
    user_reasoning: str
    follow_up_intent: Optional[str]  # "clarification" | "drill_down" | etc.


@dataclass
class MultiTurnTrajectory:
    """Complete multi-turn conversation trajectory."""
    conversation_id: str
    seed_query: str
    user_persona: str

    turns: List[Turn]

    # Final outcome
    total_turns: int
    final_decision: str
    final_satisfaction: float

    # Metadata
    timestamp: str
    agent_model: str
    user_model: str
    dynamically_loaded_servers: List[str]


# ============================================================================
# Simulated User LLM
# ============================================================================

class SimulatedUser:
    """LLM-based user simulator with persona."""

    def __init__(self, llm, persona_name: str):
        self.llm = llm
        self.persona_name = persona_name
        self.persona = USER_PERSONAS[persona_name]

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

    def _build_prompt(
        self,
        seed_query: str,
        turn_number: int,
        conversation_history: List[Turn],
        agent_response: str,
        current_turn_tool_calls: List[Dict] = None,
    ) -> str:
        """Build prompt for simulated user at current turn."""

        # Build conversation context with tool usage info
        history_text = ""
        if conversation_history:
            for turn in conversation_history:
                history_text += f"\nTurn {turn.turn_number}:\n"
                history_text += f"  You asked: {turn.query}\n"
                history_text += f"  Agent responded: {turn.agent_response[:200]}...\n"

                # Add tool usage information
                if turn.tool_calls:
                    tools_used = [f"{tc.get('server', 'unknown')}/{tc.get('tool', 'unknown')}"
                                  for tc in turn.tool_calls]
                    history_text += f"  Tools used: {', '.join(tools_used)}\n"
                else:
                    history_text += f"  Tools used: NONE (answered from internal knowledge)\n"

                history_text += f"  Your satisfaction: {turn.satisfaction_level:.2f}\n"

        prompt = f"""You are simulating a {self.persona_name} who is having a conversation with an AI agent.

Persona Description: {self.persona['description']}
Behavior: {self.persona['behavior']}

Original Goal (Seed Query): {seed_query}

Conversation So Far:
{history_text}

Current Turn {turn_number}:
Agent's Latest Response:
{agent_response}

**CURRENT TURN TOOL USAGE:**
{self._format_tool_usage(current_turn_tool_calls)}

Your role now:
1. ANALYZE the agent's response to your question
2. EVALUATE how well it answered and what's missing
3. DECIDE: Continue with a follow-up question OR terminate the conversation
4. If continuing: GENERATE a natural, contextual follow-up question based on the agent's actual response

CRITICAL EVALUATION CRITERIA - Tool Usage and External Sources:
⚠️ **You must be EXTREMELY CAUTIOUS about answers that come from the agent's internal knowledge**

When evaluating the agent's response, CHECK:
1. Did the agent use ANY tools to answer the question?
   - If NO tools were used → Lower satisfaction significantly (subtract 0.3-0.5)
   - The agent should be using external tools, NOT just its training data

2. Does the answer contain specific, verifiable information that requires external lookup?
   - Real-time data (current events, weather, prices, etc.) → MUST use tools
   - Specific database lookups (papers, spells, company info) → MUST use tools
   - General knowledge questions → Tools preferred but not always required

3. Is the information grounded in tool results or made up?
   - If the agent provides specific details without using tools → Be VERY suspicious
   - Lower satisfaction if answer seems plausible but lacks tool backing

4. Examples of UNACCEPTABLE responses (no tool usage):
   ❌ "Here's what I know about D&D spells..." (internal knowledge)
   ❌ "Based on my training data..." (not using tools)
   ❌ "Here are some climate change effects..." (general knowledge, should search)

5. Examples of ACCEPTABLE responses:
   ✓ Used search_tools to find relevant MCP servers
   ✓ Used specific tools to fetch current/verified data
   ✓ Cited specific results from tool calls

**PENALTY RULES:**
- No tools used when they should be → satisfaction ≤ 0.4 (frustrated)
- Made up specific details without tools → satisfaction ≤ 0.3 (very frustrated)
- Used tools appropriately → satisfaction can be normal (0.6-1.0)

IMPORTANT - Tool Exploration Behavior:
- Your goal is to test the agent's ability to DISCOVER and USE DIFFERENT TOOLS
- After the agent uses a tool successfully, ask questions that require DIFFERENT capabilities
- Examples of follow-up patterns that encourage tool exploration:
  * If agent used a search tool → ask for details about a specific result (may need a fetch/read tool)
  * If agent fetched data → ask to analyze or transform it (may need a data processing tool)
  * If agent used one API → ask about related but different information (may need a different API)
  * If agent searched code → ask to modify or run it (may need execution tools)
  * If agent got weather data → ask about news or events in that location (different domain)

- Try to naturally expand the conversation into adjacent domains that require new tools
- Think: "What related question would need a DIFFERENT type of tool?"

Termination Guidelines:
- SATISFIED: terminate if satisfaction >= {self.persona['satisfaction_threshold']} and no more questions
- FRUSTRATED: terminate if satisfaction <= {self.persona['frustration_threshold']}
- NATURAL_END: terminate if the conversation naturally concluded
- MAX_TURNS: you're at turn {turn_number}/{self.persona['max_turns']}

Output Format (strict JSON):
{{
  "decision": "CONTINUE" | "TERMINATE",
  "termination_reason": "satisfied|frustrated|natural_end|max_turns" (only if TERMINATE),
  "follow_up_query": "your next question" (only if CONTINUE),
  "intent": "clarification|drill_down|expansion|verification|correction|explore_tools",
  "satisfaction_level": <float 0.0-1.0>,
  "reasoning": "Why you're asking this OR why you're done (2-3 sentences)"
}}

REMEMBER: Use "explore_tools" intent when your follow-up question is designed to make the agent discover and use a DIFFERENT type of tool than what was used before.

Respond with ONLY the JSON, no other text."""

        return prompt

    async def evaluate_and_decide(
        self,
        seed_query: str,
        turn_number: int,
        conversation_history: List[Turn],
        agent_response: str,
        current_turn_tool_calls: List[Dict] = None,
    ) -> Dict:
        """Evaluate agent's response and decide next action."""

        prompt = self._build_prompt(
            seed_query, turn_number, conversation_history, agent_response, current_turn_tool_calls
        )

        # Get LLM response
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm.generate_async(messages)
        response_text = response.strip()

        # Parse JSON
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        try:
            decision = json.loads(response_text)
            return decision
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse user decision JSON: {e}")
            print(f"Raw response: {response_text}")
            # Default to termination on parse error
            return {
                "decision": "TERMINATE",
                "termination_reason": "parse_error",
                "satisfaction_level": 0.0,
                "reasoning": f"Failed to parse user decision: {e}"
            }


# ============================================================================
# Conversation Controller
# ============================================================================

class ConversationController:
    """
    Orchestrates multi-turn conversation between DynamicReActAgent and simulated user.

    Uses existing DynamicReActAgent without modification - just wraps it with
    conversation history management and user simulation.
    """

    def __init__(
        self,
        agent: DynamicReActAgent,
        simulated_user: SimulatedUser,
        max_turns: int = 10,
    ):
        self.agent = agent
        self.simulated_user = simulated_user
        self.max_turns = max_turns

        self.turns: List[Turn] = []

    def _build_agent_context_prompt(self, turn_number: int) -> str:
        """Build context-aware prompt for agent."""

        if turn_number == 1:
            return ""  # First turn, no history

        context = "\n\nCONVERSATION HISTORY (use this context for follow-up questions):\n"
        for turn in self.turns:
            context += f"\nTurn {turn.turn_number}:\n"
            context += f"  User: {turn.query}\n"
            context += f"  You: {turn.agent_response[:150]}...\n"

        context += "\nIMPORTANT: The user's current question may reference previous turns. Use the conversation history to understand context.\n"

        return context

    async def run_conversation(self, seed_query: str) -> MultiTurnTrajectory:
        """Run complete multi-turn conversation."""

        # Reset turns for new conversation
        self.turns = []

        # No need to reset agent tool cache - each conversation uses a fresh agent instance
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n{'='*60}")
        print(f"Starting Multi-Turn Conversation")
        print(f"{'='*60}")
        print(f"ID: {conversation_id}")
        print(f"Seed Query: {seed_query}")
        print(f"User Persona: {self.simulated_user.persona_name}")
        print(f"Max Turns: {self.max_turns}")
        print(f"{'='*60}\n")

        current_query = seed_query
        turn_number = 0

        while turn_number < self.max_turns:
            turn_number += 1

            print(f"\n{'─'*60}")
            print(f"Turn {turn_number}")
            print(f"{'─'*60}")
            print(f"User Query: {current_query}\n")

            # Add conversation context to agent's query
            context_prompt = self._build_agent_context_prompt(turn_number)
            agent_query = current_query + context_prompt

            # Agent executes (using existing DynamicReActAgent)
            print("Agent is thinking and executing tools...\n")

            # Capture tool cache state at start of turn
            available_servers = list(self.agent.loaded_servers) if hasattr(self.agent, 'loaded_servers') else []
            available_tool_count = sum(
                len(tools) for tools in self.agent._tools.values()
            ) if hasattr(self.agent, '_tools') else 0

            print(f"🔧 Available servers at turn start: {', '.join(available_servers)}")
            print(f"🔧 Total tools available: {available_tool_count}\n")

            # Reset trajectory for this turn
            self.agent.trajectory = []
            self.agent.reasoning_trace = []

            agent_response = await self.agent.execute(agent_query)

            print(f"Agent Response:\n{agent_response.response}\n")

            # Capture trajectory from this turn
            turn_tool_calls = list(self.agent.trajectory)  # Copy trajectory
            turn_reasoning = list(self.agent.reasoning_trace)  # Copy reasoning

            # Extract tool results from reasoning trace for easier access
            turn_tool_results = [
                {"step": entry.get("type", ""), "result": entry.get("content", "")}
                for entry in turn_reasoning
                if entry.get("type") == "result"
            ]

            # Simulated user evaluates and decides
            print("Simulated user is evaluating...\n")
            user_decision = await self.simulated_user.evaluate_and_decide(
                seed_query=seed_query,
                turn_number=turn_number,
                conversation_history=self.turns,
                agent_response=agent_response.response,
                current_turn_tool_calls=turn_tool_calls,  # Pass current turn's tool usage
            )

            print(f"User Decision: {user_decision['decision']}")
            print(f"Satisfaction: {user_decision['satisfaction_level']:.2f}")
            print(f"Reasoning: {user_decision['reasoning']}\n")

            # Record turn
            turn = Turn(
                turn_number=turn_number,
                query=current_query,
                agent_response=agent_response.response,
                tool_calls=turn_tool_calls,
                reasoning_trace=turn_reasoning,
                tool_results=turn_tool_results,
                available_servers=available_servers,
                available_tool_count=available_tool_count,
                user_decision=user_decision["decision"],
                termination_reason=user_decision.get("termination_reason"),
                satisfaction_level=user_decision["satisfaction_level"],
                user_reasoning=user_decision["reasoning"],
                follow_up_intent=user_decision.get("intent"),
            )
            self.turns.append(turn)

            # Check termination
            if user_decision["decision"] == "TERMINATE":
                print(f"{'='*60}")
                print(f"Conversation Terminated: {user_decision['termination_reason']}")
                print(f"{'='*60}\n")
                break

            # Continue with follow-up
            current_query = user_decision["follow_up_query"]

        # Build final trajectory
        agent_llm = getattr(self.agent, '_llm', None)
        user_llm = self.simulated_user.llm

        trajectory = MultiTurnTrajectory(
            conversation_id=conversation_id,
            seed_query=seed_query,
            user_persona=self.simulated_user.persona_name,
            turns=self.turns,
            total_turns=len(self.turns),
            final_decision=self.turns[-1].user_decision,
            final_satisfaction=self.turns[-1].satisfaction_level,
            timestamp=datetime.now().isoformat(),
            agent_model=agent_llm.config.model_name if agent_llm and hasattr(agent_llm, 'config') else "unknown",
            user_model=user_llm.config.model_name if hasattr(user_llm, 'config') else "unknown",
            dynamically_loaded_servers=list(self.agent.dynamically_loaded_servers),
        )

        return trajectory


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-turn agent with real-time user simulation"
    )
    parser.add_argument(
        "seed_query",
        nargs="?",
        help="Initial user query to start conversation (or use --input for JSON file)"
    )
    parser.add_argument(
        "--input",
        help="JSON file with queries (format: {metadata: {...}, items: [{query: ...}, ...]})"
    )
    parser.add_argument(
        "--persona",
        default="curious_researcher",
        choices=list(USER_PERSONAS.keys()),
        help="User persona to simulate",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="Model for agent (OpenRouter format)",
    )
    parser.add_argument(
        "--user-model",
        default="anthropic/claude-3.5-sonnet",
        help="Model for simulated user (OpenRouter format)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max agent iterations per turn",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save conversation trajectory to JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process from JSON file (for testing)",
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.seed_query and not args.input:
        parser.error("Must provide either seed_query or --input")
    if args.seed_query and args.input:
        parser.error("Cannot provide both seed_query and --input")

    # Load environment
    load_dotenv(str(ORCHESTRATOR_DIR / ".env"))

    # Load queries
    queries = []
    if args.input:
        print(f"Loading queries from {args.input}...")
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        queries = [item['query'] for item in data['items']]
        if args.limit:
            queries = queries[:args.limit]
            print(f"Processing first {args.limit} queries (limited)")
        print(f"✓ Loaded {len(queries)} queries\n")
    else:
        queries = [args.seed_query]

    print(f"\n{'='*60}")
    print(f"Multi-Turn Agent with User Simulation")
    print(f"{'='*60}")
    print(f"Mode: {'Batch' if args.input else 'Single'}")
    print(f"Queries: {len(queries)}")
    print(f"User Persona: {args.persona}")
    print(f"Agent Model: {args.model}")
    print(f"User Model: {args.user_model}")
    print(f"{'='*60}\n")

    # Load server configs
    configs_path = PROJECT_ROOT / "MCP_INFO_MGR" / "mcp_data" / "usable" / "remote_server_configs.json"
    print(f"Loading server configs from {configs_path}...")
    with configs_path.open("r") as f:
        all_server_configs = json.load(f)
    print(f"✓ Loaded {len(all_server_configs)} server configurations\n")

    # Initialize MCP Manager
    print("Initializing MCP Manager...")
    mcp_manager = MCPManager()

    # Add Meta-MCP server
    meta_mcp_config = {
        "stdio": {
            "command": "python",
            "args": [str(PROJECT_ROOT / "meta_mcp_server" / "server.py")],
        }
    }
    mcp_manager.add_server_config("meta-mcp", meta_mcp_config)
    print("✓ Added Meta-MCP server\n")

    # Initialize LLMs
    print(f"Initializing models...")
    model_manager = ModelManager()

    agent_llm = model_manager.build_model("openrouter", config={"model_name": args.model})
    user_llm = model_manager.build_model("openrouter", config={"model_name": args.user_model})
    print("✓ Models ready\n")

    # Create DynamicReActAgent (existing agent, no modifications)
    react_config = {
        "name": "multiturn-react-agent",
        "instruction": """You are an intelligent agent in a multi-turn conversation.

IMPORTANT - Context Awareness:
- The user may ask follow-up questions referencing previous turns
- Pay attention to pronouns like "it", "that one", "the first result"
- Use conversation history provided to understand context

Tool Discovery Workflow:
1. For EACH new user query, evaluate if you have the right tools
2. Use 'meta-mcp/search_tools' to discover tools when:
   - The query requires capabilities you don't currently have
   - The user asks about a different domain/topic than previous turns
   - You need more specialized or different types of tools
3. Read search results carefully for server and tool names
4. Use the discovered tools with appropriate arguments
5. Provide clear, helpful responses

CRITICAL: Don't just reuse existing tools - actively search for NEW tools via meta-mcp when the query requires different capabilities. You ALWAYS have access to meta-mcp/search_tools.

The conversation will continue until the user is satisfied or decides to stop.""",
        "max_iterations": args.max_iterations,
    }

    print("Creating DynamicReActAgent...")
    agent = DynamicReActAgent(
        mcp_manager=mcp_manager,
        llm=agent_llm,
        server_configs=all_server_configs,
        config=react_config,
    )

    await agent.initialize(mcp_servers=[{"name": "meta-mcp"}])
    print("✓ Agent ready\n")

    # Create simulated user
    print(f"Creating simulated user with persona: {args.persona}...")
    simulated_user = SimulatedUser(user_llm, args.persona)
    print("✓ Simulated user ready\n")

    # Create conversation controller
    max_turns = USER_PERSONAS[args.persona]["max_turns"]
    controller = ConversationController(
        agent=agent,
        simulated_user=simulated_user,
        max_turns=max_turns,
    )

    # Process queries
    trajectories = []
    results_summary = []

    for query_idx, query in enumerate(queries):
        print(f"\n{'='*80}")
        print(f"Processing Query {query_idx + 1}/{len(queries)}")
        print(f"{'='*80}")
        print(f"Query: {query}\n")

        try:
            # Create a fresh agent for each conversation to avoid async cleanup issues
            print(f"Creating fresh agent for conversation {query_idx + 1}...")
            fresh_agent = DynamicReActAgent(
                mcp_manager=mcp_manager,
                llm=agent_llm,
                server_configs=all_server_configs,
                config=react_config,
            )
            await fresh_agent.initialize(mcp_servers=[{"name": "meta-mcp"}])
            print("✓ Fresh agent ready\n")

            # Create fresh controller with new agent
            fresh_controller = ConversationController(
                agent=fresh_agent,
                simulated_user=simulated_user,
                max_turns=max_turns,
            )

            # Run conversation
            trajectory = await fresh_controller.run_conversation(query)

            # Cleanup this agent
            try:
                await fresh_agent.cleanup()
            except Exception as e:
                print(f"⚠️  Warning during agent cleanup: {e}")

            # Print summary
            print(f"\n{'='*60}")
            print(f"Conversation Summary")
            print(f"{'='*60}")
            print(f"Total Turns: {trajectory.total_turns}")
            print(f"Final Decision: {trajectory.final_decision}")
            print(f"Final Satisfaction: {trajectory.final_satisfaction:.2f}")
            print(f"Dynamically Loaded Servers: {len(trajectory.dynamically_loaded_servers)}")
            print(f"{'='*60}\n")

            trajectories.append(trajectory)
            results_summary.append({
                "query_index": query_idx,
                "query": query,
                "success": trajectory.final_decision in ["satisfied", "natural_end"],
                "total_turns": trajectory.total_turns,
                "final_decision": trajectory.final_decision,
                "final_satisfaction": trajectory.final_satisfaction,
            })

            # Save individual trajectory
            if args.save_trajectory:
                trajectory_dir = PROJECT_ROOT / "trajectories" / "multiturn"
                trajectory_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if args.input:
                    filename = f"multiturn_{query_idx:03d}_{timestamp}.json"
                else:
                    filename = f"multiturn_{timestamp}.json"
                filepath = trajectory_dir / filename

                # Convert trajectory to dict
                trajectory_dict = {
                    "conversation_id": trajectory.conversation_id,
                    "seed_query": trajectory.seed_query,
                    "user_persona": trajectory.user_persona,
                    "turns": [
                        {
                            "turn_number": t.turn_number,
                            "query": t.query,
                            "agent_response": t.agent_response,
                            "tool_calls": t.tool_calls,
                            "reasoning_trace": t.reasoning_trace,
                            "tool_results": t.tool_results,
                            "available_servers": t.available_servers,
                            "available_tool_count": t.available_tool_count,
                            "user_decision": t.user_decision,
                            "termination_reason": t.termination_reason,
                            "satisfaction_level": t.satisfaction_level,
                            "user_reasoning": t.user_reasoning,
                            "follow_up_intent": t.follow_up_intent,
                        }
                        for t in trajectory.turns
                    ],
                    "total_turns": trajectory.total_turns,
                    "final_decision": trajectory.final_decision,
                    "final_satisfaction": trajectory.final_satisfaction,
                    "timestamp": trajectory.timestamp,
                    "agent_model": trajectory.agent_model,
                    "user_model": trajectory.user_model,
                    "dynamically_loaded_servers": trajectory.dynamically_loaded_servers,
                }

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(trajectory_dict, f, indent=2, ensure_ascii=False)

                print(f"✓ Trajectory saved to: {filepath}\n")

        except Exception as e:
            print(f"\n✗ Error processing query {query_idx + 1}: {e}\n")
            results_summary.append({
                "query_index": query_idx,
                "query": query,
                "success": False,
                "error": str(e),
            })

    # Print batch summary if multiple queries
    if len(queries) > 1:
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"Total queries: {len(queries)}")
        print(f"Successful: {sum(1 for r in results_summary if r.get('success', False))}")
        print(f"Failed: {sum(1 for r in results_summary if not r.get('success', False))}")
        print(f"{'='*80}\n")

        # Save batch summary
        if args.save_trajectory:
            summary_file = PROJECT_ROOT / "trajectories" / "multiturn" / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "persona": args.persona,
                    "total_queries": len(queries),
                    "results": results_summary,
                }, f, indent=2, ensure_ascii=False)
            print(f"✓ Batch summary saved to: {summary_file}\n")

    # Cleanup
    print("Cleaning up...")
    await agent.cleanup()
    print("✓ Done\n")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
