# Multi-Turn Agent Environment Specification

## Overview

This specification defines a **multi-turn conversational agent environment** where:
1. A **seed query** initiates the conversation (from benchmark or manual input)
2. An **agent** (MCP-powered ReAct agent) responds using dynamic tool discovery
3. A **simulated user LLM** analyzes the agent's response and generates contextual follow-up questions OR decides to terminate
4. The conversation continues dynamically until the user LLM signals satisfaction or max turns reached
5. Complete conversation trajectories are captured and evaluated

**Key Design Decision**: Multi-turn conversations use **real-time LLM simulation** where follow-ups are generated based on the agent's actual responses. This ensures:
- **Realistic interactions**: Follow-ups adapt to what the agent actually said
- **Dynamic adaptation**: User can react to agent mistakes, clarifications, or good answers
- **True conversational testing**: Tests agent's ability to maintain context across adaptive turns
- **Authentic termination**: User decides when they're satisfied based on actual agent performance

This extends the existing single-turn agent architecture to support realistic multi-turn interactions, enabling evaluation of:
- Context retention across turns
- Clarification handling based on actual agent responses
- Information refinement through iterative questioning
- Multi-step reasoning with adaptive follow-ups
- Tool reuse and adaptation to conversation flow

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│               Real-Time Multi-Turn Environment               │
│                                                              │
│  ┌────────────────────────────────────────────┐            │
│  │       Conversation Controller              │            │
│  │  ┌──────────────────────────────────────┐  │            │
│  │  │  Turn Loop (until termination)       │  │            │
│  │  │                                      │  │            │
│  │  │  1. Feed query to Agent             │  │            │
│  │  │         ▼                            │  │            │
│  │  │  ┌──────────────┐                   │  │            │
│  │  │  │ ReAct Agent  │                   │  │            │
│  │  │  │ (MCP-powered)│                   │  │            │
│  │  │  └──────────────┘                   │  │            │
│  │  │         │                            │  │            │
│  │  │         ▼ (response)                 │  │            │
│  │  │  2. Pass response to User LLM       │  │            │
│  │  │         ▼                            │  │            │
│  │  │  ┌──────────────┐                   │  │            │
│  │  │  │ User LLM     │                   │  │            │
│  │  │  │ (Simulator)  │                   │  │            │
│  │  │  └──────────────┘                   │  │            │
│  │  │         │                            │  │            │
│  │  │         ▼ (follow-up OR done)       │  │            │
│  │  │  3. Check termination               │  │            │
│  │  │         ▼                            │  │            │
│  │  │  If continue: loop back to 1        │  │            │
│  │  │  If done: exit                      │  │            │
│  │  └──────────────────────────────────────┘  │            │
│  │                                             │            │
│  │  - Maintains conversation history          │            │
│  │  - Records trajectory per turn             │            │
│  │  - Tracks satisfaction levels              │            │
│  └────────────────────────────────────────────┘            │
│                          │                                  │
│                          ▼                                  │
│              ┌────────────────────┐                        │
│              │  Multi-Turn        │                        │
│              │  Trajectory JSON   │                        │
│              └────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 1. Simulated User LLM (Runtime)

**Purpose**: Analyze agent responses in real-time and generate contextual follow-up questions OR decide to terminate.

**Responsibilities**:
- **Analyze** agent's response for:
  - Completeness (did it answer the question?)
  - Correctness (are there errors or hallucinations?)
  - Opportunities (interesting points to dig deeper)
  - Gaps (missing information)
- **Generate** natural follow-up questions that:
  - Request clarification on ambiguous points
  - Dig deeper into interesting aspects
  - Request additional information
  - Explore related topics
  - Challenge incorrect information
  - Ask for verification/sources
- **Decide termination** based on:
  - Task completion (got all needed information)
  - Agent failure (repeated errors/inability to help)
  - Satisfaction level (accumulated over turns)

**Configuration**:
```json
{
  "model": "openai/gpt-4o-mini",
  "temperature": 0.7,
  "user_persona": "curious_researcher",
  "max_turns": 10,
  "satisfaction_threshold": 0.85,
  "frustration_threshold": 0.3
}
```

**User Personas** (affect question style and termination patience):
- `curious_researcher`: Asks detailed, exploratory questions; patient (8-10 turns)
- `impatient_user`: Wants quick answers; terminates early if unsatisfied (3-5 turns)
- `thorough_analyst`: Requests comprehensive information and verification (7-12 turns)
- `casual_user`: Asks simple follow-ups, easily satisfied (2-4 turns)
- `skeptical_user`: Challenges responses, asks for evidence (5-8 turns)

### 2. Conversation Controller (Runtime)

**Purpose**: Orchestrate the real-time multi-turn conversation loop.

**Responsibilities**:
- Initialize conversation with seed query (from benchmark or manual input)
- Execute turn loop:
  1. Feed current query to agent
  2. Collect agent's response
  3. Pass response + conversation history to user LLM
  4. Get user's decision (follow-up OR terminate)
  5. If continue, loop back; if terminate, exit
- Maintain conversation history (all turns)
- Track satisfaction levels across turns
- Record complete trajectory with metadata
- Handle errors (agent failures, LLM timeouts)

**Turn Flow**:
```
Turn 0: Seed Query (from benchmark/manual input)
  ↓
Turn 1: Agent responds to seed query
  ↓
        User LLM analyzes response
        → satisfaction_level: 0.5
        → decision: CONTINUE
        → follow_up: "Tell me more about X?"
  ↓
Turn 2: Agent responds to follow-up
  ↓
        User LLM analyzes response
        → satisfaction_level: 0.75
        → decision: CONTINUE
        → follow_up: "What about Y?"
  ↓
Turn 3: Agent responds to follow-up
  ↓
        User LLM analyzes response
        → satisfaction_level: 0.9
        → decision: TERMINATE
        → reason: "Task completed successfully"
  ↓
Conversation ends: Save trajectory
```

### 3. Termination Conditions (User LLM Runtime Decision)

The conversation ends when the **User LLM** decides to terminate OR hard limits are reached:

**User LLM Termination Reasons**:
1. **SATISFIED**: Got all needed information, task complete
   - satisfaction_level >= satisfaction_threshold (default: 0.85)
   - User signals: "I have everything I need"

2. **FRUSTRATED**: Agent repeatedly failing or unable to help
   - satisfaction_level <= frustration_threshold (default: 0.3)
   - User signals: "This isn't working, giving up"

3. **NATURAL_END**: Conversation reached logical conclusion
   - No more reasonable follow-ups to ask
   - Topic exhausted

**Hard Limits** (Controller enforced):
4. **MAX_TURNS**: Reached maximum conversation length (default: 10 turns)
5. **LOOP_DETECTED**: User asking same question >2 times (stuck)
6. **ERROR**: Agent crashed or LLM timeout

---

## Data Structures

### Multi-Turn Trajectory

Extends the existing single-turn trajectory format:

```json
{
  "metadata": {
    "conversation_id": "conv_20251107_143256",
    "seed_query": "Find recent ML papers on arXiv",
    "seed_reference_tools": [
      {"server": "arxiv", "tool": "search_papers", "why": "..."}
    ],
    "model_agent": "anthropic/claude-3.5-sonnet",
    "model_user": "openai/gpt-4o-mini",
    "user_persona": "curious_researcher",
    "total_turns": 5,
    "termination_reason": "user_satisfied",
    "timestamp_start": "2025-11-07T14:32:56.123Z",
    "timestamp_end": "2025-11-07T14:35:12.456Z",
    "duration_seconds": 136.33
  },

  "turns": [
    {
      "turn_number": 0,
      "speaker": "user",
      "query": "Find recent ML papers on arXiv",
      "timestamp": "2025-11-07T14:32:56.123Z"
    },
    {
      "turn_number": 1,
      "speaker": "agent",
      "query": "Find recent ML papers on arXiv",
      "reasoning_trace": [
        {"type": "thought", "content": "I need to search for ML papers..."},
        {"type": "action", "content": "Using tool `search_papers`..."}
      ],
      "response": "I found 10 recent papers on machine learning...",
      "tool_calls": [
        {
          "server": "arxiv",
          "tool": "search_papers",
          "arguments": {"query": "machine learning", "max_results": 10},
          "status": "success",
          "dynamically_loaded": true
        }
      ],
      "servers_loaded": ["arxiv"],
      "timestamp": "2025-11-07T14:33:15.789Z",
      "duration_seconds": 19.666
    },
    {
      "turn_number": 2,
      "speaker": "user",
      "query": "Can you tell me more about the paper on small language models?",
      "generation_context": {
        "previous_response": "I found 10 recent papers...",
        "user_intent": "drill_down",
        "satisfaction_level": 0.6
      },
      "timestamp": "2025-11-07T14:33:20.123Z"
    },
    {
      "turn_number": 3,
      "speaker": "agent",
      "query": "Can you tell me more about the paper on small language models?",
      "reasoning_trace": [...],
      "response": "The paper 'Efficient Small LMs' discusses...",
      "tool_calls": [...],
      "timestamp": "2025-11-07T14:34:02.456Z",
      "duration_seconds": 42.333
    }
  ],

  "summary": {
    "total_user_queries": 3,
    "total_agent_responses": 3,
    "total_tool_calls": 7,
    "unique_tools_used": 3,
    "unique_servers_used": 2,
    "context_retention_score": 0.85,
    "task_completion_score": 0.9
  }
}
```

### User Follow-up Generation Context

Input to the simulated user LLM:

```json
{
  "conversation_history": [
    {"speaker": "user", "content": "..."},
    {"speaker": "agent", "content": "..."}
  ],
  "current_agent_response": "...",
  "seed_query": "Original user query",
  "turn_number": 3,
  "user_persona": "curious_researcher",
  "previous_satisfaction_levels": [0.5, 0.6, 0.7]
}
```

Output from the simulated user LLM:

```json
{
  "follow_up_query": "Can you tell me more about X?",
  "intent": "clarification",
  "satisfaction_level": 0.7,
  "should_continue": true,
  "reasoning": "The agent provided good info but I want to know more about..."
}
```

---

## Prompts

### Simulated User System Prompt (Real-Time)

```
You are simulating a {user_persona} who is having a conversation with an AI agent.

Your role at each turn:
1. ANALYZE the agent's response to your most recent question
2. EVALUATE how well it answered and what's missing
3. DECIDE: Continue with a follow-up question OR terminate the conversation
4. If continuing: GENERATE a natural, contextual follow-up question

Analysis Criteria:
- Completeness: Did the agent fully answer my question?
- Correctness: Are there errors, hallucinations, or unclear statements?
- Usefulness: Is this information helpful for my goal?
- Gaps: What important information is still missing?
- Opportunities: What interesting details could I explore further?

Conversation Guidelines:
- Ask ONE follow-up question at a time
- Base questions on the agent's ACTUAL response, not generic curiosity
- Reference specific details from the agent's answer
- Don't repeat questions already answered
- Terminate when you've got what you need OR agent is failing repeatedly

Follow-up Intent Types:
- CLARIFICATION: "What did you mean by X?" (agent was unclear)
- DRILL_DOWN: "Tell me more about Y" (want details on interesting point)
- EXPANSION: "What about Z?" (explore related topic)
- VERIFICATION: "Can you confirm X?" (check correctness)
- CORRECTION: "Actually, that seems wrong because..." (challenge errors)
- COMPLETION: Signal satisfaction and end (got everything needed)

Termination Conditions:
- SATISFIED: You have all the information you need (satisfaction ≥ 0.85)
- FRUSTRATED: Agent failing repeatedly or giving bad answers (satisfaction ≤ 0.3)
- NATURAL_END: No more reasonable questions to ask

User Persona: {user_persona}
{persona_description}

Seed Query (Original Goal):
{seed_query}

Conversation History:
{conversation_history}

Agent's Latest Response (Turn {turn_number}):
{agent_response}

Output Format (strict JSON):
{
  "decision": "CONTINUE" | "TERMINATE",
  "termination_reason": "satisfied|frustrated|natural_end" (only if TERMINATE),
  "follow_up_query": "your next question" (only if CONTINUE),
  "intent": "clarification|drill_down|expansion|verification|correction",
  "satisfaction_level": <float 0.0-1.0>,
  "reasoning": "Why you're asking this OR why you're done (2-3 sentences)"
}

IMPORTANT:
- If decision is CONTINUE, you MUST include "follow_up_query"
- If decision is TERMINATE, you MUST include "termination_reason"
- satisfaction_level should reflect cumulative progress toward your goal
- Base your decision on the agent's ACTUAL performance, not ideal expectations

Generate your decision now.
```

### Multi-Turn Agent System Prompt

Extends the existing ReAct prompt:

```
You are an intelligent agent that can discover and use MCP tools dynamically.
You are currently in a MULTI-TURN conversation with a user.

IMPORTANT - Context Awareness:
- This is turn {turn_number} of an ongoing conversation
- Previous conversation history is provided below
- The user's current question may reference previous responses
- Reuse information and tools from earlier turns when appropriate
- Don't re-fetch information you already obtained unless asked

Previous Conversation:
{conversation_history}

Current User Query:
{current_query}

[Rest of existing ReAct prompt...]
```

---

## API Specification

### ConversationController

```python
class ConversationController:
    """Orchestrates multi-turn conversations between simulated user and agent."""

    def __init__(
        self,
        agent: DynamicReActAgent,
        user_llm: BaseLLM,
        user_persona: str = "curious_researcher",
        max_turns: int = 10,
        termination_threshold: float = 0.9
    ):
        """
        Args:
            agent: The MCP-powered ReAct agent
            user_llm: LLM for simulating user behavior
            user_persona: Type of simulated user
            max_turns: Maximum conversation turns
            termination_threshold: Satisfaction level to auto-terminate
        """
        pass

    async def run_conversation(
        self,
        seed_query: str,
        seed_reference_tools: List[Dict] = None
    ) -> MultiTurnTrajectory:
        """
        Execute a complete multi-turn conversation.

        Args:
            seed_query: Initial user query to start conversation
            seed_reference_tools: Ground truth tools for seed query (optional)

        Returns:
            Complete multi-turn trajectory with all turns and metadata
        """
        pass

    async def _generate_user_followup(
        self,
        conversation_history: List[Dict],
        agent_response: str
    ) -> Dict:
        """
        Use simulated user LLM to generate next follow-up question.

        Returns:
            {
                "follow_up_query": str,
                "intent": str,
                "satisfaction_level": float,
                "should_continue": bool,
                "reasoning": str
            }
        """
        pass

    def _check_termination(
        self,
        turn_number: int,
        user_decision: Dict
    ) -> Tuple[bool, str]:
        """
        Check if conversation should terminate.

        Returns:
            (should_terminate: bool, reason: str)
        """
        pass
```

### Usage Example

```python
from mcpuniverse.conversation import ConversationController
from mcpuniverse.agent.dynamic_react import DynamicReActAgent
from mcpuniverse.llm.manager import ModelManager

# Initialize agent (existing setup)
agent = DynamicReActAgent(...)
await agent.initialize(mcp_servers=[{"name": "meta-mcp"}])

# Initialize simulated user
model_manager = ModelManager()
user_llm = model_manager.build_model("openrouter", config={
    "model_name": "openai/gpt-4o-mini"
})

# Create conversation controller
controller = ConversationController(
    agent=agent,
    user_llm=user_llm,
    user_persona="curious_researcher",
    max_turns=10,
    termination_threshold=0.85
)

# Run multi-turn conversation
trajectory = await controller.run_conversation(
    seed_query="Find recent papers on small language models",
    seed_reference_tools=[
        {"server": "arxiv", "tool": "search_papers", "why": "..."}
    ]
)

# Save trajectory
with open(f"trajectories/multiturn_{trajectory.id}.json", "w") as f:
    json.dump(trajectory.to_dict(), f, indent=2)
```

---

## Evaluation Framework

### Multi-Turn Specific Metrics

Extends the existing 5-dimension rubric with multi-turn metrics:

```python
{
  # Existing single-turn metrics (averaged across all turns)
  "avg_task_alignment": 0.85,
  "avg_grounding": 0.90,
  "avg_tool_planning": 0.88,
  "avg_execution_recovery": 0.92,

  # Multi-turn specific metrics
  "context_retention": {
    "score": 0.85,
    "explanation": "Agent remembered user preferences from turn 2..."
  },

  "information_reuse": {
    "score": 0.78,
    "explanation": "Agent reused arXiv results instead of re-fetching..."
  },

  "clarification_handling": {
    "score": 0.90,
    "explanation": "Agent correctly interpreted follow-up question..."
  },

  "conversation_coherence": {
    "score": 0.88,
    "explanation": "Responses logically built on previous turns..."
  },

  "user_satisfaction_trajectory": [0.5, 0.6, 0.75, 0.85, 0.9],

  "conversation_efficiency": {
    "score": 0.82,
    "redundant_tool_calls": 1,
    "optimal_turn_count": 4,
    "actual_turn_count": 5
  }
}
```

### Evaluation Script

```bash
# Evaluate single multi-turn trajectory
export PYTHONPATH="/Users/xiziqiao/Documents/MCP-Research/MCP-R/Orchestrator:$PYTHONPATH"

python Orchestrator/mcpuniverse/evaluator/multiturn_judge.py \
  --trajectory trajectories/multiturn_conv_20251107_143256.json \
  --model openai/gpt-4o \
  --save_json evaluation/multiturn_result.json

# Batch evaluation
python Orchestrator/mcpuniverse/evaluator/multiturn_judge.py \
  --traj_dir trajectories/multiturn \
  --model anthropic/claude-3.5-sonnet \
  --save_json evaluation/multiturn_batch_results.json
```

---

## Benchmark Generation

### Multi-Turn Seed Queries

For real-time simulation, we only need to generate **seed queries** that are likely to lead to multi-turn conversations. The actual follow-ups are generated dynamically by the user LLM.

**Good Seed Query Properties**:
1. **Open-ended or exploratory**: Invites follow-up questions
2. **Not fully specified**: Missing details the user might want to clarify
3. **Multi-faceted**: Has several aspects that could be explored
4. **Interesting results expected**: Agent's answer will likely spark curiosity

**Examples**:
- "Find recent research on small language models" → User will likely drill down into specific papers
- "What's the weather in Tokyo and are there any events happening?" → User might ask about specific dates/venues
- "Search for machine learning tools on GitHub" → User will want details on top results

**Generation Script** (`mcp_generate/query_generate_multiturn.py`):

```python
MULTITURN_SEED_SYSTEM_PROMPT = """
Generate realistic seed queries designed to naturally lead to multi-turn conversations.

A good seed query:
- Is open-ended or exploratory (not fully specified)
- Would benefit from follow-up questions
- Has multiple aspects that could be explored
- Will return interesting results the user wants to know more about

Bad examples (too specific, no follow-ups needed):
- "Get weather for Tokyo on 2025-11-10"
- "Search for paper ID 12345 on arXiv"

Good examples (invite natural follow-ups):
- "Find recent AI research papers" → "Tell me about the top one" → "Who are the authors?"
- "What's happening in New York this weekend?" → "Tell me about the concerts" → "Check availability"

Generate seed query in this format:
{
  "seed_query": "...",
  "reference_tools": [{"server": "...", "tool": "...", "why": "..."}],
  "likely_followup_types": ["drill_down", "clarification", "expansion"]
}
```

**Usage**:
```bash
python mcp_generate/query_generate_multiturn.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/prompt/multiturn_seeds.json \
  --num-queries 50
```

**Note**: Unlike single-turn benchmarks, we don't pre-generate the full conversation—just the seed. The user LLM generates follow-ups at runtime based on actual agent responses.

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create `ConversationController` class
- [ ] Implement simulated user LLM integration
- [ ] Define multi-turn trajectory data structure
- [ ] Add conversation history management

### Phase 2: User Simulation (Week 2)
- [ ] Implement user persona system
- [ ] Create follow-up generation prompts
- [ ] Add termination detection logic
- [ ] Implement satisfaction scoring

### Phase 3: Agent Enhancement (Week 3)
- [ ] Extend agent prompt for multi-turn awareness
- [ ] Add conversation context to agent state
- [ ] Implement information reuse detection
- [ ] Add turn-aware logging

### Phase 4: Evaluation (Week 4)
- [ ] Create `multiturn_judge.py` evaluator
- [ ] Implement multi-turn metrics
- [ ] Add conversation coherence scoring
- [ ] Build visualization tools

### Phase 5: Benchmarking (Week 5)
- [ ] Create multi-turn query generator
- [ ] Generate 100-query benchmark
- [ ] Run baseline experiments
- [ ] Document results

---

## File Structure

```
MCP-R/
├── Orchestrator/
│   └── mcpuniverse/
│       ├── conversation/
│       │   ├── __init__.py
│       │   ├── controller.py           # ConversationController
│       │   ├── simulated_user.py       # User simulation logic
│       │   └── termination.py          # Termination conditions
│       │
│       ├── agent/
│       │   └── dynamic_react.py        # Enhanced for multi-turn
│       │
│       └── evaluator/
│           ├── commonllmjudge.py       # Existing single-turn
│           └── multiturn_judge.py      # New multi-turn evaluator
│
├── mcp_generate/
│   ├── query_generate_multiturn.py     # Multi-turn seed generation
│   └── prompt/
│       └── multiturn_benchmark.json    # Multi-turn benchmark
│
├── runtime/
│   └── run_multiturn_conversation.py   # CLI for multi-turn runs
│
├── trajectories/
│   └── multiturn/                      # Multi-turn trajectories
│
└── docs/
    └── MULTI_TURN_AGENT_SPEC.md        # This document
```

---

## Example Multi-Turn Conversation

### Seed Query
"Find recent machine learning papers on arXiv"

### Turn 1 - Agent Response
```
I found 10 recent papers on machine learning:
1. "Efficient Small Language Models" (2025-11-01)
2. "Neural Architecture Search" (2025-10-28)
...
```

### Turn 2 - User Follow-up (Simulated)
```json
{
  "follow_up_query": "Can you tell me more about the first paper on small language models?",
  "intent": "drill_down",
  "satisfaction_level": 0.6,
  "should_continue": true,
  "reasoning": "The list is helpful but I want details on the most interesting paper"
}
```

### Turn 3 - Agent Response
```
The paper "Efficient Small Language Models" discusses techniques for
training high-quality language models with <1B parameters. Key findings:
- Uses knowledge distillation from larger models
- Achieves 95% of GPT-3.5 performance with 10x fewer parameters
Authors: Smith et al. from Stanford University
```

### Turn 4 - User Follow-up
```json
{
  "follow_up_query": "Are there any similar papers by the same research group?",
  "intent": "expansion",
  "satisfaction_level": 0.75,
  "should_continue": true,
  "reasoning": "Good info, now curious about related work from same team"
}
```

### Turn 5 - Agent Response
```
Yes, I found 3 more papers from Smith's group at Stanford:
1. "Distillation Techniques for LLMs" (2025-09-15)
2. "Efficient Training Methods" (2025-08-20)
...
```

### Turn 6 - User Termination
```json
{
  "follow_up_query": "DONE",
  "intent": "completion",
  "satisfaction_level": 0.9,
  "should_continue": false,
  "reasoning": "I have all the information I need about this research area"
}
```

**Termination**: User satisfied (satisfaction_level = 0.9 > 0.85 threshold)

---

## Configuration Examples

### Curious Researcher Persona
```json
{
  "user_persona": "curious_researcher",
  "temperature": 0.7,
  "max_patience": 8,
  "satisfaction_threshold": 0.85,
  "follow_up_probability": {
    "drill_down": 0.3,
    "clarification": 0.2,
    "expansion": 0.25,
    "verification": 0.15,
    "new_aspect": 0.1
  }
}
```

### Impatient User Persona
```json
{
  "user_persona": "impatient_user",
  "temperature": 0.5,
  "max_patience": 3,
  "satisfaction_threshold": 0.7,
  "follow_up_probability": {
    "drill_down": 0.1,
    "clarification": 0.6,
    "expansion": 0.1,
    "verification": 0.1,
    "new_aspect": 0.1
  }
}
```

---

## User Persona Descriptions

### 1. Curious Researcher
**Personality**: Deeply interested in learning, patient, thorough
**Questioning Style**: Exploratory, detailed, builds on previous answers
**Patience**: High (8-10 turns before giving up)
**Satisfaction Threshold**: 0.85 (wants comprehensive information)

**Example Behavior**:
- Seed: "Find recent AI papers"
- Turn 2: "Tell me more about the first paper on small LLMs"
- Turn 3: "Who are the authors of that paper?"
- Turn 4: "Are there other papers by the same team?"
- Turn 5: "What methods did they use for training?"
- Terminates: When has deep understanding of the topic

**Prompt Addition**:
```
You are a curious researcher who values depth and thoroughness. You ask detailed questions,
build on previous answers, and explore topics systematically. You're patient and will continue
asking until you have a comprehensive understanding. You appreciate concrete details, data,
and sources.
```

### 2. Impatient User
**Personality**: Busy, wants quick answers, low tolerance for failure
**Questioning Style**: Direct, focused, no-nonsense
**Patience**: Low (3-5 turns before giving up)
**Satisfaction Threshold**: 0.7 (good enough is acceptable)

**Example Behavior**:
- Seed: "Find restaurants in Tokyo"
- Turn 2: "Which ones have Michelin stars?"
- Turn 3: "Can I get a reservation for tonight?"
- Terminates early if: Agent is slow, gives vague answers, or makes errors

**Prompt Addition**:
```
You are an impatient user who needs quick, actionable answers. You don't have time for
lengthy explanations. If the agent is slow, unclear, or makes mistakes, you'll give up
quickly. You want specific, practical information you can use immediately.
```

### 3. Thorough Analyst
**Personality**: Meticulous, wants verification, checks everything
**Questioning Style**: Verification-focused, asks for sources, challenges claims
**Patience**: Medium-High (7-12 turns)
**Satisfaction Threshold**: 0.90 (very high bar)

**Example Behavior**:
- Seed: "What's the unemployment rate?"
- Turn 2: "Can you verify that with another source?"
- Turn 3: "How does that compare to last year?"
- Turn 4: "What's the methodology for calculating this?"
- Turn 5: "Are there any caveats or limitations to this data?"
- Terminates: When has verified, cross-checked information

**Prompt Addition**:
```
You are a thorough analyst who verifies everything. You ask for sources, cross-check claims,
and look for potential issues with data or methodology. You're skeptical but fair - good
evidence satisfies you, but you need solid proof. You check the agent's work carefully.
```

### 4. Casual User
**Personality**: Relaxed, easily satisfied, not very demanding
**Questioning Style**: Simple, surface-level, one or two follow-ups max
**Patience**: Low-Medium (2-4 turns)
**Satisfaction Threshold**: 0.65 (easily satisfied)

**Example Behavior**:
- Seed: "What's the weather like today?"
- Turn 2: "What about tomorrow?"
- Terminates: After getting basic info

**Prompt Addition**:
```
You are a casual user who just needs basic information. You're not very demanding and are
easily satisfied with surface-level answers. You typically ask one or two simple follow-ups
and then move on. You're forgiving of small errors or vagueness.
```

### 5. Skeptical User
**Personality**: Distrustful, challenges responses, looks for errors
**Questioning Style**: Challenging, correction-focused, devil's advocate
**Patience**: Medium (5-8 turns)
**Satisfaction Threshold**: 0.75 (convinced only by solid evidence)

**Example Behavior**:
- Seed: "Find evidence that X is true"
- Turn 2: "That source seems biased, do you have others?"
- Turn 3: "Actually, I think your interpretation is wrong because..."
- Turn 4: "Can you explain the contradiction between X and Y?"
- Terminates: When convinced OR frustrated with poor answers

**Prompt Addition**:
```
You are a skeptical user who questions and challenges responses. You look for errors,
inconsistencies, and weak evidence. You play devil's advocate and point out problems. You
need strong, well-supported answers to be convinced. You're quick to spot hallucinations
or logical errors.
```

---

## Open Questions & Future Work

1. **Context Window Management**: How to handle very long conversations that exceed LLM context limits?
   - Proposed: Implement conversation summarization after every 5 turns

2. **Tool State Persistence**: Should tools maintain state across turns?
   - Proposed: Agent tracks previous tool results for potential reuse

3. **User Intent Classification**: Should we explicitly classify user intents?
   - Proposed: Use structured output from user LLM for intent tracking

4. **Dynamic Persona Switching**: Should user persona evolve during conversation?
   - Future work: Implement persona drift based on agent performance

5. **Multi-User Scenarios**: Can we simulate multiple users in one conversation?
   - Future work: Team collaboration scenarios

---

## Success Criteria

A successful multi-turn environment implementation will:

1. ✅ Support 3+ turn conversations with coherent context
2. ✅ Generate realistic follow-up questions using LLM simulation
3. ✅ Track complete conversation trajectories with all metadata
4. ✅ Implement at least 3 user personas
5. ✅ Provide multi-turn specific evaluation metrics
6. ✅ Integrate seamlessly with existing single-turn infrastructure
7. ✅ Generate benchmark with 50+ multi-turn seed queries
8. ✅ Demonstrate measurable improvement in agent performance across turns

---

## References

- Single-turn agent: `/runtime/run_react_agent.py`
- Trajectory format: `/trajectories/trajectory_*.json`
- Evaluation framework: `/Orchestrator/mcpuniverse/evaluator/commonllmjudge.py`
- Query generation: `/mcp_generate/query_generate.py`

---

**Version**: 1.0
**Last Updated**: 2025-11-07
**Status**: Draft Specification
