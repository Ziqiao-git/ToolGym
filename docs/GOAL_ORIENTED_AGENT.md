# Goal-Oriented Multi-Turn Agent

A goal-driven conversational agent system where the simulated user has a hidden goal that drives follow-up questions and evaluation.

## Overview

Unlike the standard multi-turn agent (which asks exploratory follow-ups), the goal-oriented agent simulates a user with a specific objective that requires multiple information-gathering steps to complete.

### Key Features

- **Hidden User Goal**: User has a concrete goal (e.g., "plan a trip") hidden from the agent
- **Sub-Goal Decomposition**: Goal is automatically broken into 3-6 measurable sub-goals
- **Progress Tracking**: Tracks which sub-goals are completed each turn
- **Goal-Based Evaluation**: Satisfaction considers goal progress + tool usage + response quality
- **Goal-Driven Follow-Ups**: Questions work toward completing remaining sub-goals
- **Early Termination**: Conversation ends when goal achieved (even if fewer turns than max)

## Architecture

```
User Goal: "Plan trip to Bodrum"
      ↓
   Decompose
      ↓
Sub-Goals: [Events, Weather, Restaurants, Hotels]
      ↓
Turn 1: "What events in Bodrum?" → Events ✅ (25% progress)
Turn 2: "What's the weather?" → Weather ✅ (50% progress)
Turn 3: "Recommend restaurants?" → Restaurants ✅ (75% progress)
Turn 4: "Where to stay?" → Hotels ✅ (100% progress → GOAL ACHIEVED)
```

## Components

### 1. GoalTracker

Manages goal decomposition and progress tracking:

```python
class GoalTracker:
    async def decompose_goal(self) -> List[str]:
        """Break goal into 3-6 measurable sub-goals"""

    async def evaluate_progress(self, response, tools) -> Dict:
        """Determine which sub-goals were completed"""

    @property
    def progress_percentage(self) -> float:
        """Overall completion (0.0-1.0)"""

    @property
    def is_complete(self) -> bool:
        """All sub-goals done?"""
```

### 2. GoalOrientedUser

Simulated user with goal-aware evaluation:

```python
class GoalOrientedUser:
    async def evaluate_and_decide(
        self,
        query, response, history,
        tool_calls,
        completed_sub_goals,
        remaining_sub_goals,
        goal_progress
    ) -> Dict:
        """
        Evaluate response considering:
        - Goal progress (most important)
        - Tool usage
        - Response quality

        Returns: {decision, satisfaction, follow_up, reasoning}
        """
```

### 3. GoalOrientedController

Orchestrates goal-driven conversations:

```python
class GoalOrientedController:
    async def run_conversation(self, seed_query) -> GoalTrajectory:
        """
        1. Decompose goal into sub-goals
        2. Run multi-turn loop:
           - Agent responds
           - Evaluate goal progress
           - User decides next action
        3. Return complete trajectory with goal data
        """
```

## Data Structures

### GoalTurn

Single conversation turn with goal tracking:

```python
@dataclass
class GoalTurn:
    # Standard turn data
    turn_number: int
    query: str
    agent_response: str
    tool_calls: List[Dict]

    # Goal tracking (NEW)
    completed_sub_goals: List[str]
    remaining_sub_goals: List[str]
    goal_progress: float  # 0.0-1.0

    # User evaluation
    satisfaction_level: float
    user_decision: str
```

### GoalTrajectory

Complete conversation with goal completion data:

```python
@dataclass
class GoalTrajectory:
    # Goal data (NEW)
    user_goal: str
    sub_goals: List[str]
    goal_completion_rate: float
    goal_achieved: bool

    # Conversation data
    turns: List[GoalTurn]
    total_turns: int
    final_satisfaction: float
```

## Evaluation Logic

Satisfaction calculation weights goal progress heavily:

```python
Base Satisfaction (0.0-1.0)
  + Tool usage: Used tools → +0.0, No tools → -0.3 to -0.5
  + Goal progress: Advanced goal → +0.1 to +0.2
  + Goal regression: No progress → -0.2 to -0.3
  + Goal completion: All done → +0.2
= Final Satisfaction

Termination:
- goal_progress == 1.0 → TERMINATE (goal_achieved)
- satisfaction >= threshold → TERMINATE (satisfied)
- satisfaction <= frustration → TERMINATE (frustrated)
- turns >= max_turns → TERMINATE (max_turns)
```

## Usage

### Generate Goal-Oriented Seeds

```bash
python mcp_generate/query_generate_goaloriented.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/requests/goaloriented_seeds.json \
  --num-queries 20
```

Output format:

```json
{
  "items": [
    {
      "query": "What events are in Bodrum?",
      "goal": "I'm planning a trip and need events, weather, restaurants, hotels.",
      "reference_tools": [...],
      "goal_category": "trip_planning",
      "complexity": "medium"
    }
  ]
}
```

### Run Goal-Oriented Agent

```bash
# From seeds file
python runtime/run_goaloriented_agent.py \
  --seeds mcp_generate/requests/goaloriented_seeds.json \
  --persona curious_researcher \
  --max-turns 8 \
  --save-trajectory

# Single query + goal
python runtime/run_goaloriented_agent.py \
  "What events are in Bodrum?" \
  --goal "I'm planning a trip and need events, weather, restaurants, hotels" \
  --persona curious_researcher \
  --save-trajectory
```

## Goal Categories

The system supports various goal types:

1. **Trip Planning**
   - Events + Weather + Restaurants + Hotels
   - Example: "Plan weekend trip to Bodrum"

2. **Research**
   - Papers + Methods + Researchers + Trends
   - Example: "Understand small language models landscape"

3. **Investment Decision**
   - Price + News + Financials + Competitors
   - Example: "Evaluate Tesla stock investment"

4. **Product Comparison**
   - Reviews + Specs + Prices + Availability
   - Example: "Choose best laptop for development"

5. **Problem Diagnosis**
   - Causes + Solutions + Similar Cases + Verification
   - Example: "Fix Python memory leak issue"

6. **Event Discovery**
   - Events + Topics + Tickets + Speakers
   - Example: "Find tech conferences to attend in 2025"

## Comparison with Multi-Turn Agent

| Feature | Multi-Turn | Goal-Oriented |
|---------|-----------|---------------|
| User Goal | Exploratory | Specific hidden goal |
| Follow-ups | Reactive to responses | Goal-driven |
| Progress Tracking | No | Yes (sub-goals) |
| Evaluation | Quality + tools | **Goal progress** + quality + tools |
| Termination | Satisfaction-based | **Goal completion** OR satisfaction |
| Use Case | Open exploration | Task completion |

## Example Trajectory

```json
{
  "metadata": {
    "seed_query": "What events are in Bodrum?",
    "user_goal": "I'm planning a trip to Bodrum and need events, weather, restaurants, hotels.",
    "sub_goals": [
      "Find upcoming events in Bodrum",
      "Check weather forecast",
      "Get restaurant recommendations",
      "Find hotel options"
    ],
    "goal_completion_rate": 1.0,
    "goal_achieved": true
  },
  "turns": [
    {
      "turn_number": 1,
      "query": "What events are in Bodrum?",
      "completed_sub_goals": ["Find upcoming events in Bodrum"],
      "remaining_sub_goals": ["Check weather forecast", "Get restaurant recommendations", "Find hotel options"],
      "goal_progress": 0.25,
      "satisfaction_level": 0.7
    },
    {
      "turn_number": 2,
      "query": "What's the weather forecast?",
      "completed_sub_goals": ["Check weather forecast"],
      "remaining_sub_goals": ["Get restaurant recommendations", "Find hotel options"],
      "goal_progress": 0.5,
      "satisfaction_level": 0.75
    },
    {
      "turn_number": 3,
      "query": "Can you recommend restaurants?",
      "completed_sub_goals": ["Get restaurant recommendations"],
      "remaining_sub_goals": ["Find hotel options"],
      "goal_progress": 0.75,
      "satisfaction_level": 0.8
    },
    {
      "turn_number": 4,
      "query": "Where can I find hotels?",
      "completed_sub_goals": ["Find hotel options"],
      "remaining_sub_goals": [],
      "goal_progress": 1.0,
      "satisfaction_level": 0.9,
      "termination_reason": "goal_achieved"
    }
  ],
  "summary": {
    "total_turns": 4,
    "final_satisfaction": 0.9,
    "goal_completion_rate": 1.0
  }
}
```

## Files

- **Runtime**: `runtime/run_goaloriented_agent.py` (~850 lines)
- **Query Generation**: `mcp_generate/query_generate_goaloriented.py` (~400 lines)
- **Trajectories**: `trajectories/goaloriented/goal_*.json`
- **Documentation**: `docs/GOAL_ORIENTED_AGENT.md` (this file)

## Testing

```bash
# Generate 5 test seeds
python mcp_generate/query_generate_goaloriented.py \
  --in MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \
  --out mcp_generate/requests/goaloriented_seeds_test.json \
  --num-queries 5

# Run on test seeds
python runtime/run_goaloriented_agent.py \
  --seeds mcp_generate/requests/goaloriented_seeds_test.json \
  --persona casual_user \
  --max-turns 6 \
  --save-trajectory

# Check trajectories
ls -lh trajectories/goaloriented/
cat trajectories/goaloriented/goal_*.json | jq '.metadata'
```

## Implementation Notes

### Goal Decomposition

The `GoalTracker.decompose_goal()` uses an LLM to break down the user's goal:

- Input: Natural language goal description
- Output: 3-6 specific, measurable sub-goals
- Fallback: If decomposition fails, uses the full goal as a single sub-goal

### Progress Evaluation

The `GoalTracker.evaluate_progress()` determines sub-goal completion:

- Checks if agent's response + tools address a sub-goal
- Requires tool usage (not internal knowledge)
- Requires actionable information
- Returns list of completed sub-goals this turn

### Satisfaction Bonus/Penalty

Goal progress affects satisfaction:

```python
if completed_sub_goals_this_turn:
    satisfaction += 0.1 to 0.2  # Progress bonus
else:
    satisfaction -= 0.2 to 0.3  # No progress penalty

if goal_progress == 1.0:
    satisfaction += 0.2  # Completion bonus
    → TERMINATE with goal_achieved
```

## Future Enhancements

Potential improvements:

1. **Sub-Goal Prioritization**: Not all sub-goals equally important
2. **Partial Credit**: Sub-goals can be partially completed
3. **Goal Revision**: User adjusts goal mid-conversation
4. **Tool Suggestion**: Recommend tools for remaining sub-goals
5. **Multi-Agent Collaboration**: Multiple agents working toward shared goal

---

*Created: 2025-11-11*
*Author: Goal-Oriented Agent Development Team*
