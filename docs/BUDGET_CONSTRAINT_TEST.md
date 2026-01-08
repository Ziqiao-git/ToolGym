# Budget Constraint Test - Tool Call Efficiency vs Quality Tradeoff

Research the tradeoff between **tool call efficiency** and **answer quality** when agents operate under limited tool call budgets.

## 📋 Experiment Goals

By constraining the number of non-search tool calls available to the agent, we aim to observe:
1. **Cost-Awareness**: Whether agents can adhere to budget constraints
2. **Quality Degradation**: How budget limitations impact answer quality
3. **Strategy Changes**: How agents adapt their tool selection and usage patterns under constraints

## 🚀 Quick Start

### Step 1: Generate Budget-Constrained Queries

Use the `add_budget_constraint.py` script to add budget constraint instructions to original queries:

```bash
# Budget 3 (tight): Max 3 non-search tool calls
python task_creation_engine/add_budget_constraint.py \
    --input task_creation_engine/queries10.json \
    --output task_creation_engine/queries_budget_3.json \
    --budget 300 \
    --cost-per-call 100

# Budget 5 (medium): Max 5 non-search tool calls
python task_creation_engine/add_budget_constraint.py \
    --input task_creation_engine/queries10.json \
    --output task_creation_engine/queries_budget_5.json \
    --budget 500 \
    --cost-per-call 100

# Budget 7 (loose): Max 7 non-search tool calls
python task_creation_engine/add_budget_constraint.py \
    --input task_creation_engine/queries10.json \
    --output task_creation_engine/queries_budget_7.json \
    --budget 700 \
    --cost-per-call 100
```

### Step 2: Run Experiments

#### Option 1: Budget 3 (Tight Budget)

```bash
python runtime/budget_constraint_test.py \
    --query-file task_creation_engine/queries_budget_3.json \
    --budget-level budget_3 \
    --model anthropic/claude-3.5-sonnet \
    --pass-number 1 \
    --max-concurrent 3
```

#### Option 2: Budget 5 (Medium Budget)

```bash
python runtime/budget_constraint_test.py \
    --query-file task_creation_engine/queries_budget_5.json \
    --budget-level budget_5 \
    --model anthropic/claude-3.5-sonnet \
    --pass-number 1 \
    --max-concurrent 3
```

#### Option 3: Budget 7 (Loose Budget)

```bash
python runtime/budget_constraint_test.py \
    --query-file task_creation_engine/queries_budget_7.json \
    --budget-level budget_7 \
    --model anthropic/claude-3.5-sonnet \
    --pass-number 1 \
    --max-concurrent 3
```

#### Option 4: Baseline (No Budget Constraint)

```bash
python runtime/budget_constraint_test.py \
    --query-file task_creation_engine/queries10.json \
    --budget-level baseline \
    --model anthropic/claude-3.5-sonnet \
    --pass-number 1 \
    --max-concurrent 3
```

### 🔁 Run All Budget Levels (Recommended)

```bash
# Create all budget-constrained query files
for budget_val in 300 500 700; do
    level=$((budget_val / 100))
    python task_creation_engine/add_budget_constraint.py \
        --input task_creation_engine/queries10.json \
        --output task_creation_engine/queries_budget_${level}.json \
        --budget ${budget_val} \
        --cost-per-call 100
done

# Run all budget levels
for budget in budget_3 budget_5 budget_7; do
    python runtime/budget_constraint_test.py \
        --query-file task_creation_engine/queries_${budget}.json \
        --budget-level ${budget} \
        --model anthropic/claude-3.5-sonnet \
        --pass-number 1 \
        --max-concurrent 3
done

# Run baseline
python runtime/budget_constraint_test.py \
    --query-file task_creation_engine/queries10.json \
    --budget-level baseline \
    --model anthropic/claude-3.5-sonnet \
    --pass-number 1 \
    --max-concurrent 3
```

## 📊 Budget Level Descriptions

### Budget 3: Tight Budget 🔴

- **Total Budget**: 300 credits
- **Cost per Call**: 100 credits
- **Max Calls**: 3 non-search tool calls
- **Search Tools**: FREE (no budget consumed)
- **Use Case**: Test agent performance under extreme budget pressure
- **Expected Behavior**:
  - Agent must carefully plan each tool call
  - Increased use of free search tools for exploration
  - May sacrifice some quality to complete the task

### Budget 5: Medium Budget 🟡

- **Total Budget**: 500 credits
- **Cost per Call**: 100 credits
- **Max Calls**: 5 non-search tool calls
- **Search Tools**: FREE
- **Use Case**: Test impact of moderate budget constraints
- **Expected Behavior**:
  - More balanced quality-efficiency tradeoff
  - Still requires strategic tool call planning
  - Most tasks should be completable

### Budget 7: Loose Budget 🟢

- **Total Budget**: 700 credits
- **Cost per Call**: 100 credits
- **Max Calls**: 7 non-search tool calls
- **Search Tools**: FREE
- **Use Case**: Test impact of light budget constraints
- **Expected Behavior**:
  - Performance close to baseline
  - Minimal impact from budget constraints
  - Should maintain high quality in most cases

### Baseline: Unlimited ⭐

- **Total Budget**: Unlimited
- **Cost per Call**: 0 (no cost)
- **Max Calls**: Unlimited
- **Use Case**: Control group providing performance baseline
- **Expected Behavior**:
  - Agent completely ignores budget
  - Free use of all tools
  - Provides quality upper bound reference

## 🎯 Budget Constraint Mechanism

### Query Rewriting

Original queries are automatically appended with budget constraint instructions:

```
---
RESOURCE CONSTRAINT:
- Total Budget: 300 credits
- Each non-search tool call costs: 100 credits
- Search tools (search_tools, find_tools, etc.) are FREE (exploration is encouraged)
- Maximum supported non-search tool calls: 3 times

Please solve this task efficiently within your budget.
You currently have 300 credits available.
```

### Budget Calculation

- **Search Tools (FREE)**: `search_tools`, `find_tools` and other exploratory tools are free
- **Execution Tools (CHARGED)**: All other tools cost a fixed amount per call (default 100 credits)
- **Budget Exhausted**: When max calls are reached, agent is expected to stop using paid tools

### Evaluation Metrics

The experiment tracks the following data for analysis:

```python
trajectory_metadata = {
    "budget_config": {
        "total_budget": 500,
        "cost_per_call": 100,
        "max_allowed_calls": 5,
    },
    "budget_usage": {
        "actual_non_search_calls": 4,      # Actual calls made
        "final_remaining_budget": 100,      # Remaining budget
        "efficiency": 0.8,                  # Budget utilization (4/5)
        "budget_exhausted": False,          # Whether budget was exhausted
        "exceeded_budget": False,           # Whether budget was exceeded
        "termination_reason": "task_completed"
    },
    "tool_call_breakdown": {
        "search_tools": 3,     # Free search calls
        "other_tools": 4,      # Paid tool calls
        "total_calls": 7
    }
}
```

## 📂 Output Structure

### Trajectory File Locations

```
trajectories/Budget_test/
├── anthropic-claude-3.5-sonnet/          # Model name
│   └── pass@1/                           # Pass number
│       ├── budget_3/                     # Budget level: 3 calls
│       │   ├── trajectory_72a11620-d156-11f0-8107-3abd0a1b915e.json
│       │   ├── trajectory_72a1f04a-d156-11f0-8107-3abd0a1b915e.json
│       │   └── ...
│       ├── budget_5/                     # Budget level: 5 calls
│       │   └── ...
│       ├── budget_7/                     # Budget level: 7 calls
│       │   └── ...
│       └── baseline/                     # Unlimited control group
│           └── ...
└── budget_test_budget_3_pass1_a1b2c3d4_20260114_153045.json  # Batch summary
```

### Trajectory File Content

Each trajectory file contains:

```json
{
  "query_uuid": "72a11620-d156-11f0-8107-3abd0a1b915e",
  "query": "Original query + budget constraint instructions",
  "model": "anthropic/claude-3.5-sonnet",
  "final_answer": "Agent's final answer",
  "reasoning_trace": [...],
  "metadata": {
    "budget_config": {
      "total_budget": 300,
      "cost_per_call": 100,
      "max_allowed_calls": 3
    },
    "budget_usage": {
      "actual_non_search_calls": 3,
      "final_remaining_budget": 0,
      "efficiency": 1.0,
      "budget_exhausted": true,
      "exceeded_budget": false
    },
    "tool_call_breakdown": {
      "search_tools": 5,
      "other_tools": 3,
      "total_calls": 8
    }
  }
}
```

### Batch Summary File

```json
{
  "metadata": {
    "batch_id": "a1b2c3d4",
    "experiment": "Budget_test",
    "budget_level": "budget_3",
    "budget_config": {
      "total_budget": 300,
      "cost_per_call": 100,
      "max_allowed_calls": 3
    },
    "timestamp": "2026-01-14T15:30:45.123456",
    "query_file": "task_creation_engine/queries_budget_3.json",
    "total_queries": 10,
    "successful": 9,
    "failed": 1,
    "model": "anthropic/claude-3.5-sonnet",
    "max_iterations": 20,
    "pass_number": 1
  },
  "results": [
    {
      "index": 1,
      "uuid": "72a11620-d156-11f0-8107-3abd0a1b915e",
      "budget_level": "budget_3",
      "status": "success"
    },
    ...
  ]
}
```

## 🔧 Parameter Reference

### `add_budget_constraint.py` Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--input` | ✅ | - | Input original query JSON file |
| `--output` | ✅ | - | Output budget-constrained query JSON file |
| `--budget` | ✅ | - | Total budget in credits (e.g., 300, 500, 700) |
| `--cost-per-call` | ❌ | 100 | Cost per non-search tool call in credits |

### `budget_constraint_test.py` Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--query-file` | ✅ | - | JSON file containing test queries |
| `--budget-level` | ✅ | - | Budget level: `budget_3`, `budget_5`, `budget_7`, `baseline` |
| `--max-iterations` | ❌ | 20 | Maximum iterations per query |
| `--model` | ❌ | `anthropic/claude-3.5-sonnet` | Model to use |
| `--pass-number` | ❌ | 1 | Pass number for multiple runs |
| `--max-concurrent` | ❌ | 5 | Maximum concurrent queries |

## 📈 Post-Experiment Analysis

After running experiments, analyze results using the following dimensions:

### 1. Quality vs Budget Curve

```python
# Compare average quality scores across budget levels
budget_levels = ["budget_3", "budget_5", "budget_7", "baseline"]
avg_scores = [0.65, 0.78, 0.82, 0.85]  # Example data

# Plot curve to observe quality degradation trend
```

### 2. Cost-Awareness Analysis

```python
# Calculate rate of budget violations
exceeded_budget_rate = {
    "budget_3": 0.15,  # 15% of queries exceeded budget
    "budget_5": 0.05,  # 5% of queries exceeded budget
    "budget_7": 0.02,  # 2% of queries exceeded budget
}
```

### 3. Strategy Change Analysis

```python
# Compare tool usage patterns with/without budget constraints
tool_usage = {
    "baseline": {
        "avg_search_calls": 2.5,
        "avg_other_calls": 6.3,
        "tool_diversity": 0.7
    },
    "budget_3": {
        "avg_search_calls": 4.2,  # More free search
        "avg_other_calls": 2.9,   # Fewer paid calls
        "tool_diversity": 0.5     # More conservative tool selection
    }
}
```

### 4. Marginal Returns Analysis

- Budget 3 → Budget 5: +20% quality gain (0.65 → 0.78)
- Budget 5 → Budget 7: +5% quality gain (0.78 → 0.82)
- Budget 7 → Baseline: +3.6% quality gain (0.82 → 0.85)

**Conclusion**: Diminishing returns exist; Budget 5 may be the optimal cost-performance point

## ⚠️ Important Notes

1. **"Soft" Budget Constraints**: Currently implemented via query instructions, relying on the model's instruction-following ability. For "hard" constraints, runtime-level enforcement would be needed.

2. **Search Tools are Free**: `search_tools` and other exploratory tools don't count toward budget, encouraging agents to explore before executing.

3. **Query Difficulty Variance**: The same budget constraint impacts queries of different difficulty levels differently. Consider grouping by query complexity during analysis.

4. **Model Differences**: Different models may vary significantly in understanding and adhering to budget constraints (e.g., Claude vs GPT).

## 🔗 Related Experiments

- **Emergency Test** (`docs/EMERGENCY_TEST.md`): Robustness testing under tool failures
- **Multi-turn Agent** (`docs/MULTI_TURN_AGENT_SPEC.md`): Multi-turn conversation agent specification
- **Goal-oriented Agent** (`docs/GOAL_ORIENTED_AGENT.md`): Goal-oriented agent specification

## 📝 Example Console Output

```
======================================================================
Budget Constraint Test - Parallel Generation
======================================================================
Batch ID:         a1b2c3d4
Budget level:     budget_3
Budget config:    300 credits, 100 per call, max 3 calls
Total queries:    10
Model:            anthropic/claude-3.5-sonnet
Max iterations:   20
Pass number:      1
Max concurrent:   3
Output:           trajectories/Budget_test/anthropic-claude-3.5-sonnet/pass@1/budget_3/
======================================================================

Starting 10 queries with max concurrency of 3...

[1] Starting query 72a11620-d156-11f0-8107-3abd0a1b915e (budget: budget_3)...
[2] Starting query 72a1f04a-d156-11f0-8107-3abd0a1b915e (budget: budget_3)...
[3] Starting query 72a17642-d156-11f0-8107-3abd0a1b915e (budget: budget_3)...
[1] ✓ Query 72a11620-d156-11f0-8107-3abd0a1b915e completed successfully
[4] Starting query 72a23096-d156-11f0-8107-3abd0a1b915e (budget: budget_3)...
[2] ✓ Query 72a1f04a-d156-11f0-8107-3abd0a1b915e completed successfully
...

======================================================================
Budget Test Generation Summary (budget_3)
======================================================================
Total:       10
Successful:  9
Failed:      1
======================================================================

✓ Batch ID:      a1b2c3d4
✓ Summary saved: trajectories/Budget_test/budget_test_budget_3_pass1_a1b2c3d4_20260114_153045.json
```

## 🎓 Research Questions

This experiment can help answer the following research questions:

1. **How much does budget constraint impact quality?**
   - Quantify quality degradation at different budget levels

2. **How cost-aware are models?**
   - Do models truly understand and respect budget constraints?
   - What is the frequency of budget violations?

3. **What strategy changes occur?**
   - How do tool selection patterns change under budget pressure?
   - Do agents use more free exploratory tools?

4. **Where is the optimal cost-performance ratio?**
   - Which budget level provides the best quality/cost tradeoff?
   - Is there a point of diminishing returns?

5. **How do different models compare?**
   - Differences in budget awareness between Claude, GPT, and other models
   - Which models are better at optimizing under constraints?
