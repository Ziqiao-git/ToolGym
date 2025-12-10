# MCP Evaluation Analysis Scripts

This folder contains analysis scripts for the MCP-R (Model Context Protocol Research) project. These scripts analyze trajectory and evaluation data to understand model performance in MCP tool usage scenarios.

## Overview

| Script | Purpose |
|--------|---------|
| `analyze_evaluations.py` | Main entry point with unified interface to all analyses |
| `analyze_tool_stats.py` | Tool call success rate analysis |
| `analyze_retry.py` | Retry behavior analysis |
| `analyze_auto_correction.py` | Auto-correction behavior analysis |
| `analyze_weighted.py` | Weighted score analysis (penalizes no-tool trajectories) |
| `data_loaders.py` | Shared data loading functions |

## Data Sources

### Trajectories (`../trajectories/`)
- `trajectory_*.json` files containing tool call sequences
- Each trajectory records: tool calls, results, errors, timing

### Evaluations (`../evaluation/`)
- `eval_*.json` files with per-query evaluation results
- `_summary.json` files with aggregated scores

## Scripts

### 1. analyze_evaluations.py (Main Entry Point)

The main script that provides access to all analysis functionality.

```bash
# Basic usage - show model comparison
python analyze_scripts/analyze_evaluations.py

# Filter by model
python analyze_scripts/analyze_evaluations.py --models claude-3.5 Gemini-3pro

# Show pass@1/2/3 breakdown
python analyze_scripts/analyze_evaluations.py --passes

# Show detailed per-query analysis
python analyze_scripts/analyze_evaluations.py --detailed

# Show score distributions
python analyze_scripts/analyze_evaluations.py --distribution

# Export to CSV
python analyze_scripts/analyze_evaluations.py --export results.csv

# Filter by judge model
python analyze_scripts/analyze_evaluations.py --judge gpt4omini
```

### 2. analyze_tool_stats.py

Analyzes tool call success rates from trajectory files.

```bash
# Basic usage
python analyze_scripts/analyze_tool_stats.py

# Filter by model
python analyze_scripts/analyze_tool_stats.py --model gemini-2.5pro
```

**Output includes:**
- Overall success rate
- Success rates by model (ranking)
- Success rates by MCP server
- Tools with most failures
- Dynamic loading statistics
- Call duration statistics
- Error analysis (MODEL_ERROR vs SERVER_ERROR)

**Error Classification:**
| Category | Description | Subcategories |
|----------|-------------|---------------|
| MODEL_ERROR | LLM called tool incorrectly | missing_required_field, wrong_type, invalid_schema, invalid_arguments, validation_error, invalid_date_range |
| SERVER_ERROR | MCP server issues | rate_limit, quota_exceeded, null_reference, data_processing_error, index_error, not_found, network_error, server_unavailable, execution_error |
| UNKNOWN | Cannot determine cause | unclassified |

**Methodology:**
1. Load all trajectory files and extract `tool_calls`
2. Check if `status != "success"` or result contains `"isError=True"`
3. Classify errors using regex patterns on error messages
4. Aggregate statistics by model, server, and tool

### 3. analyze_retry.py

Analyzes how models behave when tool calls fail.

```bash
# Basic usage
python analyze_scripts/analyze_retry.py

# Filter by model
python analyze_scripts/analyze_retry.py --model claude-3.5
```

**Output includes:**
- Behavior after tool failure (retry same, try different, give up)
- Retry rate percentages by model
- Consecutive same-tool call patterns
- Tool usage diversity
- Queries with most/fewest tool calls
- Hardest/easiest queries across models

**Key Metrics:**
| Metric | Description |
|--------|-------------|
| Retry Same % | How often model retries the exact same tool after failure |
| Switch Tool % | How often model tries a different tool after failure |
| Give Up % | How often trajectory ends after a failure |
| Avg Consecutive | Average times same tool called in a row |
| Tool Diversity | Average unique tools per trajectory |

**Methodology:**
1. For each trajectory, build a `tool_sequence` from `tool_calls`
2. When a tool call fails (`is_error=True`), check what happens next:
   - Same tool called again → "retry_same_tool"
   - Different tool called → "retry_different_tool"
   - No more calls → "gave_up"
3. Track consecutive same-tool calls to measure persistence
4. Calculate tool diversity (unique tools per trajectory)

### 4. analyze_auto_correction.py

Analyzes auto-correction behavior: when a model makes an error and successfully fixes it.

```bash
# Basic usage
python analyze_scripts/analyze_auto_correction.py

# Filter by model
python analyze_scripts/analyze_auto_correction.py --model gemini-2.5pro
```

**Definition:**
Auto-correction = A sequence where:
1. Model calls tool T and gets an error
2. Model calls tool T again (possibly multiple times)
3. Eventually tool T succeeds

**Output includes:**
- Auto-correction rate by model (ranking)
- Tools most often successfully corrected
- Error types most often corrected
- Example correction sequences

**Key Metrics:**
| Metric | Description |
|--------|-------------|
| Errors | Total tool call errors encountered |
| Retried | Errors followed by retry of same tool |
| Corrected | Retries that eventually succeeded |
| Rate | Correction success rate (Corrected / Retried) |
| Avg Attempts | Average retry attempts needed to correct |

**Methodology:**
1. For each trajectory, iterate through `tool_sequence`
2. When an error is found on tool T:
   - Look ahead for subsequent calls to the same tool T
   - Track if any retry calls succeed
   - Record the number of attempts needed
3. Calculate correction rates by model, tool, and error type

### 5. analyze_weighted.py

Analyzes weighted scores that penalize trajectories without tool usage.

```bash
# Basic usage with default weights
python analyze_scripts/analyze_weighted.py

# Custom weights
python analyze_scripts/analyze_weighted.py --fa-weight 0.6 --step-weight 0.4

# Stronger penalty for no-tool trajectories
python analyze_scripts/analyze_weighted.py --no-tool-penalty 0.3

# Filter by model or judge
python analyze_scripts/analyze_weighted.py --model claude-3.5 --judge gpt4omini
```

**Formula:**
```
base_score = (FA × fa_weight) + (Step × step_weight)
weighted_score = base_score × penalty_multiplier
```

Where `penalty_multiplier = no_tool_penalty` if no real tools used, else `1.0`

**Default Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| fa_weight | 0.5 | Weight for final answer score |
| step_weight | 0.5 | Weight for step score |
| no_tool_penalty | 0.5 | Penalty multiplier (50% penalty) |

**Penalty Conditions:**
1. No actual MCP tools used (excluding `search_tools` which only finds tools)
2. No reasoning steps recorded

**Output includes:**
- Model comparison (original vs weighted scores)
- Ranking changes between original and weighted scoring
- Penalized trajectories breakdown
- Top non-penalized performers

**Why This Matters:**
In MCP evaluation, we want to measure tool usage capability. A model that answers correctly from internal knowledge without calling tools is not demonstrating MCP capability. The penalty ensures tool usage is properly valued.

### 6. data_loaders.py

Shared data loading functions used by all analysis scripts.

**Functions:**
| Function | Description |
|----------|-------------|
| `classify_error(result)` | Classify tool call error into MODEL_ERROR, SERVER_ERROR, or UNKNOWN |
| `load_trajectory_data(traj_dir, model_filter)` | Load tool calls and trajectories from trajectory files |
| `load_summary_files(eval_dir)` | Load all `_summary.json` files |
| `load_individual_evals(eval_dir, model, judge)` | Load individual `eval_*.json` files |

**Data Structures:**

Tool Call Record:
```python
{
    "model": str,           # Model name
    "pass_number": int,     # Pass number (1, 2, 3)
    "server": str,          # MCP server name
    "tool": str,            # Tool name
    "status": str,          # Call status
    "is_error": bool,       # Whether call errored
    "error_category": str,  # MODEL_ERROR, SERVER_ERROR, UNKNOWN
    "error_subcategory": str,  # Specific error type
    "error_message": str,   # Truncated error message
    "duration_seconds": float,
    "dynamically_loaded": bool,
    "trajectory_file": str,
    "call_index": int,
}
```

Trajectory Record:
```python
{
    "model": str,
    "pass_number": int,
    "query_uuid": str,
    "trajectory_file": str,
    "tool_sequence": List[Dict],  # List of tool calls with server, tool, is_error, error_category
    "total_tool_calls": int,
}
```

## Combined Analysis Examples

```bash
# Full analysis with all features
python analyze_scripts/analyze_evaluations.py \
    --tool-stats \
    --retry-analysis \
    --auto-correction \
    --weighted

# Compare models with weighted scoring and tool stats
python analyze_scripts/analyze_evaluations.py \
    --models claude-3.5 Gemini-3pro gpt-4o-mini \
    --weighted --no-tool-penalty 0.3 \
    --tool-stats

# Export detailed analysis
python analyze_scripts/analyze_evaluations.py \
    --detailed \
    --distribution \
    --export analysis_results.csv
```

## Directory Structure

```
analyze_scripts/
├── README.md                    # This file
├── analyze_evaluations.py       # Main entry point
├── analyze_tool_stats.py        # Tool call statistics
├── analyze_retry.py             # Retry behavior
├── analyze_auto_correction.py   # Auto-correction analysis
├── analyze_weighted.py          # Weighted scoring
└── data_loaders.py             # Shared data loading
```
