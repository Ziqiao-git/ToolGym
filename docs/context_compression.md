# Context Compression Mechanism

## Overview

To address the issue of tool results being too long and exceeding the LLM context window during long conversations in ReAct Agent, we implemented a **two-layer compression strategy**.

## Two-Layer Compression Strategy

### Layer 1: Streaming Compression for Individual Tool Results

**Trigger Condition:** When a single tool result > 50,000 characters

**Process:**
1. Split the large result into 50K chunks
2. Independently summarize each chunk using LLM (target: compress to 10% of original)
3. Combine all chunk summaries
4. If the combined summary is still > 50K, apply second-level compression (target: 10%)
5. Return the compressed string

**When it happens:** After `call_tool()` returns a result, before adding it to the trajectory

**Code location:** `ContextManager.process_tool_result()`

### Layer 2: Global Compression of Entire Trajectory

**Trigger Condition:** When total context usage > 80% of model capacity

**Total context includes:**
- System prompt (~5K)
- Agent instructions (~3K)
- All thought, action, result in trajectory

**Process:**
1. Calculate current total context size
2. If exceeds threshold (e.g., 200K × 80% = 160K), trigger global compression
3. Iterate through each turn in the entire trajectory
4. For each turn, **compress only the `result` field** (if result > 5K, target: 10%)
5. Use LLM for compression, providing thought and action as context
6. **Directly modify the `result` field value in trajectory**

**When it happens:** Check on every `add_turn()`, trigger immediately if threshold is exceeded

**Code location:** `ContextManager._compress_all_results()`

## Key Design Guarantees

1. ✅ **Data structure unchanged:** trajectory is always `List[{"thought": str, "action": dict, "result": str}]`
2. ✅ **Compress only result:** `thought` and `action` fields are completely unmodified
3. ✅ **No step deletion:** All reasoning turns are preserved
4. ✅ **Two-layer protection:**
   - Layer 1 prevents individual results from being too large
   - Layer 2 prevents cumulative total context from being too large
5. ✅ **Backward compatible:** Existing code requires no changes, trajectory structure is identical

## Usage

### Enable compression by default (recommended)

```bash
python runtime/run_react_agent.py "Your query here" --save-trajectory
```

### Customize context limit

```bash
python runtime/run_react_agent.py "Your query" \
  --context-limit 128000 \
  --save-trajectory
```

### Disable compression (not recommended)

```bash
python runtime/run_react_agent.py "Your query" \
  --disable-compression \
  --save-trajectory
```

## Parameter Configuration

### ContextManager Initialization Parameters

```python
ContextManager(
    llm=llm,                          # LLM instance for summarization
    summarizer_llm=summarizer_llm,    # Optional separate LLM for compression
    model_context_limit=200000,       # Model context window size (in characters)
    system_prompt_length=5000,        # Estimated system prompt length
    instruction_length=3000,          # Estimated instruction length
)
```

### Compression Threshold Parameters

In the `ContextManager` class:

```python
# Layer 1 parameters
self.single_result_threshold = 50000  # Trigger streaming compression if single result > 50K
self.chunk_size = 50000              # Process 50K at a time

# Layer 2 parameters
self.global_compression_threshold = 0.8  # Trigger global compression at 80% capacity
self.result_compression_min_size = 5000  # Only compress results > 5K
```

To adjust these parameters, directly modify the corresponding values in `context_manager.py`.

## Trajectory Output Format

With compression enabled, the saved trajectory JSON will include compression statistics:

```json
{
  "metadata": { ... },
  "reasoning_trace": [ ... ],
  "execution": {
    "final_response": "...",
    "tool_calls": [
      {
        "thought": "...",
        "action": {...},
        "result": "compressed result"
      }
    ]
  },
  "context_compression": {
    "enabled": true,
    "stats": {
      "total_size": 145000,
      "model_limit": 200000,
      "usage_percent": 72.5,
      "turns_count": 15,
      "compression_threshold": 160000,
      "needs_compression": false
    }
  }
}
```

## Performance Considerations

### When Layer 1 compression is triggered

- Single tool result > 50K characters
- Examples: search returns large results, reading large files, API returns detailed data

### When Layer 2 compression is triggered (global compression)

- Total context (system prompt + instructions + all trajectory) > 80% model capacity
- Example: For a 200K model, triggers when total size > 160K

### Compression Model Used

- **Default summarizer:** `openai/gpt-4o-mini` (via OpenRouter)
- **Advantages:** Fast, cheap, good compression quality
- **Customizable:** Pass `summarizer_llm` parameter during initialization

### Compression Ratio Explanation

- **Default compression target: 10%** (preserves more information compared to previous 5%)
- Layer 1 chunk compression: each chunk compressed to ~10%
- Second-level compression: combined summary compressed to ~10%
- Global compression: each result compressed to ~10%

### LLM Call Count

**Layer 1:**
- If result is 150K, will call LLM 3 times (150K ÷ 50K = 3 chunks)
  - Each chunk (50K) compressed to ~5K
  - Combined to ~15K, usually no second-level compression needed
- If combined summary is still > 50K (poor compression), additional 1 second-level compression call

**Layer 2:**
- Call LLM once for each result > 5K
- Example: 15 turns, 10 results > 5K, then 10 LLM calls

**Cost Optimization Suggestions:**
- Default uses GPT-4o-mini as summarizer_llm (fast and cheap)
- Layer 2 compression can adjust `result_compression_min_size` to raise threshold

## Error Handling

### Compression Failure

If LLM call fails:
- **Layer 1:** Use simple truncation as fallback (keep first 1000 characters)
- **Layer 2:** Keep original result, skip compression for that turn

### Logging

The compression process logs detailed information:

```
INFO: Tool result too large (125000 chars), applying streaming compression...
INFO: Breaking result into 3 chunks for summarization
INFO: Compressed 125000 → 35000 chars (28.0%)
WARNING: Context size 165000 exceeds threshold 160000, triggering global compression...
INFO: Turn 5: Compressed result 8000 → 3500 chars
INFO: Global compression complete: 8 results compressed, total size 165000 → 135000 chars (81.8%)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DynamicReActAgent                        │
│                                                             │
│  call_tool()                                                │
│    │                                                        │
│    ├─> super().call_tool()  ──> result (original)          │
│    │                                                        │
│    ├─> ContextManager.process_tool_result()  <─── Layer 1  │
│    │     └─> If > 50K: streaming compression               │
│    │                                                        │
│    ├─> Build trajectory_entry (using compressed result)    │
│    │                                                        │
│    └─> ContextManager.add_turn()  <─── Layer 2             │
│          └─> Check total context                           │
│              └─> If > 80%: global compression all results  │
└─────────────────────────────────────────────────────────────┘
```

## Common Questions

### Q: Does compression lose information?

A: Uses LLM summarization rather than truncation to maximize retention of key information. Currently set to compress to 10% (preserves more information compared to previous 5%), but some details may be lost - this is a trade-off given context window limitations.

### Q: Why not compress thought and action?

A: Because these are the core of the reasoning process, containing decision logic - compression would affect interpretability. Tool results typically occupy the most space but can be summarized.

### Q: Can I enable only Layer 1 without Layer 2?

A: Currently a unified switch, but can be achieved by modifying code:
- Set `global_compression_threshold = 1.0` to effectively disable Layer 2
- Set `single_result_threshold = float('inf')` to disable Layer 1

### Q: How to adjust compression ratio?

A: Modify the divisor in three places in `context_manager.py`:
- Layer 1 chunk compression: `len(chunk) // 10` (current 10%, change to `// 20` for 5%)
- Second-level compression: `len(combined_summary) // 10`
- Global compression: `len(result) // 10`
Larger numbers mean more aggressive compression and less information retained.

### Q: Can compressed trajectories still be used for analysis?

A: Yes. The trajectory structure is completely unchanged, only the result field content is compressed. All analysis tools still apply.

## Compression Effect Example

Assume a tool result is **300K characters**:

**Layer 1 compression:**
1. Split into 6 × 50K chunks
2. Each compressed to ~5K (10% compression ratio)
3. Combined to ~30K ✅ (below 50K threshold, no second-level compression needed)

**If compression is ineffective:**
1. 6 chunks each compressed to 10K
2. Combined to 60K ❌ (exceeds 50K threshold)
3. Trigger second-level compression: 60K → 6K

## Future Optimization Directions

1. **Smart compression strategy:** Choose different compression methods based on result type (JSON, text, tables)
2. **Tiered compression:** Preserve more information for important turns (e.g., related to final answer)
3. **Incremental compression:** Only compress new turns, avoid re-compressing already compressed content
4. **Cache compression results:** Avoid repeatedly calling LLM for same results
5. **Adaptive compression ratio:** Dynamically adjust compression target based on LLM feedback to avoid "cannot compress" errors
