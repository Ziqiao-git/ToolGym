# Emergency Test - 工具拦截实验

测试 MCP Agent 在工具突然失效时的鲁棒性和恢复能力。

## 快速开始

### 对照组: 不拦截（Control Group）

```bash
python runtime/emergency_test.py \
    --query-file task_creation_engine/generated_queries.json \
    --strategy no_interception \
    --max-iterations 20 \
    --model anthropic/claude-3.5-sonnet \
    --max-concurrent 3
```

### 方案 1: 拦截第一个非 search 工具

```bash
python runtime/emergency_test.py \
    --query-file task_creation_engine/generated_queries.json \
    --strategy first_non_search \
    --max-iterations 20 \
    --model anthropic/claude-3.5-sonnet \
    --max-concurrent 3
```

### 方案 2: 随机 20% 概率拦截

```bash
python runtime/emergency_test.py \
    --query-file task_creation_engine/generated_queries.json \
    --strategy random_20 \
    --max-iterations 20 \
    --model anthropic/claude-3.5-sonnet \
    --max-concurrent 3 \
    --random-seed 42
```

### 运行所有策略（推荐）

```bash
python runtime/emergency_test.py \
    --query-file task_creation_engine/generated_queries.json \
    --strategy all \
    --max-iterations 20 \
    --model anthropic/claude-3.5-sonnet \
    --max-concurrent 3
```

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--query-file` | ✅ | - | 包含测试 queries 的 JSON 文件 |
| `--strategy` | ❌ | `first_non_search` | 拦截策略: `no_interception`, `first_non_search`, `random_20`, `all` |
| `--max-iterations` | ❌ | 20 | 每个 query 的最大迭代次数 |
| `--model` | ❌ | `anthropic/claude-3.5-sonnet` | 使用的模型 |
| `--pass-number` | ❌ | 1 | Pass 编号（用于多次运行） |
| `--max-concurrent` | ❌ | 5 | 最大并发数 |
| `--random-seed` | ❌ | 42 | 随机种子（用于 `random_20` 策略，确保可复现） |
| `--error-message` | ❌ | `Error: Tool temporarily unavailable...` | 拦截时返回的错误消息 |

## 拦截策略说明

### Strategy 0: `no_interception` 🔵 对照组
- **触发条件**: 完全不拦截任何工具调用
- **使用场景**: 作为对照组，提供正常情况下的 baseline 性能
- **预期行为**: Agent 正常执行，所有工具调用都成功

### Strategy 1: `first_non_search` 🟡 首次失败
- **触发条件**: 拦截第一个非 `search_tools` 的工具调用（只拦截一次）
- **使用场景**: 测试 agent 在首次尝试使用工具时失败的反应
- **预期行为**: Agent 应该尝试使用其他工具或不同的方法

### Strategy 2: `random_20` 🔴 随机失败
- **触发条件**: 每次非 `search_tools` 的工具调用有 20% 概率被拦截
- **使用场景**: 模拟真实场景中工具的不稳定性，可能多次失败
- **预期行为**: Agent 应该具有持续的容错能力，在多次失败后仍能完成任务
- **可复现性**: 通过 `--random-seed` 参数确保每次运行结果一致
  - 每个 query 使用不同的 seed: `random_seed + query_index`
  - 相同的 seed 会产生相同的拦截序列
  - 不同的 query 使用不同的 seed，确保拦截模式多样化

### Strategy: `all` ⭐ 推荐
- **行为**: 自动运行以上三种策略
- **优势**: 一次性获得完整的对比数据
  - 对照组: 正常表现
  - 首次失败: 恢复能力
  - 随机失败: 持续容错能力

## 输出结构

### Trajectory 文件位置

```
trajectories/
└── Emergency_test/
    ├── {model}/                           # 例如: claude-3.5-sonnet
    │   └── pass@{N}/                      # 例如: pass@1
    │       ├── no_interception/           # 对照组 trajectories
    │       │   ├── trajectory_{uuid}_{timestamp}.json
    │       │   └── ...
    │       ├── first_non_search/          # 策略 1 的 trajectories
    │       │   ├── trajectory_{uuid}_{timestamp}.json
    │       │   └── ...
    │       └── random_20/                 # 策略 2 的 trajectories
    │           ├── trajectory_{uuid}_{timestamp}.json
    │           └── ...
    └── emergency_test_pass1_{batch_id}_{timestamp}.json  # 总结文件
```

### Trajectory JSON 结构

每个 trajectory 文件包含以下内容：

```json
{
  "metadata": {
    "timestamp": "2025-12-13T...",
    "query": "Find GitHub repositories about...",
    "model": "anthropic/claude-3.5-sonnet",
    "max_iterations": 20,
    "pass_number": 1,
    "query_uuid": "abc123...",
    "batch_id": "xyz789",
    "emergency_test": true,
    "interception_strategy": "random_20",
    "interception_stats": {
      "strategy": "random_20",
      "total_tool_calls": 5,
      "non_search_tool_calls": 3,
      "intercepted": true,
      "interception_count": 2,
      "interception_log": [
        {
          "timestamp": "2025-12-13T...",
          "strategy": "random_20",
          "tool_name": "search_repositories",
          "tool_call_count": 2,
          "non_search_call_count": 1,
          "error_message": "Error: Tool temporarily unavailable..."
        },
        {
          "timestamp": "2025-12-13T...",
          "strategy": "random_20",
          "tool_name": "get_file_contents",
          "tool_call_count": 5,
          "non_search_call_count": 3,
          "error_message": "Error: Tool temporarily unavailable..."
        }
      ]
    }
  },
  "reasoning_trace": [
    {"type": "thought", "content": "I need to search for tools..."},
    {"type": "action", "content": "{...}"},
    {"type": "observation", "content": "..."}
  ],
  "execution": {
    "final_response": "Based on my research...",
    "tool_calls": [
      {
        "type": "tool_call",
        "thought": "Let me search for GitHub tools",
        "server": "meta-mcp",
        "tool": "search_tools",
        "arguments": {...},
        "status": "success",
        "result": "..."
      },
      {
        "type": "tool_call",
        "thought": "Now I'll use the search_repositories tool",
        "server": "@smithery-ai/github",
        "tool": "search_repositories",
        "arguments": {...},
        "status": "error",
        "result": "Error: Tool temporarily unavailable (503 Service Unavailable)"
      }
    ],
    "total_tool_calls": 5
  },
  "servers": {...},
  "context_compression": {...},
  "emergency_interception": {
    // 同 metadata.interception_stats
  }
}
```

### 总结文件结构

`emergency_test_pass1_{batch_id}_{timestamp}.json`:

```json
{
  "metadata": {
    "batch_id": "xyz789",
    "test_type": "emergency_interception",
    "timestamp": "2025-12-13T...",
    "query_file": "task_creation_engine/generated_queries.json",
    "total_queries": 100,
    "total_runs": 300,
    "successful": 270,
    "failed": 30,
    "successfully_intercepted": 190,
    "interception_success_rate": 0.704,
    "model": "anthropic/claude-3.5-sonnet",
    "max_iterations": 20,
    "strategies": ["no_interception", "first_non_search", "random_20"],
    "random_seed": 42,
    "error_message": "Error: Tool temporarily unavailable..."
  },
  "results": [
    {
      "index": 1,
      "uuid": "abc123...",
      "status": "success",
      "strategy": "no_interception",
      "intercepted": false,
      "trajectory_path": "trajectories/Emergency_test/..."
    },
    ...
  ]
}
```

## 终端输出

运行时，你会看到清晰的进度输出：

```
══════════════════════════════════════════════════════════════════════
Emergency Test - Batch Trajectory Generation
══════════════════════════════════════════════════════════════════════
Batch ID: a1b2c3d4
Total queries: 100
Model: anthropic/claude-3.5-sonnet
Max iterations per query: 20
Interception strategy: all
Pass number: 1
Max concurrent: 3
Random seed: 42
Error message: Error: Tool temporarily unavailable (503 Service Unavailable)
══════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────
Running strategy: no_interception
──────────────────────────────────────────────────────────────────────
Starting 100 queries with max concurrency of 3...

[1] Starting query abc123...
[2] Starting query def456...
[3] Starting query ghi789...
[1] ✓ Query abc123 completed
[4] Starting query jkl012...
[2] ✓ Query def456 completed
[5] Starting query mno345...
...

──────────────────────────────────────────────────────────────────────
Strategy 'no_interception' Summary
──────────────────────────────────────────────────────────────────────
Total: 100
Successful: 95
Failed: 5
Intercepted: 0 / 95 successful runs
──────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────
Running strategy: first_non_search
──────────────────────────────────────────────────────────────────────
...
```

运行完成后，你会看到总体汇总：

```
══════════════════════════════════════════════════════════════════════
Emergency Test - Overall Summary
══════════════════════════════════════════════════════════════════════
Total runs: 300
Successful: 270
Failed: 30
Successfully intercepted: 190 / 270
Interception success rate: 70.4%
══════════════════════════════════════════════════════════════════════

✓ Batch ID: a1b2c3d4
✓ Summary saved to: trajectories/Emergency_test/emergency_test_pass1_a1b2c3d4_20251213_143022.json
✓ Trajectories saved to: trajectories/Emergency_test/
```

### 关键观察点

对比三种策略，你可以分析：

1. **基线性能** (`no_interception`)
   - 正常情况下的成功率
   - 平均工具调用次数
   - 平均完成时间

2. **首次失败恢复** (`first_non_search`)
   - 拦截后的成功率下降幅度
   - Agent 是否尝试了替代工具
   - 恢复所需的额外工具调用次数

3. **持续容错能力** (`random_20`)
   - 面对多次失败的成功率
   - 被拦截次数 vs 最终成功率的关系
   - Agent 的重试策略和耐心程度

4. **对比分析**
   - `no_interception` vs `first_non_search`: 单次失败的影响
   - `first_non_search` vs `random_20`: 单次失败 vs 多次失败
   - 不同策略下 `reasoning_trace` 的差异

## 实现原理

**零侵入设计** - 通过猴子补丁（Monkey Patching）实现，完全不修改原始代码：

### 核心架构

1. **`emergency_interceptor.py`** - 拦截器模块
   - 通过包装 `agent.call_tool()` 方法实现拦截
   - 支持多种拦截策略（`InterceptionStrategy` 枚举）
   - 自动排除 `search_tools`（可配置）
   - 记录详细的拦截日志和统计信息
   - 支持随机种子，确保 `random_20` 策略可复现

2. **`emergency_test.py`** - 批量测试脚本
   - 类似 `batch_generate_trajectories.py` 的并发执行
   - 使用 subprocess 隔离每个 query 的执行环境
   - 通过 `asyncio.Semaphore` 控制并发数
   - 支持 `all` 策略，一次运行所有三种策略
   - 生成汇总 JSON 文件和策略级别的统计信息

3. **`_emergency_single_run.py`** - 单个 query 执行脚本（内部使用）
   - 被 `emergency_test.py` 通过 subprocess 调用
   - 完全隔离的执行环境（独立的 stdout/stderr）
   - 在脚本顶部抑制所有冗余日志输出
   - 运行单个 query 并保存 trajectory
   - 返回退出码（0 = 成功，1 = 失败）

### 执行流程

```
emergency_test.py (主进程)
    │
    ├─> 创建 asyncio.Semaphore(max_concurrent)
    ├─> 加载所有 queries
    ├─> 遍历所有策略 (no_interception, first_non_search, random_20)
    │
    └─> 对于每个 query:
        │
        ├─> asyncio.create_subprocess_exec(
        │       _emergency_single_run.py
        │       --query-file ...
        │       --query-index ...
        │       --strategy ...
        │       --random-seed {seed + index}  # 确保可复现
        │   )
        │
        └─> _emergency_single_run.py (子进程)
            │
            ├─> 抑制日志输出
            ├─> 初始化 MCPManager
            ├─> 初始化 DynamicReActAgent
            ├─> 注入 EmergencyInterceptor
            ├─> 运行 agent.execute(query)
            ├─> 保存 trajectory 到 Emergency_test/{model}/pass@{N}/{strategy}/
            └─> 退出 (0 或 1)
```

### 为什么使用 Subprocess？

使用 subprocess 而非直接在同一进程中运行的原因：

1. **日志隔离**: 完全隔离每个 query 的 stdout/stderr，避免冗余的 Meta-MCP Server 启动日志和 Agent 调试信息混入终端输出
2. **进程隔离**: 每个 query 在独立的 Python 进程中运行，避免全局状态（如 logging 配置）互相干扰
3. **资源清理**: 每个 query 完成后，子进程退出自动清理所有资源，避免内存泄漏
4. **一致性**: 与 `batch_generate_trajectories.py` 保持一致的架构，复用成熟的模式

## 下一步

1. **运行测试**: 先运行一小批 queries 测试（比如 10 个）
2. **检查 trajectories**: 查看生成的 JSON 文件，确认拦截是否生效
3. **分析结果**: 根据实际输出设计 analysis script

## 示例：小规模测试

```bash
# 使用快速测试脚本（3 个 queries）
./runtime/test_emergency_quick.sh

# 或者手动运行小规模测试
python runtime/emergency_test.py \
    --query-file task_creation_engine/generated_queries.json \
    --strategy all \
    --max-iterations 10 \
    --model anthropic/claude-3.5-sonnet \
    --max-concurrent 2
```
