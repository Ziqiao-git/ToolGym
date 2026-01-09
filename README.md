# ToolGym

**An Open-world Tool-using Environment for Scalable Agent Testing**

[![Paper](https://img.shields.io/badge/Paper-ACL%202025-blue)](https://arxiv.org/abs/xxxx.xxxxx)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/ToolGym/ToolGym)
[![Website](https://img.shields.io/badge/Website-GitHub%20Pages-green)](https://ziqiao-git.github.io/ToolGym/)

## Overview

ToolGym is a large-scale, open-world benchmark for evaluating LLM agents' tool-using capabilities. Built on **5,571 real tools** across **204 applications**, ToolGym enables realistic testing with:

- **Long-horizon workflows**: Multi-step tasks requiring complex tool coordination
- **Wild constraints**: Natural language requirements that must be satisfied
- **Robustness testing**: State Controller for systematic perturbation testing

## Key Statistics

| Metric | Value |
|--------|-------|
| Total Tools | 5,571 |
| Applications | 204 |
| Task Instances | 3,091 |
| Avg. Tools per Task | 4.77 |
| Avg. Steps per Task | 7.46 |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ToolGym                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ Task Creation   │    │ Tool Retrieval  │    │    State     │ │
│  │    Engine       │    │     Index       │    │  Controller  │ │
│  │                 │    │                 │    │              │ │
│  │ • Workflow      │    │ • BGE-M3        │    │ • Tool-level │ │
│  │   Synthesis     │    │ • FAISS         │    │ • State-level│ │
│  │ • Constraint    │    │ • 5,571 tools   │    │ • Constraint │ │
│  │   Generation    │    │                 │    │   -level     │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    Planner-Actor Framework                       │
│  ┌─────────────────┐              ┌─────────────────────────┐   │
│  │     Planner     │ ──prompts──▶ │         Actor           │   │
│  │  (Decomposes    │              │  (Executes tools via    │   │
│  │   into subtasks)│ ◀─feedback── │   ReAct reasoning)      │   │
│  └─────────────────┘              └─────────────────────────┘   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                       LLM-as-Judge                               │
│            Multi-model evaluation with majority voting           │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Ziqiao-git/ToolGym.git
cd ToolGym

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### Running an Agent

```bash
# Basic usage with semantic tool discovery
python runtime/run_react_agent.py "Search for latest AI news"

# With trajectory logging
python runtime/run_react_agent.py "Find GitHub repos about ML" --save-trajectory

# Custom model
python runtime/run_react_agent.py "Your query" \
  --model anthropic/claude-3.5-sonnet \
  --max-iterations 10
```

## Core Components

### 1. Task Creation Engine

Synthesizes realistic, long-horizon tasks through:
- **Workflow synthesis**: Chains tool calls into coherent task sequences
- **Constraint generation**: Adds natural language requirements
- **Diversity sampling**: Ensures coverage across tool categories

Location: `task_creation_engine/`

### 2. Tool Retrieval Index

Semantic search over 5,571 tools using:
- **Embeddings**: BGE-M3 (multilingual, 1024 dimensions)
- **Index**: FAISS for efficient similarity search
- **Dynamic loading**: On-demand MCP server connections

Location: `tool_retrieval_index/`

### 3. State Controller

Systematic robustness testing with three control types:

| Control Type | Strategies |
|--------------|------------|
| **Tool-level** | Timeout, Rate limit, Unavailable, Schema change, Partial failure |
| **State-level** | Response delay, Data corruption, Truncation, Session timeout, Stale data |
| **Constraint-level** | Add constraint, Modify constraint, Tighten deadline, Resource limit |

Location: `toolgym/state_controller/`

### 4. Planner-Actor Framework

Two-stage agent architecture:
- **Planner**: Decomposes tasks into subtask sequences
- **Actor**: Executes subtasks using ReAct reasoning with tool calls

Location: `Orchestrator/mcpuniverse/agent/`

### 5. LLM-as-Judge Evaluation

Multi-dimensional evaluation with:
- **5 scoring dimensions**: Task fulfillment, Grounding, Tool choice, Tool execution, Requirement satisfaction
- **Multi-model voting**: Uses multiple LLM judges for robustness
- **Majority voting**: Final score from consensus

Location: `Orchestrator/mcpuniverse/evaluator/`

## Project Structure

```
ToolGym/
├── README.md                    # This file
├── docs/                        # GitHub Pages website
│   └── index.html              # Leaderboard & documentation
│
├── task_creation_engine/        # Task synthesis
│   └── query_generate.py       # Workflow generation
│
├── tool_retrieval_index/        # Semantic tool search
│   └── server.py               # MCP server with search
│
├── toolgym/                     # Core library
│   └── state_controller/       # Robustness testing
│
├── Orchestrator/                # Agent framework
│   └── mcpuniverse/
│       ├── agent/              # Planner-Actor implementation
│       └── evaluator/          # LLM-as-Judge
│
├── MCP_INFO_MGR/                # Tool data management
│   ├── mcp_data/               # Tool metadata
│   └── semantic_search/        # FAISS index
│
├── runtime/                     # Agent runtime
│   └── run_react_agent.py      # CLI interface
│
└── evaluation/                  # Evaluation scripts
```

## Dataset

The ToolGym dataset is available on HuggingFace:

🤗 **[ToolGym/ToolGym](https://huggingface.co/datasets/ToolGym/ToolGym)**

Contents:
- 3,091 task instances with ground-truth tool sequences
- Tool metadata for 5,571 tools across 204 applications
- Constraint annotations and perturbation configurations

## Citation

```bibtex
@inproceedings{toolgym2025,
  title={ToolGym: An Open-world Tool-using Environment for LLM Agent Evaluation},
  author={...},
  booktitle={Proceedings of ACL 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on the Model Context Protocol (MCP) ecosystem
- Tool data sourced from Smithery and other MCP registries
- Evaluation framework inspired by recent LLM-as-Judge research

---

**Website**: [https://ziqiao-git.github.io/ToolGym/](https://ziqiao-git.github.io/ToolGym/)
**Dataset**: [https://huggingface.co/datasets/ToolGym/ToolGym](https://huggingface.co/datasets/ToolGym/ToolGym)
**GitHub**: [https://github.com/Ziqiao-git/ToolGym](https://github.com/Ziqiao-git/ToolGym)
