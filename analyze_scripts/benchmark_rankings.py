# analyze_scripts/benchmark_rankings.py
from __future__ import annotations

from typing import Dict, List
import re

###############################################################################
# Canonicalization
#
# Goal:
#   Map BOTH:
#     (1) your internal model ids (from eval files), and
#     (2) external benchmark display names
#   into a shared canonical namespace so overlap is computed correctly.
#
# Important constraints (per your request):
#   - kimi-k2-thinking is NOT the same as kimi-k2 (do NOT merge).
#   - deepseek-v3.2 is NOT the same as deepseek-v3 (do NOT merge).
###############################################################################

def _normalize_name(raw: str) -> str:
    """
    Normalize a raw model name into a comparison-friendly key.

    Operations:
      - strip + lowercase
      - unify separators to '-'
      - convert "(Open)" into "open" (then parentheses removed) -> e.g. "xxx (Open)" -> "xxx-open"
      - remove parentheses
      - convert spaces to hyphens
      - collapse repeated hyphens
    """
    s = (raw or "").strip().lower()
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("_", "-")

    # Convert "(open)" marker into token "open" so we can keep it as "-open"
    s = re.sub(r"\(open\)", "open", s)
    s = s.replace("(", "").replace(")", "")

    s = s.replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s


# 1) Internal / eval model names -> canonical names
# Keys MUST be normalized form (because canonicalize_model_name() normalizes first).
_INTERNAL_MODEL_ALIAS: Dict[str, str] = {
    # Internal IDs (examples from your screenshot / pipeline)
    "grok4": "grok-4",

    "gemini-2.5pro": "gemini-2.5-pro",
    "gemini-3pro": "gemini-3-pro",

    # Keep v3.2 distinct from v3
    "deepseek-v3.2": "deepseek-v3.2",

    # Keep kimi-k2-thinking distinct from kimi-k2
    "kimi-k2-thinking": "kimi-k2-thinking",

    "claude-3.5": "claude-3.5",

    # If you want to treat gpt-4omini as o4-mini for comparisons, keep this mapping.
    "gpt-4omini": "o4-mini",
    # Optional: sometimes internal id might be missing hyphens
    "gpt4omini": "o4-mini",

    "glm-4.6v": "glm-4.6",

    "gpt-oss-120b": "gpt-oss-120b",
}


# 2) External benchmark names -> canonical names
# Keys MUST be normalized form (lowercased, parentheses removed).
_BENCHMARK_ALIAS: Dict[str, str] = {
    # Grok
    "grok-4": "grok-4",

    # OpenAI
    "gpt-5": "gpt-5",
    "o3": "o3",
    "o4-mini": "o4-mini",

    # If you consider GPT-4o-mini equivalent to o4-mini, map it.
    "gpt-4o-mini": "o4-mini",

    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4o": "gpt-4o",

    "gpt-oss-120b": "gpt-oss-120b",
    "gpt-oss-120b-open": "gpt-oss-120b",
    "gpt-oss-20b": "gpt-oss-20b",

    # Anthropic
    "claude-4.0-sonnet": "claude-sonnet-4",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    "claude-opus-4-20250514": "claude-opus-4",
    "claude-3.7-sonnet": "claude-3.7-sonnet",
    "claude-3.5": "claude-3.5",

    # Gemini
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash",  # optional merge
    "gemini-3-pro": "gemini-3-pro",

    # GLM
    "glm-4.5": "glm-4.5",
    "glm-4.5-open": "glm-4.5",
    "glm-4.6": "glm-4.6",

    # Qwen
    "qwen3-235b": "qwen3-235b-a22b",
    "qwen3-235b-open": "qwen3-235b-a22b",
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "qwen3-235b-a22b-2507": "qwen3-235b-a22b",

    "qwen3-32b": "qwen3-32b",
    "qwen3-coder-open": "qwen3-coder",
    "qwen3-coder": "qwen3-coder",
    "qwen3-30b-a3b-instruct-2507": "qwen3-30b-a3b-instruct-2507",

    # DeepSeek (keep v3 distinct; do not merge v3.2 here)
    "deepseek-v3": "deepseek-v3",
    "deepseek-v3-0324": "deepseek-v3",
    "deepseek-v3-open": "deepseek-v3",
    "deepseek-r1-0528": "deepseek-r1",
    "deepseek-r1": "deepseek-r1",

    # Kimi (NOTE: kimi-k2-thinking is intentionally not mapped here)
    "kimi-k2": "kimi-k2",
    "kimi-k2-open": "kimi-k2",

    # Other models that may appear in benchmark lists
    "qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
    "gemma-3-27b-it": "gemma-3-27b-it",
    "llama-3-70b-instruct": "llama-3-70b-instruct",
    "llama-3-1-70b-instruct": "llama-3-1-70b-instruct",
    "llama-3-2-90b-vision-instruct": "llama-3-2-90b-vision-instruct",
    "llama-3-1-8b-instruct": "llama-3-1-8b-instruct",
    "mistral-small-2503": "mistral-small-2503",
    "nova-micro-v1": "nova-micro-v1",
}


def canonicalize_model_name(name: str) -> str:
    """
    Convert any model name (internal id or benchmark display name) to a canonical name.

    Order matters:
      1) normalize (lowercase, strip, remove parentheses, etc.)
      2) apply internal mapping (your eval ids)
      3) apply benchmark/display mapping
      4) fallback to normalized string
    """
    s = _normalize_name(name)

    if s in _INTERNAL_MODEL_ALIAS:
        return _INTERNAL_MODEL_ALIAS[s]

    if s in _BENCHMARK_ALIAS:
        return _BENCHMARK_ALIAS[s]

    return s


###############################################################################
# External benchmark rankings (strongest -> weakest)
###############################################################################

MCPUNIVERSE_RANKING_STRONG_TO_WEAK: List[str] = [
    "GPT-5",
    "Grok-4",
    "Claude-4.0-Sonnet",
    "o3",
    "o4-mini",
    "GLM-4.5 (Open)",
    "Claude-3.7-Sonnet",
    "Gemini-2.5-Pro",
    "Gemini-2.5-Flash",
    "Qwen3-Coder (Open)",
    "Kimi-K2 (Open)",
    "GPT-4.1",
    "Qwen3-235B (Open)",
    "GPT-4o",
    "DeepSeek-V3 (Open)",
    "GPT-OSS-120B (Open)",
]

MCPBENCH_RANKING_STRONG_TO_WEAK: List[str] = [
    "GPT-5",
    "o3",
    "GPT-OSS-120B",
    "Gemini-2.5-Pro",
    "Claude-Sonnet-4",
    "Qwen3-235B-A22B-2507",
    "GLM-4.5",
    "GPT-OSS-20B",
    "Kimi-K2",
    "Qwen3-30B-A3B-Instruct-2507",
    "Gemini-2.5-Flash-Lite",
    "GPT-4o",
    "Gemma-3-27B-IT",
    "LLaMA-3-70B-Instruct",
    "GPT-4o-mini",
    "Mistral-Small-2503",
    "LLaMA-3-1-70B-Instruct",
    "Nova-Micro-v1",
    "LLaMA-3-2-90B-Vision-Instruct",
    "LLaMA-3-1-8B-Instruct",
]

LIVEMCPBENCH_RANKING_STRONG_TO_WEAK: List[str] = [
    "Claude-Sonnet-4-20250514",
    "Claude-Opus-4-20250514",
    "DeepSeek-R1-0528",
    "Qwen3-235B-A22B",
    "GPT-4.1-Mini",
    "Qwen2.5-72B-Instruct",
    "DeepSeek-V3-0324",
    "Gemini-2.5-Pro",
    "GPT-4.1",
    "Qwen3-32B",
]

BENCHMARK_RANKINGS: Dict[str, List[str]] = {
    "mcpuniverse": MCPUNIVERSE_RANKING_STRONG_TO_WEAK,
    "mcpbench": MCPBENCH_RANKING_STRONG_TO_WEAK,
    "livemcpbench": LIVEMCPBENCH_RANKING_STRONG_TO_WEAK,
}

def get_benchmark_rank_map(benchmark: str) -> Dict[str, float]:
    """
    Convert a benchmark ranking list into a rank map.

    Returns:
        Dict[canonical_model_name -> rank], where rank=1 is best.
    """
    ranking = BENCHMARK_RANKINGS[benchmark]
    canonical = [canonicalize_model_name(m) for m in ranking]
    return {m: i + 1 for i, m in enumerate(canonical)}
