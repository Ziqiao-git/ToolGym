#!/usr/bin/env python3
"""
Goal Enforcer Analysis for Multiturn Trajectories

This script evaluates whether an agent persists in solving the same problem after
a tool call fails. It measures "goal enforcement" by analyzing if consecutive
tool calls continue addressing the same objective after failures.

The goal enforcer metric is calculated using three scenarios:

1. **Same tool retried**: Similarity = 1.0 (perfect goal enforcement)
   - Example: clickup::get_spaces fails → retries clickup::get_spaces

2. **Different tool used**: Calculate semantic similarity between tools
   - Uses BGE-M3 embeddings for accurate semantic comparison (optional)
   - Falls back to rule-based similarity (same server=0.5, same tool name=0.4)
   - Example: trello::get_board → trello::get_boards = 0.5 (same server)

3. **search_tools called**: Parse search results and find best match
   - Extracts top 3 tools found by search_tools
   - Calculates similarity between failed tool and each found tool
   - Returns maximum similarity
   - Example: github::search fails → search_tools finds "gitlab::search" = 0.4

Similarity Calculation Methods:

A. **BGE-M3 Embeddings** (default, requires sentence-transformers):
   - State-of-the-art multilingual semantic similarity
   - Cosine similarity on 1024-dimensional embeddings
   - Accurately captures semantic relationships between tools
   - Example: "search_repositories" vs "find_repos" → high similarity

B. **Rule-based** (fallback or --no-embeddings):
   - Same tool: 1.0
   - Same server: 0.5
   - Same tool name: 0.4
   - Similar tool words (Jaccard>0.5): 0.3
   - Otherwise: 0.0

Usage:
    # Basic usage with BGE-M3 embeddings (most accurate)
    python analyze_scripts/evaluate_goal_enforcer.py

    # Faster mode without embeddings
    python analyze_scripts/evaluate_goal_enforcer.py --no-embeddings

    # Group by model and show summary table (RECOMMENDED)
    python analyze_scripts/evaluate_goal_enforcer.py --by-model --no-embeddings

    # Filter by specific model
    python analyze_scripts/evaluate_goal_enforcer.py --model gemini-3-pro-preview

    # Use custom trajectories directory
    python analyze_scripts/evaluate_goal_enforcer.py --traj-dir /path/to/trajectories/goaloriented

    # Show detailed analysis with examples
    python analyze_scripts/evaluate_goal_enforcer.py --detailed

    # Adjust similarity threshold
    python analyze_scripts/evaluate_goal_enforcer.py --threshold 0.4

    # Combine options: by-model with detailed analysis
    python analyze_scripts/evaluate_goal_enforcer.py --by-model --detailed --no-embeddings

Output:
    - Total failed tool calls and goal enforcement rate
    - Search_tools usage patterns and effectiveness
    - Similarity score distribution
    - Error pattern breakdown
    - Detailed examples with search results (if --detailed)
    - Per-model summary table (if --by-model):
      * Model name
      * Number of failures
      * Number enforced
      * Enforcement rate %
      * Average similarity score
"""
from __future__ import annotations

import sys
import argparse
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# Try to import BGE embeddings (optional, will fall back to simple similarity if not available)
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("Warning: sentence-transformers not available, using simple similarity calculation")

# Default trajectories directory
DEFAULT_TRAJECTORIES_DIR = Path(__file__).parent.parent / "trajectories" / "goaloriented"


class SemanticSimilarityCalculator:
    """
    Calculate semantic similarity using BGE-M3 embeddings.

    Uses cosine similarity on normalized embeddings for accurate semantic comparison.
    """

    def __init__(self, use_embeddings: bool = True):
        """
        Initialize the semantic similarity calculator.

        Args:
            use_embeddings: Whether to use BGE embeddings (requires sentence-transformers)
        """
        self.use_embeddings = use_embeddings and HAS_EMBEDDINGS
        self.model = None
        self.cache = {}  # Cache embeddings to avoid recomputation

        if self.use_embeddings:
            try:
                print("Loading BGE-M3 model for semantic similarity...")
                self.model = SentenceTransformer("BAAI/bge-m3")
                print("✓ BGE-M3 model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load BGE-M3 model: {e}")
                print("Falling back to simple similarity calculation")
                self.use_embeddings = False

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text, with caching.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector
        """
        if text in self.cache:
            return self.cache[text]

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            convert_to_numpy=True,
        )

        self.cache[text] = embedding
        return embedding

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1 (cosine similarity)
        """
        if not self.use_embeddings or self.model is None:
            # Fallback to simple similarity
            return self._simple_similarity(text1, text2)

        # Get embeddings
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Cosine similarity (dot product of normalized vectors)
        similarity = float(np.dot(emb1, emb2))

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        Simple word overlap similarity (fallback).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity score
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0


# Error patterns to detect in tool call results
ERROR_PATTERNS = [
    'error executing tool',
    'error occurred during executing tool',
    'failed to',
    'request failed',
    'status code 4',  # 400, 401, 402, 403, 404, 429, etc.
    'status code 5',  # 500, 502, 503, etc.
    'timed out',
    'timeout',
    'could not be loaded',
    'server_not_in_configs',
    'validation error',
    'unauthorized',
    'forbidden',
    '"error"',
    '"status":403',
    '"status":429',
    '"status":500',
]


def check_result_has_error(result_content: Any) -> Tuple[bool, str]:
    """
    Check if a result content indicates an error.

    Args:
        result_content: The content from reasoning_trace result

    Returns:
        Tuple of (has_error, matched_pattern)
    """
    if result_content is None:
        return False, ""

    content_str = str(result_content).lower()

    for pattern in ERROR_PATTERNS:
        if pattern in content_str:
            return True, pattern

    return False, ""


def extract_tool_intent(thought: str, tool_call: Dict[str, Any]) -> str:
    """
    Extract the intent/goal from a thought and tool call combination.

    Args:
        thought: The thought/reasoning before the tool call
        tool_call: The tool call dictionary

    Returns:
        A string describing the intent
    """
    server = tool_call.get("server", "unknown")
    tool = tool_call.get("tool", "unknown")

    # For search_tools, extract what the agent is searching for from the thought
    # This helps distinguish between different search intents
    search_keywords = ""
    if tool == "search_tools":
        # Try to extract search intent from thought
        thought_lower = thought.lower()
        # Look for key phrases that indicate what's being searched for
        search_indicators = ["search for", "looking for", "need", "require", "find"]
        for indicator in search_indicators:
            if indicator in thought_lower:
                # Extract context around the indicator
                idx = thought_lower.find(indicator)
                search_keywords = thought[idx:min(idx+100, len(thought))]
                break

    # Create a description of what the agent is trying to do
    if search_keywords:
        intent = f"Thought: {thought[:200]}... | Tool: {server}::{tool} | Search: {search_keywords}"
    else:
        intent = f"Thought: {thought[:200]}... | Tool: {server}::{tool}"
    return intent


def parse_search_tools_results(result_content: Any) -> List[str]:
    """
    Parse search_tools results to extract the tool names found.

    Args:
        result_content: The result content from search_tools

    Returns:
        List of tool names in format "server::tool" (up to 3)
    """
    tools_found = []

    if result_content is None:
        return tools_found

    result_str = str(result_content)

    # Parse format like: "1. **@server/name** / `tool_name`"
    # Pattern to match server and tool names
    # Example: "1. **@alexcz-a11y/jina-httpstreamable** / `search_web`"
    pattern = r'\*\*(@[^*]+)\*\*\s*/\s*`([^`]+)`'
    matches = re.findall(pattern, result_str)

    for server, tool in matches[:3]:  # Only take top 3
        # Clean server name (remove @)
        server_clean = server.strip()
        if server_clean.startswith('@'):
            server_clean = server_clean[1:]
        tool_clean = tool.strip()
        tools_found.append(f"{server_clean}::{tool_clean}")

    return tools_found


def create_tool_description(tool_full_name: str) -> str:
    """
    Create a text description for a tool.

    Args:
        tool_full_name: Tool name in format "server::tool"

    Returns:
        Descriptive text for the tool
    """
    if "::" not in tool_full_name:
        return tool_full_name

    server, tool = tool_full_name.split("::", 1)

    # Clean server name (remove @ if present)
    server_clean = server.lstrip("@")

    # Create readable description
    # Convert underscores to spaces for better semantic matching
    tool_readable = tool.replace("_", " ")
    server_readable = server_clean.replace("-", " ").replace("/", " ")

    description = f"{server_readable} server tool: {tool_readable}"
    return description


def calculate_tool_similarity(
    tool1: str,
    tool2: str,
    similarity_calc: SemanticSimilarityCalculator = None
) -> float:
    """
    Calculate similarity between two tools.

    Args:
        tool1: First tool in format "server::tool"
        tool2: Second tool in format "server::tool"
        similarity_calc: Optional semantic similarity calculator (uses BGE embeddings)

    Returns:
        Similarity score between 0 and 1
    """
    if tool1 == tool2:
        return 1.0

    # If we have a semantic similarity calculator, use it
    if similarity_calc is not None and similarity_calc.use_embeddings:
        desc1 = create_tool_description(tool1)
        desc2 = create_tool_description(tool2)
        return similarity_calc.calculate_similarity(desc1, desc2)

    # Fallback to rule-based similarity
    # Split into server and tool parts
    parts1 = tool1.split("::")
    parts2 = tool2.split("::")

    if len(parts1) != 2 or len(parts2) != 2:
        return 0.0

    server1, tool_name1 = parts1
    server2, tool_name2 = parts2

    # Same server, different tool
    if server1 == server2:
        return 0.5

    # Different servers, same tool name (e.g., both have "search")
    if tool_name1 == tool_name2:
        return 0.4

    # Check if tool names are similar (e.g., "search_web" vs "web_search")
    words1 = set(tool_name1.lower().replace('_', ' ').split())
    words2 = set(tool_name2.lower().replace('_', ' ').split())

    if words1 and words2:
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union)

        if jaccard > 0.5:
            return 0.3

    return 0.0


def calculate_semantic_similarity_simple(
    intent1: str,
    intent2: str,
    current_tool_full: str,
    next_tool_full: str,
    search_results: List[str] = None,
    similarity_calc: SemanticSimilarityCalculator = None
) -> float:
    """
    Calculate similarity based on new logic:

    1. If next tool is the SAME as failed tool -> similarity = 1.0 (full persistence)
    2. If next tool is DIFFERENT (not search_tools) -> use tool similarity
    3. If next action is search_tools:
       - Parse the search results (top 3 tools found)
       - Calculate similarity between failed tool and each found tool
       - Return the maximum similarity found

    Args:
        intent1: First intent string (not used in new logic, kept for compatibility)
        intent2: Second intent string (not used in new logic, kept for compatibility)
        current_tool_full: Current failed tool in "server::tool" format
        next_tool_full: Next tool in "server::tool" format
        search_results: List of tools found by search_tools (if applicable)
        similarity_calc: Optional semantic similarity calculator (uses BGE embeddings)

    Returns:
        Similarity score between 0 and 1
    """
    # Case 1: Same tool retried -> perfect similarity
    if current_tool_full == next_tool_full:
        return 1.0

    # Check if next action is search_tools
    is_next_search = "search_tools" in next_tool_full

    if is_next_search:
        # Case 3: Next action is search_tools
        # Compare failed tool with tools found in search results
        if search_results:
            max_similarity = 0.0
            for found_tool in search_results:
                similarity = calculate_tool_similarity(
                    current_tool_full,
                    found_tool,
                    similarity_calc=similarity_calc
                )
                max_similarity = max(max_similarity, similarity)
            return max_similarity
        else:
            # No search results available, assume low similarity
            return 0.1

    else:
        # Case 2: Different tool (not search)
        # Direct tool-to-tool similarity
        return calculate_tool_similarity(
            current_tool_full,
            next_tool_full,
            similarity_calc=similarity_calc
        )


def analyze_goal_enforcement(
    trajectories: List[Dict[str, Any]],
    similarity_threshold: float = 0.3,
    detailed: bool = False,
    use_embeddings: bool = True
) -> Dict[str, Any]:
    """
    Analyze goal enforcement after tool call failures.

    Args:
        trajectories: List of trajectory dictionaries
        similarity_threshold: Threshold for considering intents as "same goal"
        detailed: Whether to collect detailed examples
        use_embeddings: Whether to use BGE-M3 embeddings for semantic similarity

    Returns:
        Dictionary containing goal enforcement statistics
    """
    # Initialize semantic similarity calculator
    similarity_calc = SemanticSimilarityCalculator(use_embeddings=use_embeddings)

    stats = {
        "total_failures": 0,
        "failures_with_next_action": 0,  # Failures that have a next action to analyze
        "goal_enforced_count": 0,  # Times agent persisted with same goal
        "goal_switched_count": 0,  # Times agent switched to different goal
        "enforcement_rate": 0.0,
        "error_pattern_distribution": defaultdict(int),
        "examples_enforced": [],
        "examples_switched": [],
        "similarity_scores": [],
        # Search tools specific stats
        "search_tools_failures": 0,  # Number of search_tools failures
        "search_to_search_count": 0,  # search_tools -> search_tools
        "search_to_tool_count": 0,  # search_tools -> actual tool (found what they needed)
        "tool_to_search_count": 0,  # tool failure -> search_tools (looking for alternatives)
        "search_enforced_count": 0,  # Search-related actions that enforced goal
    }

    for traj in trajectories:
        turns = traj.get("turns", [])

        # We need to track across turns since tool calls span multiple turns
        all_actions = []  # List of (thought, tool_call, result, is_success)

        for turn in turns:
            tool_calls = turn.get("tool_calls", [])
            reasoning_trace = turn.get("reasoning_trace", [])

            # Extract thoughts and results from reasoning trace
            thoughts = [t.get("content", "") for t in reasoning_trace if t.get("type") == "thought"]
            results = [t.get("content") for t in reasoning_trace if t.get("type") == "result"]

            # Match thoughts to tool calls (should be 1:1 correspondence)
            for idx, call in enumerate(tool_calls):
                thought = thoughts[idx] if idx < len(thoughts) else ""
                result = results[idx] if idx < len(results) else None

                # Check if this call failed
                status = call.get("status", "unknown")
                has_error, error_pattern = check_result_has_error(result)
                is_success = status == "success" and not has_error

                all_actions.append({
                    "thought": thought,
                    "tool_call": call,
                    "result": result,
                    "is_success": is_success,
                    "error_pattern": error_pattern,
                })

        # Now analyze consecutive actions
        for i in range(len(all_actions) - 1):
            current = all_actions[i]
            next_action = all_actions[i + 1]

            if not current["is_success"]:
                stats["total_failures"] += 1
                stats["failures_with_next_action"] += 1

                if current["error_pattern"]:
                    stats["error_pattern_distribution"][current["error_pattern"]] += 1

                # Track search_tools patterns
                current_tool_name = current["tool_call"].get("tool", "")
                next_tool_name = next_action["tool_call"].get("tool", "")

                is_current_search = current_tool_name == "search_tools"
                is_next_search = next_tool_name == "search_tools"

                if is_current_search:
                    stats["search_tools_failures"] += 1
                    if is_next_search:
                        stats["search_to_search_count"] += 1
                    else:
                        stats["search_to_tool_count"] += 1
                elif is_next_search:
                    stats["tool_to_search_count"] += 1

                # Extract tool names
                current_tool_full = f"{current['tool_call'].get('server', '')}::{current_tool_name}"
                next_tool_full = f"{next_action['tool_call'].get('server', '')}::{next_tool_name}"

                # Parse search results if next action is search_tools
                search_results = None
                if is_next_search:
                    next_result = next_action.get("result")
                    search_results = parse_search_tools_results(next_result)

                # Extract intents (for backwards compatibility)
                current_intent = extract_tool_intent(current["thought"], current["tool_call"])
                next_intent = extract_tool_intent(next_action["thought"], next_action["tool_call"])

                # Calculate similarity using new logic with embeddings
                similarity = calculate_semantic_similarity_simple(
                    current_intent,
                    next_intent,
                    current_tool_full,
                    next_tool_full,
                    search_results,
                    similarity_calc=similarity_calc
                )
                stats["similarity_scores"].append(similarity)

                # Determine if goal was enforced
                if similarity >= similarity_threshold:
                    stats["goal_enforced_count"] += 1

                    # Track search-related enforcement
                    if is_current_search or is_next_search:
                        stats["search_enforced_count"] += 1

                    if detailed and len(stats["examples_enforced"]) < 5:
                        example = {
                            "current_thought": current["thought"][:200],
                            "current_tool": current_tool_full,
                            "next_thought": next_action["thought"][:200],
                            "next_tool": next_tool_full,
                            "similarity": similarity,
                        }
                        if search_results:
                            example["search_results"] = search_results
                        stats["examples_enforced"].append(example)
                else:
                    stats["goal_switched_count"] += 1

                    if detailed and len(stats["examples_switched"]) < 5:
                        example = {
                            "current_thought": current["thought"][:200],
                            "current_tool": current_tool_full,
                            "next_thought": next_action["thought"][:200],
                            "next_tool": next_tool_full,
                            "similarity": similarity,
                        }
                        if search_results:
                            example["search_results"] = search_results
                        stats["examples_switched"].append(example)

    # Calculate enforcement rate
    if stats["failures_with_next_action"] > 0:
        stats["enforcement_rate"] = stats["goal_enforced_count"] / stats["failures_with_next_action"]

    return stats


def group_trajectories_by_model(trajectories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group trajectories by model name.

    Args:
        trajectories: List of trajectory dictionaries

    Returns:
        Dictionary mapping model name to list of trajectories
    """
    grouped = defaultdict(list)

    for traj in trajectories:
        agent_model = traj.get("metadata", {}).get("agent_model", "unknown")
        grouped[agent_model].append(traj)

    return dict(grouped)


def load_multiturn_trajectories(traj_dir: Path, model_filter: str = None) -> List[Dict[str, Any]]:
    """
    Load all multiturn trajectory files from the goaloriented directory.

    Args:
        traj_dir: Path to trajectories/goaloriented directory
        model_filter: Optional model name filter (case-insensitive)

    Returns:
        List of trajectory dictionaries
    """
    trajectories = []

    if not traj_dir.exists():
        print(f"Warning: Directory does not exist: {traj_dir}")
        return trajectories

    # Find only trajectory JSON files (exclude batch summaries, etc.)
    for json_file in traj_dir.rglob("trajectory_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Apply model filter if specified
            if model_filter:
                agent_model = data.get("metadata", {}).get("agent_model", "")
                if model_filter.lower() not in agent_model.lower():
                    continue

            trajectories.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return trajectories


def print_goal_enforcer_analysis(stats: Dict[str, Any], detailed: bool = False):
    """Print comprehensive goal enforcer analysis."""

    print("\n" + "=" * 80)
    print("GOAL ENFORCER ANALYSIS")
    print("=" * 80)

    print(f"\nTotal tool call failures: {stats['total_failures']}")
    print(f"Failures with next action: {stats['failures_with_next_action']}")

    if stats['failures_with_next_action'] == 0:
        print("\nNo failures with follow-up actions to analyze!")
        return

    print("\n" + "-" * 80)
    print("GOAL ENFORCEMENT METRICS")
    print("-" * 80)

    print(f"Times agent persisted with same goal: {stats['goal_enforced_count']}")
    print(f"Times agent switched to different goal: {stats['goal_switched_count']}")
    print(f"Goal Enforcement Rate: {stats['enforcement_rate'] * 100:.1f}%")

    # Search tools analysis
    print("\n" + "-" * 80)
    print("SEARCH_TOOLS ANALYSIS")
    print("-" * 80)
    print(f"Total search_tools failures: {stats['search_tools_failures']}")
    if stats['search_tools_failures'] > 0:
        print(f"  search_tools → search_tools: {stats['search_to_search_count']} ({stats['search_to_search_count']/stats['search_tools_failures']*100:.1f}%)")
        print(f"  search_tools → actual tool: {stats['search_to_tool_count']} ({stats['search_to_tool_count']/stats['search_tools_failures']*100:.1f}%)")
    print(f"Tool failure → search_tools: {stats['tool_to_search_count']}")
    print(f"Search-related goal enforcement: {stats['search_enforced_count']}")

    # Similarity score distribution
    if stats['similarity_scores']:
        print("\n" + "-" * 80)
        print("SIMILARITY SCORE DISTRIBUTION")
        print("-" * 80)
        print(f"Average similarity: {statistics.mean(stats['similarity_scores']):.3f}")
        print(f"Median similarity: {statistics.median(stats['similarity_scores']):.3f}")
        print(f"Min similarity: {min(stats['similarity_scores']):.3f}")
        print(f"Max similarity: {max(stats['similarity_scores']):.3f}")

    # Error pattern breakdown
    if stats['error_pattern_distribution']:
        print("\n" + "-" * 80)
        print("ERROR PATTERN DISTRIBUTION")
        print("-" * 80)
        sorted_patterns = sorted(
            stats['error_pattern_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for pattern, count in sorted_patterns[:10]:
            print(f"  '{pattern}': {count}")

    # Detailed examples
    if detailed:
        if stats['examples_enforced']:
            print("\n" + "-" * 80)
            print("EXAMPLES: GOAL ENFORCED (Agent Persisted)")
            print("-" * 80)
            for i, ex in enumerate(stats['examples_enforced'], 1):
                print(f"\nExample {i}:")
                print(f"  Failed Action:")
                print(f"    Thought: {ex['current_thought']}")
                print(f"    Tool: {ex['current_tool']}")
                print(f"  Next Action:")
                print(f"    Thought: {ex['next_thought']}")
                print(f"    Tool: {ex['next_tool']}")
                if 'search_results' in ex and ex['search_results']:
                    print(f"    Search found: {ex['search_results']}")
                print(f"  Similarity: {ex['similarity']:.3f}")

        if stats['examples_switched']:
            print("\n" + "-" * 80)
            print("EXAMPLES: GOAL SWITCHED (Agent Changed Direction)")
            print("-" * 80)
            for i, ex in enumerate(stats['examples_switched'], 1):
                print(f"\nExample {i}:")
                print(f"  Failed Action:")
                print(f"    Thought: {ex['current_thought']}")
                print(f"    Tool: {ex['current_tool']}")
                print(f"  Next Action:")
                print(f"    Thought: {ex['next_thought']}")
                print(f"    Tool: {ex['next_tool']}")
                if 'search_results' in ex and ex['search_results']:
                    print(f"    Search found: {ex['search_results']}")
                print(f"  Similarity: {ex['similarity']:.3f}")

    print("\n" + "=" * 80)


def print_model_summary(model_stats: Dict[str, Dict[str, Any]]):
    """
    Print summary statistics grouped by model.

    Args:
        model_stats: Dictionary mapping model name to statistics
    """
    print("\n" + "=" * 80)
    print("SUMMARY BY MODEL")
    print("=" * 80)

    # Create summary table
    print("\n{:<35} {:>10} {:>12} {:>12} {:>15}".format(
        "Model", "Failures", "Enforced", "Rate", "Avg Similarity"
    ))
    print("-" * 85)

    # Sort by model name
    for model_name in sorted(model_stats.keys()):
        stats = model_stats[model_name]

        failures = stats.get("failures_with_next_action", 0)
        enforced = stats.get("goal_enforced_count", 0)
        rate = stats.get("enforcement_rate", 0.0) * 100

        # Calculate average similarity
        similarity_scores = stats.get("similarity_scores", [])
        avg_similarity = statistics.mean(similarity_scores) if similarity_scores else 0.0

        print("{:<35} {:>10} {:>12} {:>11.1f}% {:>15.3f}".format(
            model_name[:34],  # Truncate long model names
            failures,
            enforced,
            rate,
            avg_similarity
        ))

    print("-" * 85)

    # Overall statistics
    total_failures = sum(s.get("failures_with_next_action", 0) for s in model_stats.values())
    total_enforced = sum(s.get("goal_enforced_count", 0) for s in model_stats.values())
    all_similarities = []
    for s in model_stats.values():
        all_similarities.extend(s.get("similarity_scores", []))

    overall_rate = (total_enforced / total_failures * 100) if total_failures > 0 else 0.0
    overall_avg_sim = statistics.mean(all_similarities) if all_similarities else 0.0

    print("{:<35} {:>10} {:>12} {:>11.1f}% {:>15.3f}".format(
        "OVERALL",
        total_failures,
        total_enforced,
        overall_rate,
        overall_avg_sim
    ))

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze goal enforcement after tool call failures in multiturn trajectories"
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=DEFAULT_TRAJECTORIES_DIR,
        help=f"Trajectories directory (default: {DEFAULT_TRAJECTORIES_DIR})",
    )
    parser.add_argument(
        "--model",
        help="Filter to specific model (case-insensitive)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Similarity threshold for considering intents as 'same goal' (default: 0.3)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed examples of enforced vs. switched goals",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable BGE-M3 embeddings and use simple similarity (faster but less accurate)",
    )
    parser.add_argument(
        "--by-model",
        action="store_true",
        help="Group results by model and show summary statistics for each",
    )

    args = parser.parse_args()

    if not args.traj_dir.exists():
        print(f"Error: Trajectories directory not found: {args.traj_dir}")
        return 1

    print(f"Loading trajectory data from: {args.traj_dir}")
    trajectories = load_multiturn_trajectories(args.traj_dir, args.model)
    print(f"Found {len(trajectories)} trajectories")

    if args.model:
        print(f"Filtered to model: {args.model}")

    if not trajectories:
        print("\nNo trajectories found matching criteria!")
        return 1

    print(f"Analyzing with similarity threshold: {args.threshold}")
    if not args.no_embeddings and HAS_EMBEDDINGS:
        print("Using BGE-M3 embeddings for semantic similarity")
    else:
        print("Using simple rule-based similarity")

    # If --by-model is specified, analyze each model separately
    if args.by_model:
        grouped = group_trajectories_by_model(trajectories)
        print(f"\nFound {len(grouped)} different models")

        model_stats = {}
        for model_name in sorted(grouped.keys()):
            model_trajs = grouped[model_name]
            print(f"\nAnalyzing {model_name}: {len(model_trajs)} trajectories...")

            stats = analyze_goal_enforcement(
                model_trajs,
                similarity_threshold=args.threshold,
                detailed=False,  # Don't show detailed examples for each model
                use_embeddings=not args.no_embeddings
            )
            model_stats[model_name] = stats

        # Print summary table
        print_model_summary(model_stats)

        # Optionally print detailed analysis for one model or overall
        if args.detailed:
            print("\n" + "=" * 80)
            print("DETAILED ANALYSIS (All Models Combined)")
            print("=" * 80)
            overall_stats = analyze_goal_enforcement(
                trajectories,
                similarity_threshold=args.threshold,
                detailed=True,
                use_embeddings=not args.no_embeddings
            )
            print_goal_enforcer_analysis(overall_stats, detailed=True)
    else:
        # Original behavior: analyze all trajectories together
        stats = analyze_goal_enforcement(
            trajectories,
            similarity_threshold=args.threshold,
            detailed=args.detailed,
            use_embeddings=not args.no_embeddings
        )
        print_goal_enforcer_analysis(stats, detailed=args.detailed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
