#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_failures.py

Extract actionable insights from LLM judge evaluation results.

Usage:
  python evaluation/stats/analyze_failures.py evaluation/results/judge_results.json
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List


def analyze_failures(results_path: Path) -> Dict[str, Any]:
    """Analyze evaluation results to extract actionable insights."""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Categorize issues
    issues = {
        "zero_tool_calls": [],  # Discovery gap
        "low_tool_choice": [],  # Wrong tools selected
        "low_grounding": [],    # Results not used properly
        "low_execution": [],    # Tool execution failures
        "server_errors": [],    # Server access issues
        "empty_results": [],    # Tools return empty despite success
    }

    # Score thresholds
    LOW_SCORE = 5.0  # Out of 10

    for result in results:
        task_id = result.get("task_id", "unknown")
        query = result.get("query", "")[:80]
        explanation = result.get("explanation", "").lower()

        # Check for zero tool calls (need to cross-reference with trajectories)
        if "prompt_idx" in task_id:
            issues["zero_tool_calls"].append({
                "task_id": task_id,
                "query": query,
                "issue": "No trajectory generated - semantic search failed to find relevant tools"
            })

        # Check dimension scores
        tool_choice = result.get("tool_choice", 10)
        grounding = result.get("grounding", 10)
        execution = result.get("tool_execution", 10)

        if tool_choice and tool_choice < LOW_SCORE:
            issues["low_tool_choice"].append({
                "task_id": task_id,
                "query": query,
                "score": tool_choice,
                "explanation": result.get("explanation", "")[:200]
            })

        if grounding and grounding < LOW_SCORE:
            issues["low_grounding"].append({
                "task_id": task_id,
                "query": query,
                "score": grounding,
                "explanation": result.get("explanation", "")[:200]
            })

        if execution and execution < LOW_SCORE:
            issues["low_execution"].append({
                "task_id": task_id,
                "query": query,
                "score": execution,
                "explanation": result.get("explanation", "")[:200]
            })

        # Check for specific issues in explanations
        if "server" in explanation and ("error" in explanation or "access" in explanation or "failed" in explanation):
            issues["server_errors"].append({
                "task_id": task_id,
                "query": query,
                "explanation": result.get("explanation", "")[:200]
            })

        if "empty" in explanation and "result" in explanation:
            issues["empty_results"].append({
                "task_id": task_id,
                "query": query,
                "explanation": result.get("explanation", "")[:200]
            })

    return issues


def print_actionable_insights(issues: Dict[str, Any]) -> None:
    """Print actionable insights in a clear format."""

    print("=" * 80)
    print("ACTIONABLE INSIGHTS FROM LLM JUDGE EVALUATION")
    print("=" * 80)

    # Priority 1: Discovery Gap (no tools found)
    if issues["zero_tool_calls"]:
        print(f"\n🔴 CRITICAL: Discovery Gap ({len(issues['zero_tool_calls'])} cases)")
        print("   Problem: Semantic search failed to find any relevant tools")
        print("   Impact: Agent gives up without attempting any actions")
        print("\n   ACTION ITEMS:")
        print("   1. Review tool descriptions in FAISS index - are they accurate?")
        print("   2. Check semantic search threshold - is it too strict?")
        print("   3. Improve tool metadata and descriptions")
        print("   4. Consider fallback strategy when no tools found")
        print(f"\n   Failed queries:")
        for item in issues["zero_tool_calls"][:5]:
            print(f"   - {item['task_id']}: {item['query']}...")
        if len(issues["zero_tool_calls"]) > 5:
            print(f"   ... and {len(issues['zero_tool_calls']) - 5} more")

    # Priority 2: Wrong tool selection
    if issues["low_tool_choice"]:
        print(f"\n🟡 HIGH PRIORITY: Poor Tool Selection ({len(issues['low_tool_choice'])} cases)")
        print("   Problem: Agent selects irrelevant or redundant tools")
        print("   Impact: Wastes time on wrong tools, misses correct ones")
        print("\n   ACTION ITEMS:")
        print("   1. Debug semantic search - why are wrong tools ranking high?")
        print("   2. Improve tool descriptions to better match user intent")
        print("   3. Add tool usage examples to improve matching")
        print("   4. Consider re-ranking or filtering strategies")
        print(f"\n   Examples:")
        for item in sorted(issues["low_tool_choice"], key=lambda x: x["score"])[:3]:
            print(f"   - {item['task_id']} (score: {item['score']}/10)")
            print(f"     Query: {item['query']}...")
            print(f"     Issue: {item['explanation'][:150]}...")
            print()

    # Priority 3: Empty results despite success
    if issues["empty_results"]:
        print(f"\n🟡 HIGH PRIORITY: Empty Tool Results ({len(issues['empty_results'])} cases)")
        print("   Problem: Tools execute successfully but return empty/useless data")
        print("   Impact: High technical success rate but low task completion")
        print("\n   ACTION ITEMS:")
        print("   1. Review tool test results - are these tools actually working?")
        print("   2. Filter out broken tools from the index")
        print("   3. Add result quality validation (not just status checking)")
        print("   4. Consider re-testing tools with realistic queries")
        print(f"\n   Examples:")
        for item in issues["empty_results"][:3]:
            print(f"   - {item['task_id']}")
            print(f"     Query: {item['query']}...")
            print(f"     Issue: {item['explanation'][:150]}...")
            print()

    # Priority 4: Server errors
    if issues["server_errors"]:
        print(f"\n🟠 MEDIUM PRIORITY: Server Access Issues ({len(issues['server_errors'])} cases)")
        print("   Problem: MCP servers failing to load or respond")
        print("   Impact: Tool execution failures even with correct selection")
        print("\n   ACTION ITEMS:")
        print("   1. Test server connectivity and health")
        print("   2. Add server health monitoring")
        print("   3. Implement retry logic for transient failures")
        print("   4. Remove consistently failing servers from index")
        print(f"\n   Examples:")
        for item in issues["server_errors"][:3]:
            print(f"   - {item['task_id']}")
            print(f"     Query: {item['query']}...")
            print(f"     Issue: {item['explanation'][:150]}...")
            print()

    # Priority 5: Poor grounding
    if issues["low_grounding"]:
        print(f"\n🔵 LOW PRIORITY: Poor Result Usage ({len(issues['low_grounding'])} cases)")
        print("   Problem: Agent doesn't use tool results effectively")
        print("   Impact: Good data retrieved but not incorporated in answer")
        print("\n   ACTION ITEMS:")
        print("   1. This is a ReAct agent issue, not a tool issue")
        print("   2. Consider improving agent prompts")
        print("   3. Add result synthesis step")

    # Priority 6: Execution issues
    if issues["low_execution"]:
        print(f"\n🔵 MEDIUM PRIORITY: Tool Execution Issues ({len(issues['low_execution'])} cases)")
        print("   Problem: Tools called incorrectly or outputs not used")
        print("   Impact: Correct tools available but not used properly")
        print("\n   ACTION ITEMS:")
        print("   1. Review tool parameter schemas - are they clear?")
        print("   2. Add tool usage examples")
        print("   3. Improve error messages from tools")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal issues found:")
    print(f"  Discovery Gap:        {len(issues['zero_tool_calls'])} cases (CRITICAL)")
    print(f"  Wrong Tool Selection: {len(issues['low_tool_choice'])} cases (HIGH)")
    print(f"  Empty Results:        {len(issues['empty_results'])} cases (HIGH)")
    print(f"  Server Errors:        {len(issues['server_errors'])} cases (MEDIUM)")
    print(f"  Poor Grounding:       {len(issues['low_grounding'])} cases (LOW)")
    print(f"  Execution Issues:     {len(issues['low_execution'])} cases (MEDIUM)")

    print("\n🎯 RECOMMENDED NEXT STEPS:")
    print("1. Fix discovery gap: Improve tool descriptions and semantic search")
    print("2. Filter broken tools: Re-test and remove tools that return empty results")
    print("3. Debug tool selection: Analyze why wrong tools rank high in search")
    print("4. Monitor server health: Add health checks for MCP servers")
    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract actionable insights from LLM judge evaluation"
    )
    parser.add_argument(
        "results_file",
        help="Path to judge_results.json file"
    )

    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: File not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    issues = analyze_failures(results_path)
    print_actionable_insights(issues)


if __name__ == "__main__":
    main()
