#!/usr/bin/env python3
"""
Add Budget Constraint to Queries

This script takes an existing query JSON file and adds budget constraints to each query.
The budget constraint tells the agent it has limited tool call budget.

Usage:
    python mcp_generate/add_budget_constraint.py \
        --input mcp_generate/queries10.json \
        --output mcp_generate/queries_budget_3.json \
        --budget 300 \
        --cost-per-call 100
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any


BUDGET_CONSTRAINT_TEMPLATE = """
---
RESOURCE CONSTRAINT:
- Total Budget: {total_budget} credits
- Each non-search tool call costs: {cost_per_call} credits
- Search tools (search_tools, find_tools, etc.) are FREE (exploration is encouraged)
- Maximum supported non-search tool calls: {max_calls} times

Please solve this task efficiently within your budget.
You currently have {total_budget} credits available.
"""


def add_budget_constraint_to_query(
    query_item: Dict[str, Any],
    total_budget: int,
    cost_per_call: int,
) -> Dict[str, Any]:
    """
    Add budget constraint to a single query item.

    Args:
        query_item: Original query item with 'query' field
        total_budget: Total budget in credits
        cost_per_call: Cost per non-search tool call

    Returns:
        Modified query item with budget constraint appended
    """
    max_calls = total_budget // cost_per_call

    # Get original query
    original_query = query_item.get("query", "")

    # Add budget constraint text
    budget_text = BUDGET_CONSTRAINT_TEMPLATE.format(
        total_budget=total_budget,
        cost_per_call=cost_per_call,
        max_calls=max_calls
    )

    # Create new query with budget constraint
    new_query = original_query + budget_text

    # Create modified item
    modified_item = query_item.copy()
    modified_item["query"] = new_query

    # Add metadata about budget
    if "metadata" not in modified_item:
        modified_item["metadata"] = {}

    modified_item["metadata"]["budget_config"] = {
        "total_budget": total_budget,
        "cost_per_call": cost_per_call,
        "max_allowed_calls": max_calls,
    }

    return modified_item


def main():
    parser = argparse.ArgumentParser(
        description="Add budget constraints to query file"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input query JSON file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output query JSON file with budget constraints"
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Total budget in credits (e.g., 300, 500, 700)"
    )
    parser.add_argument(
        "--cost-per-call",
        type=int,
        default=100,
        help="Cost per non-search tool call in credits (default: 100)"
    )

    args = parser.parse_args()

    # Load input queries
    input_path = Path(args.input)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "items" not in data:
        raise ValueError(f"Input file must have 'items' field")

    # Process each query
    modified_items = []
    for item in data["items"]:
        modified_item = add_budget_constraint_to_query(
            item,
            args.budget,
            args.cost_per_call
        )
        modified_items.append(modified_item)

    # Create output data
    output_data = data.copy()
    output_data["items"] = modified_items

    # Update metadata
    if "metadata" not in output_data:
        output_data["metadata"] = {}

    output_data["metadata"]["budget_constraint"] = {
        "total_budget": args.budget,
        "cost_per_call": args.cost_per_call,
        "max_allowed_calls": args.budget // args.cost_per_call,
    }
    output_data["metadata"]["source_file"] = str(input_path)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    max_calls = args.budget // args.cost_per_call
    print(f"\n{'='*70}")
    print(f"Budget Constraint Added Successfully")
    print(f"{'='*70}")
    print(f"Input file:       {input_path}")
    print(f"Output file:      {output_path}")
    print(f"Total queries:    {len(modified_items)}")
    print(f"Budget:           {args.budget} credits")
    print(f"Cost per call:    {args.cost_per_call} credits")
    print(f"Max calls:        {max_calls}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
