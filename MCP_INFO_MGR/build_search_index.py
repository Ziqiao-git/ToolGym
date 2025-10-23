"""
Build FAISS search index from tool descriptions.

This script:
1. Loads tool descriptions from NDJSON file
2. Generates BGE-M3 embeddings for each tool
3. Builds FAISS index for fast similarity search
4. Saves index and metadata for use by the Meta-MCP server

Usage:
    python MCP_INFO_MGR/build_search_index.py \
        --input MCP_INFO_MGR/tool_descriptions.ndjson \
        --output MCP_INFO_MGR/semantic_search/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from MCP_INFO_MGR.semantic_search.embeddings import BGEEmbedder, create_tool_text
from MCP_INFO_MGR.semantic_search.faiss_backend import FAISSIndex


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build FAISS search index from MCP tool descriptions"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("MCP_INFO_MGR/tool_descriptions.ndjson"),
        help="NDJSON file with tool descriptions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("MCP_INFO_MGR/semantic_search/"),
        help="Output directory for index files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model inference (cpu, cuda, mps, or None for auto)",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip tools with errors instead of failing",
    )
    return parser.parse_args()


def load_tool_descriptions(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load tool descriptions from NDJSON file.

    Args:
        input_path: Path to tool_descriptions.ndjson

    Returns:
        List of tool description dictionaries
    """
    tools = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                server_data = json.loads(line)
                # Only process successful server responses
                if server_data.get("status") == "ok":
                    server_name = server_data["qualifiedName"]
                    for tool in server_data.get("tools", []):
                        tools.append({
                            "server": server_name,
                            "tool": tool["name"],
                            "description": tool.get("description", ""),
                            "inputSchema": tool.get("inputSchema", {}),
                        })
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue

    return tools


def main():
    """Main entry point."""
    args = parse_args()

    print(f"{'='*60}")
    print(f"Building FAISS Search Index")
    print(f"{'='*60}\n")

    # Load tool descriptions
    print(f"Loading tool descriptions from {args.input}...")
    tools = load_tool_descriptions(args.input)
    print(f"✓ Loaded {len(tools)} tools from {args.input}")

    if len(tools) == 0:
        print("Error: No tools found in input file")
        return 1

    # Initialize BGE-M3 embedder
    print(f"\nInitializing BGE-M3 embedder...")
    embedder = BGEEmbedder(device=args.device)
    model_info = embedder.get_model_info()
    print(f"✓ Model loaded: {model_info['model_name']}")
    print(f"  - Dimensions: {model_info['dimensions']}")
    print(f"  - Device: {model_info['device']}")
    print(f"  - Max sequence length: {model_info['max_seq_length']}")

    # Prepare texts for embedding
    print(f"\nPreparing tool texts for embedding...")
    tool_texts = []
    metadata = []

    for tool in tools:
        try:
            # Create combined text for embedding
            text = create_tool_text(
                tool_name=tool["tool"],
                tool_description=tool["description"],
                server_name=tool["server"],
                input_schema=tool.get("inputSchema"),
            )
            tool_texts.append(text)
            metadata.append(tool)
        except Exception as e:
            if args.skip_errors:
                print(f"Warning: Skipping tool {tool.get('server')}/{tool.get('tool')}: {e}")
                continue
            else:
                raise

    print(f"✓ Prepared {len(tool_texts)} tool texts")

    # Generate embeddings
    print(f"\nGenerating embeddings (batch_size={args.batch_size})...")
    print(f"This may take a few minutes...")
    embeddings = embedder.encode_documents(
        tool_texts,
        batch_size=args.batch_size,
        show_progress=True,
    )
    print(f"✓ Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    # Verify normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  - Norm range: [{norms.min():.6f}, {norms.max():.6f}] (should be ~1.0)")

    # Build FAISS index
    print(f"\nBuilding FAISS index...")
    index = FAISSIndex(dimension=model_info["dimensions"])
    index.add(embeddings, metadata)
    stats = index.get_stats()
    print(f"✓ Index built successfully")
    print(f"  - Total tools: {stats['total_tools']}")
    print(f"  - Dimension: {stats['dimension']}")
    print(f"  - Index type: {stats['index_type']}")

    # Save index
    print(f"\nSaving index to {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)
    index.save(args.output)

    # Save model info
    model_info_path = args.output / "model_info.json"
    model_info_data = {
        **model_info,
        "build_date": datetime.utcnow().isoformat(),
        "total_tools": stats['total_tools'],
        "input_file": str(args.input),
    }
    with model_info_path.open("w", encoding="utf-8") as f:
        json.dump(model_info_data, f, indent=2)
    print(f"✓ Saved model info to {model_info_path}")

    # Test the index with a sample query
    print(f"\n{'='*60}")
    print(f"Testing index with sample queries...")
    print(f"{'='*60}\n")

    test_queries = [
        "search GitHub repositories",
        "fetch web page content",
        "analyze stock prices",
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        query_embedding = embedder.encode_query(query)
        results = index.search(query_embedding, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['server']}/{result['tool']}")
            print(f"     Score: {result['similarity_score']:.4f}")
            print(f"     Desc: {result['description'][:80]}...")
        print()

    # Final summary
    print(f"{'='*60}")
    print(f"Index Build Complete!")
    print(f"{'='*60}")
    print(f"Total tools indexed: {stats['total_tools']}")
    print(f"Output directory: {args.output}")
    print(f"Files created:")
    print(f"  - index.faiss (FAISS index)")
    print(f"  - metadata.json (Tool metadata)")
    print(f"  - config.json (Index configuration)")
    print(f"  - model_info.json (Model information)")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())