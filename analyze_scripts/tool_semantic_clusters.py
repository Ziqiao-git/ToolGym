#!/usr/bin/env python3
"""
Semantic clustering of MCP tools using the existing FAISS index (no re-embedding).

Requirements:
  pip install scikit-learn faiss-cpu

Usage:
  python analyze_scripts/tool_semantic_clusters.py \
    --index-dir MCP_INFO_MGR/mcp_data/working/semantic_index \
    --clusters 12 \
    --sample 3000 \
    --seed 42
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss  # type: ignore
import numpy as np
from sklearn.cluster import KMeans
try:
    from sklearn.decomposition import PCA  # type: ignore
except ImportError:
    PCA = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:
    plt = None


def load_metadata(meta_path: Path) -> List[Dict[str, Any]]:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic clustering of MCP tools (using existing FAISS index).")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("MCP_INFO_MGR/mcp_data/working/semantic_index"),
        help="Directory containing index.faiss and metadata.json",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=12,
        help="Number of clusters (default: 12)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample this many tools (default: use all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling/kmeans",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a bar chart of cluster sizes to cluster_sizes.png (requires matplotlib)",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Save a 2D scatter plot of clusters to cluster_scatter.png (requires matplotlib, sklearn PCA)",
    )
    args = parser.parse_args()

    index_path = args.index_dir / "index.faiss"
    meta_path = args.index_dir / "metadata.json"

    if not index_path.exists() or not meta_path.exists():
        raise SystemExit(f"Missing index or metadata in {args.index_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    meta = load_metadata(meta_path)
    index = faiss.read_index(str(index_path))

    total = index.ntotal
    if len(meta) != total:
        print(f"Warning: metadata count ({len(meta)}) != index count ({total})")

    # Choose indices to use
    all_indices = list(range(total))
    if args.sample and args.sample < total:
        idxs = random.sample(all_indices, args.sample)
    else:
        idxs = all_indices

    # Reconstruct embeddings for chosen indices
    if len(idxs) == total and hasattr(index, "reconstruct_n"):
        try:
            embeddings = index.reconstruct_n(0, total)
        except Exception:
            embeddings = np.stack([index.reconstruct(i) for i in idxs])
    else:
        embeddings = np.stack([index.reconstruct(i) for i in idxs])
    tools = [(meta[i].get("server", "unknown"), meta[i].get("tool", ""), meta[i].get("description", "")) for i in idxs]

    print(f"Clustering {len(tools)} tools into {args.clusters} clusters...")

    kmeans = KMeans(n_clusters=args.clusters, n_init="auto", random_state=args.seed)
    labels = kmeans.fit_predict(embeddings)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Report cluster sizes and sample tool names
    print("\nCluster summary:")
    for label, idxs in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        sample_idxs = idxs[:5]
        sample_names = [tools[i][1] for i in sample_idxs]
        print(f"  Cluster {label}: {len(idxs)} tools | examples: {', '.join(sample_names)}")

    if args.plot:
        if plt is None:
            print("matplotlib is not installed; cannot plot. Install with `pip install matplotlib`.")
            return
        sizes = [(label, len(idxs)) for label, idxs in sorted(clusters.items())]
        labels_plot = [str(lbl) for lbl, _ in sizes]
        counts_plot = [cnt for _, cnt in sizes]
        plt.figure(figsize=(10, 5))
        plt.bar(labels_plot, counts_plot)
        plt.xlabel("Cluster")
        plt.ylabel("Tool count")
        plt.title("MCP tool clusters (FAISS index)")
        plt.tight_layout()
        out_path = Path("cluster_sizes.png")
        plt.savefig(out_path)
        print(f"Saved cluster size plot to {out_path}")

    if args.scatter:
        if plt is None:
            print("matplotlib is not installed; cannot scatter plot. Install with `pip install matplotlib`.")
            return
        if PCA is None:
            print("scikit-learn PCA not available; cannot scatter plot. Install scikit-learn.")
            return
        # 2D projection
        pca = PCA(n_components=2, random_state=args.seed)
        coords = pca.fit_transform(embeddings)
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=8, alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("MCP tool clusters (PCA 2D)")
        plt.tight_layout()
        out_path = Path("cluster_scatter.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved cluster scatter plot to {out_path}")


if __name__ == "__main__":
    main()
