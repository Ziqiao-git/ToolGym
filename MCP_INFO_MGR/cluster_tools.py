"""
Cluster MCP servers and tools by their semantic similarity.

Uses BGE-M3 embeddings + K-means/HDBSCAN clustering to group similar tools together.
Outputs cluster analysis and visualizations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional
import argparse

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from MCP_INFO_MGR.semantic_search.embeddings import BGEEmbedder


def load_servers(input_path: Path) -> list[dict]:
    """Load servers from NDJSON file."""
    servers = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if data.get("status") == "ok" and data.get("tools"):
                    servers.append(data)
    return servers


def build_server_texts(servers: list[dict], max_chars: int = 2000) -> tuple[list[str], list[str], list[dict]]:
    """
    Build text representations for each server.
    Returns: (texts, server_names, server_data)

    Args:
        servers: List of server data dictionaries
        max_chars: Maximum characters per server text (to avoid memory issues)
    """
    texts = []
    names = []
    data = []

    for server in servers:
        name = server.get("qualifiedName", "unknown")
        tools = server.get("tools", [])

        # Combine all tool names and descriptions (limit to first N tools and truncate)
        tool_texts = []
        for tool in tools[:20]:  # Limit to first 20 tools
            tool_name = tool.get("name", "")
            tool_desc = (tool.get("description") or "")[:200]  # Truncate descriptions
            if tool_name or tool_desc:
                tool_texts.append(f"{tool_name}: {tool_desc}")

        combined = f"Server: {name}\n" + "\n".join(tool_texts)
        # Truncate to max_chars
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        texts.append(combined)
        names.append(name)
        data.append(server)

    return texts, names, data


def cluster_servers(
    embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 15,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """Cluster embeddings using specified method."""
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings)
    elif method == "hdbscan":
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels


def analyze_clusters(
    labels: np.ndarray,
    server_names: list[str],
    server_data: list[dict],
) -> dict:
    """Analyze cluster composition."""
    clusters = defaultdict(list)

    for idx, label in enumerate(labels):
        clusters[int(label)].append({
            "name": server_names[idx],
            "tools": server_data[idx].get("tools", []),
            "tool_count": len(server_data[idx].get("tools", [])),
            "use_count": server_data[idx].get("useCount", 0),
        })

    # Compute cluster summaries
    summaries = {}
    for cluster_id, members in sorted(clusters.items()):
        if cluster_id == -1:
            cluster_name = "Noise/Outliers"
        else:
            cluster_name = f"Cluster {cluster_id}"

        # Get common keywords from tool names
        all_tool_names = []
        for m in members:
            for t in m["tools"]:
                name = t.get("name", "")
                # Split camelCase and snake_case
                parts = name.replace("_", " ").replace("-", " ")
                all_tool_names.extend(parts.lower().split())

        # Filter common words
        stopwords = {"get", "set", "list", "create", "update", "delete", "fetch", "search",
                     "the", "a", "an", "to", "from", "for", "and", "or", "tool", "mcp", "server"}
        keywords = [w for w in all_tool_names if len(w) > 2 and w not in stopwords]
        top_keywords = [w for w, _ in Counter(keywords).most_common(8)]

        summaries[cluster_name] = {
            "size": len(members),
            "servers": [m["name"] for m in members],
            "total_tools": sum(m["tool_count"] for m in members),
            "total_usage": sum(m["use_count"] for m in members),
            "keywords": top_keywords,
        }

    return summaries


def generate_cluster_labels(summaries: dict) -> dict[str, str]:
    """Generate human-readable labels for clusters based on keywords."""
    label_map = {
        "Noise/Outliers": "Miscellaneous",
    }

    # Keyword -> category mapping
    category_hints = {
        ("github", "git", "repo", "commit", "branch"): "Code & Version Control",
        ("file", "read", "write", "directory", "path"): "File System",
        ("search", "web", "browse", "url", "fetch"): "Web & Search",
        ("database", "sql", "query", "postgres", "mysql", "mongo"): "Database",
        ("api", "rest", "http", "request", "response"): "API & HTTP",
        ("slack", "discord", "telegram", "chat", "message"): "Communication",
        ("email", "gmail", "mail", "send"): "Email",
        ("image", "photo", "vision", "screenshot"): "Image & Vision",
        ("audio", "voice", "speech", "music"): "Audio & Voice",
        ("weather", "forecast", "temperature"): "Weather",
        ("finance", "stock", "crypto", "trading", "price"): "Finance & Trading",
        ("news", "article", "content", "blog"): "News & Content",
        ("calendar", "schedule", "event", "meeting"): "Calendar & Scheduling",
        ("docs", "documentation", "learn", "microsoft", "azure"): "Documentation",
        ("code", "python", "javascript", "typescript"): "Programming",
        ("ai", "llm", "model", "generate", "prompt"): "AI & ML",
        ("map", "location", "geo", "place"): "Maps & Location",
        ("translate", "language", "text"): "Translation",
        ("pdf", "document", "convert"): "Document Processing",
        ("notion", "obsidian", "note", "knowledge"): "Notes & Knowledge",
        ("aws", "cloud", "azure", "gcp"): "Cloud Services",
        ("docker", "kubernetes", "container"): "DevOps",
        ("test", "debug", "lint", "format"): "Development Tools",
        ("figma", "design", "ui", "canvas"): "Design",
        ("game", "play", "steam", "epic"): "Gaming",
        ("sport", "mlb", "nba", "football"): "Sports",
        ("health", "medical", "clinical", "trial"): "Health & Medical",
        ("research", "paper", "academic", "scholar"): "Research & Academic",
    }

    for cluster_name, info in summaries.items():
        if cluster_name in label_map:
            continue

        keywords_set = set(info["keywords"])
        best_match = None
        best_score = 0

        for hint_words, category in category_hints.items():
            score = len(keywords_set.intersection(hint_words))
            if score > best_score:
                best_score = score
                best_match = category

        if best_match and best_score >= 1:
            label_map[cluster_name] = best_match
        else:
            # Use top keywords as label
            label_map[cluster_name] = " / ".join(info["keywords"][:3]).title() if info["keywords"] else "General"

    return label_map


def plot_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    server_names: list[str],
    label_map: dict[str, str],
    output_path: Path,
    method: str = "tsne",
):
    """Create 2D visualization of clusters."""
    # Reduce to 2D
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    else:
        reducer = PCA(n_components=2, random_state=42)

    coords = reducer.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Get unique labels and colors
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_name = f"Cluster {label}" if label >= 0 else "Noise/Outliers"
        display_name = label_map.get(cluster_name, cluster_name)

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[i]],
            label=f"{display_name} ({mask.sum()})",
            alpha=0.7,
            s=50,
        )

    ax.set_title("MCP Server Clusters by Tool Similarity", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster visualization to {output_path}")


def print_cluster_report(summaries: dict, label_map: dict[str, str]):
    """Print formatted cluster report."""
    print("\n" + "=" * 80)
    print("MCP SERVER CLUSTER ANALYSIS")
    print("=" * 80)

    # Sort by size
    sorted_clusters = sorted(summaries.items(), key=lambda x: -x[1]["size"])

    for cluster_name, info in sorted_clusters:
        display_name = label_map.get(cluster_name, cluster_name)
        print(f"\n📦 {display_name} ({info['size']} servers, {info['total_tools']} tools)")
        print(f"   Keywords: {', '.join(info['keywords'][:6])}")
        print(f"   Total Usage: {info['total_usage']:,}")
        print(f"   Servers:")
        for server in sorted(info["servers"])[:10]:
            print(f"      - {server}")
        if len(info["servers"]) > 10:
            print(f"      ... and {len(info['servers']) - 10} more")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_servers = sum(info["size"] for info in summaries.values())
    total_tools = sum(info["total_tools"] for info in summaries.values())
    print(f"Total Servers: {total_servers}")
    print(f"Total Tools: {total_tools}")
    print(f"Number of Clusters: {len(summaries)}")
    print(f"Average Cluster Size: {total_servers / len(summaries):.1f} servers")


def save_cluster_json(summaries: dict, label_map: dict[str, str], output_path: Path):
    """Save cluster data as JSON."""
    output = {
        "clusters": [],
        "summary": {
            "total_servers": sum(info["size"] for info in summaries.values()),
            "total_tools": sum(info["total_tools"] for info in summaries.values()),
            "num_clusters": len(summaries),
        }
    }

    for cluster_name, info in sorted(summaries.items(), key=lambda x: -x[1]["size"]):
        output["clusters"].append({
            "id": cluster_name,
            "label": label_map.get(cluster_name, cluster_name),
            "size": info["size"],
            "total_tools": info["total_tools"],
            "total_usage": info["total_usage"],
            "keywords": info["keywords"],
            "servers": info["servers"],
        })

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved cluster data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster MCP servers by tool similarity")
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "mcp_data/working/tool_descriptions.ndjson",
        help="Input NDJSON file with tool descriptions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "cluster_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=15,
        help="Number of clusters for K-means (default: 15)"
    )
    parser.add_argument(
        "--method",
        choices=["kmeans", "hdbscan"],
        default="kmeans",
        help="Clustering method (default: kmeans)"
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading servers from {args.input}...")
    servers = load_servers(args.input)
    print(f"Loaded {len(servers)} servers with tools")

    if len(servers) < 5:
        print("Error: Not enough servers to cluster")
        return 1

    # Build texts
    print("Building text representations...")
    texts, names, data = build_server_texts(servers)

    # Get embeddings
    print("Computing embeddings with BGE-M3...")
    embedder = BGEEmbedder()
    embeddings = embedder.encode_documents(texts)
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # Cluster
    n_clusters = min(args.n_clusters, len(servers) // 2)
    print(f"Clustering with {args.method} (n_clusters={n_clusters})...")
    labels = cluster_servers(embeddings, method=args.method, n_clusters=n_clusters)

    # Analyze
    print("Analyzing clusters...")
    summaries = analyze_clusters(labels, names, data)
    label_map = generate_cluster_labels(summaries)

    # Output
    print_cluster_report(summaries, label_map)

    # Save JSON
    json_path = args.output_dir / "clusters.json"
    save_cluster_json(summaries, label_map, json_path)

    # Plot
    plot_path = args.output_dir / "cluster_visualization.png"
    plot_clusters(embeddings, labels, names, label_map, plot_path)

    print(f"\nDone! Results saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
