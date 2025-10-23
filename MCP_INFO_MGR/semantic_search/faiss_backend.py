"""
FAISS-based vector search backend for MCP tool discovery.

This module provides efficient similarity search using Facebook AI Similarity Search (FAISS).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np


class FAISSIndex:
    """
    FAISS-based vector index for semantic search.

    Uses IndexFlatIP (inner product) for exact cosine similarity search
    on normalized vectors.

    Features:
    - Fast similarity search
    - Metadata storage and filtering
    - Save/load functionality
    - Batch operations
    """

    def __init__(self, dimension: int = 1024):
        """
        Initialize FAISS index.

        Args:
            dimension: Embedding dimension (1024 for BGE-M3)
        """
        self.dimension = dimension
        # IndexFlatIP: Inner Product (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Add embeddings and their metadata to the index.

        Args:
            embeddings: Array of shape (N, D) with normalized embeddings
            metadata: List of N metadata dictionaries, one per embedding
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        assert embeddings.shape[0] == len(
            metadata
        ), f"Embeddings ({embeddings.shape[0]}) and metadata ({len(metadata)}) count mismatch"
        assert (
            embeddings.shape[1] == self.dimension
        ), f"Embedding dimension {embeddings.shape[1]} != index dimension {self.dimension}"

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Store metadata
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar tools.

        Args:
            query_embedding: Query embedding of shape (D,) or (1, D)
            top_k: Number of results to return
            filters: Optional filters to apply:
                - server: Filter by server name (exact match)
                - min_score: Minimum similarity score (0-1)

        Returns:
            List of results with metadata and similarity scores
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search in FAISS index
        # We search for more results than needed to account for filtering
        search_k = min(top_k * 10, self.index.ntotal) if filters else top_k
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)

        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            result = {
                **self.metadata[idx],
                "similarity_score": float(score),
            }

            # Apply filters
            if filters:
                # Server filter
                if "server" in filters and result.get("server") != filters["server"]:
                    continue

                # Minimum score filter
                if "min_score" in filters and result["similarity_score"] < filters["min_score"]:
                    continue

            results.append(result)

            # Stop if we have enough results
            if len(results) >= top_k:
                break

        return results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar tools in batch.

        Args:
            query_embeddings: Query embeddings of shape (N, D)
            top_k: Number of results per query
            filters: Optional filters to apply

        Returns:
            List of N result lists, one per query
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Search in FAISS index
        search_k = min(top_k * 10, self.index.ntotal) if filters else top_k
        scores, indices = self.index.search(query_embeddings.astype(np.float32), search_k)

        # Prepare results for each query
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx == -1:
                    continue

                result = {
                    **self.metadata[idx],
                    "similarity_score": float(score),
                }

                # Apply filters
                if filters:
                    if "server" in filters and result.get("server") != filters["server"]:
                        continue
                    if "min_score" in filters and result["similarity_score"] < filters["min_score"]:
                        continue

                results.append(result)

                if len(results) >= top_k:
                    break

            all_results.append(results)

        return all_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        return {
            "total_tools": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "IndexFlatIP (cosine similarity)",
            "total_metadata_entries": len(self.metadata),
        }

    def save(self, output_dir: Path) -> None:
        """
        Save the index and metadata to disk.

        Creates three files:
        - index.faiss: FAISS index
        - metadata.json: Tool metadata
        - config.json: Index configuration

        Args:
            output_dir: Directory to save files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = output_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        print(f"✓ Saved FAISS index to {index_path}")

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")

        # Save configuration
        config_path = output_dir / "config.json"
        config = {
            "dimension": self.dimension,
            "total_tools": self.index.ntotal,
            "index_type": "IndexFlatIP",
        }
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config to {config_path}")

    @classmethod
    def load(cls, index_dir: Path) -> FAISSIndex:
        """
        Load index and metadata from disk.

        Args:
            index_dir: Directory containing saved index files

        Returns:
            Loaded FAISSIndex instance
        """
        index_dir = Path(index_dir)

        # Load configuration
        config_path = index_dir / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        # Create instance
        instance = cls(dimension=config["dimension"])

        # Load FAISS index
        index_path = index_dir / "index.faiss"
        instance.index = faiss.read_index(str(index_path))
        print(f"✓ Loaded FAISS index from {index_path}")

        # Load metadata
        metadata_path = index_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as f:
            instance.metadata = json.load(f)
        print(f"✓ Loaded metadata from {metadata_path}")

        print(f"✓ Index ready with {instance.index.ntotal} tools")
        return instance


if __name__ == "__main__":
    # Example usage
    print("Creating FAISS index...")
    index = FAISSIndex(dimension=1024)

    # Example embeddings (normally from BGE-M3)
    embeddings = np.random.randn(10, 1024).astype(np.float32)
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Example metadata
    metadata = [
        {
            "server": "@smithery-ai/github",
            "tool": f"tool_{i}",
            "description": f"This is tool number {i}",
            "inputSchema": {},
        }
        for i in range(10)
    ]

    # Add to index
    index.add(embeddings, metadata)
    print(f"Added {len(metadata)} tools to index")

    # Search
    query = np.random.randn(1024).astype(np.float32)
    query = query / np.linalg.norm(query)

    results = index.search(query, top_k=3)
    print(f"\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['tool']} (score: {result['similarity_score']:.4f})")

    # Statistics
    print(f"\nIndex stats: {index.get_stats()}")

    # Save and load
    from tempfile import mkdtemp

    temp_dir = Path(mkdtemp())
    print(f"\nSaving to {temp_dir}...")
    index.save(temp_dir)

    print(f"\nLoading from {temp_dir}...")
    loaded_index = FAISSIndex.load(temp_dir)
    print(f"Loaded index stats: {loaded_index.get_stats()}")