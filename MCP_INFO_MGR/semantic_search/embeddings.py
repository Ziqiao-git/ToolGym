"""
BGE-M3 embedding generation for multilingual semantic search.

This module provides a wrapper around the BGE-M3 model from BAAI for generating
high-quality multilingual embeddings optimized for retrieval tasks.
"""
from __future__ import annotations

import numpy as np
from typing import Union, List
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """
    Wrapper for BGE-M3 (BAAI General Embedding) model.

    BGE-M3 is a state-of-the-art multilingual embedding model that supports:
    - 100+ languages
    - Up to 8192 token context length
    - Dense, sparse, and multi-vector retrieval

    Features:
    - Normalized embeddings for cosine similarity
    - Query instruction prefix for better retrieval
    - Batch processing for efficiency
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    MODEL_NAME = "BAAI/bge-m3"
    DIMENSIONS = 1024

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the BGE-M3 embedder.

        Args:
            model_name: Model identifier (default: BAAI/bge-m3)
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.model_name = model_name or self.MODEL_NAME
        self.device = device

        print(f"Loading BGE-M3 model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"✓ Model loaded on device: {self.model.device}")

    def encode_query(
        self,
        query: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a search query or list of queries.

        Queries are prefixed with an instruction for better retrieval performance.

        Args:
            query: Single query string or list of query strings
            batch_size: Batch size for processing multiple queries
            show_progress: Show progress bar for batch processing

        Returns:
            Normalized embedding(s) of shape (D,) for single query or (N, D) for multiple
        """
        # Add instruction prefix
        if isinstance(query, str):
            text = self.QUERY_INSTRUCTION + query
        else:
            text = [self.QUERY_INSTRUCTION + q for q in query]

        embeddings = self.model.encode(
            text,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def encode_documents(
        self,
        documents: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode document(s) for indexing.

        Documents do not receive an instruction prefix (BGE-M3 best practice).

        Args:
            documents: Single document string or list of document strings
            batch_size: Batch size for processing multiple documents
            show_progress: Show progress bar for batch processing

        Returns:
            Normalized embedding(s) of shape (D,) for single doc or (N, D) for multiple
        """
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "dimensions": self.DIMENSIONS,
            "max_seq_length": self.model.max_seq_length,
            "device": str(self.model.device),
            "query_instruction": self.QUERY_INSTRUCTION,
        }


def create_tool_text(
    tool_name: str,
    tool_description: str,
    server_name: str,
    input_schema: dict = None,
) -> str:
    """
    Create a combined text representation of a tool for embedding.

    This combines multiple fields into a single searchable text that captures
    the tool's purpose, functionality, and context.

    Args:
        tool_name: Name of the tool
        tool_description: Description of what the tool does
        server_name: Name of the MCP server providing the tool
        input_schema: Optional JSON schema of tool inputs

    Returns:
        Combined text string optimized for semantic search
    """
    parts = []

    # Add server name for context
    parts.append(f"Server: {server_name}")

    # Add tool name (often contains useful keywords)
    parts.append(f"Tool: {tool_name}")

    # Add description (main searchable content)
    if tool_description:
        parts.append(f"Description: {tool_description}")

    # Add schema summary if available
    if input_schema and isinstance(input_schema, dict):
        # Extract parameter names and descriptions
        properties = input_schema.get("properties", {})
        if properties:
            param_names = list(properties.keys())
            parts.append(f"Parameters: {', '.join(param_names)}")

            # Add parameter descriptions if available
            for param, schema in properties.items():
                if isinstance(schema, dict) and "description" in schema:
                    parts.append(f"{param}: {schema['description']}")

    return " | ".join(parts)


if __name__ == "__main__":
    # Example usage
    embedder = BGEEmbedder()

    # Example tool
    tool_text = create_tool_text(
        tool_name="search_repositories",
        tool_description="Search for GitHub repositories using various filters",
        server_name="@smithery-ai/github",
        input_schema={
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "per_page": {"type": "number", "description": "Results per page"},
            }
        },
    )

    print(f"\nTool text: {tool_text}\n")

    # Generate document embedding
    doc_embedding = embedder.encode_documents(tool_text)
    print(f"Document embedding shape: {doc_embedding.shape}")

    # Generate query embedding
    query_embedding = embedder.encode_query("search GitHub repositories")
    print(f"Query embedding shape: {query_embedding.shape}")

    # Calculate similarity
    similarity = np.dot(query_embedding, doc_embedding)
    print(f"Similarity score: {similarity:.4f}")

    # Model info
    print(f"\nModel info: {embedder.get_model_info()}")
