"""
Semantic search components for MCP tool discovery.

This package provides:
- BGE-M3 embeddings for multilingual text
- FAISS-based vector similarity search
- Tool indexing and search functionality
"""

from .embeddings import BGEEmbedder
from .faiss_backend import FAISSIndex

__all__ = ["BGEEmbedder", "FAISSIndex"]