"""
Text Embedding Service — consolidated re-export.

Previously this module defined a second EmbeddingService class that loaded the
same 80MB all-MiniLM-L6-v2 model independently, causing double memory usage and
divergent cache behaviour (this module had an @lru_cache on .encode(); the
canonical service in app.services.embeddings did not).

All callers now use the canonical singleton from app.services.embeddings, which
is the single source of truth for embedding generation. This module re-exports
that singleton for backwards compatibility with existing import paths.
"""
from app.services.embeddings import EmbeddingService, embedding_service  # noqa: F401

__all__ = ["EmbeddingService", "embedding_service"]
