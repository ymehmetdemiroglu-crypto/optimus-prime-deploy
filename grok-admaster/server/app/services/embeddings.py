"""
Embedding Service — The Semantic Core
Generates vector embeddings for text using a local SentenceTransformer model.
No external API calls. Runs entirely on your server.
"""
import logging
import numpy as np
from typing import List, Optional
from functools import lru_cache

logger = logging.getLogger("embedding_service")


class EmbeddingService:
    """
    Singleton embedding generator using sentence-transformers.
    Loads the model once into memory and reuses for all requests.
    
    Model: all-MiniLM-L6-v2 (384 dimensions, ~80MB, runs on CPU)
    """
    
    _instance: Optional["EmbeddingService"] = None
    _model = None
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_model(self):
        """Lazy-load model on first use to avoid import-time overhead."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.MODEL_NAME} ...")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.MODEL_NAME)
                logger.info(f"Model loaded successfully ({self.EMBEDDING_DIM} dimensions)")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: The text to embed (search term, product title, etc.)
            
        Returns:
            List of 384 floats representing the semantic vector.
        """
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        Much faster than calling embed_text() in a loop.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Processing batch size (memory vs speed tradeoff).
            
        Returns:
            List of embedding vectors.
        """
        self._load_model()
        if not texts:
            return []
        
        logger.info(f"Embedding batch of {len(texts)} texts (batch_size={batch_size})")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()
    
    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Returns:
            Float between -1 and 1 (1 = identical meaning, 0 = unrelated).
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)
    
    def semantic_distance(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Compute semantic distance (1 - cosine_similarity).
        Higher = more different.
        
        Returns:
            Float between 0 and 2 (0 = identical, 2 = opposite).
        """
        return 1.0 - self.cosine_similarity(vec_a, vec_b)


# Global singleton
embedding_service = EmbeddingService()
