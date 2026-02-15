"""
Text Embedding Service for generating vector representations of keywords.
Uses sentence-transformers to encode text into dense vectors.
"""
import logging
from typing import List, Dict, Union
import numpy as np
from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Singleton service for generating text embeddings.
    Uses 'all-MiniLM-L6-v2' model for fast and efficient embeddings (384-d).
    """
    _instance = None
    MODEL_NAME = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    
    def __init__(self):
        self.model = None
        if SentenceTransformer:
            try:
                logger.info(f"Loading embedding model: {self.MODEL_NAME}")
                self.model = SentenceTransformer(self.MODEL_NAME)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
        else:
            logger.warning("sentence-transformers not installed. Embedding features disabled.")
            
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EmbeddingService()
        return cls._instance
        
    @lru_cache(maxsize=10000)
    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        Cached for performance.
        """
        if not self.model or not text:
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            
        try:
            vector = self.model.encode(text)
            return vector
        except Exception as e:
            logger.error(f"Error encoding text '{text}': {e}")
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        """
        if not self.model or not texts:
            return np.zeros((len(texts), self.EMBEDDING_DIM), dtype=np.float32)
            
        try:
            vectors = self.model.encode(texts)
            return vectors
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            return np.zeros((len(texts), self.EMBEDDING_DIM), dtype=np.float32)

# Global accessor
embedding_service = EmbeddingService.get_instance()
