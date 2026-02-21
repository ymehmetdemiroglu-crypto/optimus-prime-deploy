"""
Text Embedding Service for generating vector representations of keywords.
Uses sentence-transformers to encode text into dense vectors.
"""
import logging
from typing import List, Dict, Union
import numpy as np
from functools import lru_cache

import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Singleton service for generating text embeddings.
    Uses 'all-MiniLM-L6-v2' model for fast and efficient embeddings (384-d).
    Uses 'intfloat/e5-large-v2' model lazily for high-quality e-commerce intent (1024-d).
    """
    _instance = None
    MODEL_NAME = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    
    INTENT_MODEL_NAME = 'intfloat/e5-large-v2'
    INTENT_EMBEDDING_DIM = 1024
    
    def __init__(self):
        self.model = None
        self._intent_model = None
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
        
    def get_intent_model(self):
        """Lazy load the intent model using fp16 to save memory."""
        if self._intent_model is None and SentenceTransformer:
            try:
                logger.info(f"Loading intent embedding model: {self.INTENT_MODEL_NAME} in float16")
                # e5 requires query: or passage: prefix, handled in the encode method
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._intent_model = SentenceTransformer(
                    self.INTENT_MODEL_NAME, 
                    model_kwargs={"torch_dtype": torch.float16},
                    device=device
                )
                logger.info("Intent embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load intent embedding model: {e}")
        return self._intent_model
        
    @lru_cache(maxsize=10000)
    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string using the core model.
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
        Generate embeddings for a batch of texts using the core model.
        """
        if not self.model or not texts:
            return np.zeros((len(texts), self.EMBEDDING_DIM), dtype=np.float32)
            
        try:
            vectors = self.model.encode(texts)
            return vectors
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            return np.zeros((len(texts), self.EMBEDDING_DIM), dtype=np.float32)

    @lru_cache(maxsize=5000)
    def encode_intent(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        Generate e-commerce intent embedding.
        e5-large-v2 expects 'query: ' or 'passage: ' prefix.
        """
        intent_model = self.get_intent_model()
        if not intent_model or not text:
            return np.zeros(self.INTENT_EMBEDDING_DIM, dtype=np.float32)
            
        prefix = "query: " if is_query else "passage: "
        try:
            vector = intent_model.encode(prefix + text)
            return vector
        except Exception as e:
            logger.error(f"Error encoding intent text '{text}': {e}")
            return np.zeros(self.INTENT_EMBEDDING_DIM, dtype=np.float32)

    def encode_batch_intent(self, texts: List[str], is_query: bool = True) -> np.ndarray:
        """
        Generate e-commerce intent embeddings for a batch.
        """
        intent_model = self.get_intent_model()
        if not intent_model or not texts:
            return np.zeros((len(texts), self.INTENT_EMBEDDING_DIM), dtype=np.float32)
            
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [prefix + t for t in texts]
        
        try:
            vectors = intent_model.encode(prefixed_texts)
            return vectors
        except Exception as e:
            logger.error(f"Error encoding intent batch: {e}")
            return np.zeros((len(texts), self.INTENT_EMBEDDING_DIM), dtype=np.float32)

# Global accessor
embedding_service = EmbeddingService.get_instance()
