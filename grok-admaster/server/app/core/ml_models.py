
from typing import Optional
import logging
from functools import lru_cache

# Import ML models
# Note: These imports might be slow if they load heavy libraries, but that is acceptable at startup
from app.modules.amazon_ppc.ml.bid_optimizer import BidOptimizer
from app.modules.amazon_ppc.ml.rl_agent import PPCRLAgent
from app.modules.amazon_ppc.ml.deep_optimizer import DeepBidOptimizer
from app.modules.amazon_ppc.ml.bandits import BidBanditOptimizer
from app.modules.amazon_ppc.ml.ensemble import ModelEnsemble

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Singleton cache for ML models to prevent re-initialization on every request.
    This significantly reduces latency and file I/O operations.
    """
    _instance = None
    
    def __init__(self):
        self._bid_optimizer: Optional[BidOptimizer] = None
        self._deep_optimizer: Optional[DeepBidOptimizer] = None
        self._rl_agent: Optional[PPCRLAgent] = None
        self._bandit_optimizer: Optional[BidBanditOptimizer] = None
        self._model_ensemble: Optional[ModelEnsemble] = None
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelCache()
        return cls._instance

    def get_bid_optimizer(self) -> BidOptimizer:
        if self._bid_optimizer is None:
            logger.info("Initializing BidOptimizer (Lazy Load)")
            self._bid_optimizer = BidOptimizer()
        return self._bid_optimizer

    def get_deep_optimizer(self) -> DeepBidOptimizer:
        if self._deep_optimizer is None:
            logger.info("Initializing DeepBidOptimizer (Lazy Load)")
            self._deep_optimizer = DeepBidOptimizer()
        return self._deep_optimizer

    def get_rl_agent(self) -> PPCRLAgent:
        if self._rl_agent is None:
            logger.info("Initializing PPCRLAgent (Lazy Load)")
            self._rl_agent = PPCRLAgent()
        return self._rl_agent
        
    def get_bandit_optimizer(self) -> BidBanditOptimizer:
        if self._bandit_optimizer is None:
            logger.info("Initializing BidBanditOptimizer (Lazy Load)")
            self._bandit_optimizer = BidBanditOptimizer()
        return self._bandit_optimizer

    def get_model_ensemble(self) -> ModelEnsemble:
        if self._model_ensemble is None:
            logger.info("Initializing ModelEnsemble (Lazy Load)")
            self._model_ensemble = ModelEnsemble()
        return self._model_ensemble
        
    def preload_all(self):
        """Force load all models into memory (e.g., at startup)."""
        logger.info("Preloading all ML models...")
        self.get_bid_optimizer()
        self.get_deep_optimizer()
        self.get_rl_agent()
        self.get_bandit_optimizer()
        self.get_model_ensemble()
        logger.info("All ML models preloaded.")

# Global accessor
model_cache = ModelCache.get_instance()
