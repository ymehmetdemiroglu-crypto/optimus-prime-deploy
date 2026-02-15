"""
Multi-Armed Bandit for exploration/exploitation in bid optimization.
Implements Thompson Sampling and UCB algorithms.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class BidArm:
    """Represents a bid multiplier as a bandit arm."""
    multiplier: float
    pulls: int = 0
    total_reward: float = 0.0
    
    # For Thompson Sampling (Beta distribution)
    alpha: float = 1.0
    beta: float = 1.0
    
    # For UCB
    sum_squared_reward: float = 0.0
    
    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0


class ThompsonSampler:
    """
    Thompson Sampling for bid optimization.
    Uses Beta distribution for binary rewards (hit target or not).
    """
    
    def __init__(self, arms: List[float] = None):
        # Default arms: bid multipliers from 0.5x to 1.5x
        self.arm_multipliers = arms or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        self.arms: Dict[int, BidArm] = {
            idx: BidArm(multiplier=m) 
            for idx, m in enumerate(self.arm_multipliers)
        }
    
    def select_arm(self) -> Tuple[int, float]:
        """Select arm using Thompson Sampling."""
        samples = []
        for arm_id, arm in self.arms.items():
            # Sample from Beta distribution
            sample = np.random.beta(arm.alpha, arm.beta)
            samples.append((arm_id, sample))
        
        # Select arm with highest sample
        best_arm_id = max(samples, key=lambda x: x[1])[0]
        return best_arm_id, self.arms[best_arm_id].multiplier
    
    def update(self, arm_id: int, reward: float):
        """
        Update arm statistics.
        reward: 1 if hit target ACoS, 0 otherwise (or continuous in [0,1])
        """
        arm = self.arms[arm_id]
        arm.pulls += 1
        arm.total_reward += reward
        
        # Update Beta parameters
        if reward > 0.5:
            arm.alpha += 1
        else:
            arm.beta += 1
    
    def get_statistics(self) -> List[Dict[str, Any]]:
        """Get arm statistics."""
        return [
            {
                'multiplier': arm.multiplier,
                'pulls': arm.pulls,
                'mean_reward': arm.mean_reward,
                'alpha': arm.alpha,
                'beta': arm.beta,
                'expected_value': arm.alpha / (arm.alpha + arm.beta)
            }
            for arm in self.arms.values()
        ]


class UCBBandit:
    """
    Upper Confidence Bound algorithm for bid optimization.
    Balances exploration and exploitation with confidence bounds.
    """
    
    def __init__(self, arms: List[float] = None, exploration_factor: float = 2.0):
        self.arm_multipliers = arms or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        self.arms: Dict[int, BidArm] = {
            idx: BidArm(multiplier=m) 
            for idx, m in enumerate(self.arm_multipliers)
        }
        self.exploration_factor = exploration_factor
        self.total_pulls = 0
    
    def select_arm(self) -> Tuple[int, float]:
        """Select arm using UCB1 algorithm."""
        self.total_pulls += 1
        
        # First, try each arm at least once
        for arm_id, arm in self.arms.items():
            if arm.pulls == 0:
                return arm_id, arm.multiplier
        
        # Calculate UCB for each arm
        ucb_values = []
        for arm_id, arm in self.arms.items():
            mean = arm.mean_reward
            exploration_bonus = self.exploration_factor * np.sqrt(
                np.log(self.total_pulls) / arm.pulls
            )
            ucb = mean + exploration_bonus
            ucb_values.append((arm_id, ucb))
        
        best_arm_id = max(ucb_values, key=lambda x: x[1])[0]
        return best_arm_id, self.arms[best_arm_id].multiplier
    
    def update(self, arm_id: int, reward: float):
        """Update arm statistics with observed reward."""
        arm = self.arms[arm_id]
        arm.pulls += 1
        arm.total_reward += reward
        arm.sum_squared_reward += reward ** 2
    
    def get_best_arm(self) -> Tuple[int, float]:
        """Get the arm with highest mean reward."""
        best_arm = max(self.arms.values(), key=lambda a: a.mean_reward)
        return list(self.arms.keys())[list(self.arms.values()).index(best_arm)], best_arm.multiplier


class ContextualBandit:
    """
    Contextual bandit that considers features when selecting arms.
    Uses linear regression for each arm.
    """
    
    CONTEXT_FEATURES = ['acos_7d', 'ctr_7d', 'conversion_rate_7d', 'momentum', 'is_weekend']
    
    def __init__(self, arms: List[float] = None, regularization: float = 1.0):
        self.arm_multipliers = arms or [0.7, 0.85, 1.0, 1.15, 1.3]
        self.n_features = len(self.CONTEXT_FEATURES)
        self.regularization = regularization
        
        # Linear regression weights for each arm
        self.arm_weights: Dict[int, np.ndarray] = {
            idx: np.zeros(self.n_features + 1)  # +1 for bias
            for idx in range(len(self.arm_multipliers))
        }
        
        # For online ridge regression
        self.arm_A: Dict[int, np.ndarray] = {
            idx: np.eye(self.n_features + 1) * regularization
            for idx in range(len(self.arm_multipliers))
        }
        self.arm_b: Dict[int, np.ndarray] = {
            idx: np.zeros(self.n_features + 1)
            for idx in range(len(self.arm_multipliers))
        }
    
    def _get_context(self, features: Dict[str, Any]) -> np.ndarray:
        """Extract context vector from features."""
        context = [1.0]  # Bias term
        for feat in self.CONTEXT_FEATURES:
            val = features.get(feat, 0)
            if isinstance(val, bool):
                val = int(val)
            context.append(float(val) if val is not None else 0.0)
        return np.array(context)
    
    def select_arm(self, features: Dict[str, Any]) -> Tuple[int, float]:
        """Select arm based on context using LinUCB."""
        context = self._get_context(features)
        
        best_arm = 0
        best_ucb = -float('inf')
        
        for arm_id in range(len(self.arm_multipliers)):
            A_inv = np.linalg.inv(self.arm_A[arm_id])
            theta = A_inv @ self.arm_b[arm_id]
            
            # Expected reward
            expected = context @ theta
            
            # Confidence bound
            confidence = np.sqrt(context @ A_inv @ context)
            
            ucb = expected + confidence
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm_id
        
        return best_arm, self.arm_multipliers[best_arm]
    
    def update(self, arm_id: int, features: Dict[str, Any], reward: float):
        """Update arm model with observed reward."""
        context = self._get_context(features)
        
        self.arm_A[arm_id] += np.outer(context, context)
        self.arm_b[arm_id] += reward * context


class BidBanditOptimizer:
    """
    Combined bandit optimizer using multiple algorithms.
    Ensembles Thompson Sampling, UCB, and Contextual Bandit.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/bandit_optimizer.pkl"
        
        # Individual bandits
        self.thompson = ThompsonSampler()
        self.ucb = UCBBandit()
        self.contextual = ContextualBandit()
        
        # Keyword-specific bandits for personalization
        self.keyword_bandits: Dict[int, ThompsonSampler] = defaultdict(ThompsonSampler)
        
        self._load_model()
    
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.thompson = data.get('thompson', self.thompson)
                    self.ucb = data.get('ucb', self.ucb)
                    self.contextual = data.get('contextual', self.contextual)
                logger.info(f"Loaded bandit optimizer from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load bandit model: {e}")
    
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'thompson': self.thompson,
                'ucb': self.ucb,
                'contextual': self.contextual
            }, f)
    
    def select_bid_multiplier(
        self,
        features: Dict[str, Any],
        keyword_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Select bid multiplier using ensemble of bandits.
        """
        # Thompson Sampling selection
        ts_arm, ts_mult = self.thompson.select_arm()
        
        # UCB selection
        ucb_arm, ucb_mult = self.ucb.select_arm()
        
        # Contextual selection
        ctx_arm, ctx_mult = self.contextual.select_arm(features)
        
        # Keyword-specific if available
        if keyword_id:
            kw_arm, kw_mult = self.keyword_bandits[keyword_id].select_arm()
        else:
            kw_mult = 1.0
        
        # Ensemble (weighted average)
        weights = {
            'thompson': 0.3,
            'ucb': 0.25,
            'contextual': 0.35,
            'keyword': 0.1
        }
        
        ensemble_mult = (
            weights['thompson'] * ts_mult +
            weights['ucb'] * ucb_mult +
            weights['contextual'] * ctx_mult +
            weights['keyword'] * kw_mult
        )
        
        return {
            'ensemble_multiplier': round(ensemble_mult, 3),
            'thompson': {'arm': ts_arm, 'multiplier': ts_mult},
            'ucb': {'arm': ucb_arm, 'multiplier': ucb_mult},
            'contextual': {'arm': ctx_arm, 'multiplier': ctx_mult},
            'keyword_specific': kw_mult if keyword_id else None
        }
    
    def update_all(
        self,
        features: Dict[str, Any],
        arm_selections: Dict[str, int],
        reward: float,
        keyword_id: Optional[int] = None
    ):
        """Update all bandits with observed reward."""
        self.thompson.update(arm_selections['thompson'], reward)
        self.ucb.update(arm_selections['ucb'], reward)
        self.contextual.update(arm_selections['contextual'], features, reward)
        
        if keyword_id:
            self.keyword_bandits[keyword_id].update(
                arm_selections.get('keyword', 5),  # Default to 1.0 arm
                reward
            )
        
        self.save_model()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all bandits."""
        return {
            'thompson': self.thompson.get_statistics(),
            'ucb': [
                {
                    'multiplier': arm.multiplier,
                    'pulls': arm.pulls,
                    'mean_reward': arm.mean_reward
                }
                for arm in self.ucb.arms.values()
            ],
            'best_ucb_arm': self.ucb.get_best_arm()[1]
        }
