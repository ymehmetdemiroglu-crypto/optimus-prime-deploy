"""
Ensemble Model System - Combines multiple ML models for robust predictions.
Implements stacking, voting, and weighted averaging.
"""
import json
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

from .bid_optimizer import BidOptimizer
from .deep_optimizer import DeepBidOptimizer
from .rl_agent import PPCRLAgent
from .bandits import BidBanditOptimizer

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS_PATH = "models/ensemble_weights.json"


@dataclass
class EnsemblePrediction:
    """Combined prediction from all models."""
    final_bid: float
    confidence: float
    model_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    reasoning: str


class ModelEnsemble:
    """
    Ensemble of ML models for bid optimization.
    Combines predictions using adaptive weighting.

    Weights are persisted to disk so they survive process restarts.
    Call update_weights() with observed outcomes to adapt the ensemble.
    """

    _DEFAULT_WEIGHTS = {
        'gradient_boost': 0.30,
        'deep_nn': 0.25,
        'rl_agent': 0.25,
        'bandit': 0.20,
    }

    def __init__(self, weights_path: str = _DEFAULT_WEIGHTS_PATH):
        # Initialize all models
        self.gradient_boost = BidOptimizer()
        self.deep_nn = DeepBidOptimizer()
        self.rl_agent = PPCRLAgent()
        self.bandit = BidBanditOptimizer()
        self.weights_path = weights_path

        # Performance tracking for adaptive weighting
        self.model_performance: Dict[str, List[float]] = {
            'gradient_boost': [],
            'deep_nn': [],
            'rl_agent': [],
            'bandit': [],
        }

        # Load persisted weights if available, otherwise use defaults
        self.model_weights = self._load_weights()

    def _load_weights(self) -> Dict[str, float]:
        """Load model weights from disk; fall back to defaults on any error."""
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path) as f:
                    saved = json.load(f)
                weights = saved.get('weights', {})
                performance = saved.get('performance', {})
                # Validate all expected keys are present
                if set(weights.keys()) == set(self._DEFAULT_WEIGHTS.keys()):
                    self.model_performance = {
                        k: performance.get(k, []) for k in self._DEFAULT_WEIGHTS
                    }
                    logger.info(f"Loaded ensemble weights from {self.weights_path}: {weights}")
                    return weights
            except Exception as e:
                logger.warning(f"Failed to load ensemble weights from {self.weights_path}: {e}")
        return dict(self._DEFAULT_WEIGHTS)

    def _save_weights(self) -> None:
        """Persist current weights and performance history to disk."""
        try:
            os.makedirs(os.path.dirname(self.weights_path) or ".", exist_ok=True)
            with open(self.weights_path, "w") as f:
                json.dump({
                    'weights': self.model_weights,
                    'performance': {k: v[-100:] for k, v in self.model_performance.items()},
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist ensemble weights: {e}")
    
    def predict(
        self,
        features: Dict[str, Any],
        target_acos: float = 25.0,
        target_roas: float = 4.0
    ) -> EnsemblePrediction:
        """
        Get ensemble prediction from all models.
        """
        current_bid = features.get('current_bid', 1.0)
        predictions = {}
        
        # Gradient Boosting
        gb_pred = self.gradient_boost.predict_bid(features, target_acos, target_roas)
        predictions['gradient_boost'] = gb_pred.predicted_bid
        
        # Deep Neural Network
        if self.deep_nn.is_trained:
            nn_pred, nn_uncertainty = self.deep_nn.predict(features)
            predictions['deep_nn'] = nn_pred
        else:
            predictions['deep_nn'] = gb_pred.predicted_bid  # Fallback
        
        # Reinforcement Learning
        rl_rec = self.rl_agent.get_bid_recommendation(features, current_bid, target_acos)
        predictions['rl_agent'] = rl_rec['recommended_bid']
        
        # Multi-Armed Bandit
        bandit_rec = self.bandit.select_bid_multiplier(features)
        predictions['bandit'] = current_bid * bandit_rec['ensemble_multiplier']
        
        # Weighted average
        weighted_sum = sum(
            predictions[model] * self.model_weights[model]
            for model in predictions
        )
        total_weight = sum(self.model_weights.values())
        final_bid = weighted_sum / total_weight
        
        # Ensure reasonable bounds
        final_bid = max(0.10, min(final_bid, current_bid * 2.0))
        
        # Calculate confidence (based on model agreement)
        pred_values = list(predictions.values())
        if len(pred_values) > 1:
            std_dev = np.std(pred_values)
            mean_pred = np.mean(pred_values)
            cv = std_dev / mean_pred if mean_pred > 0 else 1
            confidence = max(0.1, 1 - cv)  # Higher agreement = higher confidence
        else:
            confidence = 0.5
        
        # Generate reasoning
        recommendations = []
        for model, pred in predictions.items():
            change = ((pred - current_bid) / current_bid * 100) if current_bid > 0 else 0
            direction = "increase" if change > 0 else "decrease" if change < 0 else "maintain"
            recommendations.append(f"{model}: {direction} by {abs(change):.1f}%")
        
        reasoning = f"Ensemble of {len(predictions)} models. " + "; ".join(recommendations)
        
        return EnsemblePrediction(
            final_bid=round(final_bid, 2),
            confidence=round(confidence, 2),
            model_predictions={k: round(v, 2) for k, v in predictions.items()},
            model_weights=self.model_weights,
            reasoning=reasoning
        )
    
    def update_weights(self, observed_outcomes: List[Dict[str, Any]]):
        """
        Update model weights based on observed outcomes.
        
        observed_outcomes: List of {model_predictions, actual_optimal_bid}
        """
        for outcome in observed_outcomes:
            predictions = outcome.get('model_predictions', {})
            actual = outcome.get('actual_optimal_bid')
            
            if actual is None:
                continue
            
            for model, pred in predictions.items():
                # Calculate error
                error = abs(pred - actual)
                # Inverse error as performance metric (lower error = higher performance)
                performance = 1 / (1 + error)
                
                if model in self.model_performance:
                    self.model_performance[model].append(performance)
                    # Keep last 100 observations
                    self.model_performance[model] = self.model_performance[model][-100:]
        
        # Recalculate weights based on average performance
        avg_performance = {}
        for model, perf_list in self.model_performance.items():
            if perf_list:
                avg_performance[model] = np.mean(perf_list)
            else:
                avg_performance[model] = 1.0
        
        total_perf = sum(avg_performance.values())
        if total_perf > 0:
            self.model_weights = {
                model: perf / total_perf
                for model, perf in avg_performance.items()
            }

        logger.info(f"Updated model weights: {self.model_weights}")
        self._save_weights()
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models in the ensemble."""
        return {
            'gradient_boost': {
                # is_trained is now a property on BidOptimizer (delegates to market_model)
                'is_trained': self.gradient_boost.is_trained,
                'weight': self.model_weights['gradient_boost'],
                'avg_performance': float(np.mean(self.model_performance['gradient_boost']))
                    if self.model_performance['gradient_boost'] else None,
            },
            'deep_nn': {
                'is_trained': self.deep_nn.is_trained,
                'weight': self.model_weights['deep_nn'],
                'avg_performance': float(np.mean(self.model_performance['deep_nn']))
                    if self.model_performance['deep_nn'] else None,
            },
            'rl_agent': {
                # DQN agent uses a neural network, not a Q-table.
                # Expose steps_done (training steps) and current epsilon instead.
                'steps_done': self.rl_agent.steps_done,
                'epsilon': round(self.rl_agent.epsilon, 4),
                'explore_steps': self.rl_agent._explore_steps,
                'weight': self.model_weights['rl_agent'],
                'avg_performance': float(np.mean(self.model_performance['rl_agent']))
                    if self.model_performance['rl_agent'] else None,
            },
            'bandit': {
                'weight': self.model_weights['bandit'],
                'avg_performance': float(np.mean(self.model_performance['bandit']))
                    if self.model_performance['bandit'] else None,
            },
        }


class StackingEnsemble:
    """
    Stacking ensemble that uses a meta-learner to combine base models.
    """
    
    def __init__(self):
        self.base_models = ModelEnsemble()
        
        # Meta-learner weights (learned via linear regression)
        self.meta_weights = np.array([0.25, 0.25, 0.25, 0.25])
        self.meta_bias = 0.0
        
        # Training data for meta-learner
        self.meta_X: List[np.ndarray] = []
        self.meta_y: List[float] = []
    
    def predict(
        self,
        features: Dict[str, Any],
        target_acos: float = 25.0
    ) -> Dict[str, Any]:
        """
        Get stacked ensemble prediction.
        """
        ensemble_pred = self.base_models.predict(features, target_acos)
        
        # Create meta-features from base predictions
        base_predictions = [
            ensemble_pred.model_predictions.get('gradient_boost', 1.0),
            ensemble_pred.model_predictions.get('deep_nn', 1.0),
            ensemble_pred.model_predictions.get('rl_agent', 1.0),
            ensemble_pred.model_predictions.get('bandit', 1.0)
        ]
        
        # Meta-learner prediction
        meta_pred = np.dot(self.meta_weights, base_predictions) + self.meta_bias
        
        return {
            'stacked_prediction': round(meta_pred, 2),
            'ensemble_prediction': ensemble_pred.final_bid,
            'base_predictions': ensemble_pred.model_predictions,
            'confidence': ensemble_pred.confidence
        }
    
    def train_meta_learner(self, outcomes: List[Dict[str, Any]]):
        """
        Train the meta-learner on observed outcomes.
        """
        for outcome in outcomes:
            preds = outcome.get('model_predictions', {})
            actual = outcome.get('actual_optimal_bid')
            
            if actual is None:
                continue
            
            x = np.array([
                preds.get('gradient_boost', 0),
                preds.get('deep_nn', 0),
                preds.get('rl_agent', 0),
                preds.get('bandit', 0)
            ])
            
            self.meta_X.append(x)
            self.meta_y.append(actual)
        
        if len(self.meta_X) < 10:
            return {'status': 'insufficient_data'}
        
        # Simple linear regression for meta-learner
        X = np.array(self.meta_X)
        y = np.array(self.meta_y)
        
        # Add bias column
        X_bias = np.column_stack([X, np.ones(len(X))])
        
        # Solve least squares
        try:
            solution = np.linalg.lstsq(X_bias, y, rcond=None)[0]
            self.meta_weights = solution[:-1]
            self.meta_bias = solution[-1]
            
            # Normalize weights to be positive
            self.meta_weights = np.maximum(0, self.meta_weights)
            weight_sum = self.meta_weights.sum()
            if weight_sum > 0:
                self.meta_weights /= weight_sum
            
            return {
                'status': 'trained',
                'weights': self.meta_weights.tolist(),
                'bias': self.meta_bias
            }
        except Exception as e:
            logger.warning(f"Meta-learner training failed: {e}")
            return {'status': 'failed', 'error': str(e)}


class VotingEnsemble:
    """
    Voting ensemble for discrete bid decisions.
    Uses majority vote for direction and median for magnitude.
    """
    
    def __init__(self):
        self.base_models = ModelEnsemble()
    
    def vote(
        self,
        features: Dict[str, Any],
        target_acos: float = 25.0
    ) -> Dict[str, Any]:
        """
        Get voting ensemble prediction.
        """
        current_bid = features.get('current_bid', 1.0)
        ensemble_pred = self.base_models.predict(features, target_acos)
        
        predictions = ensemble_pred.model_predictions
        
        # Vote on direction
        votes = {
            'increase': 0,
            'decrease': 0,
            'maintain': 0
        }
        
        for model, pred in predictions.items():
            change_pct = ((pred - current_bid) / current_bid * 100) if current_bid > 0 else 0
            
            if change_pct > 5:
                votes['increase'] += 1
            elif change_pct < -5:
                votes['decrease'] += 1
            else:
                votes['maintain'] += 1
        
        # Majority decision
        decision = max(votes.items(), key=lambda x: x[1])[0]
        
        # Get magnitude from median
        pred_values = list(predictions.values())
        median_bid = float(np.median(pred_values))
        
        return {
            'decision': decision,
            'votes': votes,
            'median_bid': round(median_bid, 2),
            'mean_bid': round(np.mean(pred_values), 2),
            'current_bid': current_bid,
            'recommended_bid': round(median_bid, 2),
            'confidence': max(votes.values()) / len(predictions)
        }
