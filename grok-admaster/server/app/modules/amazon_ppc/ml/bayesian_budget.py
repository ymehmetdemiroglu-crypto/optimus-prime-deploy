"""
Bayesian Budget Optimizer using Gaussian Process regression.
Optimizes budget allocation across campaigns and ad groups.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import os
import logging
from scipy.stats import norm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class BudgetAllocation:
    """Recommended budget allocation."""
    campaign_id: int
    current_budget: float
    recommended_budget: float
    expected_roi: float
    confidence: float
    reasoning: str


class GaussianProcessRegressor:
    """
    Simple Gaussian Process regressor for Bayesian optimization.
    Uses RBF (Radial Basis Function) kernel.
    """
    
    def __init__(self, length_scale: float = 1.0, noise: float = 0.1):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Gaussian) kernel."""
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                 np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-0.5 / self.length_scale**2 * sqdist)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the GP to training data."""
        self.X_train = X
        self.y_train = y
        
        K = self._rbf_kernel(X, X) + self.noise**2 * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation at test points.
        """
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X))
        
        K_trans = self._rbf_kernel(X, self.X_train)
        K_ss = self._rbf_kernel(X, X)
        
        # Mean prediction
        mu = K_trans @ self.K_inv @ self.y_train
        
        # Variance prediction
        var = np.diag(K_ss - K_trans @ self.K_inv @ K_trans.T)
        var = np.maximum(var, 1e-6)  # Ensure positive
        
        return mu, np.sqrt(var)


class BayesianBudgetOptimizer:
    """
    Uses Bayesian optimization to find optimal budget allocation.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/bayesian_budget.pkl"
        self.gp = GaussianProcessRegressor(length_scale=1.0, noise=0.1)
        self.history: List[Dict[str, Any]] = []
        self._load_model()
    
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.history = data.get('history', [])
                    if self.history:
                        X = np.array([[h['budget']] for h in self.history])
                        y = np.array([h['roi'] for h in self.history])
                        self.gp.fit(X, y)
                logger.info(f"Loaded Bayesian budget optimizer from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
    
    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({'history': self.history}, f)
    
    def _expected_improvement(
        self, 
        X: np.ndarray, 
        best_y: float,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        Calculate Expected Improvement acquisition function.
        """
        mu, sigma = self.gp.predict(X)
        
        # Handle zero variance
        with np.errstate(divide='warn'):
            improvement = mu - best_y - xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def suggest_budget(
        self,
        campaign_id: int,
        current_budget: float,
        budget_range: Tuple[float, float] = None,
        n_suggestions: int = 5
    ) -> BudgetAllocation:
        """
        Suggest optimal budget using Bayesian optimization.
        """
        if budget_range is None:
            budget_range = (current_budget * 0.5, current_budget * 2.0)
        
        # If no history, return current budget with exploration bonus
        if len(self.history) < 3:
            # Exploration: try a slightly higher budget
            suggested = current_budget * 1.1
            return BudgetAllocation(
                campaign_id=campaign_id,
                current_budget=current_budget,
                recommended_budget=round(suggested, 2),
                expected_roi=0.0,
                confidence=0.3,
                reasoning="Insufficient data. Exploring with 10% budget increase."
            )
        
        # Get best observed ROI
        best_roi = max(h['roi'] for h in self.history)
        
        # Generate candidate budgets
        candidates = np.linspace(budget_range[0], budget_range[1], 100).reshape(-1, 1)
        
        # Calculate Expected Improvement
        ei = self._expected_improvement(candidates, best_roi)
        
        # Select budget with highest EI
        best_idx = np.argmax(ei)
        suggested_budget = candidates[best_idx, 0]
        
        # Get predicted ROI and uncertainty
        mu, sigma = self.gp.predict(np.array([[suggested_budget]]))
        
        confidence = 1 / (1 + sigma[0])  # Higher uncertainty = lower confidence
        
        return BudgetAllocation(
            campaign_id=campaign_id,
            current_budget=current_budget,
            recommended_budget=round(suggested_budget, 2),
            expected_roi=round(mu[0], 4),
            confidence=round(confidence, 2),
            reasoning=f"Bayesian optimization suggests budget ${suggested_budget:.2f} "
                      f"with expected ROI {mu[0]:.2%} (±{sigma[0]:.2%})"
        )
    
    def record_observation(
        self,
        campaign_id: int,
        budget: float,
        roi: float
    ):
        """Record an observation of budget -> ROI."""
        self.history.append({
            'campaign_id': campaign_id,
            'budget': budget,
            'roi': roi
        })
        
        # Refit GP
        X = np.array([[h['budget']] for h in self.history])
        y = np.array([h['roi'] for h in self.history])
        self.gp.fit(X, y)
        
        self._save_model()
    
    def optimize_portfolio(
        self,
        campaigns: List[Dict[str, Any]],
        total_budget: float
    ) -> List[BudgetAllocation]:
        """
        Optimize budget allocation across multiple campaigns.
        
        Args:
            campaigns: List of campaign dicts with 'id', 'current_budget', 'historical_roi'
            total_budget: Total budget to allocate
        """
        n_campaigns = len(campaigns)
        
        if n_campaigns == 0:
            return []
        
        # Simple portfolio optimization using historical ROI
        total_roi = sum(c.get('historical_roi', 1.0) for c in campaigns)
        
        allocations = []
        remaining_budget = total_budget
        
        for i, campaign in enumerate(campaigns):
            historical_roi = campaign.get('historical_roi', 1.0)
            current_budget = campaign.get('current_budget', 0)
            
            # Allocate proportionally to ROI
            if i < n_campaigns - 1:
                allocation = total_budget * (historical_roi / total_roi)
                remaining_budget -= allocation
            else:
                allocation = remaining_budget
            
            # Constrain to reasonable range
            min_budget = current_budget * 0.5
            max_budget = current_budget * 2.0
            allocation = max(min_budget, min(allocation, max_budget))
            
            change_pct = ((allocation - current_budget) / current_budget * 100) if current_budget > 0 else 0
            
            allocations.append(BudgetAllocation(
                campaign_id=campaign['id'],
                current_budget=current_budget,
                recommended_budget=round(allocation, 2),
                expected_roi=historical_roi,
                confidence=0.7,
                reasoning=f"Portfolio allocation based on historical ROI. "
                          f"{'Increasing' if change_pct > 0 else 'Decreasing'} by {abs(change_pct):.1f}%"
            ))
        
        return allocations


class SpendPacer:
    """
    Intelligent spend pacing to optimize budget utilization throughout the day.
    """
    
    def __init__(self):
        self.hourly_multipliers = np.ones(24)
        self.learned_patterns: Dict[int, np.ndarray] = {}  # campaign_id -> hourly performance
    
    def learn_patterns(
        self,
        campaign_id: int,
        hourly_performance: List[Dict[str, Any]]
    ):
        """
        Learn hourly performance patterns for a campaign.
        
        hourly_performance: List of {hour, impressions, clicks, sales, spend}
        """
        pattern = np.zeros(24)
        counts = np.zeros(24)
        
        for record in hourly_performance:
            hour = record.get('hour', 0) % 24
            sales = record.get('sales', 0)
            spend = record.get('spend', 1)
            
            if spend > 0:
                roi = sales / spend
                pattern[hour] += roi
                counts[hour] += 1
        
        # Average and normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            pattern = np.where(counts > 0, pattern / counts, pattern.mean())
        
        # Normalize to sum to 24 (like 24 hours)
        if pattern.sum() > 0:
            pattern = pattern / pattern.sum() * 24
        
        self.learned_patterns[campaign_id] = pattern
    
    def get_bid_multiplier(
        self,
        campaign_id: int,
        hour: int
    ) -> float:
        """
        Get bid multiplier for given hour.
        Higher multiplier during high-performance hours.
        """
        if campaign_id not in self.learned_patterns:
            return 1.0
        
        pattern = self.learned_patterns[campaign_id]
        return float(pattern[hour % 24])
    
    def get_pacing_schedule(
        self,
        campaign_id: int,
        daily_budget: float
    ) -> List[Dict[str, Any]]:
        """
        Get recommended hourly budget pacing.
        """
        if campaign_id not in self.learned_patterns:
            # Even distribution
            hourly_budget = daily_budget / 24
            return [
                {'hour': h, 'budget': round(hourly_budget, 2), 'multiplier': 1.0}
                for h in range(24)
            ]
        
        pattern = self.learned_patterns[campaign_id]
        
        schedule = []
        for hour in range(24):
            multiplier = pattern[hour]
            hourly_budget = daily_budget * (multiplier / 24)
            schedule.append({
                'hour': hour,
                'budget': round(hourly_budget, 2),
                'multiplier': round(multiplier, 2)
            })
        
        return schedule
