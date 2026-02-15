"""
Bid Optimization Model using Gradient Boosting.
Predicts optimal bids based on historical performance features.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
import numpy as np

# Try importing joblib, strictly for compatibility if needed later, 
# though ModelStore handles persistence.
try:
    import joblib
except ImportError:
    joblib = None

from app.services.meta_skills.learning_rate_adapter import LearningRateAdapter
from app.modules.amazon_ppc.strategies.config import BidStrategyConfig
from app.modules.amazon_ppc.features.config import FeatureConfig
from app.modules.amazon_ppc.ml.market_response import MarketResponseModel

logger = logging.getLogger(__name__)

@dataclass
class BidPrediction:
    """Result of bid prediction."""
    keyword_id: int
    current_bid: float
    predicted_bid: float
    confidence: float
    expected_acos: float
    expected_roas: float
    reasoning: str

class BidOptimizer:
    """
    Gradient Boosting-based bid optimization model.
    Predicts optimal bids to achieve target ACoS/ROAS.
    """
    
    # Feature columns expected by the model
    FEATURE_COLS = FeatureConfig.MODEL_FEATURES
    
    FEATURE_COLS = FeatureConfig.MODEL_FEATURES

    def __init__(
        self, 
        model_artifact: Any = None, 
        learning_adapter: Optional[LearningRateAdapter] = None,
        config: Optional[BidStrategyConfig] = None
    ):
        """
        Initialize Predictive Bid Optimizer.
        """
        # Load or Init Market Model
        self.market_model = model_artifact if isinstance(model_artifact, MarketResponseModel) else MarketResponseModel()
        
        self.learning_adapter = learning_adapter or LearningRateAdapter()
        self.config = config or BidStrategyConfig()

    def train(
        self, 
        training_data: List[Dict[str, Any]],
        target_col: str = 'optimal_bid' # Deprecated, but kept for interface compat
    ) -> Dict[str, Any]:
        """
        Train the Market Response Model.
        """
        if len(training_data) < self.config.min_training_samples:
            logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return {}
            
        logger.info(f"Training Market Response Model on {len(training_data)} samples...")
        self.market_model.train(training_data, self.FEATURE_COLS)
        
        return {"status": "trained", "samples": len(training_data)}
    
    def predict_bid(
        self, 
        features: Dict[str, Any],
        config: Optional[BidStrategyConfig] = None
    ) -> BidPrediction:
        """
        Optimize bid by solving for the maximum utility using the Market Model.
        """
        cfg = config or self.config
        
        current_bid = float(features.get('current_bid', 1.0))
        # Ensure price is available, else default to revenue_per_click * clicks (approx) or fallback
        # Ideally passed in features. Logic assumes 'revenue_per_click' ~= Price * CVR if we don't have Price.
        # Let's derive Price estimate: Price = RPC / CVR (if available) or use RPC directly for Revenue calc.
        
        rpc = float(features.get('revenue_per_click', 0))
        
        if not self.market_model.is_trained or not cfg.enable_ml_prediction:
            # Fallback to Rules
            pred, reasoning = self._rule_based_prediction(current_bid, rpc, features.get('acos_30d', 0), cfg.target_acos, cfg)
            return BidPrediction(
                keyword_id=features.get('keyword_id', 0),
                current_bid=current_bid,
                predicted_bid=round(pred, 2),
                confidence=0.5,
                expected_acos=0,
                expected_roas=0,
                reasoning=reasoning
            )

        # Solver: Grid Search around current bid
        # Search range: -50% to +100% of current bid, bounded by min/max caps
        search_space = np.linspace(
            max(cfg.min_bid, current_bid * 0.5), 
            min(cfg.max_bid_cap, current_bid * 2.0), 
            num=20
        )
        
        best_bid = current_bid
        best_score = -float('inf')
        best_metrics = (0, 0)
        
        target_acos_decimal = cfg.target_acos / 100.0
        
        # Calculate Price Proxy for Revenue Estimation
        # Revenue = Price * Conversions
        # Conversions = Clicks * CVR
        # We need Price. If features doesn't have it, we estimate from RPC.
        # RPC = (Price * Orders) / Clicks = Price * CVR. So Price = RPC / CVR.
        # If we can't estimate price, we can't sim revenue unless we assume RPC is constant? 
        # But RPC is NOT constant (CPC changes). 
        # WAIT. Revenue per Conversion (AOV) is roughly constant. 
        # So 'Price' here means AOV.
        
        # Estimate AOV
        hist_cvr = features.get('conversion_rate_30d', 0.1) # Default 10%
        if hist_cvr == 0: hist_cvr = 0.01
        
        aov = rpc / hist_cvr if rpc > 0 else 25.0 # Fallback $25
        
        for bid_candidate in search_space:
            preds = self.market_model.predict(features, bid_candidate)
            
            # Simulated Outcomes
            # We assume 1000 impressions to normalize the scale
            sim_imps = 1000 
            sim_clicks = sim_imps * preds.predicted_ctr
            sim_cost = sim_clicks * preds.predicted_cpc
            sim_conversions = sim_clicks * preds.predicted_cvr
            sim_revenue = sim_conversions * aov
            
            # Objective Function: Maximize Profit while keeping ACoS <= Target
            # Or: Maximize Revenue s.t. ACoS <= Target
            # Let's use a soft constraint penalty
            
            sim_acos = (sim_cost / sim_revenue) if sim_revenue > 0 else 999.0
            sim_profit = sim_revenue - sim_cost
            
            # Utility Score
            if sim_acos <= target_acos_decimal:
                # Inside target: Utility is Profit (or Revenue depending on goal)
                # Let's say we want to Maximize Profit
                score = sim_profit
            else:
                # Violation: Penalize heavily
                # Penalty proportional to how far off we are
                score = sim_profit - (sim_cost * (sim_acos - target_acos_decimal) * 2.0)
            
            if score > best_score:
                best_score = score
                best_bid = bid_candidate
                best_metrics = (sim_acos * 100, (sim_revenue/sim_cost) if sim_cost > 0 else 0)

        return BidPrediction(
            keyword_id=features.get('keyword_id', 0),
            current_bid=current_bid,
            predicted_bid=round(best_bid, 2),
            confidence=0.8, # Higher confidence in simulator
            expected_acos=round(best_metrics[0], 2),
            expected_roas=round(best_metrics[1], 2),
            reasoning=f"Predictive Solver: Max Profit @ ACoS {best_metrics[0]:.1f}%"
        )
    
    def batch_predict(self, feature_list, config=None):
        return [self.predict_bid(f, config) for f in feature_list]    

    def _rule_based_prediction(
        self, 
        current_bid: float, 
        rpc: float, 
        current_acos: float,
        target_acos: float,
        config: BidStrategyConfig
    ) -> Tuple[float, str]:
        """Rule-based bid calculation when ML model unavailable."""
        
        if rpc > 0:
            # Revenue-based bid: Bid = RevenuePerClick * TargetACoS
            optimal_bid = rpc * (target_acos / 100)
            reasoning = f"Target ACoS bid: RPC ${rpc:.2f} × {target_acos}%"
            # Apply safety caps
            optimal_bid = min(optimal_bid, current_bid * config.max_bid_increase_factor)
            optimal_bid = max(optimal_bid, current_bid * config.max_bid_decrease_factor)
            return optimal_bid, reasoning
        
        if current_acos > config.high_acos_threshold:
            # ACoS too high - reduce bid
            reduction = 0.20 # 20% cut
            new_bid = current_bid * (1 - reduction)
            reasoning = f"Reduce bid {reduction*100:.0f}% (High ACoS {current_acos:.1f}%)"
            return max(config.min_bid, new_bid), reasoning
        
        if current_acos < config.low_acos_threshold and current_acos > 0:
            # ACoS low - room to increase
            increase = 0.20
            new_bid = current_bid * (1 + increase)
            reasoning = f"Increase bid {increase*100:.0f}% (Low ACoS {current_acos:.1f}%)"
            return min(new_bid, config.max_bid_cap), reasoning
        
        # Maintain current bid
        return current_bid, "Maintain bid - within target range"
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.market_model.is_trained:
            return {}
        
        # Aggregate importance from sub-models (simplified)
        return dict(zip(self.FEATURE_COLS, self.market_model.ctr_model.feature_importances_))
