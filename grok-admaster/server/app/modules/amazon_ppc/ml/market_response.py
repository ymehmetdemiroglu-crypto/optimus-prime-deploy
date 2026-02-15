"""
Market Response Model.
Predicts market outcomes (CTR, CVR, CPC) based on Context + Actions (Bid).
Used by the BidOptimizer to simulate "What-If" scenarios.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketPredictions:
    predicted_ctr: float
    predicted_cvr: float
    predicted_cpc: float

class MarketResponseModel:
    """
    Predicts core metrics given state and action.
    Inputs: Keyword Features, Bid
    Outputs: CTR, CVR, CPC
    """
    
    def __init__(self):
        # We use separate regressors because the relationships are different
        # CTR ~ Bid (Diminishing returns as rank saturates)
        # CPC ~ Bid (Linear/Log near 2nd price auction)
        # CVR ~ Bid (Weak correlation, mostly intrinsic, but traffic quality varies by rank)
        
        self.ctr_model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
        self.cvr_model = GradientBoostingRegressor(n_estimators=100, max_depth=3) # Simpler model for CVR
        self.cpc_model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
        
        self.is_trained = False
        self.training_columns = []

    def prepare_features(self, samples: List[Dict[str, Any]], feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract X (Features + Bid) and Y (CTR, CVR, CPC).
        """
        X = []
        Y = []
        
        for s in samples:
            # X: Context + Action
            row = [float(s.get(col, 0)) for col in feature_cols]
            bid = float(s.get('current_bid_setting', s.get('current_bid', 1.0)))
            row.append(bid) # Bid is the last feature
            X.append(row)
            
            # Y: Outcomes
            # Avoid division by zero in training data prep
            clicks = s.get('rolling_clicks', 0)
            imps = s.get('rolling_imps', 0)
            spend = s.get('rolling_spend', 0)
            orders = s.get('rolling_orders', 0) # Assumes we add this to ingestion
            # Fallback to calculated fields if raw not present
            ctr = s.get('ctr_30d', 0) if imps == 0 else (clicks / imps)
            cpc = s.get('cpc_30d', 0) if clicks == 0 else (spend / clicks)
            
            # Estimate CVR (Orders / Clicks)
            # If we don't have raw orders in 's', used derived conversion_rate if available
            cvr = s.get('conversion_rate_30d', 0)
            if 'daily_orders' in s and 'daily_clicks' in s: # Or rolling
                 cvr = orders / clicks if clicks > 0 else 0
            
            Y.append([ctr, cvr, cpc])
            
        return np.array(X), np.array(Y)

    def train(self, samples: List[Dict[str, Any]], feature_cols: List[str]):
        if not samples:
            logger.warning("No samples for MarketModel")
            return
            
        self.training_columns = feature_cols
        X, Y = self.prepare_features(samples, feature_cols)
        
        # Y columns: 0=CTR, 1=CVR, 2=CPC
        y_ctr = Y[:, 0]
        y_cvr = Y[:, 1]
        y_cpc = Y[:, 2]
        
        self.ctr_model.fit(X, y_ctr)
        self.cvr_model.fit(X, y_cvr)
        self.cpc_model.fit(X, y_cpc)
        
        self.is_trained = True
        
        # Log Metrics
        logger.info(f"Market Model Trained. CTR R2: {self.ctr_model.score(X, y_ctr):.3f}, CPC R2: {self.cpc_model.score(X, y_cpc):.3f}")

    def predict(self, features: Dict[str, Any], bid: float) -> MarketPredictions:
        if not self.is_trained:
            return MarketPredictions(0, 0, bid)
            
        # Construct vector
        row = [float(features.get(col, 0)) for col in self.training_columns]
        row.append(bid)
        X = np.array([row])
        
        ctr = float(self.ctr_model.predict(X)[0])
        cvr = float(self.cvr_model.predict(X)[0])
        cpc = float(self.cpc_model.predict(X)[0])
        
        # Safety Clamps
        ctr = max(0.0, min(ctr, 1.0))
        cvr = max(0.0, min(cvr, 1.0))
        cpc = max(0.01, cpc) 
        
        return MarketPredictions(ctr, cvr, cpc)
