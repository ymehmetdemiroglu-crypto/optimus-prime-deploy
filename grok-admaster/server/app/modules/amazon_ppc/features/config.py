"""
Feature Configuration Store.
Centralizes the definition of ML features to ensure consistency across 
Feature Engineering, Training, and Inference.
"""
from typing import List

class FeatureConfig:
    """
    Central configuration for ML features.
    Defines the exact list and order of features used by ML models.
    """
    
    # Core performance metrics (Rolling windows)
    PERFORMANCE_FEATURES = [
        'ctr_7d', 'ctr_14d', 'ctr_30d',
        'conversion_rate_7d', 'conversion_rate_14d', 'conversion_rate_30d',
        'acos_7d', 'acos_14d', 'acos_30d',
        'roas_7d', 'roas_14d', 'roas_30d',
        'cpc_7d', 'cpc_14d', 'cpc_30d',
    ]
    
    # Trend and Momentum metrics
    TREND_FEATURES = [
        'spend_trend', 
        'sales_trend', 
        'ctr_trend', 
        'momentum'
    ]
    
    # Volatility and Market metrics
    MARKET_FEATURES = [
        'cpc_volatility', 
        'impression_volatility'
    ]
    
    # Temporal / Seasonality features
    TEMPORAL_FEATURES = [
        'day_of_week', 
        'is_weekend', 
        'month', 
        'quarter', 
        'is_q4'
    ]
    
    # Contextual features
    CONTEXT_FEATURES = [
        'current_bid', 
        'revenue_per_click', 
        'data_maturity'
    ]
    
    # MASTER LIST - Order matters for Neural Networks!
    # This list defines the input vector structure.
    MODEL_FEATURES = (
        PERFORMANCE_FEATURES +
        TREND_FEATURES +
        MARKET_FEATURES +
        TEMPORAL_FEATURES +
        CONTEXT_FEATURES
    )
    
    @classmethod
    def get_feature_index(cls, feature_name: str) -> int:
        """Get the index of a feature in the input vector."""
        try:
            return cls.MODEL_FEATURES.index(feature_name)
        except ValueError:
            return -1
            
    @classmethod
    def get_input_dimension(cls) -> int:
        """Get the total size of the feature vector."""
        return len(cls.MODEL_FEATURES)
