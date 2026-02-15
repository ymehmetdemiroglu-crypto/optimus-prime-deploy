from dataclasses import dataclass
from typing import Optional

@dataclass
class BidStrategyConfig:
    """
    Configuration for Bid Optimization Strategy.
    Allows dynamic adjustment of learning parameters and safety rails.
    """
    target_acos: float = 25.0
    target_roas: float = 4.0
    
    # Safety Rails
    min_bid: float = 0.10
    max_bid_cap: float = 5.00  # Absolute max cap
    max_bid_increase_factor: float = 1.5  # Max 50% increase per update
    max_bid_decrease_factor: float = 0.7  # Max 30% decrease per update
    
    # Learning Parameters
    sufficient_data_clicks: int = 10
    lookback_window_days: int = 30
    min_training_samples: int = 50 # Minimum samples required to attempt training
    
    # Outcome values
    high_acos_threshold: float = 30.0
    low_acos_threshold: float = 15.0
    
    # Meta-Learning Controls
    enable_ml_prediction: bool = True
    enable_volatility_dampening: bool = True
