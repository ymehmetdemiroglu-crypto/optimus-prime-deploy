"""
Learning Rate Adapter for Optimus Pryme Meta-Learner
Adjusts exploration vs exploitation based on market conditions.
"""

from typing import Dict, Any

class LearningRateAdapter:
    def __init__(self, 
                 base_exploration: float = 0.2, 
                 min_exploration: float = 0.05, 
                 max_exploration: float = 0.5):
        self.base_exploration = base_exploration
        self.min_exploration = min_exploration
        self.max_exploration = max_exploration


    def calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate a volatility score from 0.0 (stable) to 1.0 (chaos).
        """
        # Normalize inputs
        cpc_variance = float(market_data.get("cpc_variance", 0))
        competitor_changes = float(market_data.get("competitor_changes_count", 0))
        conversion_std = float(market_data.get("conversion_rate_std", 0))
        
        # Sigmoid-like normalization to cap impacts
        # e.g. 5 competitor changes comes 'high' volatility
        comp_score = min(competitor_changes / 5.0, 1.0)
        cpc_score = min(cpc_variance, 1.0) # Assuming variance is already normalized or small
        cv_score = min(conversion_std * 10, 1.0) # Std of 0.1 (10%) -> 1.0
        
        # Weighted sum
        volatility = (cpc_score * 0.4) + (comp_score * 0.3) + (cv_score * 0.3)
        
        return round(min(max(volatility, 0.0), 1.0), 3)

    def adapt_learning_rate(self, market_data: Dict[str, Any], system_confidence: float) -> Dict[str, Any]:
        """
        Determine optimal exploration rate.
        
        Logic:
        - High Volatility -> Low Exploration (Stick to what works, safety first)
        - Low Volatility -> High Exploration (Safe to experiment)
        - High Confidence -> Lower Exploration (We know the answer)
        - Low Confidence -> Higher Exploration (Need to learn)
        """
        volatility = self.calculate_volatility(market_data)
        
        # Base adjustment purely on volatility
        # Invert: High volatility = Low exploration factor
        volatility_factor = 1.0 - volatility
        
        # Adjust based on confidence
        # Low confidence adds to exploration desire
        confidence_factor = 1.0 - system_confidence
        
        # Combined logic
        # Start with base
        rate = self.base_exploration
        
        # If volatile, dampen exploration significantly
        if volatility > 0.6:
            rate = max(rate * 0.5, self.min_exploration)
        elif volatility < 0.2:
            rate = min(rate * 1.5, self.max_exploration)
            
        # If we are unsure (low confidence), boost exploration
        if system_confidence < 0.4:
            rate = min(rate + 0.1, self.max_exploration)
            
        # Clamp to bounds
        final_exploration = round(max(min(rate, self.max_exploration), self.min_exploration), 4)
        final_exploitation = round(1.0 - final_exploration, 4)
        
        return {
            "exploration_rate": final_exploration,
            "exploitation_rate": final_exploitation,
            "volatility_score": volatility,
            "confidence_score": round(system_confidence, 2),
            "reasoning": self._generate_reasoning(final_exploration, volatility, system_confidence)
        }
        
    def _generate_reasoning(self, rate: float, volatility: float, confidence: float) -> str:
        if volatility > 0.6:
            return f"High market volatility ({volatility:.2f}). Reduced exploration ({rate:.1%}) to minimize risk."
        elif confidence < 0.4:
            return f"Low system confidence ({confidence:.2f}). Increased exploration ({rate:.1%}) to accelerate learning."
        elif volatility < 0.2 and rate > 0.3:
            return f"Stable market ({volatility:.2f}). Aggressive exploration ({rate:.1%}) enabled to find new opportunities."
        else:
            return f"Balanced learning approach (Exploration: {rate:.1%}) suitable for current conditions."
