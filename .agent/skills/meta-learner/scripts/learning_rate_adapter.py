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
        # Example metrics that contribute to volatility
        cpc_variance = market_data.get("cpc_variance", 0)
        competitor_changes = market_data.get("competitor_changes_count", 0)
        conversion_swings = market_data.get("conversion_rate_std", 0)
        
        # Simple weighted sum (normalized logic would be more complex)
        # Assuming inputs are already somewhat normalized
        volatility = (cpc_variance * 0.4) + (competitor_changes * 0.1) + (conversion_swings * 0.5)
        
        return min(max(volatility, 0.0), 1.0)

    def adapt_learning_rate(self, market_data: Dict[str, Any], system_confidence: float) -> Dict[str, float]:
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
            rate *= 0.5
        elif volatility < 0.2:
            rate *= 1.5
            
        # If we are unsure (low confidence), boost exploration
        if system_confidence < 0.4:
            rate += 0.1
            
        # Clamp to bounds
        final_exploration = max(min(rate, self.max_exploration), self.min_exploration)
        final_exploitation = 1.0 - final_exploration
        
        return {
            "exploration_rate": round(final_exploration, 2),
            "exploitation_rate": round(final_exploitation, 2),
            "volatility_score": round(volatility, 2),
            "confidence_score": round(system_confidence, 2),
            "reasoning": self._generate_reasoning(final_exploration, volatility, system_confidence)
        }
        
    def _generate_reasoning(self, rate: float, volatility: float, confidence: float) -> str:
        if volatility > 0.6:
            return "High market volatility estimated. Reducing exploration to protect budget measures."
        elif confidence < 0.4:
            return "Low system confidence in current strategy. Increasing exploration to discover better patterns."
        elif volatility < 0.2 and rate > 0.3:
            return "Stable market conditions detected. Increased experimentation allowed to aggressively find growth."
        else:
            return "Balanced learning approach suitable for current conditions."

if __name__ == "__main__":
    adapter = LearningRateAdapter()
    
    # Scenario 1: Stable market, new product (low confidence)
    s1 = {
        "market": {"cpc_variance": 0.1, "competitor_changes_count": 0, "conversion_rate_std": 0.1},
        "confidence": 0.3
    }
    
    result1 = adapter.adapt_learning_rate(s1["market"], s1["confidence"])
    print(f"Scenario 1 (Stable/Low Conf): Exp {result1['exploration_rate']} - {result1['reasoning']}")
    
    # Scenario 2: Volatile market, mature product (high confidence)
    s2 = {
        "market": {"cpc_variance": 0.8, "competitor_changes_count": 5, "conversion_rate_std": 0.7},
        "confidence": 0.9
    }
    
    result2 = adapter.adapt_learning_rate(s2["market"], s2["confidence"])
    print(f"Scenario 2 (Volatile/High Conf): Exp {result2['exploration_rate']} - {result2['reasoning']}")
