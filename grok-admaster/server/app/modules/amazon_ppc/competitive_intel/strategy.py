import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class UndercutPredictor:
    """
    Predicts probability of a competitor undercutting our price.
    Uses Gradient Boosting (simulating XGBoost logic).
    """
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.is_trained = False
        
        # Mapping features
        self.feature_names = [
            "price_gap_percent",     # (Our Price - Their Price) / Their Price
            "days_since_last_change",
            "is_weekend",
            "competitor_inventory_level", # Estimated (0=low, 1=high)
            "category_demand_index"       # 0-100
        ]

    def train_model(self, X: List[List[float]], y: List[int]):
        """
        Train the classifier.
        X: List of features
        y: List of 0/1 (did they undercut?)
        """
        if len(X) < 10:
            logger.warning("Not enough data to train undercut model")
            return

        try:
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("Undercut predictor trained successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")

    def predict_probability(self, features: List[float]) -> float:
        """Returns probability (0.0 - 1.0) of undercut."""
        if not self.is_trained:
            # Fallback heuristic if no model
            # If price gap is large (>10%), high probability they cut
            price_gap = features[0]
            if price_gap > 0.10: return 0.8
            if price_gap < -0.05: return 0.2
            return 0.5 

        try:
            # Predict for single sample
            prob = self.model.predict_proba([features])[0][1] # Probability of class 1 (Yes)
            return float(prob)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

class GameTheorySimulator:
    """
    Simulates strategic interactions (Nash Equilibrium).
    """
    
    def solve_nash_equilibrium(self, 
                             my_costs: float, 
                             their_costs: float, 
                             current_price: float,
                             demand_elasticity: float = 1.5) -> Dict[str, Any]:
        """
        Simulate a 2x2 Pricing Game.
        Strategies: {High Price, Low Price}
        High = Current Price
        Low = Current Price * 0.9 (10% cut)
        
        Payoff = Profit = (Price - Cost) * Volume
        Volume changes based on relative price.
        """
        
        high_price = current_price
        low_price = current_price * 0.9
        
        base_volume = 1000 # arbitrary daily units
        
        # Payoff Matrix Calculation
        # Cell 1: (High, High) - Split market 50/50
        profit_hh_me = (high_price - my_costs) * (base_volume / 2)
        profit_hh_them = (high_price - their_costs) * (base_volume / 2)
        
        # Cell 2: (High, Low) - They steal share (e.g., 80/20 split)
        # Elasticity simplified: Lower price captures more share
        share_steal = 0.8
        profit_hl_me = (high_price - my_costs) * (base_volume * (1-share_steal))
        profit_hl_them = (low_price - their_costs) * (base_volume * share_steal * 1.2) # +20% total demand due to lower price
        
        # Cell 3: (Low, High) - I steal share
        profit_lh_me = (low_price - my_costs) * (base_volume * share_steal * 1.2)
        profit_lh_them = (high_price - their_costs) * (base_volume * (1-share_steal))
        
        # Cell 4: (Low, Low) - Split market 50/50 but lower margin, slightly higher total vol
        profit_ll_me = (low_price - my_costs) * (base_volume * 1.2 / 2)
        profit_ll_them = (low_price - their_costs) * (base_volume * 1.2 / 2)
        
        matrix = {
            "HH": (profit_hh_me, profit_hh_them),
            "HL": (profit_hl_me, profit_hl_them),
            "LH": (profit_lh_me, profit_lh_them),
            "LL": (profit_ll_me, profit_ll_them)
        }
        
        # Find Nash Equilibrium (Pure Strategy)
        # Check My Best Responses
        # If they play H: I prefer (LH if LH > HH else HH)
        # If they play L: I prefer (LL if LL > HL else HL)
        
        my_best_response_to_H = "Low" if profit_lh_me > profit_hh_me else "High"
        my_best_response_to_L = "Low" if profit_ll_me > profit_hl_me else "High"
        
        their_best_response_to_H = "Low" if profit_hl_them > profit_hh_them else "High"
        their_best_response_to_L = "Low" if profit_ll_them > profit_lh_them else "High"
        
        equilibrium = []
        if my_best_response_to_H == "High" and their_best_response_to_H == "High": equilibrium.append("High-High")
        if my_best_response_to_H == "High" and their_best_response_to_L == "Low":  equilibrium.append("High-Low") # Unstable usually
        if my_best_response_to_L == "Low" and their_best_response_to_H == "High":  equilibrium.append("Low-High")
        if my_best_response_to_L == "Low" and their_best_response_to_L == "Low":   equilibrium.append("Low-Low")
        
        recommended_strategy = "High" # Default
        if "Low-Low" in equilibrium:
            # Prisoner's Dilemma detected
            recommended_strategy = "Differentiate" # Don't play the price game
        elif "High-High" in equilibrium:
            recommended_strategy = "Maintain"
        else:
             # Mixed or asymmetric
             recommended_strategy = my_best_response_to_L # Assume worst case (maximin)
             
        return {
            "matrix": matrix,
            "equilibria": equilibrium,
            "recommendation": recommended_strategy,
            "expected_payoff": profit_hh_me if recommended_strategy == "Maintain" else profit_ll_me
        }
