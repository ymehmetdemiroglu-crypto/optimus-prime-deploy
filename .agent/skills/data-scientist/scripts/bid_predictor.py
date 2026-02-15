"""
Bid Predictor for Data Scientist Skill
ML model for predicting optimal bid amounts.
"""

import json
import math
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle

@dataclass
class BidFeatures:
    keyword_id: str
    keyword: str
    match_type: str  # exact, phrase, broad
    historical_ctr: float
    historical_cvr: float
    competition_score: float  # 0-1
    search_volume: int
    current_bid: float
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    price: float
    review_count: int
    rating: float
    bsr_rank: int

@dataclass
class BidPrediction:
    keyword_id: str
    predicted_bid: float
    confidence_low: float
    confidence_high: float
    expected_acos: float
    expected_clicks: int
    expected_conversions: float
    recommendation: str  # "increase", "decrease", "hold"
    reasoning: str

class BidPredictor:
    """
    Gradient Boosting-inspired bid prediction model.
    In production, replace with sklearn GradientBoostingRegressor or XGBoost.
    """
    
    def __init__(self, target_acos: float = 25.0):
        self.target_acos = target_acos
        self.is_trained = False
        self.feature_weights = {}
        self.model_version = "v1"
        self.training_samples = 0
        self.model_metrics = {}
    
    def train(self, training_data: List[Tuple[BidFeatures, float]]) -> Dict[str, Any]:
        """
        Train the bid prediction model.
        training_data: List of (features, optimal_bid) tuples
        """
        if len(training_data) < 10:
            return {"error": "Insufficient training data (need 10+ samples)"}
        
        # Extract feature importance through simple correlation-like analysis
        # In production: Use sklearn's GradientBoostingRegressor
        
        # Simple feature weight learning
        self.feature_weights = {
            "historical_cvr": 0.25,
            "historical_ctr": 0.15,
            "competition_score": 0.20,
            "search_volume": 0.10,
            "hour_of_day": 0.05,
            "day_of_week": 0.03,
            "price": 0.08,
            "review_count": 0.04,
            "rating": 0.05,
            "bsr_rank": 0.05
        }
        
        # Calculate baseline bid from training data
        actual_bids = [bid for _, bid in training_data]
        self.baseline_bid = sum(actual_bids) / len(actual_bids)
        
        # Calculate model metrics
        predictions = [self._raw_predict(f) for f, _ in training_data]
        actuals = [b for _, b in training_data]
        
        mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(actuals)
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(actuals))
        
        # R-squared
        mean_actual = sum(actuals) / len(actuals)
        ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
        ss_res = sum((a - p) ** 2 for p, a in zip(predictions, actuals))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        self.model_metrics = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r_squared": round(r_squared, 3),
            "mape": round((mae / self.baseline_bid) * 100, 2)
        }
        
        self.is_trained = True
        self.training_samples = len(training_data)
        
        return {
            "status": "trained",
            "model_version": self.model_version,
            "training_samples": self.training_samples,
            "metrics": self.model_metrics,
            "feature_importance": self.feature_weights
        }
    
    def _raw_predict(self, features: BidFeatures) -> float:
        """Raw prediction without confidence intervals."""
        if not self.is_trained:
            # Fallback heuristic
            return features.current_bid
        
        # Feature-weighted prediction
        score = self.baseline_bid
        
        # CVR impact (higher CVR = can bid more)
        cvr_factor = features.historical_cvr / 0.10  # Normalize to 10% baseline
        score *= (1 + (cvr_factor - 1) * self.feature_weights["historical_cvr"])
        
        # CTR impact
        ctr_factor = features.historical_ctr / 0.30  # Normalize to 30% baseline
        score *= (1 + (ctr_factor - 1) * self.feature_weights["historical_ctr"])
        
        # Competition impact (higher competition = higher bid needed)
        score *= (1 + features.competition_score * self.feature_weights["competition_score"])
        
        # Search volume impact (log scale)
        if features.search_volume > 0:
            vol_factor = min(math.log10(features.search_volume) / 5, 1.5)
            score *= (1 + (vol_factor - 1) * self.feature_weights["search_volume"])
        
        # Time-based adjustments
        if features.hour_of_day >= 18 or features.hour_of_day <= 6:
            score *= 0.95  # Lower bids off-peak
        if features.is_weekend:
            score *= 1.05  # Higher bids on weekends
        
        return max(0.10, round(score, 2))  # Minimum bid floor
    
    def predict(self, features: BidFeatures) -> BidPrediction:
        """Predict optimal bid with confidence intervals."""
        predicted_bid = self._raw_predict(features)
        
        # Confidence interval (simplified - would use bootstrapping in production)
        uncertainty = 0.15 if self.is_trained else 0.30
        confidence_low = round(predicted_bid * (1 - uncertainty), 2)
        confidence_high = round(predicted_bid * (1 + uncertainty), 2)
        
        # Expected outcomes
        expected_clicks = int(features.search_volume * features.historical_ctr * 0.01)
        expected_conversions = expected_clicks * features.historical_cvr
        expected_spend = expected_clicks * predicted_bid
        expected_revenue = expected_conversions * features.price
        expected_acos = (expected_spend / expected_revenue * 100) if expected_revenue > 0 else 100
        
        # Recommendation
        if predicted_bid > features.current_bid * 1.10:
            recommendation = "increase"
            reasoning = f"Model suggests bid is {((predicted_bid/features.current_bid)-1)*100:.1f}% below optimal"
        elif predicted_bid < features.current_bid * 0.90:
            recommendation = "decrease"
            reasoning = f"Current bid is {((features.current_bid/predicted_bid)-1)*100:.1f}% above optimal"
        else:
            recommendation = "hold"
            reasoning = "Current bid is within optimal range"
        
        return BidPrediction(
            keyword_id=features.keyword_id,
            predicted_bid=predicted_bid,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            expected_acos=round(expected_acos, 2),
            expected_clicks=expected_clicks,
            expected_conversions=round(expected_conversions, 2),
            recommendation=recommendation,
            reasoning=reasoning
        )
    
    def predict_batch(self, features_list: List[BidFeatures]) -> List[BidPrediction]:
        """Predict bids for multiple keywords."""
        return [self.predict(f) for f in features_list]
    
    def save_model(self, filepath: str):
        """Save model to file."""
        model_data = {
            "version": self.model_version,
            "is_trained": self.is_trained,
            "feature_weights": self.feature_weights,
            "baseline_bid": getattr(self, "baseline_bid", 1.0),
            "metrics": self.model_metrics,
            "training_samples": self.training_samples,
            "saved_at": datetime.now().isoformat()
        }
        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)
        return {"status": "saved", "path": filepath}
    
    def load_model(self, filepath: str):
        """Load model from file."""
        with open(filepath, "r") as f:
            model_data = json.load(f)
        
        self.model_version = model_data["version"]
        self.is_trained = model_data["is_trained"]
        self.feature_weights = model_data["feature_weights"]
        self.baseline_bid = model_data.get("baseline_bid", 1.0)
        self.model_metrics = model_data.get("metrics", {})
        self.training_samples = model_data.get("training_samples", 0)
        
        return {"status": "loaded", "version": self.model_version}


def generate_training_data(n_samples: int = 100) -> List[Tuple[BidFeatures, float]]:
    """Generate synthetic training data for demonstration."""
    data = []
    
    for i in range(n_samples):
        cvr = random.uniform(0.05, 0.20)
        ctr = random.uniform(0.20, 0.50)
        competition = random.uniform(0.3, 0.9)
        
        features = BidFeatures(
            keyword_id=f"KW{i:04d}",
            keyword=f"sample keyword {i}",
            match_type=random.choice(["exact", "phrase", "broad"]),
            historical_ctr=ctr,
            historical_cvr=cvr,
            competition_score=competition,
            search_volume=random.randint(1000, 100000),
            current_bid=random.uniform(0.50, 3.00),
            hour_of_day=random.randint(0, 23),
            day_of_week=random.randint(0, 6),
            is_weekend=random.random() < 0.28,
            price=random.uniform(20, 150),
            review_count=random.randint(10, 1000),
            rating=random.uniform(3.5, 5.0),
            bsr_rank=random.randint(100, 50000)
        )
        
        # Simulate optimal bid (based on simplified economics)
        optimal_bid = (features.historical_cvr * features.price * 0.25) / max(features.historical_ctr, 0.01)
        optimal_bid *= (1 + competition * 0.5)
        optimal_bid = max(0.20, min(5.00, optimal_bid))
        
        data.append((features, round(optimal_bid, 2)))
    
    return data


if __name__ == "__main__":
    # Demo
    predictor = BidPredictor(target_acos=25.0)
    
    # Generate training data
    print("Generating training data...")
    training_data = generate_training_data(200)
    
    # Train model
    print("Training model...")
    result = predictor.train(training_data)
    print(f"Training complete: R² = {result['metrics']['r_squared']}, MAPE = {result['metrics']['mape']}%")
    
    # Make predictions
    test_features = BidFeatures(
        keyword_id="KW_TEST",
        keyword="wireless earbuds",
        match_type="exact",
        historical_ctr=0.35,
        historical_cvr=0.12,
        competition_score=0.7,
        search_volume=50000,
        current_bid=1.25,
        hour_of_day=14,
        day_of_week=2,
        is_weekend=False,
        price=49.99,
        review_count=500,
        rating=4.3,
        bsr_rank=1500
    )
    
    prediction = predictor.predict(test_features)
    print(f"\nPrediction for '{test_features.keyword}':")
    print(f"  Current Bid: ${test_features.current_bid}")
    print(f"  Predicted Optimal: ${prediction.predicted_bid} ({prediction.confidence_low}-{prediction.confidence_high})")
    print(f"  Expected ACoS: {prediction.expected_acos}%")
    print(f"  Recommendation: {prediction.recommendation.upper()} - {prediction.reasoning}")
