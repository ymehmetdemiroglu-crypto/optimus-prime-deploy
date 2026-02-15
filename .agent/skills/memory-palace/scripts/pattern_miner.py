"""
Pattern Miner - Detect recurring patterns in historical data
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json

class PatternMiner:
    """
    Mines historical data for recurring patterns
    """
    
    def __init__(self):
        self.patterns = []
        
    def detect_seasonal_patterns(
        self,
        historical_data: List[Dict[str, Any]],
        metric:str = "sales"
    ) -> List[Dict[str, Any]]:
        """
        Detect seasonal patterns in data
        
        Args:
            historical_data: List of data points with timestamps
            metric: Metric to analyze
            
        Returns:
            List of detected seasonal patterns
        """
        monthly_aggregates = defaultdict(list)
        
        for data_point in historical_data:
            timestamp = datetime.fromisoformat(data_point.get("timestamp", datetime.utcnow().isoformat()))
            month = timestamp.month
            value = data_point.get(metric, 0)
            
            monthly_aggregates[month].append(value)
        
        patterns = []
        avg_value = sum(sum(vals) for vals in monthly_aggregates.values()) / max(len(historical_data), 1)
        
        for month, values in monthly_aggregates.items():
            if not values:
                continue
                
            month_avg = sum(values) / len(values)
            multiplier = month_avg / avg_value if avg_value > 0 else 1.0
            
            if multiplier > 1.5 or multiplier < 0.7:  # Significant deviation
                patterns.append({
                    "pattern_type": "seasonal",
                    "pattern_signature": {
                        "month": month,
                        "metric": metric
                    },
                    "observed_effect": f"{multiplier:.1f}x baseline",
                    "occurrences": len(values),
                    "confidence": min(len(values) / 12.0, 1.0)  # Need multiple years for confidence
                })
        
        return patterns
    
    def detect_day_of_week_patterns(
        self,
        historical_data: List[Dict[str, Any]],
        metric: str = "ctr"
    ) -> List[Dict[str, Any]]:
        """
        Detect day-of-week performance patterns
        
        Args:
            historical_data: List of data points with timestamps
            metric: Metric to analyze
            
        Returns:
            List of detected day patterns
        """
        dow_aggregates = defaultdict(list)
        
        for data_point in historical_data:
            timestamp = datetime.fromisoformat(data_point.get("timestamp", datetime.utcnow().isoformat()))
            day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
            value = data_point.get(metric, 0)
            
            dow_aggregates[day_of_week].append(value)
        
        patterns = []
        avg_value = sum(sum(vals) for vals in dow_aggregates.values()) / max(len(historical_data), 1)
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for dow, values in dow_aggregates.items():
            if not values or len(values) < 4:  # Need at least 4 weeks
                continue
                
            dow_avg = sum(values) / len(values)
            multiplier = dow_avg / avg_value if avg_value > 0 else 1.0
            
            if multiplier > 1.2 or multiplier < 0.8:
                patterns.append({
                    "pattern_type": "day_of_week",
                    "pattern_signature": {
                        "day": day_names[dow],
                        "metric": metric
                    },
                    "observed_effect": f"{multiplier:.1f}x average",
                    "occurrences": len(values),
                    "recommendation": f"Adjust bids on {day_names[dow]}" if multiplier > 1.2 else f"Reduce spend on {day_names[dow]}"
                })
        
        return patterns
    
    def find_similar_scenarios(
        self,
        current_situation: Dict[str, Any],
        case_library: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar past scenarios using simple similarity scoring
        
        Args:
            current_situation: Current metrics and context
            case_library: Historical cases
            top_k: Number of similar cases to return
            
        Returns:
            Top K most similar cases with similarity scores
        """
        scored_cases = []
        
        current_metrics = current_situation.get("metrics", {})
        
        for case in case_library:
            past_metrics = case.get("scenario_metrics", {})
            similarity = self._calculate_similarity(current_metrics, past_metrics)
            
            scored_cases.append({
                "case": case,
                "similarity_score": similarity
            })
        
        # Sort by similarity descending
        scored_cases.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return scored_cases[:top_k]
    
    def _calculate_similarity(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float]
    ) -> float:
        """
        Simple cosine-like similarity between two metric dictionaries
        
        Args:
            metrics1: First metric set
            metrics2: Second metric set
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        common_keys = set(metrics1.keys()) & set(metrics2.keys())
        
        if not common_keys:
            return 0.0
        
        # Normalize and compare
        similarities = []
        for key in common_keys:
            v1 = metrics1[key]
            v2 = metrics2[key]
            
            if v1 == 0 and v2 == 0:
                similarities.append(1.0)
            elif v1 == 0 or v2 == 0:
                similarities.append(0.0)
            else:
                ratio = min(v1, v2) / max(v1, v2)
                similarities.append(ratio)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def learn_user_preference(
        self,
        user_action: str,
        recommendation_context: Dict[str, Any],
        existing_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user preferences based on actions
        
        Args:
            user_action: 'accepted' or 'rejected'
            recommendation_context: Details of the recommendation
            existing_preferences: Current preference state
            
        Returns:
            Updated preferences
        """
        strategy = recommendation_context.get("strategy", "unknown")
        
        if strategy not in existing_preferences:
            existing_preferences[strategy] = {
                "accepted": 0,
                "rejected": 0
            }
        
        if user_action == "accepted":
            existing_preferences[strategy]["accepted"] += 1
        else:
            existing_preferences[strategy]["rejected"] += 1
        
        # Calculate acceptance rate
        total = existing_preferences[strategy]["accepted"] + existing_preferences[strategy]["rejected"]
        acceptance_rate = existing_preferences[strategy]["accepted"] / total if total > 0 else 0.5
        
        existing_preferences[strategy]["acceptance_rate"] = acceptance_rate
        
        return existing_preferences


# Example usage
if __name__ == "__main__":
    miner = PatternMiner()
    
    # Simulate historical data
    historical = [
        {"timestamp": "2025-12-15T00:00:00Z", "sales": 1000},
        {"timestamp": "2024-12-20T00:00:00Z", "sales": 950},
        {"timestamp": "2025-06-10T00:00:00Z", "sales": 300},
        {"timestamp": "2024-06-15T00:00:00Z", "sales": 320},
    ]
    
    patterns = miner.detect_seasonal_patterns(historical, metric="sales")
    print("Seasonal Patterns:")
    print(json.dumps(patterns, indent=2))
