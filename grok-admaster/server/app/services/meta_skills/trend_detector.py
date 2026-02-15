"""
Trend Detector for Optimus Pryme Knowledge Synthesizer
Identifies market trends, emerging opportunities, and anomalies.
"""

import math
from typing import List, Dict, Any, Optional

class TrendDetector:
    def __init__(self):
        pass

    def calculate_momentum(self, data_points: List[float]) -> float:
        """
        Calculate momentum score based on recent data vs historical baseline.
        Positive = Growing trend, Negative = Declining trend.
        """
        if len(data_points) < 2:
            return 0.0
            
        # Split data: recent (last 20%) vs baseline (first 80%)
        # For small datasets, just compare last vs avg
        split_idx = int(len(data_points) * 0.8)
        if split_idx == 0:
            split_idx = len(data_points) // 2
        
        baseline = data_points[:split_idx]
        recent = data_points[split_idx:]
        
        baseline_avg = sum(baseline) / len(baseline) if baseline else 0
        recent_avg = sum(recent) / len(recent) if recent else 0
        
        if baseline_avg == 0:
            return 1.0 if recent_avg > 0 else 0.0
            
        growth = (recent_avg - baseline_avg) / baseline_avg
        return growth

    def detect_anomalies(self, data_points: List[float], threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies using Z-score.
        """
        if len(data_points) < 5:
            return []
            
        mean = sum(data_points) / len(data_points)
        variance = sum([((x - mean) ** 2) for x in data_points]) / len(data_points)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return []
            
        anomalies = []
        for i, val in enumerate(data_points):
            z_score = (val - mean) / std_dev
            if abs(z_score) > threshold:
                anomalies.append({
                    "index": i,
                    "value": val,
                    "z_score": z_score,
                    "type": "spike" if z_score > 0 else "drop"
                })
                
        return anomalies

    def simple_linear_regression(self, y: List[float]) -> Dict[str, float]:
        """
        Calculate simple linear regression (y = mx + c) to get slope.
        Uses scipy if available, otherwise falls back to basic math.
        """
        n = len(y)
        if n < 2:
            return {"slope": 0, "intercept": 0, "r_squared": 0, "direction": "flat"}
            
        from scipy import stats
        x = list(range(n))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "direction": "up" if slope > 0 else "down"
        }

    def analyze_keyword_trends(self, keyword_data: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of keywords for potential opportunities.
        """
        results = []
        for term, volumes in keyword_data.items():
            momentum = self.calculate_momentum(volumes)
            regression = self.simple_linear_regression(volumes)
            anomalies = self.detect_anomalies(volumes)
            
            score = (momentum * 0.5) + (regression['slope'] * 0.1)  # Simplified weighting
            
            results.append({
                "keyword": term,
                "momentum_score": momentum,
                "trend_slope": regression['slope'],
                "trend_strength": regression['r_squared'],
                "anomalies_count": len(anomalies),
                "is_emerging": momentum > 0.2 and regression['r_squared'] > 0.6
            })
            
        # Sort by momentum
        results.sort(key=lambda x: x["momentum_score"], reverse=True)
        return results

    def analyze_market_context(
        self,
        internal_cpc_history: List[float],
        competitor_price_history: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze correlation between Internal CPC and Competitor Pricing.
        Detects 'Price Wars', 'Opportunity Windows', etc.
        """
        cpc_momentum = self.calculate_momentum(internal_cpc_history)
        price_momentum = self.calculate_momentum(competitor_price_history)
        
        status = "normal"
        confidence = "low"
        
        # Logic: If Competitor Price DROPS and our CPC RISES -> PRICE WAR
        if price_momentum < -0.05 and cpc_momentum > 0.05:
            status = "price_war_risk"
            confidence = "high"
        
        # Logic: If Competitor Price RISES and our CPC is STABLE -> MARGIN OPPORTUNITY
        elif price_momentum > 0.05 and abs(cpc_momentum) < 0.05:
            status = "margin_opportunity"
            confidence = "medium"
            
        return {
            "status": status,
            "confidence": confidence,
            "metrics": {
                "cpc_trend": cpc_momentum,
                "competitor_price_trend": price_momentum
            }
        }
