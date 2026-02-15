"""
Trend Detector for Optimus Pryme Knowledge Synthesizer
Identifies market trends, emerging opportunities, and anomalies.
"""

import math
from typing import List, Dict, Any

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
        """
        n = len(y)
        if n < 2:
            return {"slope": 0, "intercept": 0, "r_squared": 0}
            
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum([xi * yi for xi, yi in zip(x, y)])
        sum_xx = sum([xi ** 2 for xi in x])
        
        # Calculate slope (m) and intercept (c)
        denominator = (n * sum_xx - sum_x ** 2)
        if denominator == 0:
             return {"slope": 0, "intercept": 0, "r_squared": 0}
             
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        y_pred = [slope * xi + intercept for xi in x]
        y_mean = sum_y / n
        ss_tot = sum([(yi - y_mean) ** 2 for yi in y])
        ss_res = sum([(yi - y_predi) ** 2 for yi, y_predi in zip(y, y_pred)])
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
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

if __name__ == "__main__":
    detector = TrendDetector()
    
    # Example: Keyword search volume over 12 weeks
    data = {
        "eco friendly packaging": [100, 105, 110, 108, 120, 130, 145, 160, 180, 210, 250, 290], # Strong growth
        "plastic straws": [500, 480, 460, 450, 420, 400, 380, 350, 320, 300, 280, 250],   # Decline
        "steady eddie": [100, 102, 98, 101, 100, 99, 103, 100, 101, 98, 100, 102],       # Flat
        "viral spike": [50, 55, 52, 58, 60, 800, 1200, 600, 200, 80, 65, 60]             # Anomaly
    }
    
    analysis = detector.analyze_keyword_trends(data)
    
    print("Trend Analysis Results:")
    for item in analysis:
        status = "🔥 EMERGING" if item["is_emerging"] else "stable"
        print(f"[{item['momentum_score']*100:+.1f}%] {item['keyword']} (Slope: {item['trend_slope']:.2f}) {status}")
        
    print("\nAnomaly Detection (Viral Spike):")
    spikes = detector.detect_anomalies(data["viral spike"])
    for spike in spikes:
        print(f"Week {spike['index']}: Value {spike['value']} (Z-Score: {spike['z_score']:.2f})")
