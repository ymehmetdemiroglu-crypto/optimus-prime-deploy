"""
Anomaly Detector for Data Scientist Skill
Multi-dimensional anomaly detection for campaign performance.
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class AnomalySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AnomalyType(Enum):
    SPIKE = "spike"
    DROP = "drop"
    TREND_BREAK = "trend_break"
    OUTLIER = "outlier"

@dataclass
class DataPoint:
    timestamp: datetime
    value: float
    metric_name: str
    entity_id: str  # campaign_id, keyword_id, etc.
    entity_type: str

@dataclass
class Anomaly:
    anomaly_id: str
    entity_id: str
    entity_type: str
    metric_name: str
    detected_value: float
    expected_value: float
    expected_range: Tuple[float, float]
    deviation_sigma: float
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    detected_at: datetime
    probable_causes: List[Dict[str, Any]] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

class AnomalyDetector:
    def __init__(self, lookback_periods: int = 30, sigma_threshold: float = 2.5):
        self.lookback_periods = lookback_periods
        self.sigma_threshold = sigma_threshold
        self.anomaly_count = 0
    
    def detect_statistical_anomaly(self, data_points: List[DataPoint]) -> List[Anomaly]:
        """
        Detect anomalies using Z-score method.
        """
        if len(data_points) < self.lookback_periods:
            return []
        
        anomalies = []
        values = [dp.value for dp in data_points]
        
        # Calculate rolling statistics
        for i in range(self.lookback_periods, len(data_points)):
            window = values[i - self.lookback_periods:i]
            current = values[i]
            current_dp = data_points[i]
            
            mean = sum(window) / len(window)
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            std = math.sqrt(variance) if variance > 0 else 0.001
            
            z_score = (current - mean) / std if std > 0 else 0
            
            if abs(z_score) > self.sigma_threshold:
                self.anomaly_count += 1
                anomaly = self._create_anomaly(
                    current_dp,
                    current,
                    mean,
                    (mean - 2*std, mean + 2*std),
                    z_score
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_multi_metric_anomaly(self, 
                                    metric_data: Dict[str, List[DataPoint]]) -> List[Anomaly]:
        """
        Detect anomalies across multiple metrics simultaneously.
        Useful for catching correlated issues.
        """
        all_anomalies = []
        
        for metric_name, data_points in metric_data.items():
            anomalies = self.detect_statistical_anomaly(data_points)
            all_anomalies.extend(anomalies)
        
        # Check for correlated anomalies (same time, multiple metrics)
        return self._correlate_anomalies(all_anomalies)
    
    def _correlate_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Add context when multiple metrics show anomalies together."""
        if len(anomalies) < 2:
            return anomalies
        
        # Group by timestamp (within 1 hour window)
        time_groups: Dict[str, List[Anomaly]] = {}
        for a in anomalies:
            key = a.detected_at.strftime("%Y-%m-%d-%H")
            if key not in time_groups:
                time_groups[key] = []
            time_groups[key].append(a)
        
        # Upgrade severity for correlated anomalies
        for key, group in time_groups.items():
            if len(group) >= 2:
                for a in group:
                    if a.severity == AnomalySeverity.MEDIUM:
                        a.severity = AnomalySeverity.HIGH
                    elif a.severity == AnomalySeverity.LOW:
                        a.severity = AnomalySeverity.MEDIUM
                    
                    a.probable_causes.append({
                        "cause": "Correlated multi-metric anomaly",
                        "confidence": 0.85,
                        "related_metrics": [x.metric_name for x in group if x != a]
                    })
        
        return anomalies
    
    def _create_anomaly(self, 
                        data_point: DataPoint, 
                        current: float, 
                        expected: float,
                        expected_range: Tuple[float, float],
                        z_score: float) -> Anomaly:
        """Create an anomaly object with full context."""
        
        # Determine type
        if z_score > 0:
            anomaly_type = AnomalyType.SPIKE
        else:
            anomaly_type = AnomalyType.DROP
        
        # Determine severity
        abs_z = abs(z_score)
        if abs_z > 4:
            severity = AnomalySeverity.CRITICAL
        elif abs_z > 3:
            severity = AnomalySeverity.HIGH
        elif abs_z > 2.5:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW
        
        # Generate probable causes
        probable_causes = self._infer_causes(data_point.metric_name, anomaly_type, abs_z)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            data_point.metric_name, anomaly_type, severity
        )
        
        return Anomaly(
            anomaly_id=f"ANM{self.anomaly_count:05d}",
            entity_id=data_point.entity_id,
            entity_type=data_point.entity_type,
            metric_name=data_point.metric_name,
            detected_value=round(current, 4),
            expected_value=round(expected, 4),
            expected_range=(round(expected_range[0], 4), round(expected_range[1], 4)),
            deviation_sigma=round(z_score, 2),
            anomaly_type=anomaly_type,
            severity=severity,
            detected_at=data_point.timestamp,
            probable_causes=probable_causes,
            recommended_actions=recommendations
        )
    
    def _infer_causes(self, metric: str, anomaly_type: AnomalyType, 
                      deviation: float) -> List[Dict[str, Any]]:
        """Infer probable causes based on metric and anomaly type."""
        causes = []
        
        metric_causes = {
            "ctr": {
                AnomalyType.DROP: [
                    ("Main image quality issue", 0.70),
                    ("Competitor launched better creative", 0.55),
                    ("Ad fatigue", 0.45)
                ],
                AnomalyType.SPIKE: [
                    ("Seasonal interest increase", 0.60),
                    ("Viral mention or trend", 0.50),
                    ("Competitor stockout", 0.45)
                ]
            },
            "cvr": {
                AnomalyType.DROP: [
                    ("Price competitiveness issue", 0.65),
                    ("Negative review impact", 0.55),
                    ("Listing content changed", 0.50),
                    ("Stock issue / delivery time", 0.40)
                ],
                AnomalyType.SPIKE: [
                    ("Successful promotion", 0.70),
                    ("Positive review momentum", 0.50),
                    ("Competitor price increase", 0.45)
                ]
            },
            "cpc": {
                AnomalyType.SPIKE: [
                    ("Competitor bid increase", 0.75),
                    ("Seasonal demand surge", 0.55),
                    ("Quality score degradation", 0.40)
                ],
                AnomalyType.DROP: [
                    ("Competitor budget exhausted", 0.60),
                    ("Off-peak timing", 0.50)
                ]
            },
            "acos": {
                AnomalyType.SPIKE: [
                    ("CVR drop while CPC stable", 0.70),
                    ("Bid too high for current performance", 0.55)
                ],
                AnomalyType.DROP: [
                    ("CVR improvement", 0.65),
                    ("CPC decrease", 0.50)
                ]
            },
            "spend": {
                AnomalyType.SPIKE: [
                    ("Unexpected high impression volume", 0.60),
                    ("Bid setting error", 0.55)
                ],
                AnomalyType.DROP: [
                    ("Budget exhausted early", 0.70),
                    ("Campaign paused accidentally", 0.50),
                    ("Low relevance score", 0.40)
                ]
            }
        }
        
        if metric.lower() in metric_causes:
            cause_list = metric_causes[metric.lower()].get(anomaly_type, [])
            for cause, confidence in cause_list:
                causes.append({"cause": cause, "confidence": confidence})
        
        # Adjust confidence based on deviation
        if deviation > 4:
            for c in causes:
                c["confidence"] = min(0.95, c["confidence"] + 0.1)
        
        return sorted(causes, key=lambda x: x["confidence"], reverse=True)[:3]
    
    def _generate_recommendations(self, metric: str, anomaly_type: AnomalyType,
                                   severity: AnomalySeverity) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Severity-based priority
        if severity == AnomalySeverity.CRITICAL:
            recommendations.append("URGENT: Investigate immediately")
        
        # Metric-specific recommendations
        if metric.lower() == "ctr" and anomaly_type == AnomalyType.DROP:
            recommendations.extend([
                "Review main product image for quality issues",
                "Check if competitors launched new creative",
                "Consider A/B testing new ad headlines"
            ])
        elif metric.lower() == "cvr" and anomaly_type == AnomalyType.DROP:
            recommendations.extend([
                "Audit recent listing changes",
                "Check new negative reviews",
                "Verify competitive pricing",
                "Confirm stock availability"
            ])
        elif metric.lower() == "cpc" and anomaly_type == AnomalyType.SPIKE:
            recommendations.extend([
                "Review competitor advertising activity",
                "Consider dayparting to avoid peak competition",
                "Evaluate keyword relevance scores"
            ])
        elif metric.lower() == "spend" and anomaly_type == AnomalyType.SPIKE:
            recommendations.extend([
                "Set emergency budget caps",
                "Review bid settings for anomalies",
                "Check for bot traffic patterns"
            ])
        
        return recommendations[:4]
    
    def get_anomaly_summary(self, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Generate a summary of detected anomalies."""
        if not anomalies:
            return {"total": 0, "status": "healthy"}
        
        severity_counts = {s: 0 for s in AnomalySeverity}
        type_counts = {t: 0 for t in AnomalyType}
        metric_counts: Dict[str, int] = {}
        
        for a in anomalies:
            severity_counts[a.severity] += 1
            type_counts[a.anomaly_type] += 1
            metric_counts[a.metric_name] = metric_counts.get(a.metric_name, 0) + 1
        
        return {
            "total": len(anomalies),
            "by_severity": {s.value: c for s, c in severity_counts.items() if c > 0},
            "by_type": {t.value: c for t, c in type_counts.items() if c > 0},
            "by_metric": metric_counts,
            "most_affected_metric": max(metric_counts.items(), key=lambda x: x[1])[0] if metric_counts else None,
            "critical_count": severity_counts[AnomalySeverity.CRITICAL],
            "high_count": severity_counts[AnomalySeverity.HIGH],
            "requires_immediate_attention": severity_counts[AnomalySeverity.CRITICAL] > 0
        }


def generate_sample_data(metric: str, n_points: int = 100, 
                         inject_anomaly: bool = True) -> List[DataPoint]:
    """Generate sample time-series data with optional anomaly injection."""
    import random
    
    data = []
    base_value = {"ctr": 0.35, "cvr": 0.10, "cpc": 1.50, "acos": 25.0, "spend": 100.0}
    base = base_value.get(metric.lower(), 50.0)
    std = base * 0.15
    
    start_time = datetime.now() - timedelta(days=n_points)
    
    for i in range(n_points):
        value = base + random.gauss(0, std)
        
        # Inject anomaly at specific point
        if inject_anomaly and i == n_points - 5:
            if random.random() < 0.5:
                value = base * 0.5  # Drop
            else:
                value = base * 2.0  # Spike
        
        data.append(DataPoint(
            timestamp=start_time + timedelta(days=i),
            value=max(0.01, value),
            metric_name=metric,
            entity_id="CAMP_001",
            entity_type="campaign"
        ))
    
    return data


if __name__ == "__main__":
    detector = AnomalyDetector(lookback_periods=30, sigma_threshold=2.5)
    
    # Generate and analyze sample data
    print("Generating sample data with injected anomalies...")
    
    metrics = ["CTR", "CVR", "CPC"]
    all_data = {m: generate_sample_data(m, 100, inject_anomaly=True) for m in metrics}
    
    print("Running multi-metric anomaly detection...")
    anomalies = detector.detect_multi_metric_anomaly(all_data)
    
    summary = detector.get_anomaly_summary(anomalies)
    print(f"\nAnomaly Summary:")
    print(f"  Total Detected: {summary['total']}")
    print(f"  Critical: {summary['critical_count']}, High: {summary['high_count']}")
    print(f"  Requires Immediate Attention: {summary['requires_immediate_attention']}")
    
    if anomalies:
        print(f"\nTop Anomaly Detail:")
        top = anomalies[0]
        print(f"  Metric: {top.metric_name}")
        print(f"  Value: {top.detected_value} (expected: {top.expected_value})")
        print(f"  Severity: {top.severity.value}")
        print(f"  Probable Cause: {top.probable_causes[0]['cause'] if top.probable_causes else 'Unknown'}")
        print(f"  Recommended Action: {top.recommended_actions[0] if top.recommended_actions else 'Monitor'}")
