"""
Anomaly Detection for PPC Performance.
Identifies unusual patterns and potential issues.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of anomalies."""
    SPEND_SPIKE = "spend_spike"
    SPEND_DROP = "spend_drop"
    CTR_ANOMALY = "ctr_anomaly"
    CONVERSION_ANOMALY = "conversion_anomaly"
    ACOS_SPIKE = "acos_spike"
    IMPRESSION_DROP = "impression_drop"
    PERFORMANCE_DETERIORATION = "performance_deterioration"
    UNUSUAL_PATTERN = "unusual_pattern"


class AnomalySeverity(str, Enum):
    """Severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    entity_type: str  # 'campaign', 'keyword', 'account'
    entity_id: int
    metric: str
    expected_value: float
    actual_value: float
    deviation: float
    detected_at: datetime
    message: str
    recommended_action: str


class IsolationForest:
    """
    Isolation Forest for unsupervised anomaly detection.
    """
    
    def __init__(self, n_trees: int = 100, sample_size: int = 256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = []
    
    def fit(self, X: np.ndarray):
        """Fit isolation forest."""
        n_samples = len(X)
        self.sample_size = min(self.sample_size, n_samples)
        
        self.trees = []
        for _ in range(self.n_trees):
            # Random sample
            indices = np.random.choice(n_samples, self.sample_size, replace=False)
            sample = X[indices]
            
            # Build tree
            tree = self._build_tree(sample, 0, int(np.ceil(np.log2(self.sample_size))))
            self.trees.append(tree)
    
    def _build_tree(self, X: np.ndarray, depth: int, max_depth: int) -> Dict:
        """Build isolation tree recursively."""
        n_samples, n_features = X.shape
        
        if depth >= max_depth or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}
        
        # Random split
        feature = np.random.randint(n_features)
        min_val, max_val = X[:, feature].min(), X[:, feature].max()
        
        if min_val == max_val:
            return {'type': 'leaf', 'size': n_samples}
        
        split_value = np.random.uniform(min_val, max_val)
        
        left_mask = X[:, feature] < split_value
        right_mask = ~left_mask
        
        return {
            'type': 'node',
            'feature': feature,
            'split': split_value,
            'left': self._build_tree(X[left_mask], depth + 1, max_depth),
            'right': self._build_tree(X[right_mask], depth + 1, max_depth)
        }
    
    def _path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """Calculate path length for a sample."""
        if tree['type'] == 'leaf':
            # Average path length for remaining samples
            n = tree['size']
            if n <= 1:
                return depth
            return depth + 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        
        if x[tree['feature']] < tree['split']:
            return self._path_length(x, tree['left'], depth + 1)
        return self._path_length(x, tree['right'], depth + 1)
    
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores (higher = more anomalous)."""
        scores = np.zeros(len(X))
        
        # Average path length normalization
        c_n = 2 * (np.log(self.sample_size - 1) + 0.5772156649) - 2 * (self.sample_size - 1) / self.sample_size
        
        for i, x in enumerate(X):
            avg_path = np.mean([self._path_length(x, tree) for tree in self.trees])
            scores[i] = 2 ** (-avg_path / c_n)
        
        return scores
    
    def predict(self, X: np.ndarray, threshold: float = 0.6) -> np.ndarray:
        """Predict anomalies (1 = anomaly, 0 = normal)."""
        scores = self.anomaly_score(X)
        return (scores > threshold).astype(int)


class ZScoreDetector:
    """
    Z-Score based anomaly detection for time series.
    """
    
    def __init__(self, threshold: float = 2.5, window: int = 30):
        self.threshold = threshold
        self.window = window
    
    def detect(
        self,
        values: List[float],
        dates: Optional[List[datetime]] = None
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in time series."""
        if len(values) < self.window:
            return []
        
        anomalies = []
        values = np.array(values)
        
        for i in range(self.window, len(values)):
            window_values = values[i-self.window:i]
            mean = np.mean(window_values)
            std = np.std(window_values)
            
            if std == 0:
                continue
            
            z_score = abs(values[i] - mean) / std
            
            if z_score > self.threshold:
                anomalies.append({
                    'index': i,
                    'value': values[i],
                    'expected': mean,
                    'z_score': round(z_score, 2),
                    'direction': 'up' if values[i] > mean else 'down',
                    'date': dates[i].isoformat() if dates else None
                })
        
        return anomalies


class ChangePointDetector:
    """
    Detects sudden changes in time series (change points).
    Uses CUSUM (Cumulative Sum) algorithm.
    """
    
    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
    
    def detect(self, values: List[float]) -> List[Dict[str, Any]]:
        """Detect change points in time series."""
        if len(values) < 10:
            return []
        
        values = np.array(values)
        mean = np.mean(values[:10])  # Initial reference
        
        s_pos = 0  # Positive CUSUM
        s_neg = 0  # Negative CUSUM
        
        change_points = []
        
        for i, x in enumerate(values):
            s_pos = max(0, s_pos + x - mean - self.drift)
            s_neg = max(0, s_neg - x + mean - self.drift)
            
            if s_pos > self.threshold:
                change_points.append({
                    'index': i,
                    'value': x,
                    'direction': 'increase',
                    'cusum': s_pos
                })
                s_pos = 0  # Reset
                mean = x  # Update reference
            
            if s_neg > self.threshold:
                change_points.append({
                    'index': i,
                    'value': x,
                    'direction': 'decrease',
                    'cusum': s_neg
                })
                s_neg = 0  # Reset
                mean = x  # Update reference
        
        return change_points


class PPCAnomalyDetector:
    """
    Comprehensive anomaly detection for PPC campaigns.
    """
    
    def __init__(self):
        self.isolation_forest = IsolationForest(n_trees=50)
        self.zscore_detector = ZScoreDetector(threshold=2.5)
        self.changepoint_detector = ChangePointDetector()
    
    def detect_campaign_anomalies(
        self,
        campaign_id: int,
        historical_data: List[Dict[str, Any]],
        current_data: Dict[str, Any]
    ) -> List[Anomaly]:
        """
        Detect anomalies for a campaign.
        """
        anomalies = []
        
        if not historical_data:
            return anomalies
        
        # Extract time series for each metric
        metrics = ['spend', 'sales', 'clicks', 'impressions', 'acos']
        
        for metric in metrics:
            values = [d.get(metric, 0) for d in historical_data]
            current = current_data.get(metric, 0)
            
            if not values:
                continue
            
            # Z-Score detection
            mean = np.mean(values)
            std = np.std(values)
            
            if std > 0:
                z_score = abs(current - mean) / std
                
                if z_score > 2.5:
                    severity = self._determine_severity(z_score, metric)
                    anomaly_type = self._determine_anomaly_type(metric, current > mean)
                    
                    anomalies.append(Anomaly(
                        anomaly_type=anomaly_type,
                        severity=severity,
                        entity_type='campaign',
                        entity_id=campaign_id,
                        metric=metric,
                        expected_value=round(mean, 2),
                        actual_value=round(current, 2),
                        deviation=round(z_score, 2),
                        detected_at=datetime.now(),
                        message=f"{metric.upper()} is {z_score:.1f} standard deviations from normal",
                        recommended_action=self._recommend_action(anomaly_type, metric)
                    ))
        
        # Multi-dimensional anomaly detection
        if len(historical_data) >= 20:
            X = self._prepare_features(historical_data)
            self.isolation_forest.fit(X)
            
            current_features = self._prepare_features([current_data])
            score = self.isolation_forest.anomaly_score(current_features)[0]
            
            if score > 0.65:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.UNUSUAL_PATTERN,
                    severity=AnomalySeverity.MEDIUM if score < 0.8 else AnomalySeverity.HIGH,
                    entity_type='campaign',
                    entity_id=campaign_id,
                    metric='overall',
                    expected_value=0.5,
                    actual_value=round(score, 2),
                    deviation=round(score, 2),
                    detected_at=datetime.now(),
                    message=f"Unusual overall performance pattern detected (score: {score:.2f})",
                    recommended_action="Review all metrics for potential issues"
                ))
        
        return anomalies
    
    def _prepare_features(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare feature matrix for isolation forest."""
        features = []
        for d in data:
            features.append([
                d.get('spend', 0),
                d.get('sales', 0),
                d.get('clicks', 0),
                d.get('impressions', 0),
                d.get('acos', 0),
                d.get('ctr', 0),
                d.get('conversion_rate', 0)
            ])
        return np.array(features)
    
    def _determine_severity(self, z_score: float, metric: str) -> AnomalySeverity:
        """Determine severity based on z-score and metric importance."""
        critical_metrics = ['spend', 'acos']
        
        if z_score > 4:
            return AnomalySeverity.CRITICAL
        elif z_score > 3:
            return AnomalySeverity.HIGH if metric in critical_metrics else AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.MEDIUM if metric in critical_metrics else AnomalySeverity.LOW
    
    def _determine_anomaly_type(self, metric: str, is_increase: bool) -> AnomalyType:
        """Determine anomaly type based on metric and direction."""
        if metric == 'spend':
            return AnomalyType.SPEND_SPIKE if is_increase else AnomalyType.SPEND_DROP
        elif metric == 'acos':
            return AnomalyType.ACOS_SPIKE if is_increase else AnomalyType.PERFORMANCE_DETERIORATION
        elif metric == 'impressions':
            return AnomalyType.IMPRESSION_DROP if not is_increase else AnomalyType.UNUSUAL_PATTERN
        elif metric == 'ctr':
            return AnomalyType.CTR_ANOMALY
        elif metric == 'conversion_rate':
            return AnomalyType.CONVERSION_ANOMALY
        return AnomalyType.UNUSUAL_PATTERN
    
    def _recommend_action(self, anomaly_type: AnomalyType, metric: str) -> str:
        """Recommend action based on anomaly type."""
        recommendations = {
            AnomalyType.SPEND_SPIKE: "Review bid settings and budget caps",
            AnomalyType.SPEND_DROP: "Check campaign status and bid amounts",
            AnomalyType.CTR_ANOMALY: "Review ad copy and targeting",
            AnomalyType.CONVERSION_ANOMALY: "Check landing pages and product listing",
            AnomalyType.ACOS_SPIKE: "Reduce bids or pause poor performers",
            AnomalyType.IMPRESSION_DROP: "Check bid competitiveness and budget",
            AnomalyType.PERFORMANCE_DETERIORATION: "Comprehensive campaign review needed",
            AnomalyType.UNUSUAL_PATTERN: "Monitor closely and investigate"
        }
        return recommendations.get(anomaly_type, "Investigate metric changes")
    
    def detect_trend_changes(
        self,
        values: List[float],
        metric_name: str = "metric"
    ) -> List[Dict[str, Any]]:
        """Detect significant trend changes."""
        return self.changepoint_detector.detect(values)
