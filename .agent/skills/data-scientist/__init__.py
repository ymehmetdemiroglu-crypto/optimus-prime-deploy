"""
Data Scientist Skill Package
Personalized ML models for bid prediction, anomaly detection, and customer segmentation.
"""

from .scripts.bid_predictor import BidPredictor, BidFeatures, BidPrediction
from .scripts.anomaly_detector import AnomalyDetector, Anomaly, AnomalySeverity, AnomalyType
from .scripts.customer_segmenter import CustomerSegmenter, CustomerSegment, CustomerOrder

__all__ = [
    "BidPredictor",
    "BidFeatures",
    "BidPrediction",
    "AnomalyDetector",
    "Anomaly",
    "AnomalySeverity",
    "AnomalyType",
    "CustomerSegmenter",
    "CustomerSegment",
    "CustomerOrder"
]

__version__ = "1.0.0"
