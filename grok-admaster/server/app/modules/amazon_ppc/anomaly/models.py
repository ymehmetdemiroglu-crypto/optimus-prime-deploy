"""
Database models for anomaly detection.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base
from ..accounts.models import Profile


class AnomalyAlert(Base):
    """
    Stores detected anomalies for real-time alerting.
    
    Retention: 90 days (alerts older than 90 days auto-archived to AnomalyHistory)
    """
    __tablename__ = "anomaly_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Entity identification
    entity_type = Column(String(50), nullable=False, index=True)  # keyword, campaign, portfolio
    entity_id = Column(String(100), nullable=False, index=True)
    entity_name = Column(String(500))
    
    # Profile/account association
    profile_id = Column(String, ForeignKey("profiles.profile_id"), nullable=False, index=True)
    
    # Anomaly details
    anomaly_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
    metric_name = Column(String(100), nullable=False)
    
    # Values
    actual_value = Column(Float)
    expected_value = Column(Float)
    reconstruction_error = Column(Float)
    
    # Detection metadata
    detector_type = Column(String(50), nullable=False)  # lstm, streaming, isolation_forest, ensemble
    detection_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Explanations and root cause
    explanation = Column(JSON)  # {feature_name: contribution}
    root_causes = Column(JSON)  # [cause1, cause2, ...]
    
    # Alert management
    is_acknowledged = Column(Boolean, default=False, index=True)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(200))
    
    is_resolved = Column(Boolean, default=False, index=True)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    # Relationships
    profile = relationship("Profile", back_populates="anomaly_alerts")
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('ix_anomaly_alerts_profile_severity', 'profile_id', 'severity'),
        Index('ix_anomaly_alerts_entity_type_id', 'entity_type', 'entity_id'),
        Index('ix_anomaly_alerts_unresolved', 'is_resolved', 'detection_timestamp'),
    )


class AnomalyHistory(Base):
    """
    Historical anomaly records for long-term analysis and ML training.
    
    Retention: Indefinite (used for pattern recognition and model training)
    """
    __tablename__ = "anomaly_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Entity identification
    entity_type = Column(String(50), nullable=False, index=True)
    entity_id = Column(String(100), nullable=False, index=True)
    entity_name = Column(String(500))
    
    # Profile/account association
    profile_id = Column(String, ForeignKey("profiles.profile_id"), nullable=False, index=True)
    
    # Anomaly details
    anomaly_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False)
    
    # Values at time of detection
    actual_value = Column(Float)
    expected_value = Column(Float)
    reconstruction_error = Column(Float)
    
    # Detection metadata
    detector_type = Column(String(50), nullable=False)
    detection_timestamp = Column(DateTime, nullable=False, index=True)
    
    # Context at time of detection
    explanation = Column(JSON)
    root_causes = Column(JSON)
    market_conditions = Column(JSON)  # Competitive intelligence context
    campaign_settings = Column(JSON)  # Bid, budget, status at detection time
    
    # Resolution tracking
    was_resolved = Column(Boolean, default=False)
    resolution_time_minutes = Column(Integer)  # Time to resolution
    resolution_action = Column(String(100))  # bid_adjustment, pause, budget_change, etc.
    
    # Impact measurement
    revenue_impact = Column(Float)  # $ impact if measurable
    performance_degradation = Column(Float)  # % drop in key metric
    
    # Relationship
    profile = relationship("Profile", back_populates="anomaly_history")
    
    # Composite indexes
    __table_args__ = (
        Index('ix_anomaly_history_profile_date', 'profile_id', 'detection_timestamp'),
        Index('ix_anomaly_history_entity_type_id', 'entity_type', 'entity_id'),
        Index('ix_anomaly_history_severity_date', 'severity', 'detection_timestamp'),
    )


class AnomalyTrainingData(Base):
    """
    Stores labeled data for model retraining and validation.
    """
    __tablename__ = "anomaly_training_data"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Reference
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(100), nullable=False)
    profile_id = Column(String, ForeignKey("profiles.profile_id"), nullable=False)
    
    # Time series data
    sequence_data = Column(JSON, nullable=False)  # 14-day sequences
    feature_snapshot = Column(JSON, nullable=False)
    
    # Labels
    is_true_anomaly = Column(Boolean, nullable=False)  # Human-verified
    labeled_by = Column(String(200))
    labeled_at = Column(DateTime, default=datetime.utcnow)
    
    # Model performance tracking
    predicted_score = Column(Float)
    was_correctly_classified = Column(Boolean)
    
    # Relationship
    profile = relationship("Profile")
    
    __table_args__ = (
        Index('ix_training_profile_labeled', 'profile_id', 'labeled_at'),
    )
