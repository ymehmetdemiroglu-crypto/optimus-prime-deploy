"""
Pydantic schemas for anomaly detection API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class SeverityLevel(str, Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EntityType(str, Enum):
    """Entity types that can have anomalies."""
    KEYWORD = "keyword"
    CAMPAIGN = "campaign"
    PORTFOLIO = "portfolio"
    ACCOUNT = "account"


class DetectorType(str, Enum):
    """Anomaly detector types."""
    LSTM = "lstm"
    STREAMING = "streaming"
    ISOLATION_FOREST = "isolation_forest"
    ENSEMBLE = "ensemble"


# ═══════════════════════════════════════════════════════════════════════
#  API Request/Response Schemas
# ═══════════════════════════════════════════════════════════════════════

class AnomalyDetectionRequest(BaseModel):
    """Request to detect anomalies for specific entities."""
    entity_type: EntityType
    entity_ids: Optional[List[str]] = None  # None = check all
    profile_id: str
    detector_type: DetectorType = DetectorType.ENSEMBLE
    include_explanation: bool = True
    include_root_cause: bool = True


class AnomalyDetectionResponse(BaseModel):
    """Response from anomaly detection."""
    total_entities_checked: int
    anomalies_detected: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    alerts: List["AnomalyAlertRead"]
    execution_time_ms: float


class FeatureContribution(BaseModel):
    """Individual feature contribution to anomaly score."""
    name: str
    contribution: float
    actual_value: float
    baseline_value: float
    direction: str  # increase, decrease, neutral


# ═══════════════════════════════════════════════════════════════════════
#  Database Model Schemas
# ═══════════════════════════════════════════════════════════════════════

class AnomalyAlertBase(BaseModel):
    """Base schema for anomaly alerts."""
    entity_type: str
    entity_id: str
    entity_name: Optional[str]
    profile_id: str
    anomaly_score: float
    threshold: float
    severity: SeverityLevel
    metric_name: str
    actual_value: Optional[float]
    expected_value: Optional[float]
    reconstruction_error: Optional[float]
    detector_type: DetectorType


class AnomalyAlertRead(AnomalyAlertBase):
    """Schema for reading anomaly alerts."""
    id: int
    detection_timestamp: datetime
    explanation: Optional[Dict[str, float]]
    root_causes: Optional[List[str]]
    is_acknowledged: bool
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    is_resolved: bool
    resolved_at: Optional[datetime]
    resolution_notes: Optional[str]
    
    class Config:
        from_attributes = True


class AnomalyAlertUpdate(BaseModel):
    """Schema for updating anomaly alert status."""
    is_acknowledged: Optional[bool] = None
    acknowledged_by: Optional[str] = None
    is_resolved: Optional[bool] = None
    resolution_notes: Optional[str] = None


class AnomalyHistoryBase(BaseModel):
    """Base schema for anomaly history."""
    entity_type: str
    entity_id: str
    entity_name: Optional[str]
    profile_id: str
    anomaly_score: float
    threshold: float
    severity: SeverityLevel
    metric_name: str
    detector_type: DetectorType
    detection_timestamp: datetime


class AnomalyHistoryRead(AnomalyHistoryBase):
    """Schema for reading anomaly history."""
    id: int
    actual_value: Optional[float]
    expected_value: Optional[float]
    explanation: Optional[Dict[str, float]]
    root_causes: Optional[List[str]]
    market_conditions: Optional[Dict[str, Any]]
    was_resolved: bool
    resolution_time_minutes: Optional[int]
    resolution_action: Optional[str]
    revenue_impact: Optional[float]
    performance_degradation: Optional[float]
    
    class Config:
        from_attributes = True


# ═══════════════════════════════════════════════════════════════════════
#  Dashboard & Analytics Schemas
# ═══════════════════════════════════════════════════════════════════════

class AnomalyStatistics(BaseModel):
    """Aggregated anomaly statistics for dashboard."""
    total_alerts: int
    unresolved_count: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    avg_resolution_time_minutes: Optional[float]
    most_common_entity_type: str
    most_common_root_cause: Optional[str]
    detection_rate_last_24h: int
    detection_rate_last_7d: int


class AnomalyTrend(BaseModel):
    """Anomaly trends over time."""
    date: datetime
    count: int
    critical_count: int
    high_count: int
    avg_score: float


class EntityAnomalyProfile(BaseModel):
    """Anomaly profile for a specific entity."""
    entity_type: EntityType
    entity_id: str
    entity_name: str
    total_anomalies: int
    last_anomaly_date: Optional[datetime]
    avg_severity_score: float  # 1-4 (low to critical)
    common_metrics: List[str]
    risk_score: float  # 0-1, higher = more prone to anomalies


class RootCauseAnalysis(BaseModel):
    """Aggregated root cause analysis."""
    root_cause: str
    occurrence_count: int
    affected_entities: int
    avg_severity: str
    resolution_rate: float  # % resolved
    avg_resolution_time_minutes: Optional[float]
