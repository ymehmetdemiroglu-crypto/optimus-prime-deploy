"""
Anomaly Detection API Router.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from app.core.database import get_db
from .service import anomaly_service
from .schemas import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    AnomalyAlertRead,
    AnomalyAlertUpdate,
    AnomalyStatistics,
    EntityType,
    SeverityLevel,
)

router = APIRouter(tags=["Anomaly Detection"])


@router.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Detect anomalies for specified entities.
    
    Uses ensemble detection (LSTM + Streaming + Isolation Forest) with
    root cause analysis.
    """
    return await anomaly_service.detect_anomalies(db, request)


@router.get("/alerts/active", response_model=List[AnomalyAlertRead])
async def get_active_alerts(
    profile_id: str = Query(..., description="Profile ID"),
    severity: Optional[SeverityLevel] = Query(None, description="Filter by severity"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get active (unresolved) anomaly alerts.
    """
    alerts = await anomaly_service.get_active_alerts(db, profile_id, severity, limit)
    return [AnomalyAlertRead.model_validate(a) for a in alerts]


@router.patch("/alerts/{alert_id}/acknowledge", response_model=AnomalyAlertRead)
async def acknowledge_alert(
    alert_id: int,
    acknowledged_by: str = Query(..., description="User who acknowledged"),
    db: AsyncSession = Depends(get_db),
):
    """
    Mark an alert as acknowledged.
    """
    alert = await anomaly_service.acknowledge_alert(db, alert_id, acknowledged_by)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return AnomalyAlertRead.model_validate(alert)


@router.patch("/alerts/{alert_id}/resolve", response_model=AnomalyAlertRead)
async def resolve_alert(
    alert_id: int,
    update: AnomalyAlertUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Mark an alert as resolved.
    """
    alert = await anomaly_service.resolve_alert(
        db,
        alert_id,
        resolution_notes=update.resolution_notes,
    )
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return AnomalyAlertRead.model_validate(alert)


@router.get("/statistics", response_model=AnomalyStatistics)
async def get_statistics(
    profile_id: str = Query(..., description="Profile ID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get aggregated anomaly statistics for dashboard.
    """
    return await anomaly_service.get_statistics(db, profile_id)
