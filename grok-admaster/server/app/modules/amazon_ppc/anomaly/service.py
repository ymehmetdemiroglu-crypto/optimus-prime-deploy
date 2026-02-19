"""
Anomaly Detection Service - Integrates advanced anomaly detection into PPC workflow.
"""
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, update
from sqlalchemy.orm import selectinload
import numpy as np

from ..ml.advanced_anomaly import (
    TimeSeriesAnomalyDetector,
    StreamingAnomalyDetector,
    EnsembleAnomalyDetector,
    RootCauseAnalyzer,
    AnomalyResult,
    TimestampedAnomaly,
)
from ..models import PPCKeyword, PPCCampaign, PerformanceRecord
from .models import AnomalyAlert, AnomalyHistory, AnomalyTrainingData
from .schemas import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    AnomalyAlertRead,
    AnomalyStatistics,
    EntityType,
    DetectorType,
    SeverityLevel,
)

logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """
    Production anomaly detection service.
    
    Features:
        - Continuous monitoring of keywords, campaigns, portfolios
        - Ensemble detection (LSTM + Streaming + Isolation Forest)
        - Root cause analysis with dependency graphs
        - Real-time alerting and historical tracking
        - Model persistence and retraining
    """
    
    def __init__(self):
        """Initialize with pre-configured detectors."""
        self.ensemble = EnsembleAnomalyDetector()
        self.root_cause_analyzer = RootCauseAnalyzer()
        
        # Model paths
        self.lstm_model_path = "models/anomaly/lstm_autoencoder.pth"
        self.isoforest_model_path = "models/anomaly/isolation_forest.pkl"
        
        # Load pre-trained models if available
        self._load_trained_models()
    
    def _load_trained_models(self):
        """Load pre-trained models from disk."""
        try:
            # Load LSTM if exists
            if self.ensemble.lstm_detector is not None:
                import os
                if os.path.exists(self.lstm_model_path):
                    self.ensemble.lstm_detector.load_model(self.lstm_model_path)
                    logger.info("[AnomalyService] Loaded pre-trained LSTM model")
                else:
                    logger.info("[AnomalyService] No pre-trained LSTM model found")
        except Exception as e:
            logger.warning(f"[AnomalyService] Failed to load models: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    #  Main Detection Methods
    # ═══════════════════════════════════════════════════════════════════
    
    async def detect_anomalies(
        self,
        db: AsyncSession,
        request: AnomalyDetectionRequest,
    ) -> AnomalyDetectionResponse:
        """
        Detect anomalies for specified entities.
        
        Workflow:
            1. Fetch historical time-series data
            2. Build dependency graph for root cause analysis
            3. Run ensemble detection on each entity
            4. Save alerts to database
            5. Return summary response
        """
        start_time = time.time()
        
        # Fetch entities
        entities = await self._fetch_entities(
            db,
            request.entity_type,
            request.profile_id,
            request.entity_ids,
        )
        
        if not entities:
            return AnomalyDetectionResponse(
                total_entities_checked=0,
                anomalies_detected=0,
                critical_count=0,
                high_count=0,
                medium_count=0,
                low_count=0,
                alerts=[],
                execution_time_ms=0.0,
            )
        
        # Build dependency graph
        await self._build_dependency_graph(db, request.profile_id)
        
        # Detect anomalies for each entity
        alerts = []
        for entity in entities:
            try:
                result = await self._detect_entity_anomaly(
                    db,
                    entity,
                    request.entity_type,
                    request.detector_type,
                    request.include_explanation,
                    request.include_root_cause,
                )
                
                if result and result.is_anomalous:
                    # Save alert to database
                    alert = await self._save_alert(db, result, request.profile_id)
                    alerts.append(alert)
                    
            except Exception as e:
                logger.error(
                    f"[AnomalyService] Error detecting anomaly for "
                    f"{request.entity_type} {entity.get('id')}: {e}"
                )
        
        # Aggregate statistics
        severity_counts = {
            "critical": sum(1 for a in alerts if a.severity == "critical"),
            "high": sum(1 for a in alerts if a.severity == "high"),
            "medium": sum(1 for a in alerts if a.severity == "medium"),
            "low": sum(1 for a in alerts if a.severity == "low"),
        }
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return AnomalyDetectionResponse(
            total_entities_checked=len(entities),
            anomalies_detected=len(alerts),
            critical_count=severity_counts["critical"],
            high_count=severity_counts["high"],
            medium_count=severity_counts["medium"],
            low_count=severity_counts["low"],
            alerts=[AnomalyAlertRead.from_orm(a) for a in alerts],
            execution_time_ms=execution_time_ms,
        )
    
    async def _fetch_entities(
        self,
        db: AsyncSession,
        entity_type: EntityType,
        profile_id: str,
        entity_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch entities to check for anomalies."""
        if entity_type == EntityType.KEYWORD:
            query = select(PPCKeyword).join(PPCCampaign).where(PPCCampaign.profile_id == profile_id)
            if entity_ids:
                query = query.where(PPCKeyword.id.in_(entity_ids))
            result = await db.execute(query)
            keywords = result.scalars().all()
            
            return [
                {
                    "id": str(kw.id),
                    "name": kw.keyword_text,
                    "campaign_id": str(kw.campaign_id),
                    "model": kw,
                }
                for kw in keywords
            ]
        
        elif entity_type == EntityType.CAMPAIGN:
            query = select(PPCCampaign).where(PPCCampaign.profile_id == profile_id)
            if entity_ids:
                query = query.where(PPCCampaign.id.in_(entity_ids))
            result = await db.execute(query)
            campaigns = result.scalars().all()
            
            return [
                {
                    "id": str(camp.id),
                    "name": camp.name,
                    "portfolio_id": str(camp.portfolio_id) if camp.portfolio_id else None,
                    "model": camp,
                }
                for camp in campaigns
            ]
        
        # Portfolio and account support can be added here
        return []
    
    async def _build_dependency_graph(self, db: AsyncSession, profile_id: str):
        """Build dependency graph for root cause analysis."""
        # Keywords -> Campaigns
        query = select(PPCKeyword).join(PPCCampaign).where(PPCCampaign.profile_id == profile_id)
        result = await db.execute(query)
        keywords = result.scalars().all()
        
        for keyword in keywords:
            if keyword.campaign_id:
                self.root_cause_analyzer.add_dependency(
                    child_id=str(keyword.id),
                    parent_id=str(keyword.campaign_id),
                    entity_type="keyword",
                    parent_type="campaign",
                )
        
        # Campaigns -> Portfolios (if applicable)
        query = select(PPCCampaign).where(
            and_(
                PPCCampaign.profile_id == profile_id,
                PPCCampaign.portfolio_id.isnot(None)
            )
        )
        result = await db.execute(query)
        campaigns = result.scalars().all()
        
        for campaign in campaigns:
            if campaign.portfolio_id:
                self.root_cause_analyzer.add_dependency(
                    child_id=str(campaign.id),
                    parent_id=str(campaign.portfolio_id),
                    entity_type="campaign",
                    parent_type="portfolio",
                )
    
    async def _detect_entity_anomaly(
        self,
        db: AsyncSession,
        entity: Dict[str, Any],
        entity_type: EntityType,
        detector_type: DetectorType,
        include_explanation: bool,
        include_root_cause: bool,
    ) -> Optional[AnomalyResult]:
        """Detect anomaly for a single entity."""
        # Fetch historical time-series data (14 days)
        sequence = await self._fetch_time_series(
            db,
            entity_type,
            entity["id"],
            days=14,
        )
        
        if sequence is None or len(sequence) < 3:
            return None  # Insufficient data
        
        # Fetch current features for streaming detector
        features = await self._extract_features(db, entity_type, entity["id"])
        
        if not features:
            return None
        
        # Find related anomalies for root cause analysis
        related_anomalies = []
        if include_root_cause:
            related_anomalies = await self._fetch_recent_anomalies(
                db,
                entity_type,
                exclude_entity_id=entity["id"],
                hours=1,  # Last hour
            )
        
        # Run detection
        result = self.ensemble.detect_with_explanation(
            sequence=sequence,
            features=features,
            entity_id=entity["id"],
            entity_type=entity_type.value,
            related_anomalies=related_anomalies,
        )
        
        # Add entity name
        result.metric_name = f"{entity_type.value}_{entity['name']}"
        
        return result
    
    async def _fetch_time_series(
        self,
        db: AsyncSession,
        entity_type: EntityType,
        entity_id: str,
        days: int = 14,
    ) -> Optional[np.ndarray]:
        """Fetch historical time-series data for LSTM detector."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Query performance records using FK-based filters
        if entity_type == EntityType.KEYWORD:
            filter_clause = and_(
                PerformanceRecord.keyword_id == int(entity_id),
                PerformanceRecord.date >= since_date,
            )
        else:  # CAMPAIGN
            filter_clause = and_(
                PerformanceRecord.campaign_id == int(entity_id),
                PerformanceRecord.date >= since_date,
            )
        query = select(PerformanceRecord).where(filter_clause).order_by(PerformanceRecord.date.asc())
        
        result = await db.execute(query)
        metrics = result.scalars().all()
        
        if not metrics:
            return None
        
        # Convert to numpy array (5 features: impressions, clicks, spend, sales, orders)
        sequence = np.array([
            [
                float(m.impressions or 0),
                float(m.clicks or 0),
                float(m.spend or 0),
                float(m.sales or 0),
                float(m.orders or 0),
            ]
            for m in metrics
        ])
        
        return sequence
    
    async def _extract_features(
        self,
        db: AsyncSession,
        entity_type: EntityType,
        entity_id: str,
    ) -> Optional[Dict[str, float]]:
        """Extract current features for streaming detector."""
        # Get latest performance record using FK-based filters
        if entity_type == EntityType.KEYWORD:
            filter_clause = PerformanceRecord.keyword_id == int(entity_id)
        else:  # CAMPAIGN
            filter_clause = PerformanceRecord.campaign_id == int(entity_id)
        query = select(PerformanceRecord).where(filter_clause).order_by(PerformanceRecord.date.desc()).limit(1)
        
        result = await db.execute(query)
        metric = result.scalar_one_or_none()
        
        if not metric:
            return None
        
        impressions = float(metric.impressions or 0)
        clicks = float(metric.clicks or 0)
        spend = float(metric.spend or 0)
        sales = float(metric.sales or 0)
        
        return {
            "impressions": impressions,
            "clicks": clicks,
            "spend": spend,
            "sales": sales,
            "ctr": clicks / impressions if impressions > 0 else 0,
            "acos": (spend / sales * 100) if sales > 0 else 0,
            "cpc": spend / clicks if clicks > 0 else 0,
        }
    
    async def _fetch_recent_anomalies(
        self,
        db: AsyncSession,
        entity_type: EntityType,
        exclude_entity_id: str,
        hours: int = 1,
    ) -> List[Tuple[str, str]]:
        """Fetch recent anomalies for root cause analysis."""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = select(AnomalyAlert).where(
            and_(
                AnomalyAlert.detection_timestamp >= since_time,
                AnomalyAlert.entity_id != exclude_entity_id,
            )
        )
        
        result = await db.execute(query)
        alerts = result.scalars().all()
        
        return [(alert.entity_id, alert.entity_type) for alert in alerts]
    
    async def _save_alert(
        self,
        db: AsyncSession,
        result: AnomalyResult,
        profile_id: str,
    ) -> AnomalyAlert:
        """Save anomaly alert to database."""
        alert = AnomalyAlert(
            entity_type=result.metric_name.split("_")[0],  # Extract type
            entity_id=result.metric_name.split("_")[-1],    # Extract ID (simplified)
            entity_name=result.metric_name,
            profile_id=profile_id,
            anomaly_score=result.anomaly_score,
            threshold=result.threshold,
            severity=result.severity,
            metric_name=result.metric_name,
            actual_value=result.actual_value,
            expected_value=result.expected_value,
            reconstruction_error=result.reconstruction_error,
            detector_type="ensemble",
            detection_timestamp=result.timestamp,
            explanation=result.explanation,
            root_causes=result.root_causes,
        )
        
        db.add(alert)
        await db.commit()
        await db.refresh(alert)
        
        # Also archive to history
        await self._archive_to_history(db, alert)
        
        return alert
    
    async def _archive_to_history(self, db: AsyncSession, alert: AnomalyAlert):
        """Archive alert to history table for long-term analysis."""
        history = AnomalyHistory(
            entity_type=alert.entity_type,
            entity_id=alert.entity_id,
            entity_name=alert.entity_name,
            profile_id=alert.profile_id,
            anomaly_score=alert.anomaly_score,
            threshold=alert.threshold,
            severity=alert.severity,
            metric_name=alert.metric_name,
            actual_value=alert.actual_value,
            expected_value=alert.expected_value,
            reconstruction_error=alert.reconstruction_error,
            detector_type=alert.detector_type,
            detection_timestamp=alert.detection_timestamp,
            explanation=alert.explanation,
            root_causes=alert.root_causes,
        )
        
        db.add(history)
        await db.commit()
    
    # ═══════════════════════════════════════════════════════════════════
    #  Alert Management Methods
    # ═══════════════════════════════════════════════════════════════════
    
    async def get_active_alerts(
        self,
        db: AsyncSession,
        profile_id: str,
        severity: Optional[SeverityLevel] = None,
        limit: int = 100,
    ) -> List[AnomalyAlert]:
        """Get active (unresolved) alerts."""
        query = select(AnomalyAlert).where(
            and_(
                AnomalyAlert.profile_id == profile_id,
                AnomalyAlert.is_resolved == False,
            )
        ).order_by(desc(AnomalyAlert.detection_timestamp))
        
        if severity:
            query = query.where(AnomalyAlert.severity == severity.value)
        
        query = query.limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def acknowledge_alert(
        self,
        db: AsyncSession,
        alert_id: int,
        acknowledged_by: str,
    ) -> AnomalyAlert:
        """Mark alert as acknowledged."""
        query = select(AnomalyAlert).where(AnomalyAlert.id == alert_id)
        result = await db.execute(query)
        alert = result.scalar_one_or_none()
        
        if alert:
            alert.is_acknowledged = True
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            await db.commit()
            await db.refresh(alert)
        
        return alert
    
    async def resolve_alert(
        self,
        db: AsyncSession,
        alert_id: int,
        resolution_notes: Optional[str] = None,
    ) -> AnomalyAlert:
        """Mark alert as resolved."""
        query = select(AnomalyAlert).where(AnomalyAlert.id == alert_id)
        result = await db.execute(query)
        alert = result.scalar_one_or_none()
        
        if alert:
            alert.is_resolved = True
            alert.resolved_at = datetime.utcnow()
            alert.resolution_notes = resolution_notes
            
            # Update history with resolution info
            time_to_resolve = (alert.resolved_at - alert.detection_timestamp).total_seconds() / 60
            
            history_query = update(AnomalyHistory).where(
                and_(
                    AnomalyHistory.entity_id == alert.entity_id,
                    AnomalyHistory.detection_timestamp == alert.detection_timestamp,
                )
            ).values(
                was_resolved=True,
                resolution_time_minutes=int(time_to_resolve),
            )
            
            await db.execute(history_query)
            await db.commit()
            await db.refresh(alert)
        
        return alert
    
    async def get_statistics(
        self,
        db: AsyncSession,
        profile_id: str,
    ) -> AnomalyStatistics:
        """Get aggregated anomaly statistics."""
        # Total alerts
        total_query = select(func.count()).select_from(AnomalyAlert).where(
            AnomalyAlert.profile_id == profile_id
        )
        total = await db.scalar(total_query)
        
        # Unresolved
        unresolved_query = select(func.count()).select_from(AnomalyAlert).where(
            and_(
                AnomalyAlert.profile_id == profile_id,
                AnomalyAlert.is_resolved == False,
            )
        )
        unresolved = await db.scalar(unresolved_query)
        
        # By severity
        severity_query = select(
            AnomalyAlert.severity,
            func.count()
        ).where(
            and_(
                AnomalyAlert.profile_id == profile_id,
                AnomalyAlert.is_resolved == False,
            )
        ).group_by(AnomalyAlert.severity)
        
        result = await db.execute(severity_query)
        severity_counts = dict(result.all())
        
        # Avg resolution time
        avg_resolution_query = select(
            func.avg(AnomalyHistory.resolution_time_minutes)
        ).where(
            and_(
                AnomalyHistory.profile_id == profile_id,
                AnomalyHistory.was_resolved == True,
            )
        )
        avg_resolution = await db.scalar(avg_resolution_query)
        
        # Most common entity type
        entity_type_query = select(
            AnomalyAlert.entity_type,
            func.count().label("count")
        ).where(
            AnomalyAlert.profile_id == profile_id
        ).group_by(AnomalyAlert.entity_type).order_by(desc("count")).limit(1)
        entity_type_result = await db.execute(entity_type_query)
        entity_type_row = entity_type_result.first()
        most_common_entity_type = entity_type_row[0] if entity_type_row else "keyword"

        # Most common root cause (root_causes is a JSON list of strings per alert)
        root_cause_query = select(AnomalyAlert.root_causes).where(
            and_(
                AnomalyAlert.profile_id == profile_id,
                AnomalyAlert.root_causes.isnot(None),
            )
        )
        root_cause_result = await db.execute(root_cause_query)
        all_root_causes = root_cause_result.scalars().all()
        cause_counts: Dict[str, int] = {}
        for causes in all_root_causes:
            if causes:
                for cause in causes:
                    cause_counts[cause] = cause_counts.get(cause, 0) + 1
        most_common_root_cause = max(cause_counts, key=cause_counts.get) if cause_counts else None

        # Detection rates
        now = datetime.utcnow()
        rate_24h_query = select(func.count()).select_from(AnomalyAlert).where(
            and_(
                AnomalyAlert.profile_id == profile_id,
                AnomalyAlert.detection_timestamp >= now - timedelta(hours=24),
            )
        )
        detection_rate_last_24h = await db.scalar(rate_24h_query) or 0

        rate_7d_query = select(func.count()).select_from(AnomalyAlert).where(
            and_(
                AnomalyAlert.profile_id == profile_id,
                AnomalyAlert.detection_timestamp >= now - timedelta(days=7),
            )
        )
        detection_rate_last_7d = await db.scalar(rate_7d_query) or 0

        return AnomalyStatistics(
            total_alerts=total or 0,
            unresolved_count=unresolved or 0,
            critical_count=severity_counts.get("critical", 0),
            high_count=severity_counts.get("high", 0),
            medium_count=severity_counts.get("medium", 0),
            low_count=severity_counts.get("low", 0),
            avg_resolution_time_minutes=float(avg_resolution) if avg_resolution else None,
            most_common_entity_type=most_common_entity_type,
            most_common_root_cause=most_common_root_cause,
            detection_rate_last_24h=detection_rate_last_24h,
            detection_rate_last_7d=detection_rate_last_7d,
        )


# Singleton instance
anomaly_service = AnomalyDetectionService()
