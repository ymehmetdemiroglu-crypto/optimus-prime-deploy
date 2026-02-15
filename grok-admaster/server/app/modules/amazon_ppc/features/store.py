"""
Feature Store - Caches computed features for fast retrieval.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from app.core.database import Base

logger = logging.getLogger(__name__)

class FeatureSnapshot(Base):
    """
    Cached feature vectors for campaigns.
    Updated periodically to avoid recomputation.
    """
    __tablename__ = "feature_snapshots"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    campaign_id = Column(Integer, index=True)
    
    # Feature categories stored as JSON for flexibility
    rolling_metrics = Column(JSON)
    seasonality = Column(JSON)
    trends = Column(JSON)
    competition = Column(JSON)
    
    # Full flattened feature vector
    feature_vector = Column(JSON)
    
    # Metadata
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    is_stale = Column(Boolean, default=False)


class FeatureStore:
    """
    Manages feature storage and retrieval.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.stale_threshold_hours = 24  # Features older than this are stale
    
    async def save_features(
        self, 
        campaign_id: int, 
        features: Dict[str, Any],
        rolling: Dict[str, float] = None,
        seasonality: Dict[str, Any] = None,
        trends: Dict[str, float] = None,
        competition: Dict[str, float] = None
    ) -> FeatureSnapshot:
        """
        Save computed features to the store.
        """
        snapshot = FeatureSnapshot(
            campaign_id=campaign_id,
            feature_vector=features,
            rolling_metrics=rolling,
            seasonality=seasonality,
            trends=trends,
            competition=competition,
            computed_at=datetime.now()
        )
        
        self.db.add(snapshot)
        await self.db.commit()
        await self.db.refresh(snapshot)
        
        return snapshot
    
    async def get_latest_features(
        self, 
        campaign_id: int,
        max_age_hours: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent feature vector for a campaign.
        Returns None if no features exist or if they're too stale.
        """
        if max_age_hours is None:
            max_age_hours = self.stale_threshold_hours
        
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        query = (
            select(FeatureSnapshot)
            .where(
                and_(
                    FeatureSnapshot.campaign_id == campaign_id,
                    FeatureSnapshot.computed_at >= cutoff
                )
            )
            .order_by(FeatureSnapshot.computed_at.desc())
            .limit(1)
        )
        
        result = await self.db.execute(query)
        snapshot = result.scalars().first()
        
        if snapshot:
            return snapshot.feature_vector
        return None
    
    async def get_features_batch(
        self, 
        campaign_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get features for multiple campaigns at once.
        """
        cutoff = datetime.now() - timedelta(hours=self.stale_threshold_hours)
        
        query = (
            select(FeatureSnapshot)
            .where(
                and_(
                    FeatureSnapshot.campaign_id.in_(campaign_ids),
                    FeatureSnapshot.computed_at >= cutoff
                )
            )
            .order_by(FeatureSnapshot.computed_at.desc())
        )
        
        result = await self.db.execute(query)
        snapshots = result.scalars().all()
        
        # Get latest per campaign
        latest = {}
        for s in snapshots:
            if s.campaign_id not in latest:
                latest[s.campaign_id] = s.feature_vector
        
        return latest
    
    async def mark_stale(self, campaign_id: int):
        """
        Mark features for a campaign as stale (needs refresh).
        """
        query = (
            select(FeatureSnapshot)
            .where(FeatureSnapshot.campaign_id == campaign_id)
            .order_by(FeatureSnapshot.computed_at.desc())
            .limit(1)
        )
        
        result = await self.db.execute(query)
        snapshot = result.scalars().first()
        
        if snapshot:
            snapshot.is_stale = True
            await self.db.commit()
    
    async def cleanup_old_snapshots(self, retention_days: int = 30):
        """
        Remove feature snapshots older than retention period.
        """
        cutoff = datetime.now() - timedelta(days=retention_days)
        
        from sqlalchemy import delete
        stmt = delete(FeatureSnapshot).where(FeatureSnapshot.computed_at < cutoff)
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        logger.info(f"Cleaned up {result.rowcount} old feature snapshots")
