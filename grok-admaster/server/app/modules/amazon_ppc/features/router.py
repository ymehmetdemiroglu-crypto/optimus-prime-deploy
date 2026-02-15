"""
API endpoints for feature engineering operations.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import date

from app.core.database import get_db
from ..features import FeatureEngineer, FeatureStore, KeywordFeatureEngineer

router = APIRouter()

@router.get("/campaign/{campaign_id}")
async def get_campaign_features(
    campaign_id: int,
    refresh: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Get computed features for a campaign.
    If refresh=True, recompute features even if cached.
    """
    store = FeatureStore(db)
    
    # Try to get cached features first
    if not refresh:
        cached = await store.get_latest_features(campaign_id)
        if cached:
            return {"source": "cache", "features": cached}
    
    # Compute fresh features
    engineer = FeatureEngineer(db)
    features = await engineer.compute_full_feature_vector(campaign_id)
    
    # Cache the result
    rolling = await engineer.compute_rolling_metrics(campaign_id)
    seasonality = engineer.compute_seasonality_features()
    trends = await engineer.compute_trend_features(campaign_id)
    competition = await engineer.compute_competition_features(campaign_id)
    
    await store.save_features(
        campaign_id=campaign_id,
        features=features,
        rolling=rolling,
        seasonality=seasonality,
        trends=trends,
        competition=competition
    )
    
    return {"source": "computed", "features": features}


@router.get("/campaign/{campaign_id}/rolling")
async def get_rolling_metrics(
    campaign_id: int,
    windows: str = "7,14,30",
    db: AsyncSession = Depends(get_db)
):
    """
    Get rolling average metrics for specific time windows.
    windows: comma-separated list of days (e.g., "7,14,30")
    """
    window_list = [int(w.strip()) for w in windows.split(",")]
    
    engineer = FeatureEngineer(db)
    metrics = await engineer.compute_rolling_metrics(campaign_id, window_list)
    
    return {"campaign_id": campaign_id, "windows": window_list, "metrics": metrics}


@router.get("/campaign/{campaign_id}/trends")
async def get_trend_features(
    campaign_id: int,
    short_window: int = 7,
    long_window: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    Get trend indicators comparing short-term vs long-term performance.
    """
    engineer = FeatureEngineer(db)
    trends = await engineer.compute_trend_features(campaign_id, short_window, long_window)
    
    return {"campaign_id": campaign_id, "trends": trends}


@router.get("/seasonality")
async def get_seasonality_features(
    target_date: Optional[str] = None
):
    """
    Get seasonality features for a given date.
    target_date format: YYYY-MM-DD
    """
    from ..features import FeatureEngineer
    
    engineer = FeatureEngineer(None)  # No DB needed for seasonality
    
    if target_date:
        parsed_date = date.fromisoformat(target_date)
    else:
        parsed_date = None
    
    features = engineer.compute_seasonality_features(parsed_date)
    
    return {"date": str(parsed_date or date.today()), "seasonality": features}


@router.get("/keyword/{keyword_id}")
async def get_keyword_features(
    keyword_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get computed features for a specific keyword.
    """
    engineer = KeywordFeatureEngineer(db)
    features = await engineer.compute_keyword_features(keyword_id)
    
    return {"keyword_id": keyword_id, "features": features}


@router.get("/keyword/{keyword_id}/bid-recommendations")
async def get_bid_recommendations(
    keyword_id: int,
    target_acos: float = 25.0,
    target_roas: float = 4.0,
    db: AsyncSession = Depends(get_db)
):
    """
    Get bid optimization recommendations for a keyword.
    """
    engineer = KeywordFeatureEngineer(db)
    recommendations = await engineer.compute_bid_recommendations(
        keyword_id=keyword_id,
        target_acos=target_acos,
        target_roas=target_roas
    )
    
    return recommendations


@router.get("/campaign/{campaign_id}/keywords")
async def get_all_keyword_features(
    campaign_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get features for all keywords in a campaign.
    """
    engineer = KeywordFeatureEngineer(db)
    features = await engineer.bulk_compute_features(campaign_id)
    
    return {
        "campaign_id": campaign_id,
        "keyword_count": len(features),
        "keywords": features
    }


@router.post("/batch-compute")
async def batch_compute_features(
    db: AsyncSession = Depends(get_db)
):
    """
    Compute and cache features for all active campaigns.
    """
    engineer = FeatureEngineer(db)
    all_features = await engineer.compute_features_for_all_campaigns()
    
    # Save to store
    store = FeatureStore(db)
    saved_count = 0
    
    for features in all_features:
        campaign_id = features.get('campaign_id')
        if campaign_id:
            await store.save_features(campaign_id=campaign_id, features=features)
            saved_count += 1
    
    return {
        "status": "completed",
        "campaigns_processed": len(all_features),
        "features_cached": saved_count
    }


@router.delete("/cleanup")
async def cleanup_old_features(
    retention_days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    Remove old feature snapshots beyond retention period.
    """
    store = FeatureStore(db)
    await store.cleanup_old_snapshots(retention_days)
    
    return {"status": "cleanup_completed", "retention_days": retention_days}
