"""
Feature Definitions

Concrete feature implementations for Amazon PPC campaigns.

Add new features here and register them with the feature store.
"""

from datetime import datetime, timedelta
from typing import Any, Dict
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.feature_store.registry import Feature, FeatureGroup, FeatureType
from app.modules.amazon_ppc.accounts.models import Account, Profile
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Campaign Performance Features
# ============================================================================

async def compute_acos_7d(db: AsyncSession, campaign_id: int, **kwargs) -> float:
    """
    Compute 7-day Advertising Cost of Sale (ACoS).

    ACoS = (Ad Spend / Sales) * 100

    Args:
        db: Database session
        campaign_id: Campaign ID

    Returns:
        float: ACoS percentage
    """
    # TODO: Implement actual query to metrics table
    # This is a placeholder implementation
    # In production, query from campaign_metrics table with date filter

    # Example query:
    # result = await db.execute(
    #     select(
    #         func.sum(Metric.spend),
    #         func.sum(Metric.sales)
    #     )
    #     .where(Metric.campaign_id == campaign_id)
    #     .where(Metric.date >= datetime.utcnow() - timedelta(days=7))
    # )
    # spend, sales = result.one()
    # return (spend / sales * 100) if sales > 0 else 0.0

    # Placeholder
    return 15.5


async def compute_roas_30d(db: AsyncSession, campaign_id: int, **kwargs) -> float:
    """
    Compute 30-day Return on Ad Spend (ROAS).

    ROAS = Sales / Ad Spend

    Args:
        db: Database session
        campaign_id: Campaign ID

    Returns:
        float: ROAS ratio
    """
    # TODO: Implement actual query
    # Placeholder
    return 6.7


async def compute_ctr_7d(db: AsyncSession, campaign_id: int, **kwargs) -> float:
    """
    Compute 7-day Click-Through Rate (CTR).

    CTR = (Clicks / Impressions) * 100

    Args:
        db: Database session
        campaign_id: Campaign ID

    Returns:
        float: CTR percentage
    """
    # TODO: Implement actual query
    # Placeholder
    return 1.85


async def compute_conversion_rate_7d(db: AsyncSession, campaign_id: int, **kwargs) -> float:
    """
    Compute 7-day conversion rate.

    Conversion Rate = (Conversions / Clicks) * 100

    Args:
        db: Database session
        campaign_id: Campaign ID

    Returns:
        float: Conversion rate percentage
    """
    # TODO: Implement actual query
    # Placeholder
    return 8.2


async def compute_avg_cpc_7d(db: AsyncSession, campaign_id: int, **kwargs) -> float:
    """
    Compute 7-day average Cost Per Click (CPC).

    CPC = Total Spend / Total Clicks

    Args:
        db: Database session
        campaign_id: Campaign ID

    Returns:
        float: Average CPC in currency
    """
    # TODO: Implement actual query
    # Placeholder
    return 0.85


# ============================================================================
# Account-Level Features
# ============================================================================

async def compute_account_total_spend_30d(db: AsyncSession, account_id: int, **kwargs) -> float:
    """
    Compute total ad spend across all campaigns for an account (30 days).

    Args:
        db: Database session
        account_id: Account ID

    Returns:
        float: Total spend
    """
    # TODO: Implement actual query
    # Placeholder
    return 12847.32


async def compute_account_avg_acos_30d(db: AsyncSession, account_id: int, **kwargs) -> float:
    """
    Compute average ACoS across all campaigns for an account (30 days).

    Args:
        db: Database session
        account_id: Account ID

    Returns:
        float: Average ACoS percentage
    """
    # TODO: Implement actual query
    # Placeholder
    return 14.7


async def compute_active_campaign_count(db: AsyncSession, account_id: int, **kwargs) -> int:
    """
    Count number of active campaigns for an account.

    Args:
        db: Database session
        account_id: Account ID

    Returns:
        int: Number of active campaigns
    """
    # TODO: Implement actual query
    # Placeholder
    return 23


# ============================================================================
# Time-Based Features
# ============================================================================

async def compute_spend_trend_7d(db: AsyncSession, campaign_id: int, **kwargs) -> float:
    """
    Compute spend trend over last 7 days.

    Returns positive for increasing spend, negative for decreasing.

    Args:
        db: Database session
        campaign_id: Campaign ID

    Returns:
        float: Percentage change in daily spend
    """
    # TODO: Implement actual query
    # Compute: (avg_last_3_days - avg_previous_4_days) / avg_previous_4_days * 100
    # Placeholder
    return 5.2


async def compute_sales_momentum(db: AsyncSession, campaign_id: int, **kwargs) -> float:
    """
    Compute sales momentum score.

    Higher score indicates increasing sales velocity.

    Args:
        db: Database session
        campaign_id: Campaign ID

    Returns:
        float: Momentum score (0-100)
    """
    # TODO: Implement actual query
    # Could be based on exponential moving average of sales
    # Placeholder
    return 72.5


# ============================================================================
# Competitive Features
# ============================================================================

async def compute_market_share_estimate(db: AsyncSession, campaign_id: int, **kwargs) -> float:
    """
    Estimate market share based on impression share.

    Args:
        db: Database session
        campaign_id: Campaign ID

    Returns:
        float: Estimated market share percentage
    """
    # TODO: Implement actual query
    # Based on impression share data from Amazon API
    # Placeholder
    return 12.3


# ============================================================================
# Feature Groups
# ============================================================================

campaign_features = FeatureGroup(
    name="campaign_performance",
    description="Campaign-level performance features",
    features=[
        Feature(
            name="acos_7d",
            description="7-day Advertising Cost of Sale (%)",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_acos_7d,
            version="1.0.0",
            ttl_seconds=3600,  # Cache for 1 hour
            tags=["campaign", "performance", "acos"]
        ),
        Feature(
            name="roas_30d",
            description="30-day Return on Ad Spend (ratio)",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_roas_30d,
            version="1.0.0",
            ttl_seconds=3600,
            tags=["campaign", "performance", "roas"]
        ),
        Feature(
            name="ctr_7d",
            description="7-day Click-Through Rate (%)",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_ctr_7d,
            version="1.0.0",
            ttl_seconds=3600,
            tags=["campaign", "performance", "engagement"]
        ),
        Feature(
            name="conversion_rate_7d",
            description="7-day Conversion Rate (%)",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_conversion_rate_7d,
            version="1.0.0",
            ttl_seconds=3600,
            tags=["campaign", "performance", "conversion"]
        ),
        Feature(
            name="avg_cpc_7d",
            description="7-day Average Cost Per Click",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_avg_cpc_7d,
            version="1.0.0",
            ttl_seconds=3600,
            tags=["campaign", "cost", "efficiency"]
        ),
        Feature(
            name="spend_trend_7d",
            description="7-day Spend Trend (% change)",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_spend_trend_7d,
            version="1.0.0",
            ttl_seconds=3600,
            tags=["campaign", "trend", "spend"]
        ),
        Feature(
            name="sales_momentum",
            description="Sales Momentum Score (0-100)",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_sales_momentum,
            version="1.0.0",
            ttl_seconds=3600,
            tags=["campaign", "trend", "sales"]
        ),
    ]
)


account_features = FeatureGroup(
    name="account_metrics",
    description="Account-level aggregate features",
    features=[
        Feature(
            name="account_total_spend_30d",
            description="Total account spend over 30 days",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_account_total_spend_30d,
            version="1.0.0",
            ttl_seconds=7200,  # Cache for 2 hours
            tags=["account", "spend"]
        ),
        Feature(
            name="account_avg_acos_30d",
            description="Average ACoS across all campaigns (30 days)",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_account_avg_acos_30d,
            version="1.0.0",
            ttl_seconds=7200,
            tags=["account", "acos", "performance"]
        ),
        Feature(
            name="active_campaign_count",
            description="Number of active campaigns",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_active_campaign_count,
            version="1.0.0",
            ttl_seconds=7200,
            tags=["account", "campaigns"]
        ),
    ]
)


def register_all_features(store):
    """
    Register all feature groups with the feature store.

    Call this during application startup.

    Args:
        store: FeatureStore instance
    """
    store.register_group(campaign_features)
    store.register_group(account_features)

    logger.info(
        f"Registered {len(store._features)} features across {len(store._groups)} groups"
    )
