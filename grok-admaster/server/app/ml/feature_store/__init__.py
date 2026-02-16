"""
Feature Store

Centralized feature engineering and management for ML models.

Benefits:
- Reusable features across models
- Feature versioning for reproducibility
- Cached computation for performance
- Consistent features between training and production

Usage:
    from app.ml.feature_store import FeatureStore, Feature

    store = FeatureStore()
    features = await store.get_features(
        account_id=123,
        feature_names=["acos_7d", "roas_30d", "click_through_rate"]
    )
"""

from app.ml.feature_store.registry import FeatureStore, Feature, FeatureGroup, feature_store
from app.ml.feature_store.definitions import campaign_features, account_features

__all__ = [
    "FeatureStore",
    "feature_store",
    "Feature",
    "FeatureGroup",
    "campaign_features",
    "account_features",
]
