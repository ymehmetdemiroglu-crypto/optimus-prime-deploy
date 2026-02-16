"""
Feature Store Registry

Manages feature definitions, versioning, and computation.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class FeatureType(str, Enum):
    """Feature data types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    ARRAY = "array"


@dataclass
class Feature:
    """
    Feature definition.

    Attributes:
        name: Unique feature name (e.g., "acos_7d")
        description: Human-readable description
        feature_type: Data type of the feature
        compute_fn: Async function that computes the feature value
        dependencies: List of feature names this feature depends on
        version: Feature version for reproducibility
        ttl_seconds: Cache time-to-live (None = cache forever)
        tags: Optional tags for grouping/filtering
    """

    name: str
    description: str
    feature_type: FeatureType
    compute_fn: Callable
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    ttl_seconds: Optional[int] = 3600  # 1 hour default cache
    tags: List[str] = field(default_factory=list)

    @property
    def cache_key(self) -> str:
        """Generate cache key for this feature"""
        return f"feature:{self.name}:v{self.version}"

    def compute_fingerprint(self, params: Dict[str, Any]) -> str:
        """
        Generate unique fingerprint for feature computation.

        Used to cache feature values based on input parameters.

        Args:
            params: Parameters used to compute the feature

        Returns:
            str: SHA256 hash of (name, version, params)
        """
        data = {
            "name": self.name,
            "version": self.version,
            "params": params
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class FeatureGroup:
    """
    Group of related features.

    Used to organize features logically and compute multiple features efficiently.
    """

    name: str
    description: str
    features: List[Feature] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def add_feature(self, feature: Feature) -> None:
        """Add a feature to this group"""
        self.features.append(feature)

    def get_feature(self, name: str) -> Optional[Feature]:
        """Get a feature by name from this group"""
        for feature in self.features:
            if feature.name == name:
                return feature
        return None


class FeatureStore:
    """
    Central feature store for managing and computing features.

    Usage:
        store = FeatureStore()

        # Register features
        store.register(Feature(
            name="acos_7d",
            description="7-day ACoS",
            feature_type=FeatureType.NUMERIC,
            compute_fn=compute_acos_7d
        ))

        # Get features
        features = await store.get_features(
            account_id=123,
            feature_names=["acos_7d", "roas_30d"]
        )
    """

    def __init__(self, redis_client=None):
        """
        Initialize feature store.

        Args:
            redis_client: Optional Redis client for caching
        """
        self._features: Dict[str, Feature] = {}
        self._groups: Dict[str, FeatureGroup] = {}
        self._redis = redis_client
        logger.info("Feature store initialized")

    def register(self, feature: Feature) -> None:
        """
        Register a feature in the store.

        Args:
            feature: Feature to register

        Raises:
            ValueError: If feature name already exists
        """
        if feature.name in self._features:
            logger.warning(f"Feature {feature.name} already registered, overwriting")

        self._features[feature.name] = feature
        logger.info(
            f"Registered feature: {feature.name} (v{feature.version}) - {feature.description}"
        )

    def register_group(self, group: FeatureGroup) -> None:
        """
        Register a feature group.

        Args:
            group: Feature group to register
        """
        if group.name in self._groups:
            logger.warning(f"Feature group {group.name} already registered, overwriting")

        self._groups[group.name] = group

        # Register individual features
        for feature in group.features:
            self.register(feature)

        logger.info(
            f"Registered feature group: {group.name} with {len(group.features)} features"
        )

    def get_feature(self, name: str) -> Optional[Feature]:
        """
        Get a feature by name.

        Args:
            name: Feature name

        Returns:
            Feature or None if not found
        """
        return self._features.get(name)

    def list_features(self, tags: Optional[List[str]] = None) -> List[Feature]:
        """
        List all features, optionally filtered by tags.

        Args:
            tags: Optional list of tags to filter by

        Returns:
            List of features
        """
        features = list(self._features.values())

        if tags:
            features = [
                f for f in features
                if any(tag in f.tags for tag in tags)
            ]

        return features

    async def get_features(
        self,
        db: AsyncSession,
        entity_id: int,
        feature_names: List[str],
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Compute and return requested features for an entity.

        Args:
            db: Database session
            entity_id: ID of the entity (account, campaign, etc.)
            feature_names: List of feature names to compute
            params: Optional parameters for feature computation
            use_cache: Whether to use cached values

        Returns:
            Dict mapping feature names to computed values

        Example:
            features = await store.get_features(
                db=db,
                entity_id=123,
                feature_names=["acos_7d", "roas_30d"],
                params={"date_range": 7}
            )
            # Returns: {"acos_7d": 15.2, "roas_30d": 6.5}
        """
        params = params or {}
        results = {}

        for name in feature_names:
            feature = self.get_feature(name)
            if not feature:
                logger.warning(f"Feature {name} not found in registry")
                continue

            # Try to get from cache
            if use_cache and feature.ttl_seconds and self._redis:
                cached_value = await self._get_cached(feature, entity_id, params)
                if cached_value is not None:
                    results[name] = cached_value
                    continue

            # Compute feature
            try:
                value = await feature.compute_fn(db, entity_id, **params)
                results[name] = value

                # Cache the result
                if use_cache and feature.ttl_seconds and self._redis:
                    await self._cache_value(feature, entity_id, params, value)

                logger.debug(f"Computed feature {name} for entity {entity_id}: {value}")

            except Exception as e:
                logger.error(f"Error computing feature {name}: {e}")
                results[name] = None

        return results

    async def _get_cached(
        self,
        feature: Feature,
        entity_id: int,
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached feature value"""
        if not self._redis:
            return None

        try:
            fingerprint = feature.compute_fingerprint({"entity_id": entity_id, **params})
            cache_key = f"{feature.cache_key}:{fingerprint}"

            cached = await self._redis.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {feature.name}")
                return json.loads(cached)

        except Exception as e:
            logger.warning(f"Error getting cached value: {e}")

        return None

    async def _cache_value(
        self,
        feature: Feature,
        entity_id: int,
        params: Dict[str, Any],
        value: Any
    ) -> None:
        """Cache computed feature value"""
        if not self._redis:
            return

        try:
            fingerprint = feature.compute_fingerprint({"entity_id": entity_id, **params})
            cache_key = f"{feature.cache_key}:{fingerprint}"

            await self._redis.setex(
                cache_key,
                feature.ttl_seconds,
                json.dumps(value)
            )

            logger.debug(f"Cached {feature.name} for {feature.ttl_seconds}s")

        except Exception as e:
            logger.warning(f"Error caching value: {e}")

    async def invalidate_cache(
        self,
        feature_name: str,
        entity_id: Optional[int] = None
    ) -> None:
        """
        Invalidate cached feature values.

        Args:
            feature_name: Name of feature to invalidate
            entity_id: Optional entity ID (if None, invalidates all)
        """
        if not self._redis:
            return

        feature = self.get_feature(feature_name)
        if not feature:
            return

        try:
            pattern = f"{feature.cache_key}:*"
            if entity_id:
                pattern = f"{feature.cache_key}:*entity_id*{entity_id}*"

            # Delete matching keys
            async for key in self._redis.scan_iter(match=pattern):
                await self._redis.delete(key)

            logger.info(f"Invalidated cache for {feature_name}")

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")


# Global feature store instance
feature_store = FeatureStore()
