"""
Redis Caching Layer

Provides caching decorators and utilities for expensive operations.

Usage:
    from app.core.cache import cached, invalidate_cache

    @cached(ttl=3600, key_prefix="user")
    async def get_user_by_id(user_id: int):
        # Expensive database query
        return await db.query(User).filter_by(id=user_id).first()

    # Invalidate cache
    await invalidate_cache("user", user_id)
"""

from typing import Optional, Any, Callable, Union
from functools import wraps
import json
import hashlib
import pickle
from datetime import timedelta
import redis.asyncio as redis

from app.core.logging_config import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class CacheClient:
    """
    Redis cache client with graceful fallback.

    Handles Redis connection failures gracefully and provides
    caching utilities.
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize cache client.

        Args:
            redis_url: Redis connection URL (defaults to settings.REDIS_URL)
        """
        self._redis_url = redis_url or getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        self._client: Optional[redis.Redis] = None
        self._enabled = False

    async def connect(self) -> None:
        """
        Connect to Redis.

        Gracefully handles connection failures.
        """
        try:
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._client.ping()
            self._enabled = True
            logger.info("[OK] Redis cache client connected")

        except Exception as e:
            logger.warning(f"[WARN] Redis not available: {e}")
            logger.warning("   Caching disabled (will compute all values)")
            self._enabled = False
            self._client = None

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self._client:
            await self._client.close()
            logger.info("Redis cache client disconnected")

    @property
    def is_enabled(self) -> bool:
        """Check if caching is enabled"""
        return self._enabled

    def _make_key(self, key_prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from prefix and arguments.

        Args:
            key_prefix: Key prefix (e.g., "user", "campaign")
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            str: Cache key
        """
        # Create deterministic key from arguments
        key_data = {
            "args": args,
            "kwargs": {k: v for k, v in sorted(kwargs.items())}
        }

        # Hash for long keys
        json_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(json_str.encode()).hexdigest()

        return f"cache:{key_prefix}:{key_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self._enabled or not self._client:
            return None

        try:
            value = await self._client.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            else:
                logger.debug(f"Cache miss: {key}")
                return None

        except Exception as e:
            logger.warning(f"Error getting from cache: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (None = no expiration)
        """
        if not self._enabled or not self._client:
            return

        try:
            json_value = json.dumps(value, default=str)

            if ttl:
                await self._client.setex(key, ttl, json_value)
            else:
                await self._client.set(key, json_value)

            logger.debug(f"Cached: {key} (TTL: {ttl}s)")

        except Exception as e:
            logger.warning(f"Error setting cache: {e}")

    async def delete(self, key: str) -> None:
        """
        Delete key from cache.

        Args:
            key: Cache key
        """
        if not self._enabled or not self._client:
            return

        try:
            await self._client.delete(key)
            logger.debug(f"Deleted from cache: {key}")

        except Exception as e:
            logger.warning(f"Error deleting from cache: {e}")

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "cache:user:*")

        Returns:
            int: Number of keys deleted
        """
        if not self._enabled or not self._client:
            return 0

        try:
            deleted = 0
            async for key in self._client.scan_iter(match=pattern):
                await self._client.delete(key)
                deleted += 1

            logger.info(f"Deleted {deleted} keys matching pattern: {pattern}")
            return deleted

        except Exception as e:
            logger.warning(f"Error deleting pattern from cache: {e}")
            return 0

    async def clear_all(self) -> None:
        """
        Clear all cache keys.

        WARNING: Use with caution!
        """
        if not self._enabled or not self._client:
            return

        try:
            await self._client.flushdb()
            logger.warning("[WARN] Cleared entire cache database")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


# Global cache client instance
cache_client = CacheClient()


def cached(
    ttl: int = 3600,
    key_prefix: str = "default",
    skip_cache: bool = False
) -> Callable:
    """
    Decorator to cache function results in Redis.

    Args:
        ttl: Time-to-live in seconds (default: 1 hour)
        key_prefix: Prefix for cache key (e.g., "user", "campaign")
        skip_cache: If True, always compute (useful for debugging)

    Usage:
        @cached(ttl=3600, key_prefix="user")
        async def get_user_by_id(user_id: int):
            return await db.query(User).get(user_id)

        # Cache key will be: cache:user:<hash_of_args>
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Skip cache if disabled or requested
            if skip_cache or not cache_client.is_enabled:
                return await func(*args, **kwargs)

            # Generate cache key
            cache_key = cache_client._make_key(key_prefix, *args, **kwargs)

            # Try to get from cache
            cached_value = await cache_client.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Compute value
            result = await func(*args, **kwargs)

            # Cache the result
            await cache_client.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


async def invalidate_cache(key_prefix: str, *args, **kwargs) -> None:
    """
    Invalidate cache for specific key.

    Args:
        key_prefix: Key prefix (e.g., "user", "campaign")
        *args: Arguments used to generate original cache key
        **kwargs: Keyword arguments used to generate original cache key

    Usage:
        # Invalidate specific user cache
        await invalidate_cache("user", user_id=123)

        # Invalidate all user caches
        await invalidate_cache_pattern("user:*")
    """
    cache_key = cache_client._make_key(key_prefix, *args, **kwargs)
    await cache_client.delete(cache_key)
    logger.info(f"Invalidated cache: {key_prefix}")


async def invalidate_cache_pattern(pattern: str) -> int:
    """
    Invalidate all cache keys matching pattern.

    Args:
        pattern: Pattern to match (e.g., "cache:user:*")

    Returns:
        int: Number of keys deleted

    Usage:
        # Invalidate all user caches
        await invalidate_cache_pattern("cache:user:*")

        # Invalidate all caches for account 123
        await invalidate_cache_pattern("cache:*:*account_id*123*")
    """
    deleted = await cache_client.delete_pattern(pattern)
    logger.info(f"Invalidated {deleted} cache keys matching: {pattern}")
    return deleted


class CacheWarmer:
    """
    Utility for pre-warming cache with commonly accessed data.

    Usage:
        warmer = CacheWarmer()
        await warmer.warm_user_cache([1, 2, 3, 4, 5])
    """

    def __init__(self, cache: CacheClient):
        self.cache = cache

    async def warm(
        self,
        func: Callable,
        key_prefix: str,
        params_list: list,
        ttl: int = 3600
    ) -> int:
        """
        Warm cache for multiple parameter sets.

        Args:
            func: Async function to compute values
            key_prefix: Cache key prefix
            params_list: List of parameter tuples/dicts
            ttl: Cache TTL

        Returns:
            int: Number of items cached
        """
        cached_count = 0

        for params in params_list:
            try:
                # Handle both tuple and dict params
                if isinstance(params, dict):
                    result = await func(**params)
                    cache_key = self.cache._make_key(key_prefix, **params)
                else:
                    result = await func(*params)
                    cache_key = self.cache._make_key(key_prefix, *params)

                await self.cache.set(cache_key, result, ttl=ttl)
                cached_count += 1

            except Exception as e:
                logger.error(f"Error warming cache for {params}: {e}")

        logger.info(f"Warmed cache: {cached_count}/{len(params_list)} items ({key_prefix})")
        return cached_count
