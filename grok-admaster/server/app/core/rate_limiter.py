import asyncio
import redis
import logging
import os
from datetime import datetime
from typing import Optional

class APIRateLimiter:
    """Token bucket rate limiter with Redis persistence for distributed throttling."""
    
    def __init__(self, api_name: str, rate_limit: int, per_seconds: int, redis_url: Optional[str] = None):
        if not redis_url:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            
        try:
            self.redis_client = redis.from_url(redis_url)
            self.persistence_enabled = True
        except Exception as e:
            logging.error(f"RateLimiter: Redis connection failed: {e}. Falling back to in-memory.")
            self.persistence_enabled = False
            self._local_history = []
            
        self.api_name = api_name
        self.rate_limit = rate_limit
        self.per_seconds = per_seconds
        self.key = f"rate_limit:{api_name}"
        self.logger = logging.getLogger(f"rate_limiter.{api_name}")
        
    async def acquire(self):
        """Wait until rate limit allows request (Token Bucket / Sliding Window)."""
        if not self.persistence_enabled:
            return await self._acquire_local()
            
        while True:
            now = datetime.utcnow().timestamp()
            window_start = now - self.per_seconds
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            # Remove old timestamps
            pipe.zremrangebyscore(self.key, '-inf', window_start)
            # Check current count
            pipe.zcard(self.key)
            results = pipe.execute()
            
            current_count = results[1]
            
            if current_count < self.rate_limit:
                # Add timestamp - using current time as both member and score
                # We use a unique member (timestamp + random/nonce if needed) to allow multiple entries at same ms
                member = f"{now}:{os.urandom(4).hex()}"
                self.redis_client.zadd(self.key, {member: now})
                self.redis_client.expire(self.key, self.per_seconds * 2)
                return
            else:
                # Calculate wait time from the oldest element in window
                oldest = self.redis_client.zrange(self.key, 0, 0, withscores=True)
                if oldest:
                    oldest_score = oldest[0][1]
                    wait_until = oldest_score + self.per_seconds
                    wait_seconds = max(0.1, wait_until - now)
                    self.logger.info(f"Rate limit hit for {self.api_name}, throttling for {wait_seconds:.2f}s")
                    await asyncio.sleep(wait_seconds)
                else:
                    await asyncio.sleep(1)

    async def _acquire_local(self):
        """Fallback in-memory rate limiting."""
        while True:
            now = datetime.utcnow().timestamp()
            window_start = now - self.per_seconds
            
            # Clean history
            self._local_history = [t for t in self._local_history if t > window_start]
            
            if len(self._local_history) < self.rate_limit:
                self._local_history.append(now)
                return
            else:
                wait_seconds = max(0.1, (self._local_history[0] + self.per_seconds) - now)
                await asyncio.sleep(wait_seconds)
