"""
Query Profiling and Optimization

Tools for monitoring and optimizing database queries.

Features:
- Query timing and profiling
- Slow query detection
- N+1 query detection
- Eager loading helpers
- Query optimization recommendations

Usage:
    from app.core.query_profiler import profile_query, detect_n_plus_1

    @profile_query(slow_threshold_ms=100)
    async def get_accounts():
        return await db.query(Account).all()

    with detect_n_plus_1():
        accounts = await get_accounts()
        for account in accounts:
            # This will trigger N+1 warning if not eager-loaded
            credentials = await account.credentials
"""

from typing import Optional, Any, Callable, List, Dict
from functools import wraps
from contextvars import ContextVar
from datetime import datetime
import time
import warnings

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# Context variables for query tracking
query_count: ContextVar[int] = ContextVar("query_count", default=0)
query_times: ContextVar[List[float]] = ContextVar("query_times", default=[])


class QueryStats:
    """
    Container for query statistics.

    Tracks:
    - Total queries executed
    - Query execution times
    - Slow queries
    - Duplicate queries
    """

    def __init__(self):
        self.total_queries = 0
        self.total_time_ms = 0.0
        self.slow_queries: List[Dict[str, Any]] = []
        self.duplicate_queries: Dict[str, int] = {}
        self.n_plus_1_detected = False

    def add_query(self, query_str: str, duration_ms: float, threshold_ms: float = 100):
        """
        Record a query execution.

        Args:
            query_str: SQL query string
            duration_ms: Execution time in milliseconds
            threshold_ms: Threshold for slow query detection
        """
        self.total_queries += 1
        self.total_time_ms += duration_ms

        # Track slow queries
        if duration_ms >= threshold_ms:
            self.slow_queries.append({
                "query": query_str,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Track duplicate queries (potential N+1)
        normalized_query = self._normalize_query(query_str)
        self.duplicate_queries[normalized_query] = \
            self.duplicate_queries.get(normalized_query, 0) + 1

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query by removing parameter values.

        This helps detect duplicate query patterns.
        """
        # Simple normalization: replace numbers and strings with placeholders
        import re
        normalized = re.sub(r'\d+', '?', query)
        normalized = re.sub(r"'[^']*'", '?', normalized)
        return normalized

    def detect_n_plus_1(self) -> bool:
        """
        Detect potential N+1 query problems.

        Returns:
            True if N+1 pattern detected
        """
        # Check for queries executed many times
        for query, count in self.duplicate_queries.items():
            if count > 10:  # Threshold for N+1 detection
                logger.warning(
                    f"[WARN] Potential N+1 query detected! "
                    f"Query executed {count} times: {query[:100]}..."
                )
                self.n_plus_1_detected = True
                return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """
        Get query statistics summary.

        Returns:
            dict: Summary statistics
        """
        return {
            "total_queries": self.total_queries,
            "total_time_ms": round(self.total_time_ms, 2),
            "average_time_ms": round(
                self.total_time_ms / self.total_queries if self.total_queries > 0 else 0,
                2
            ),
            "slow_queries_count": len(self.slow_queries),
            "slow_queries": self.slow_queries,
            "n_plus_1_detected": self.n_plus_1_detected,
            "unique_queries": len(self.duplicate_queries)
        }


# Global query stats collector
query_stats = QueryStats()


def profile_query(
    slow_threshold_ms: float = 100.0,
    log_all: bool = False
) -> Callable:
    """
    Decorator to profile database query performance.

    Args:
        slow_threshold_ms: Log queries slower than this (milliseconds)
        log_all: If True, log all queries regardless of speed

    Usage:
        @profile_query(slow_threshold_ms=50)
        async def get_accounts(db: AsyncSession):
            return await db.query(Account).all()
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result

            finally:
                duration_ms = (time.time() - start_time) * 1000

                # Log slow queries
                if duration_ms >= slow_threshold_ms or log_all:
                    logger.warning(
                        f"🐢 Slow query in {func.__name__}: {duration_ms:.2f}ms"
                    )

                # Track in stats
                query_stats.add_query(
                    query_str=f"{func.__module__}.{func.__name__}",
                    duration_ms=duration_ms,
                    threshold_ms=slow_threshold_ms
                )

        return wrapper

    return decorator


class QueryProfiler:
    """
    Context manager for profiling queries within a block.

    Usage:
        async with QueryProfiler(name="get_accounts") as profiler:
            accounts = await db.query(Account).all()
            for account in accounts:
                creds = await account.credentials  # Potential N+1

        print(profiler.get_summary())
    """

    def __init__(
        self,
        name: str = "query_block",
        slow_threshold_ms: float = 100.0,
        detect_n_plus_1: bool = True
    ):
        self.name = name
        self.slow_threshold_ms = slow_threshold_ms
        self.detect_n_plus_1 = detect_n_plus_1
        self.stats = QueryStats()
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.time()
        logger.debug(f"Query profiler started: {self.name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000

        # Check for N+1
        if self.detect_n_plus_1:
            self.stats.detect_n_plus_1()

        # Log summary
        summary = self.stats.get_summary()
        logger.info(
            f"Query profiler [{self.name}]: "
            f"{summary['total_queries']} queries, "
            f"{summary['total_time_ms']:.2f}ms total, "
            f"{summary['average_time_ms']:.2f}ms avg"
        )

        if summary['slow_queries_count'] > 0:
            logger.warning(
                f"[WARN] {summary['slow_queries_count']} slow queries detected "
                f"(>{self.slow_threshold_ms}ms)"
            )

        if summary['n_plus_1_detected']:
            logger.error(
                "[ERROR] N+1 query problem detected! "
                "Consider using eager loading (selectinload/joinedload)"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get profiler summary statistics"""
        return self.stats.get_summary()


# SQLAlchemy event listeners for automatic query logging


def setup_query_logging(engine: Engine, echo_queries: bool = False) -> None:
    """
    Set up SQLAlchemy query logging.

    Args:
        engine: SQLAlchemy engine
        echo_queries: If True, echo all queries to console
    """

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        """Log query start time"""
        conn.info.setdefault("query_start_time", []).append(time.time())

    @event.listens_for(engine.sync_engine, "after_cursor_execute")
    def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        """Log query execution time"""
        total_time_ms = (time.time() - conn.info["query_start_time"].pop()) * 1000

        # Track in stats
        query_stats.add_query(statement, total_time_ms)

        # Log if slow or echo enabled
        if echo_queries or total_time_ms >= 100:
            logger.debug(
                f"Query executed in {total_time_ms:.2f}ms: "
                f"{statement[:200]}{'...' if len(statement) > 200 else ''}"
            )


# Eager loading helpers


def eager_load_account_relations():
    """
    Get options for eager-loading Account relations.

    Usage:
        query = select(Account).options(*eager_load_account_relations())
        accounts = await db.execute(query)
    """
    return [
        selectinload("profiles"),
        selectinload("credentials"),
    ]


def eager_load_campaign_relations():
    """
    Get options for eager-loading Campaign relations.

    Usage:
        query = select(Campaign).options(*eager_load_campaign_relations())
        campaigns = await db.execute(query)
    """
    return [
        joinedload("account"),  # One-to-one, use joinedload
        selectinload("ad_groups"),  # One-to-many, use selectinload
        selectinload("keywords"),
    ]


# Query optimization recommendations


class QueryOptimizer:
    """
    Provides query optimization recommendations.

    Usage:
        optimizer = QueryOptimizer()
        recommendations = optimizer.analyze_query(query_str)
    """

    def analyze_query(self, query: str) -> List[str]:
        """
        Analyze query and provide optimization recommendations.

        Args:
            query: SQL query string

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for SELECT *
        if "SELECT *" in query.upper():
            recommendations.append(
                "AVOID: SELECT * - Specify only needed columns to reduce data transfer"
            )

        # Check for missing WHERE clause
        if "WHERE" not in query.upper() and "SELECT" in query.upper():
            recommendations.append(
                "WARNING: No WHERE clause - Consider if you need all rows"
            )

        # Check for missing LIMIT
        if "LIMIT" not in query.upper() and "SELECT" in query.upper():
            recommendations.append(
                "CONSIDER: Adding LIMIT clause to prevent accidentally loading too many rows"
            )

        # Check for OR conditions (can prevent index usage)
        if " OR " in query.upper():
            recommendations.append(
                "OPTIMIZATION: OR conditions can prevent index usage. "
                "Consider UNION or restructuring the query"
            )

        # Check for LIKE with leading wildcard
        if "LIKE '%" in query or 'LIKE "%' in query:
            recommendations.append(
                "OPTIMIZATION: LIKE with leading wildcard (LIKE '%...') "
                "prevents index usage. Consider full-text search instead"
            )

        # Check for missing indexes (simplified)
        if "JOIN" in query.upper() and "ON" in query.upper():
            recommendations.append(
                "CHECK: Ensure JOIN columns are indexed for optimal performance"
            )

        return recommendations


# Convenience function for getting query stats


def get_query_stats() -> Dict[str, Any]:
    """
    Get global query statistics.

    Returns:
        dict: Query stats summary
    """
    return query_stats.get_summary()


def reset_query_stats() -> None:
    """Reset global query statistics"""
    global query_stats
    query_stats = QueryStats()
    logger.info("Query stats reset")
