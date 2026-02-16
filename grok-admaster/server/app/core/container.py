"""
Dependency Injection Container

This module provides a centralized dependency injection container for the application.
Uses dependency-injector for managing service lifecycles and dependencies.

Benefits:
- Easier testing with dependency overrides
- Clear service dependencies
- Singleton management for shared resources
- Lazy initialization of expensive services
"""

from dependency_injector import containers, providers
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from typing import AsyncIterator
import redis.asyncio as redis

from app.core.config import settings
from app.core.credentials import CredentialManager
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class Container(containers.DeclarativeContainer):
    """
    Application dependency injection container.

    Manages lifecycle of:
    - Database connections
    - Redis connections
    - Service instances
    - Feature store
    - Cache layer
    """

    # Configuration
    config = providers.Configuration()

    # Database Engine (Singleton)
    db_engine = providers.Singleton(
        create_async_engine,
        settings.ASYNC_DATABASE_URL,
        echo=settings.ENV == "development",
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

    # Database Session Factory
    session_factory = providers.Singleton(
        async_sessionmaker,
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Redis Client (Singleton)
    redis_client = providers.Singleton(
        redis.from_url,
        settings.REDIS_URL if hasattr(settings, 'REDIS_URL') else "redis://localhost:6379/0",
        encoding="utf-8",
        decode_responses=True,
    )

    # Credential Manager (Singleton)
    credential_manager = providers.Singleton(
        CredentialManager,
    )


# Global container instance
container = Container()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """
    Dependency for getting database sessions.

    Usage in FastAPI endpoints:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...

    Yields:
        AsyncSession: Database session that auto-closes after request
    """
    session_maker = container.session_factory()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_redis() -> redis.Redis:
    """
    Dependency for getting Redis client.

    Usage in FastAPI endpoints:
        @app.get("/cached")
        async def get_cached(redis_client: redis.Redis = Depends(get_redis)):
            ...

    Returns:
        redis.Redis: Redis client instance
    """
    return container.redis_client()


async def get_credential_manager() -> CredentialManager:
    """
    Dependency for getting credential manager.

    Usage in FastAPI endpoints:
        @app.get("/credentials/{account_id}")
        async def get_creds(
            account_id: int,
            cred_mgr: CredentialManager = Depends(get_credential_manager)
        ):
            ...

    Returns:
        CredentialManager: Credential manager instance
    """
    return container.credential_manager()


async def init_container() -> None:
    """
    Initialize the dependency injection container.

    Call this during application startup.
    """
    logger.info("Initializing dependency injection container...")

    # Wire container to modules (for @inject decorator support)
    # container.wire(modules=[__name__])

    # Test database connection
    try:
        engine = container.db_engine()
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        logger.info("✅ Database connection verified via DI container")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

    # Test Redis connection (optional)
    try:
        redis_client = container.redis_client()
        await redis_client.ping()
        logger.info("✅ Redis connection verified via DI container")
    except Exception as e:
        logger.warning(f"⚠️  Redis connection failed: {e}")
        logger.warning("   Application will continue without Redis caching")

    logger.info("✅ Dependency injection container initialized")


async def shutdown_container() -> None:
    """
    Shutdown the dependency injection container.

    Call this during application shutdown.
    """
    logger.info("Shutting down dependency injection container...")

    # Close database engine
    try:
        engine = container.db_engine()
        await engine.dispose()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

    # Close Redis connection
    try:
        redis_client = container.redis_client()
        await redis_client.close()
        logger.info("✅ Redis connections closed")
    except Exception as e:
        logger.warning(f"Error closing Redis: {e}")

    logger.info("✅ Dependency injection container shutdown complete")
