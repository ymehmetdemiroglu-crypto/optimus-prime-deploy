from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import asynccontextmanager
from app.core.config import settings

# Production-grade async connection pool.
# NullPool was previously used here but it disables pooling entirely, creating
# a new DB connection on every request — expensive under any real load.
# AsyncAdaptedQueuePool (the default) maintains a pool of persistent connections.
engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    echo=False,
    pool_size=10,          # Persistent connections kept alive
    max_overflow=20,       # Burst connections allowed above pool_size
    pool_pre_ping=True,    # Verify connection health before use
    pool_recycle=3600,     # Recycle connections after 1 hour to avoid stale TCP
    future=True,
)

# Create Session Factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Alias for background tasks
async_session_maker = AsyncSessionLocal

# Base class for models
Base = declarative_base()

# Dependency for API endpoints
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Context manager for background tasks and non-API usage
@asynccontextmanager
async def get_db_session():
    """Robust session context manager for background tasks."""
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
