from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import asynccontextmanager
from app.core.config import settings

# Create Async Engine with production-grade connection pooling
engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    echo=False,  # Set to False in production
    poolclass=QueuePool,
    pool_size=10,              # Base connections
    max_overflow=20,           # Extra connections during burst
    pool_pre_ping=True,        # Verify connections before use
    pool_recycle=3600,         # Recycle connections every hour
    future=True
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
