
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from app.main import app
from app.core.database import get_db, Base


# Teach SQLite how to handle PostgreSQL UUID columns
@compiles(PG_UUID, "sqlite")
def _compile_pg_uuid_for_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


# In-memory SQLite engine for tests (no real Postgres needed)
_test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

_TestSessionLocal = sessionmaker(
    bind=_test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def _override_get_db():
    """Yield an async session backed by in-memory SQLite."""
    async with _TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@pytest.fixture(scope="session", autouse=True)
def _apply_db_override():
    """Override the get_db dependency for every test that uses the client."""
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _create_test_tables():
    """Create all tables in the in-memory SQLite DB before tests run."""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await _test_engine.dispose()


@pytest.fixture(scope="session")
def client():
    """Create a TestClient instance for the module."""
    with TestClient(app) as c:
        yield c
