
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client():
    """Create a TestClient instance for the module."""
    with TestClient(app) as c:
        yield c
