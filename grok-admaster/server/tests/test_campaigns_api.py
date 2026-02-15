"""Integration tests for campaigns API. Require DB (and optionally seed) to return data."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

def test_campaigns_list(client):
    """GET /campaigns returns 200 and a list (may be empty without seed)."""
    response = client.get("/api/v1/campaigns")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    for c in data[:1]:
        if c:
            assert "id" in c
            assert "name" in c
            assert "status" in c
            assert "ai_mode" in c
            assert "daily_budget" in c
            assert "spend" in c
            assert "sales" in c
            assert "acos" in c


def test_campaigns_patch_strategy_404_when_missing(client):
    """PATCH /campaigns/nonexistent/strategy returns 404."""
    response = client.patch(
        "/api/v1/campaigns/999999/strategy",
        json={"ai_mode": "auto_pilot"},
    )
    assert response.status_code == 404
