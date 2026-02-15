"""Integration tests for dashboard API (summary and chart-data do not require DB)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

def test_dashboard_summary(client):
    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_sales" in data
    assert "ad_spend" in data
    assert "acos" in data
    assert "roas" in data
    assert data["velocity_trend"] in ("up", "down", "flat")


def test_dashboard_chart_data_7d(client):
    response = client.get("/api/v1/dashboard/chart-data?range=7d")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "timestamp" in data[0]
        assert "organic_sales" in data[0]
        assert "ad_sales" in data[0]
        assert "spend" in data[0]
        assert "impressions" in data[0]


def test_dashboard_chart_data_30d(client):
    response = client.get("/api/v1/dashboard/chart-data?range=30d")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_dashboard_ai_actions(client):
    """GET /ai-actions returns list (may be empty if no active campaigns in DB)."""
    response = client.get("/api/v1/dashboard/ai-actions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    for item in data[:1]:
        if item:
            assert "id" in item
            assert "action_type" in item
            assert "description" in item
            assert "timestamp" in item
