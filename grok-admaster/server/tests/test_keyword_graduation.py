"""Unit tests for keyword graduation."""
import pytest
from app.services.keyword_graduation import compute_graduation


def test_to_graduate_enough_orders_and_good_acos():
    terms = [
        {"clicks": 80, "orders": 8, "spend": 40.0, "sales": 320.0, "keyword_text": "winner", "campaign_id": "c1", "campaign_name": "Camp"},
    ]
    result = compute_graduation(terms, target_acos=25.0, min_orders_to_graduate=3, prob_acos_threshold_graduate=0.5)
    assert len(result["to_graduate"]) >= 1
    assert result["to_graduate"][0]["keyword_text"] == "winner"
    assert result["to_graduate"][0]["suggested_match_type"] == "exact"


def test_to_negate_many_clicks_zero_orders():
    terms = [
        {"clicks": 25, "orders": 0, "spend": 15.0, "sales": 0.0, "keyword_text": "waste", "campaign_id": "c1", "campaign_name": "Camp"},
    ]
    result = compute_graduation(terms, target_acos=25.0, min_clicks_to_negate=10)
    assert len(result["to_negate"]) >= 1
    assert result["to_negate"][0]["keyword_text"] == "waste"
    assert "0 orders" in result["to_negate"][0]["reason"] or "negative" in result["to_negate"][0]["reason"].lower()


def test_to_negate_high_acos_enough_spend():
    terms = [
        {"clicks": 50, "orders": 2, "spend": 60.0, "sales": 80.0, "keyword_text": "highacos", "campaign_id": "c1", "campaign_name": "Camp"},
    ]
    result = compute_graduation(terms, target_acos=20.0, acos_ceiling_for_negate=0.50, min_spend_for_negate_by_acos=5.0)
    assert len(result["to_negate"]) >= 1
    assert result["to_negate"][0]["keyword_text"] == "highacos"


def test_empty_terms():
    result = compute_graduation([], target_acos=20.0)
    assert result["to_graduate"] == []
    assert result["to_negate"] == []


def test_no_graduate_when_orders_too_low():
    terms = [
        {"clicks": 20, "orders": 1, "spend": 5.0, "sales": 30.0, "keyword_text": "low", "campaign_id": "c1", "campaign_name": "Camp"},
    ]
    result = compute_graduation(terms, target_acos=25.0, min_orders_to_graduate=5)
    assert len(result["to_graduate"]) == 0
