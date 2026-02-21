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

def test_to_graduate_rufus_rules():
    terms = [
        # Rufus keyword with generous orders but bad probability -> shouldn't graduate (prob < 0.50)
        {"clicks": 50, "orders": 6, "spend": 30.0, "sales": 100.0, "keyword_text": "rufus bad prob", "campaign_id": "c1", "query_source": "rufus"},
        # Rufus keyword with good probability but not enough orders (4 < 5)
        {"clicks": 50, "orders": 4, "spend": 20.0, "sales": 100.0, "keyword_text": "rufus low orders", "campaign_id": "c1", "query_source": "rufus"},
    ]
    # We fake the bayesian score internally. If we pass ACoS=20%, a spend of 30 and sales 100 is 30%.
    # Actually, we can just test if the parameters are used. Wait, score_keywords_bayesian computes prob_below.
    # To reliably test without mocking the bayesian function, we'll just test the organic ones and assume the threshold logic is sound,
    # or mock it. The file uses `compute_graduation` which calls `score_keywords_bayesian`. Let's mock score_keywords_bayesian.

    import unittest.mock as mock
    with mock.patch('app.services.keyword_graduation.score_keywords_bayesian') as mock_score:
        mock_score.return_value = [
            {"keyword_text": "rufus pass", "query_source": "rufus", "orders": 5, "prob_acos_below_target": 0.55, "campaign_id": "c1"},
            {"keyword_text": "rufus fail prob", "query_source": "rufus", "orders": 5, "prob_acos_below_target": 0.45, "campaign_id": "c1"},
            {"keyword_text": "rufus fail orders", "query_source": "rufus", "orders": 4, "prob_acos_below_target": 0.60, "campaign_id": "c1"},
            {"keyword_text": "organic pass", "query_source": "organic", "orders": 3, "prob_acos_below_target": 0.65, "campaign_id": "c1"},
            {"keyword_text": "organic fail prob", "query_source": "organic", "orders": 3, "prob_acos_below_target": 0.55, "campaign_id": "c1"},
        ]
        
        result = compute_graduation([], target_acos=25.0, min_orders_to_graduate=3, prob_acos_threshold_graduate=0.60)
        
        graduates = [r["keyword_text"] for r in result["to_graduate"]]
        assert "rufus pass" in graduates
        assert "rufus fail prob" not in graduates
        assert "rufus fail orders" not in graduates
        assert "organic pass" in graduates
        assert "organic fail prob" not in graduates

