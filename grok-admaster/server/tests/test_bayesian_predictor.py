"""Unit tests for Bayesian predictor (CVR and ACoS)."""
import pytest
from app.services.bayesian_predictor import (
    update_cvr_posterior,
    predict_cvr,
    update_acos_posterior,
    predict_acos,
    score_keywords_bayesian,
)


# --- CVR ---

def test_cvr_posterior_no_data():
    a, b = update_cvr_posterior(0, 0, 1.0, 9.0)
    assert a == 1.0
    assert b == 9.0


def test_cvr_posterior_with_orders():
    a, b = update_cvr_posterior(100, 5, 1.0, 9.0)
    assert a == 6.0
    assert b == 104.0


def test_predict_cvr_mean_and_prob_above():
    out = predict_cvr(100, 10, 1.0, 9.0, threshold=0.05)
    assert "mean" in out
    assert "std" in out
    assert "prob_above" in out
    assert 0 <= out["mean"] <= 1
    assert out["mean"] == (1 + 10) / (1 + 9 + 100)
    assert out["prob_above"] > 0.5


def test_predict_cvr_zero_clicks():
    out = predict_cvr(0, 0, 1.0, 9.0)
    assert out["mean"] == 1.0 / (1.0 + 9.0)


# --- ACoS ---

def test_acos_posterior():
    a, b = update_acos_posterior(20.0, 100.0, 2.0, 8.0, weight=1.0)
    assert a > 2.0
    assert b > 0
    acos_obs = 20.0 / 100.0
    assert a == pytest.approx(2.0 + acos_obs, rel=0.01)
    assert b == pytest.approx(8.0 + (1.0 - acos_obs), rel=0.01)


def test_predict_acos_mean_and_prob_below_target():
    out = predict_acos(20.0, 100.0, target_acos=25.0, prior_alpha=2.0, prior_beta=8.0)
    assert "mean" in out
    assert "prob_below_target" in out
    assert 0 <= out["mean"] <= 1
    assert 0 <= out["prob_below_target"] <= 1
    assert out["prob_below_target"] > 0.5


def test_predict_acos_zero_sales():
    out = predict_acos(10.0, 0.0, target_acos=20.0)
    assert out["mean"] == 2.0 / (2.0 + 8.0)


# --- score_keywords_bayesian ---

def test_score_keywords_bayesian_returns_sorted_list():
    terms = [
        {"clicks": 50, "orders": 5, "spend": 30.0, "sales": 150.0, "keyword_text": "a", "campaign_id": "c1", "campaign_name": "C1"},
        {"clicks": 20, "orders": 2, "spend": 40.0, "sales": 80.0, "keyword_text": "b", "campaign_id": "c1", "campaign_name": "C1"},
    ]
    result = score_keywords_bayesian(terms, target_acos=25.0)
    assert len(result) == 2
    assert result[0]["score"] >= result[1]["score"]
    assert result[0]["keyword_text"] == "a"
    assert result[0]["predicted_cvr"] > 0
    assert result[0]["predicted_acos"] > 0
    assert "prob_acos_below_target" in result[0]
    assert "term_id" in result[0] or "keyword_text" in result[0]


def test_score_keywords_bayesian_empty_list():
    assert score_keywords_bayesian([], target_acos=20.0) == []


def test_score_keywords_bayesian_expected_keys():
    terms = [{"clicks": 10, "orders": 1, "spend": 5.0, "sales": 25.0, "keyword_text": "x", "campaign_id": "c1", "campaign_name": "Camp"}]
    result = score_keywords_bayesian(terms, target_acos=25.0)
    assert len(result) == 1
    r = result[0]
    required = ["term_id", "keyword_text", "campaign_id", "campaign_name", "predicted_cvr", "predicted_acos", "prob_acos_below_target", "score"]
    for k in required:
        assert k in r
