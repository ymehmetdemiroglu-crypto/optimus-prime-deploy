"""
Tests for the Adaptive Threshold system (intent-aware graduation + per-account overrides).
"""
import pytest

from app.services.ml.intent_classifier import ShoppingIntent, INTENT_THRESHOLDS
from app.services.ml.adaptive_thresholds import (
    AdaptiveThresholdManager,
    THRESHOLD_KEYS,
)
from app.services.keyword_graduation import compute_graduation


# ── AdaptiveThresholdManager tests ──────────────────────────────────


class TestAdaptiveThresholdManager:

    def setup_method(self):
        self.mgr = AdaptiveThresholdManager()

    def test_global_defaults_returned_without_account(self):
        t = self.mgr.get_thresholds(ShoppingIntent.TRANSACTIONAL)
        assert t["bleed_threshold"] == 0.35
        assert t["opportunity_floor"] == 0.75
        assert t["min_orders_to_graduate"] == 3

    def test_rufus_defaults(self):
        t = self.mgr.get_thresholds(ShoppingIntent.INFORMATIONAL_RUFUS)
        assert t["bleed_threshold"] == 0.20
        assert t["min_orders_to_graduate"] == 5
        assert t["min_clicks_to_negate"] == 20

    def test_set_override_merges_with_defaults(self):
        self.mgr.set_override(42, ShoppingIntent.TRANSACTIONAL, "bleed_threshold", 0.50)
        t = self.mgr.get_thresholds(ShoppingIntent.TRANSACTIONAL, account_id=42)
        # Override applied
        assert t["bleed_threshold"] == 0.50
        # Other values still default
        assert t["opportunity_floor"] == 0.75

    def test_override_does_not_affect_other_accounts(self):
        self.mgr.set_override(42, ShoppingIntent.TRANSACTIONAL, "bleed_threshold", 0.50)
        t_other = self.mgr.get_thresholds(ShoppingIntent.TRANSACTIONAL, account_id=99)
        assert t_other["bleed_threshold"] == 0.35  # still default

    def test_override_does_not_affect_other_intents(self):
        self.mgr.set_override(42, ShoppingIntent.TRANSACTIONAL, "bleed_threshold", 0.50)
        t_rufus = self.mgr.get_thresholds(ShoppingIntent.INFORMATIONAL_RUFUS, account_id=42)
        assert t_rufus["bleed_threshold"] == 0.20  # still Rufus default

    def test_bulk_overrides(self):
        self.mgr.set_overrides_bulk(
            42, ShoppingIntent.DISCOVERY,
            {"bleed_threshold": 0.10, "min_orders_to_graduate": 8}
        )
        t = self.mgr.get_thresholds(ShoppingIntent.DISCOVERY, account_id=42)
        assert t["bleed_threshold"] == 0.10
        assert t["min_orders_to_graduate"] == 8

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Unknown threshold key"):
            self.mgr.set_override(42, ShoppingIntent.TRANSACTIONAL, "fake_key", 0.5)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            self.mgr.set_override(42, ShoppingIntent.TRANSACTIONAL, "bleed_threshold", 99.0)

    def test_reset_account(self):
        self.mgr.set_override(42, ShoppingIntent.TRANSACTIONAL, "bleed_threshold", 0.80)
        self.mgr.reset_account(42)
        t = self.mgr.get_thresholds(ShoppingIntent.TRANSACTIONAL, account_id=42)
        assert t["bleed_threshold"] == 0.35  # back to default

    def test_reset_intent(self):
        self.mgr.set_override(42, ShoppingIntent.TRANSACTIONAL, "bleed_threshold", 0.80)
        self.mgr.set_override(42, ShoppingIntent.DISCOVERY, "bleed_threshold", 0.05)
        self.mgr.reset_intent(42, ShoppingIntent.TRANSACTIONAL)
        # Transactional reset
        t = self.mgr.get_thresholds(ShoppingIntent.TRANSACTIONAL, account_id=42)
        assert t["bleed_threshold"] == 0.35
        # Discovery still overridden
        t2 = self.mgr.get_thresholds(ShoppingIntent.DISCOVERY, account_id=42)
        assert t2["bleed_threshold"] == 0.05

    def test_get_account_overrides(self):
        self.mgr.set_override(42, ShoppingIntent.TRANSACTIONAL, "bleed_threshold", 0.80)
        overrides = self.mgr.get_account_overrides(42)
        assert "transactional" in overrides
        assert overrides["transactional"]["bleed_threshold"] == 0.80

    def test_get_full_profile(self):
        profile = self.mgr.get_full_profile(account_id=None)
        assert len(profile) == 4
        for intent in ["transactional", "informational_rufus", "navigational", "discovery"]:
            assert intent in profile
            assert "bleed_threshold" in profile[intent]
            assert "min_orders_to_graduate" in profile[intent]

    def test_get_global_defaults(self):
        defaults = self.mgr.get_global_defaults()
        assert "transactional" in defaults
        assert defaults["transactional"]["bleed_threshold"] == 0.35


# ── INTENT_THRESHOLDS completeness tests ────────────────────────────


class TestIntentThresholdsCompleteness:
    """Ensure all intents have all required threshold keys."""

    def test_all_intents_have_graduation_keys(self):
        required = [
            "min_orders_to_graduate",
            "prob_acos_threshold",
            "suggested_match_type",
            "min_clicks_to_negate",
            "acos_ceiling_for_negate",
            "min_spend_for_negate_by_acos",
        ]
        for intent in ShoppingIntent:
            t = INTENT_THRESHOLDS[intent]
            for key in required:
                assert key in t, f"Missing '{key}' in {intent.value} thresholds"

    def test_rufus_more_lenient_negation(self):
        """Rufus terms should require more clicks before negation."""
        rufus = INTENT_THRESHOLDS[ShoppingIntent.INFORMATIONAL_RUFUS]
        trans = INTENT_THRESHOLDS[ShoppingIntent.TRANSACTIONAL]
        assert rufus["min_clicks_to_negate"] > trans["min_clicks_to_negate"]
        assert rufus["acos_ceiling_for_negate"] > trans["acos_ceiling_for_negate"]
        assert rufus["min_spend_for_negate_by_acos"] > trans["min_spend_for_negate_by_acos"]

    def test_discovery_more_lenient_than_transactional(self):
        disc = INTENT_THRESHOLDS[ShoppingIntent.DISCOVERY]
        trans = INTENT_THRESHOLDS[ShoppingIntent.TRANSACTIONAL]
        assert disc["min_clicks_to_negate"] > trans["min_clicks_to_negate"]
        assert disc["min_orders_to_graduate"] >= trans["min_orders_to_graduate"]

    def test_rufus_graduates_to_phrase(self):
        """Rufus terms should default to phrase match, not exact."""
        rufus = INTENT_THRESHOLDS[ShoppingIntent.INFORMATIONAL_RUFUS]
        assert rufus["suggested_match_type"] == "phrase"

    def test_discovery_graduates_to_broad(self):
        disc = INTENT_THRESHOLDS[ShoppingIntent.DISCOVERY]
        assert disc["suggested_match_type"] == "broad"

    def test_transactional_graduates_to_exact(self):
        trans = INTENT_THRESHOLDS[ShoppingIntent.TRANSACTIONAL]
        assert trans["suggested_match_type"] == "exact"


# ── Intent-aware keyword graduation tests ───────────────────────────


class TestIntentAwareGraduation:
    """Test that compute_graduation respects intent-specific thresholds."""

    def _make_term(self, keyword_text, clicks, orders, spend, sales):
        return {
            "keyword_text": keyword_text,
            "campaign_id": "camp_1",
            "campaign_name": "Test Campaign",
            "clicks": clicks,
            "orders": orders,
            "spend": spend,
            "sales": sales,
        }

    def test_transactional_graduates_at_3_orders(self):
        """A transactional term with 3+ orders and good ACoS should graduate."""
        terms = [self._make_term("nike air max 90 size 11", 100, 5, 20.0, 150.0)]
        result = compute_graduation(terms, target_acos=25.0, intent_aware=True)
        assert len(result["to_graduate"]) == 1
        assert result["to_graduate"][0]["suggested_match_type"] == "exact"
        assert result["to_graduate"][0]["intent_type"] == "transactional"

    def test_rufus_term_not_graduated_at_3_orders(self):
        """A Rufus-style term needs 5 orders to graduate, so 3 is not enough."""
        terms = [self._make_term(
            "what is the best wireless earbuds for running",
            100, 3, 20.0, 150.0
        )]
        result = compute_graduation(terms, target_acos=25.0, intent_aware=True)
        # 3 orders < 5 required for Rufus -> should NOT graduate
        assert len(result["to_graduate"]) == 0

    def test_rufus_term_graduates_at_5_orders_as_phrase(self):
        """A Rufus-style term with 5+ orders should graduate as phrase match."""
        terms = [self._make_term(
            "what is the best running shoe for flat feet",
            200, 6, 40.0, 300.0
        )]
        result = compute_graduation(terms, target_acos=25.0, intent_aware=True)
        assert len(result["to_graduate"]) == 1
        assert result["to_graduate"][0]["suggested_match_type"] == "phrase"

    def test_rufus_term_not_negated_at_10_clicks(self):
        """Rufus terms need 20 clicks (not 10) before negation."""
        terms = [self._make_term(
            "which laptop is best for college students",
            15, 0, 8.0, 0.0
        )]
        result = compute_graduation(terms, target_acos=25.0, intent_aware=True)
        # 15 clicks < 20 threshold for Rufus -> NOT negated
        assert len(result["to_negate"]) == 0

    def test_rufus_term_negated_at_20_clicks(self):
        """Rufus terms ARE negated after 20+ clicks with zero orders."""
        terms = [self._make_term(
            "how to choose the best running shoe",
            25, 0, 12.0, 0.0
        )]
        result = compute_graduation(terms, target_acos=25.0, intent_aware=True)
        assert len(result["to_negate"]) == 1
        assert "threshold=20" in result["to_negate"][0]["reason"]

    def test_transactional_negated_at_10_clicks(self):
        """Transactional terms are negated at the standard 10 clicks."""
        terms = [self._make_term("cheap phone case", 12, 0, 6.0, 0.0)]
        result = compute_graduation(terms, target_acos=25.0, intent_aware=True)
        assert len(result["to_negate"]) == 1

    def test_intent_aware_false_uses_defaults(self):
        """When intent_aware=False, all terms use the same default thresholds."""
        terms = [self._make_term(
            "what is the best running shoe",
            15, 0, 8.0, 0.0
        )]
        result = compute_graduation(
            terms, target_acos=25.0,
            intent_aware=False, min_clicks_to_negate=10
        )
        # 15 clicks >= 10 default -> should negate (ignoring Rufus intent)
        assert len(result["to_negate"]) == 1

    def test_empty_terms(self):
        result = compute_graduation([], target_acos=25.0)
        assert result == {"to_graduate": [], "to_negate": []}

    def test_discovery_high_acos_ceiling(self):
        """Discovery terms have 75% ACoS ceiling before negation (vs 50% for transactional)."""
        # ACoS = 15/20 = 75% exactly at ceiling -> should NOT negate
        terms = [self._make_term(
            "unique gifts for hikers essentials items",
            50, 2, 15.0, 20.0
        )]
        result = compute_graduation(terms, target_acos=25.0, intent_aware=True)
        # Should not be in to_negate because 75% == 0.75 ceiling (not >)
        negated = [n for n in result["to_negate"] if n["keyword_text"] == terms[0]["keyword_text"]]
        assert len(negated) == 0


# ── Threshold API tests ─────────────────────────────────────────────


class TestThresholdAPI:
    """Test the threshold API endpoints via TestClient."""

    def test_get_thresholds_default(self, client):
        resp = client.get("/api/v1/semantic/thresholds")
        assert resp.status_code == 200
        data = resp.json()
        assert "effective_thresholds" in data
        assert "transactional" in data["effective_thresholds"]
        assert "informational_rufus" in data["effective_thresholds"]

    def test_get_thresholds_with_account(self, client):
        resp = client.get("/api/v1/semantic/thresholds?account_id=42")
        assert resp.status_code == 200
        data = resp.json()
        assert data["account_id"] == 42

    def test_set_and_get_override(self, client):
        # Set override
        resp = client.put("/api/v1/semantic/thresholds", json={
            "account_id": 99,
            "intent": "informational_rufus",
            "overrides": {"bleed_threshold": 0.25}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["effective_thresholds"]["bleed_threshold"] == 0.25

        # Verify it shows up in GET
        resp2 = client.get("/api/v1/semantic/thresholds?account_id=99")
        assert resp2.status_code == 200
        assert resp2.json()["account_overrides"]["informational_rufus"]["bleed_threshold"] == 0.25

    def test_set_invalid_intent(self, client):
        resp = client.put("/api/v1/semantic/thresholds", json={
            "account_id": 99,
            "intent": "invalid_intent",
            "overrides": {"bleed_threshold": 0.25}
        })
        assert resp.status_code == 400

    def test_set_invalid_key(self, client):
        resp = client.put("/api/v1/semantic/thresholds", json={
            "account_id": 99,
            "intent": "transactional",
            "overrides": {"nonexistent_key": 0.5}
        })
        assert resp.status_code == 400

    def test_set_out_of_range(self, client):
        resp = client.put("/api/v1/semantic/thresholds", json={
            "account_id": 99,
            "intent": "transactional",
            "overrides": {"bleed_threshold": 99.0}
        })
        assert resp.status_code == 400

    def test_reset_account(self, client):
        # Set an override first
        client.put("/api/v1/semantic/thresholds", json={
            "account_id": 100,
            "intent": "transactional",
            "overrides": {"bleed_threshold": 0.90}
        })
        # Reset
        resp = client.delete("/api/v1/semantic/thresholds/100")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"
