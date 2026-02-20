"""
Tests for the Shopping Intent Classifier (Rufus/Cosmo optimization).
"""
import pytest
import numpy as np
from unittest.mock import MagicMock

from app.services.ml.intent_classifier import (
    IntentClassifier,
    ShoppingIntent,
    IntentResult,
    INTENT_THRESHOLDS,
    get_intent_thresholds,
)


# --- Fixtures ---

@pytest.fixture
def classifier_no_embeddings():
    """Classifier using pattern-only mode (no embedding service)."""
    return IntentClassifier(embedding_service=None)


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service that returns deterministic vectors."""
    svc = MagicMock()
    # Return a simple normalized vector based on hash of text
    def fake_encode(text):
        np.random.seed(hash(text) % (2**31))
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec
    svc.encode.side_effect = fake_encode
    return svc


@pytest.fixture
def classifier_with_embeddings(mock_embedding_service):
    """Classifier with mock embedding service for full hybrid mode."""
    return IntentClassifier(embedding_service=mock_embedding_service)


# --- Pattern Layer Tests ---

class TestPatternClassification:
    """Test the pattern-based scoring layer."""

    def test_rufus_question_query(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("what is the best wireless earbuds for running")
        assert result.intent == ShoppingIntent.INFORMATIONAL_RUFUS
        assert result.confidence > 0.3
        assert any("pattern:informational_rufus" in s for s in result.signals)

    def test_rufus_comparison_query(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("airpods pro vs galaxy buds comparison")
        assert result.intent == ShoppingIntent.INFORMATIONAL_RUFUS

    def test_rufus_best_for_pattern(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("best laptop for college students under 500")
        assert result.intent in (ShoppingIntent.INFORMATIONAL_RUFUS, ShoppingIntent.DISCOVERY)

    def test_transactional_model_number(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("B0DWK3C1R7")
        assert result.intent == ShoppingIntent.TRANSACTIONAL

    def test_transactional_with_size(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("nike air max 90 size 11 black")
        assert result.intent == ShoppingIntent.TRANSACTIONAL

    def test_transactional_with_quantity(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("duracell aa batteries 24 pack")
        assert result.intent == ShoppingIntent.TRANSACTIONAL

    def test_discovery_gift_query(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("gifts for hikers under 50")
        assert result.intent == ShoppingIntent.DISCOVERY

    def test_discovery_trending(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("trending home office accessories must have items")
        assert result.intent == ShoppingIntent.DISCOVERY

    def test_navigational_brand(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("anker official store")
        assert result.intent == ShoppingIntent.NAVIGATIONAL

    def test_empty_query_defaults_transactional(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("")
        assert result.intent == ShoppingIntent.TRANSACTIONAL
        assert result.confidence == 0.0
        assert "empty_query" in result.signals


class TestStructuralFeatures:
    """Test the structural scoring layer."""

    def test_short_query_leans_navigational(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("dyson v15")
        assert "short_query" in result.signals

    def test_long_query_leans_rufus(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify(
            "what is the best budget friendly noise cancelling headphone for airplane travel in 2026"
        )
        assert "long_query_rufus_likely" in result.signals
        assert result.intent == ShoppingIntent.INFORMATIONAL_RUFUS

    def test_question_mark_boosts_rufus(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("which protein powder is best?")
        assert "question_mark" in result.signals
        assert result.intent == ShoppingIntent.INFORMATIONAL_RUFUS

    def test_persona_target_boosts_discovery(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("cool stuff for my kids")
        assert "persona_target" in result.signals


# --- Hybrid Mode Tests ---

class TestHybridClassification:
    """Test the full hybrid classifier (patterns + embeddings)."""

    def test_embedding_service_called(self, classifier_with_embeddings, mock_embedding_service):
        classifier_with_embeddings.classify("test query")
        mock_embedding_service.encode.assert_called()

    def test_batch_classification(self, classifier_no_embeddings):
        queries = [
            "nike air max 90 size 11",
            "what is the best running shoe",
            "anker official store",
            "gifts for dad under 30",
        ]
        results = classifier_no_embeddings.classify_batch(queries)
        assert len(results) == 4
        assert all(isinstance(r, IntentResult) for r in results)

    def test_to_dict_format(self, classifier_no_embeddings):
        result = classifier_no_embeddings.classify("test query")
        d = result.to_dict()
        assert "query" in d
        assert "intent" in d
        assert "confidence" in d
        assert "scores" in d
        assert "signals" in d
        assert isinstance(d["scores"], dict)


# --- Intent Threshold Tests ---

class TestIntentThresholds:
    """Test that intent-specific thresholds are correctly configured."""

    def test_all_intents_have_thresholds(self):
        for intent in ShoppingIntent:
            thresholds = get_intent_thresholds(intent)
            assert "bleed_threshold" in thresholds
            assert "opportunity_floor" in thresholds

    def test_rufus_has_lower_bleed_threshold(self):
        rufus = get_intent_thresholds(ShoppingIntent.INFORMATIONAL_RUFUS)
        transactional = get_intent_thresholds(ShoppingIntent.TRANSACTIONAL)
        assert rufus["bleed_threshold"] < transactional["bleed_threshold"]

    def test_rufus_has_lower_opportunity_floor(self):
        rufus = get_intent_thresholds(ShoppingIntent.INFORMATIONAL_RUFUS)
        transactional = get_intent_thresholds(ShoppingIntent.TRANSACTIONAL)
        assert rufus["opportunity_floor"] < transactional["opportunity_floor"]

    def test_discovery_has_lowest_bleed_threshold(self):
        discovery = get_intent_thresholds(ShoppingIntent.DISCOVERY)
        for intent in ShoppingIntent:
            other = get_intent_thresholds(intent)
            assert discovery["bleed_threshold"] <= other["bleed_threshold"]

    def test_thresholds_are_positive(self):
        for intent in ShoppingIntent:
            t = get_intent_thresholds(intent)
            assert t["bleed_threshold"] > 0
            assert t["opportunity_floor"] > 0
            assert t["opportunity_floor"] > t["bleed_threshold"]


# --- Integration with SearchTermAnalyzer ---

class TestSearchTermAnalyzerIntegration:
    """Test that the intent analysis integrates into the search term analyzer."""

    def test_intent_breakdown_in_analysis(self):
        from app.modules.amazon_ppc.ml.search_term_analysis import SearchTermAnalyzer

        analyzer = SearchTermAnalyzer()
        data = [
            {"term": "best wireless earbuds for running", "impressions": 1000, "clicks": 50, "spend": 25.0, "sales": 100.0, "orders": 5},
            {"term": "nike air max 90 size 11", "impressions": 500, "clicks": 30, "spend": 15.0, "sales": 80.0, "orders": 3},
            {"term": "gifts for dad under 30 dollars", "impressions": 200, "clicks": 10, "spend": 5.0, "sales": 20.0, "orders": 1},
            {"term": "anker official store", "impressions": 300, "clicks": 20, "spend": 8.0, "sales": 60.0, "orders": 2},
        ]
        result = analyzer.analyze_search_terms(data, target_acos=25.0)

        assert "intent_breakdown" in result
        breakdown = result["intent_breakdown"]
        assert breakdown["available"] is True
        assert "summary" in breakdown

        # Check that all 4 intent buckets exist in summary
        for intent in ["transactional", "informational_rufus", "navigational", "discovery"]:
            assert intent in breakdown["summary"]
            bucket = breakdown["summary"][intent]
            assert "count" in bucket
            assert "spend" in bucket
            assert "sales" in bucket
            assert "spend_share" in bucket

    def test_intent_breakdown_graceful_with_empty_data(self):
        from app.modules.amazon_ppc.ml.search_term_analysis import SearchTermAnalyzer

        analyzer = SearchTermAnalyzer()
        result = analyzer.analyze_search_terms([], target_acos=25.0)
        assert "error" in result
