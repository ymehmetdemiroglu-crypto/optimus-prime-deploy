"""
Integration Tests for Phase 6: Advanced Anomaly Detection.

Tests all components:
    1. LSTM Autoencoder (training, persistence, detection)
    2. Streaming Anomaly Detector (online learning)
    3. Anomaly Explainer (SHAP-style attribution)
    4. Root Cause Analyzer (multi-hop + temporal causality)
    5. Ensemble Detector (all 3 methods + explanations)
"""
import pytest
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from typing import List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from river import anomaly as river_anomaly
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

from app.modules.amazon_ppc.ml.advanced_anomaly import (
    TimeSeriesAnomalyDetector,
    StreamingAnomalyDetector,
    AnomalyExplainer,
    RootCauseAnalyzer,
    EnsembleAnomalyDetector,
    TimestampedAnomaly,
    AnomalyResult,
    ExplanationFeature,
)


# ═══════════════════════════════════════════════════════════════════════
#  Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def normal_sequences():
    """Generate normal time series sequences for training."""
    np.random.seed(42)
    sequences = []
    
    # Generate 100 normal sequences (14 days × 5 features)
    for _ in range(100):
        # Seasonality + noise
        t = np.arange(14)
        trend = 100 + 2 * t
        seasonal = 20 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
        noise = np.random.randn(14) * 5
        
        # 5 features: impressions, clicks, spend, sales, orders
        impressions = trend + seasonal + noise
        clicks = impressions * 0.03 + np.random.randn(14) * 2
        spend = clicks * 2.5 + np.random.randn(14) * 5
        sales = spend * 2.0 + np.random.randn(14) * 10
        orders = clicks * 0.1 + np.random.randn(14) * 0.5
        
        sequence = np.column_stack([impressions, clicks, spend, sales, orders])
        sequences.append(sequence)
    
    return sequences


@pytest.fixture
def anomalous_sequence():
    """Generate a sequence with an anomaly (spike on day 10)."""
    np.random.seed(42)
    
    t = np.arange(14)
    trend = 100 + 2 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 7)
    noise = np.random.randn(14) * 5
    
    impressions = trend + seasonal + noise
    clicks = impressions * 0.03 + np.random.randn(14) * 2
    spend = clicks * 2.5 + np.random.randn(14) * 5
    sales = spend * 2.0 + np.random.randn(14) * 10
    orders = clicks * 0.1 + np.random.randn(14) * 0.5
    
    # Inject anomaly (5x spike on day 10)
    impressions[10] *= 5
    clicks[10] *= 5
    spend[10] *= 5
    sales[10] *= 2  # Lower ROAS
    orders[10] *= 3
    
    return np.column_stack([impressions, clicks, spend, sales, orders])


@pytest.fixture
def short_sequence():
    """Generate a short sequence (7 days) to test padding."""
    np.random.seed(42)
    t = np.arange(7)
    impressions = 100 + 2 * t + np.random.randn(7) * 5
    clicks = impressions * 0.03 + np.random.randn(7) * 2
    spend = clicks * 2.5 + np.random.randn(7) * 5
    sales = spend * 2.0 + np.random.randn(7) * 10
    orders = clicks * 0.1 + np.random.randn(7) * 0.5
    
    return np.column_stack([impressions, clicks, spend, sales, orders])


@pytest.fixture
def feature_dict():
    """Normal feature dictionary for streaming detector."""
    return {
        "impressions": 1200,
        "clicks": 45,
        "spend": 87.5,
        "ctr": 0.0375,
        "acos": 2.5,
    }


@pytest.fixture
def anomalous_features():
    """Anomalous feature dictionary (5x normal)."""
    return {
        "impressions": 6000,  # 5x
        "clicks": 225,        # 5x
        "spend": 437.5,       # 5x
        "ctr": 0.0375,        # Same
        "acos": 6.5,          # 2.6x (worse)
    }


# ═══════════════════════════════════════════════════════════════════════
#  LSTM Autoencoder Tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestLSTMAutoencoder:
    """Test LSTM autoencoder training, persistence, and detection."""
    
    def test_train_with_validation_split(self, normal_sequences):
        """Test training with validation split and early stopping."""
        detector = TimeSeriesAnomalyDetector(
            sequence_length=14,
            input_dim=5,
            hidden_dim=16,
            threshold_percentile=95.0,
        )
        
        # Train with validation
        detector.train(
            normal_sequences,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            patience=5,
        )
        
        # Verify threshold was set
        assert detector.threshold is not None
        assert detector.threshold > 0
        
        # Verify model is in eval mode
        assert not detector.model.training
        
        # Verify scaler was fitted
        assert detector.scaler_mean is not None
        assert detector.scaler_std is not None
        assert len(detector.scaler_mean) == 5
    
    def test_detect_normal_sequence(self, normal_sequences):
        """Test detection on normal sequence (should not flag)."""
        detector = TimeSeriesAnomalyDetector(input_dim=5)
        detector.train(normal_sequences[:80], epochs=20, validation_split=0.2)
        
        # Test on held-out normal sequence
        result = detector.detect(normal_sequences[85])
        
        assert isinstance(result, AnomalyResult)
        assert result.is_anomalous == False  # Should be normal
        assert result.anomaly_score < 1.0
        assert result.severity in ["low", "medium"]  # Not critical
    
    def test_detect_anomalous_sequence(self, normal_sequences, anomalous_sequence):
        """Test detection on anomalous sequence (should flag)."""
        detector = TimeSeriesAnomalyDetector(input_dim=5)
        detector.train(normal_sequences, epochs=20, validation_split=0.2)
        
        # Test on anomalous sequence
        result = detector.detect(anomalous_sequence)
        
        assert isinstance(result, AnomalyResult)
        assert result.is_anomalous == True  # Should detect anomaly
        assert result.anomaly_score > 0.5
        assert result.severity in ["high", "critical"]
        assert result.reconstruction_error is not None
        assert result.reconstruction_error > result.threshold
    
    def test_variable_length_sequences(self, normal_sequences, short_sequence):
        """Test padding of short sequences."""
        detector = TimeSeriesAnomalyDetector(sequence_length=14, input_dim=5)
        
        # Verify _pad_sequence works
        padded = detector._pad_sequence(short_sequence)
        
        assert padded.shape == (14, 5)
        # First 7 rows should match original
        np.testing.assert_array_almost_equal(padded[:7], short_sequence)
        # Rows 7-13 should be forward-filled from day 7
        for i in range(7, 14):
            np.testing.assert_array_almost_equal(padded[i], short_sequence[-1])
    
    def test_model_persistence(self, normal_sequences, tmp_path):
        """Test save and load of trained model."""
        detector1 = TimeSeriesAnomalyDetector(input_dim=5, hidden_dim=16)
        detector1.train(normal_sequences, epochs=20, validation_split=0.2)
        
        # Save model
        model_path = tmp_path / "lstm_model.pth"
        detector1.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load into new detector
        detector2 = TimeSeriesAnomalyDetector(input_dim=5)
        detector2.load_model(str(model_path))
        
        # Verify loaded correctly
        assert detector2.threshold == detector1.threshold
        np.testing.assert_array_almost_equal(detector2.scaler_mean, detector1.scaler_mean)
        np.testing.assert_array_almost_equal(detector2.scaler_std, detector1.scaler_std)
        
        # Verify same predictions
        test_seq = normal_sequences[0]
        result1 = detector1.detect(test_seq)
        result2 = detector2.detect(test_seq)
        
        assert abs(result1.reconstruction_error - result2.reconstruction_error) < 1e-5


# ═══════════════════════════════════════════════════════════════════════
#  Streaming Detector Tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not RIVER_AVAILABLE, reason="River not installed")
class TestStreamingAnomalyDetector:
    """Test online streaming anomaly detection."""
    
    def test_online_learning_warmup(self, feature_dict):
        """Test that detector warms up and calibrates threshold."""
        detector = StreamingAnomalyDetector(window_size=100)
        
        # Feed 150 normal observations
        scores = []
        for i in range(150):
            # Add small random variation
            features = {k: v + np.random.randn() * 5 for k, v in feature_dict.items()}
            result = detector.update_and_detect(features)
            scores.append(result.anomaly_score)
        
        # After 100 samples, threshold should be calibrated
        assert detector.threshold is not None
        assert 0.4 < detector.threshold < 0.8  # Reasonable range
        
        # Score history should be full
        assert len(detector.score_history) >= 100
    
    def test_detect_anomaly_online(self, feature_dict, anomalous_features):
        """Test detection of anomaly in online stream."""
        detector = StreamingAnomalyDetector()
        
        # Warmup with normal data
        for _ in range(150):
            features = {k: v + np.random.randn() * 5 for k, v in feature_dict.items()}
            detector.update_and_detect(features)
        
        # Inject anomaly
        result = detector.update_and_detect(anomalous_features)
        
        assert result.is_anomalous == True
        assert result.anomaly_score > result.threshold
        assert result.severity in ["medium", "high", "critical"]
    
    def test_score_before_learn(self, feature_dict):
        """Verify that scoring happens before learning (no leakage)."""
        detector = StreamingAnomalyDetector()
        
        # First observation
        result1 = detector.update_and_detect(feature_dict)
        
        # Same observation again (if scored after learning, would have lower score)
        result2 = detector.update_and_detect(feature_dict)
        
        # Scores should be similar (both scored on pre-update state)
        assert abs(result1.anomaly_score - result2.anomaly_score) < 0.3


# ═══════════════════════════════════════════════════════════════════════
#  Anomaly Explainer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAnomalyExplainer:
    """Test feature attribution for anomalies."""
    
    def test_explain_feature_contributions(self):
        """Test that explainer ranks features correctly."""
        # Simple scorer: only feature1 matters
        def scorer(features):
            return abs(features.get("feature1", 0)) / 100
        
        explainer = AnomalyExplainer(scorer, n_samples=100)
        
        features = {
            "feature1": 150,  # High contribution
            "feature2": 10,   # Low contribution
            "feature3": 5,    # Low contribution
        }
        
        explanations = explainer.explain(features)
        
        # Verify feature1 ranked first
        assert explanations[0].name == "feature1"
        assert explanations[0].contribution > 0
        
        # Verify all features have explanations
        assert len(explanations) == 3
        
        # Verify direction is correct
        assert explanations[0].direction in ["increase", "neutral"]
    
    def test_explain_with_baseline(self, anomalous_features):
        """Test explanation with custom baseline."""
        def scorer(f):
            # Score based on deviation from baseline
            return sum(abs(f.get(k, 0) - 100) for k in f.keys()) / len(f)
        
        explainer = AnomalyExplainer(scorer, n_samples=50)
        
        baseline = {k: 100 for k in anomalous_features.keys()}
        explanations = explainer.explain(anomalous_features, baseline=baseline)
        
        # Large deviations should have high contributions
        top_feature = explanations[0]
        assert abs(top_feature.contribution) > 0.01
        assert isinstance(top_feature, ExplanationFeature)


# ═══════════════════════════════════════════════════════════════════════
#  Root Cause Analyzer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRootCauseAnalyzer:
    """Test root cause analysis with dependency graphs."""
    
    @pytest.fixture
    def rca_with_graph(self):
        """Create RCA with sample dependency graph."""
        rca = RootCauseAnalyzer()
        
        # Build hierarchy: keywords -> campaigns -> portfolio
        # Campaign 1 keywords
        rca.add_dependency("keyword_1", "campaign_1", "keyword", "campaign")
        rca.add_dependency("keyword_2", "campaign_1", "keyword", "campaign")
        rca.add_dependency("keyword_3", "campaign_1", "keyword", "campaign")
        
        # Campaign 2 keywords
        rca.add_dependency("keyword_4", "campaign_2", "keyword", "campaign")
        rca.add_dependency("keyword_5", "campaign_2", "keyword", "campaign")
        
        # Campaigns -> portfolio
        rca.add_dependency("campaign_1", "portfolio_1", "campaign", "portfolio")
        rca.add_dependency("campaign_2", "portfolio_1", "campaign", "portfolio")
        
        # Portfolio -> account
        rca.add_dependency("portfolio_1", "account_1", "portfolio", "account")
        
        return rca
    
    def test_isolated_anomaly(self, rca_with_graph):
        """Test detection of isolated anomaly (only one keyword)."""
        related_anomalies = []  # No other anomalies
        
        causes = rca_with_graph.analyze(
            anomaly_entity_id="keyword_1",
            anomaly_entity_type="keyword",
            related_anomalies=related_anomalies,
        )
        
        assert len(causes) > 0
        assert "Isolated" in causes[0]
    
    def test_systemic_anomaly_direct_parent(self, rca_with_graph):
        """Test detection of systemic issue (campaign also anomalous)."""
        related_anomalies = [
            ("campaign_1", "campaign"),  # Parent is anomalous
        ]
        
        causes = rca_with_graph.analyze(
            anomaly_entity_id="keyword_1",
            anomaly_entity_type="keyword",
            related_anomalies=related_anomalies,
        )
        
        assert len(causes) > 0
        assert "Inherited" in causes[0] or "systemic" in causes[0].lower()
    
    def test_multi_hop_ancestor_detection(self, rca_with_graph):
        """Test multi-hop traversal finds portfolio-level issue."""
        related_anomalies = [
            ("portfolio_1", "portfolio"),  # Grandparent anomalous
        ]
        
        # Should find portfolio even though it's 2 levels up
        ancestors = rca_with_graph._find_anomalous_ancestors(
            entity_id="keyword_1",
            entity_type="keyword",
            related_anomalies=related_anomalies,
            max_depth=3,
        )
        
        # Filter out self
        ancestors = [
            (aid, atype, depth) for aid, atype, depth in ancestors
            if not (aid == "keyword_1" and atype == "keyword")
        ]
        
        assert len(ancestors) > 0
        assert ("portfolio_1", "portfolio", 2) in ancestors
    
    def test_widespread_sibling_anomalies(self, rca_with_graph):
        """Test detection of widespread issue (many siblings anomalous)."""
        # Need 6+ siblings to trigger (threshold is 5)
        related_anomalies = [
            ("keyword_2", "keyword"),
            ("keyword_3", "keyword"),
            ("keyword_4", "keyword"),
            ("keyword_5", "keyword"),
            ("keyword_6", "keyword"),  # Add extra sibling
            ("keyword_7", "keyword"),  # 6 siblings total
        ]
        
        causes = rca_with_graph.analyze(
            anomaly_entity_id="keyword_1",
            anomaly_entity_type="keyword",
            related_anomalies=related_anomalies,
        )
        
        assert len(causes) > 0
        assert "Widespread" in causes[0]
    
    def test_temporal_causality_analysis(self, rca_with_graph):
        """Test temporal root cause analysis."""
        now = datetime(2026, 2, 10, 15, 10, 0)
        
        current_anomaly = TimestampedAnomaly(
            entity_id="keyword_1",
            entity_type="keyword",
            timestamp=now,
            severity="high",
        )
        
        # Campaign anomaly occurred 5 minutes earlier
        related_anomalies = [
            TimestampedAnomaly(
                entity_id="campaign_1",
                entity_type="campaign",
                timestamp=now - timedelta(minutes=5),
                severity="critical",
            ),
            # Another keyword anomaly occurred later
            TimestampedAnomaly(
                entity_id="keyword_2",
                entity_type="keyword",
                timestamp=now + timedelta(minutes=2),
                severity="medium",
            ),
        ]
        
        causes = rca_with_graph.analyze_temporal_causality(
            anomaly=current_anomaly,
            related_anomalies=related_anomalies,
            time_window_minutes=30,
        )
        
        assert len(causes) > 0
        # Campaign should be identified as likely cause
        assert "campaign_1" in causes[0]
        assert "5.0 min earlier" in causes[0]


# ═══════════════════════════════════════════════════════════════════════
#  Ensemble Detector Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEnsembleAnomalyDetector:
    """Test integrated ensemble detector."""
    
    @pytest.mark.skipif(
        not TORCH_AVAILABLE or not RIVER_AVAILABLE,
        reason="PyTorch and River both required"
    )
    def test_ensemble_detection_all_components(
        self, normal_sequences, anomalous_sequence, anomalous_features
    ):
        """Test ensemble with all 3 detectors (LSTM, Streaming, IsoForest)."""
        ensemble = EnsembleAnomalyDetector()
        
        # Train LSTM
        if ensemble.lstm_detector is not None:
            ensemble.lstm_detector.train(
                normal_sequences, epochs=20, validation_split=0.2
            )
        
        # Warmup streaming detector
        if ensemble.streaming_detector is not None:
            for _ in range(150):
                features = {
                    "impressions": 100 + np.random.randn() * 10,
                    "clicks": 5 + np.random.randn(),
                    "spend": 10 + np.random.randn() * 2,
                    "ctr": 0.05 + np.random.randn() * 0.01,
                    "acos": 2.0 + np.random.randn() * 0.5,
                }
                ensemble.streaming_detector.update_and_detect(features)
        
        # Fit Isolation Forest
        if ensemble.isolation_forest is not None:
            # Extract features from sequences
            batch_features = np.array([
                [seq[-1, 0], seq[-1, 1], seq[-1, 2], seq[-1, 3], seq[-1, 4]]
                for seq in normal_sequences[:50]
            ])
            ensemble.fit_isolation_forest(batch_features)
        
        # Detect anomaly
        result = ensemble.detect_with_explanation(
            sequence=anomalous_sequence,
            features=anomalous_features,
            entity_id="keyword_123",
            entity_type="keyword",
            related_anomalies=[],
        )
        
        assert isinstance(result, AnomalyResult)
        assert result.anomaly_score > 0  # Should detect something
        assert result.severity in ["low", "medium", "high", "critical"]
        
        # Verify explanation was generated
        if result.is_anomalous:
            assert result.explanation is not None
            assert len(result.explanation) > 0
    
    def test_ensemble_graceful_degradation(self, anomalous_sequence, anomalous_features):
        """Test ensemble works even if some detectors unavailable."""
        ensemble = EnsembleAnomalyDetector()
        
        # Should still work (uses available detectors)
        result = ensemble.detect_with_explanation(
            sequence=anomalous_sequence,
            features=anomalous_features,
            entity_id="keyword_123",
            entity_type="keyword",
        )
        
        assert isinstance(result, AnomalyResult)
        # Should return a result even with untrained models
        assert result.metric_name == "ensemble_anomaly"
    
    def test_end_to_end_workflow(
        self, normal_sequences, anomalous_sequence, anomalous_features
    ):
        """Test complete end-to-end anomaly detection workflow."""
        ensemble = EnsembleAnomalyDetector()
        
        # Build dependency graph
        ensemble.root_cause_analyzer.add_dependency(
            "keyword_123", "campaign_45", "keyword", "campaign"
        )
        ensemble.root_cause_analyzer.add_dependency(
            "keyword_456", "campaign_45", "keyword", "campaign"
        )
        ensemble.root_cause_analyzer.add_dependency(
            "campaign_45", "portfolio_1", "campaign", "portfolio"
        )
        
        # Simulate related anomalies
        related_anomalies = [
            ("keyword_456", "keyword"),
            ("campaign_45", "campaign"),
        ]
        
        # Train LSTM if available
        if TORCH_AVAILABLE and ensemble.lstm_detector is not None:
            ensemble.lstm_detector.train(
                normal_sequences[:50], epochs=15, validation_split=0.2
            )
        
        # Detect with full context
        result = ensemble.detect_with_explanation(
            sequence=anomalous_sequence,
            features=anomalous_features,
            entity_id="keyword_123",
            entity_type="keyword",
            related_anomalies=related_anomalies,
        )
        
        # Verify comprehensive result
        assert isinstance(result, AnomalyResult)
        assert result.timestamp is not None
        
        # If anomaly detected, should have root causes
        if result.is_anomalous and related_anomalies:
            assert len(result.root_causes) > 0
            # Should identify campaign as systemic issue
            assert any("campaign" in cause.lower() for cause in result.root_causes)


# ═══════════════════════════════════════════════════════════════════════
#  Performance & Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_empty_sequence_list(self):
        """Test training with empty sequence list."""
        detector = TimeSeriesAnomalyDetector(input_dim=5)
        
        # Should handle gracefully
        detector.train([], epochs=10, validation_split=0.2)
        
        # Threshold should not be set
        assert detector.threshold is None
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_insufficient_sequences(self, normal_sequences):
        """Test training with very few sequences."""
        detector = TimeSeriesAnomalyDetector(input_dim=5)
        
        # Only 5 sequences (< 10 minimum after validation split)
        detector.train(normal_sequences[:5], epochs=10, validation_split=0.2)
        
        # Should skip training
        assert detector.threshold is None
    
    @pytest.mark.skipif(not RIVER_AVAILABLE, reason="River required")
    def test_streaming_with_missing_features(self):
        """Test streaming detector handles missing features gracefully."""
        detector = StreamingAnomalyDetector()
        
        # Warmup
        for _ in range(50):
            features = {"a": 1.0, "b": 2.0, "c": 3.0}
            detector.update_and_detect(features)
        
        # Features with different keys (should handle)
        result = detector.update_and_detect({"a": 1.0, "d": 4.0})
        
        assert isinstance(result, AnomalyResult)
    
    def test_root_cause_empty_graph(self):
        """Test root cause analyzer with empty dependency graph."""
        rca = RootCauseAnalyzer()
        
        # No dependencies added
        causes = rca.analyze(
            anomaly_entity_id="keyword_1",
            anomaly_entity_type="keyword",
            related_anomalies=[],
        )
        
        assert len(causes) > 0
        assert "Isolated" in causes[0]


# ═══════════════════════════════════════════════════════════════════════
#  Note: Performance benchmarks require pytest-benchmark
#  Install with: pip install pytest-benchmark
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
