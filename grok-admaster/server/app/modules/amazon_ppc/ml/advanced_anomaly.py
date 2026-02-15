"""
Advanced Anomaly Detection (Phase 6).

Implements state-of-the-art anomaly detection for PPC time-series:

    1. **LSTM Autoencoder** — Deep learning reconstruction-based anomaly
       detection. Learns normal patterns from historical sequences, flags
       deviations via reconstruction error.

    2. **Online Streaming Detector** — Incremental anomaly detection using
       Half-Space Trees (River library). Updates in real-time as new data
       arrives, no batch retraining needed.

    3. **Explainable Anomalies** — SHAP-style feature attribution for
       detected anomalies. Answers "why is this anomalous?" with ranked
       feature contributions.

    4. **Root Cause Analysis** — AWS Lookout-inspired causal graph analysis.
       Traces anomalies back through dependency chains (keyword → campaign
       → portfolio) to identify the root cause.

All components integrate with existing PPCAnomalyDetector for ensemble
anomaly scoring.

References:
    - Malhotra et al., "LSTM-based Encoder-Decoder for Multi-sensor
      Anomaly Detection" (ICML 2016)
    - Tan et al., "Robust Anomaly Detection in Streams" (KDD 2011)
    - Lundberg & Lee, "A Unified Approach to Interpreting Model
      Predictions" (NeurIPS 2017)
"""
from __future__ import annotations

import math
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from river import anomaly as river_anomaly
    from river import preprocessing as river_preprocessing
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomalous: bool
    anomaly_score: float          # 0-1, higher = more anomalous
    threshold: float
    timestamp: datetime
    metric_name: str
    actual_value: float
    expected_value: Optional[float] = None
    reconstruction_error: Optional[float] = None
    severity: str = "low"         # low, medium, high, critical
    explanation: Optional[Dict[str, float]] = None  # feature contributions
    root_causes: List[str] = field(default_factory=list)


@dataclass
class ExplanationFeature:
    """Feature importance for anomaly explanation."""
    name: str
    contribution: float           # Shapley value or similar
    actual_value: float
    baseline_value: float
    direction: str                # "increase", "decrease", "neutral"


# ═══════════════════════════════════════════════════════════════════════
#  1. LSTM Autoencoder for Time-Series Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class LSTMAutoencoder(nn.Module):
        """
        LSTM-based autoencoder for sequence anomaly detection.

        Architecture:
            Encoder:  LSTM(seq_len, input_dim) → hidden_dim
            Decoder:  LSTM(hidden_dim) → LSTM(input_dim) → seq_len

        Anomaly Score: reconstruction error (MSE between input and output)
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 32,
            n_layers: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers

            # Encoder
            self.encoder = nn.LSTM(
                input_dim,
                hidden_dim,
                n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )

            # Decoder
            self.decoder = nn.LSTM(
                hidden_dim,
                hidden_dim,
                n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )

            # Output projection
            self.output_layer = nn.Linear(hidden_dim, input_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : (batch, seq_len, input_dim)

            Returns
            -------
            reconstruction : (batch, seq_len, input_dim)
            """
            # Encode
            _, (hidden, cell) = self.encoder(x)

            # Decode (repeat hidden state for each time step)
            batch_size, seq_len, _ = x.shape
            decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
            decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

            # Project to input dimension
            reconstruction = self.output_layer(decoder_output)

            return reconstruction


class TimeSeriesAnomalyDetector:
    """
    Production-ready LSTM autoencoder anomaly detector.

    Workflow:
        1. Train on normal historical sequences (14-day windows)
        2. Set threshold at 95th percentile of training reconstruction errors
        3. At inference, flag sequences with error > threshold
    """

    def __init__(
        self,
        sequence_length: int = 14,
        input_dim: int = 5,
        hidden_dim: int = 32,
        threshold_percentile: float = 95.0,
    ):
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.threshold_percentile = threshold_percentile

        if TORCH_AVAILABLE:
            self.model = LSTMAutoencoder(input_dim, hidden_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            self.scaler_mean: Optional[np.ndarray] = None
            self.scaler_std: Optional[np.ndarray] = None
            self.threshold: Optional[float] = None
        else:
            self.model = None
            logger.warning("[LSTMAnomalyDetector] PyTorch not available")

    def _pad_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to target length."""
        if len(seq) < self.sequence_length:
            # Forward-fill last value
            padding = np.repeat(seq[-1:], self.sequence_length - len(seq), axis=0)
            return np.vstack([seq, padding])
        elif len(seq) > self.sequence_length:
            # Truncate to most recent
            return seq[-self.sequence_length:]
        return seq

    def train(
        self,
        sequences: List[np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 10,
    ):
        """
        Train autoencoder on normal sequences with validation and early stopping.

        Parameters
        ----------
        sequences : list of (seq_len, input_dim) arrays
            Training sequences (should be NORMAL data only)
        validation_split : float
            Fraction of data to use for validation
        patience : int
            Early stopping patience (epochs without improvement)
        """
        if not TORCH_AVAILABLE or self.model is None:
            return
        
        # Validate input
        if len(sequences) == 0:
            logger.warning("[LSTM] No sequences provided for training")
            return

        # Normalize
        all_data = np.concatenate(sequences, axis=0)
        self.scaler_mean = all_data.mean(axis=0)
        self.scaler_std = all_data.std(axis=0) + 1e-8

        # Pad/truncate all sequences to target length
        X_train = []
        for seq in sequences:
            padded = self._pad_sequence(seq)
            normalized = (padded - self.scaler_mean) / self.scaler_std
            X_train.append(normalized)

        if len(X_train) < 50:
            logger.warning(f"[LSTM] Only {len(X_train)} sequences, recommend >= 50")
            if len(X_train) < 10:
                return

        # Train/validation split
        n_val = max(1, int(len(X_train) * validation_split))
        val_sequences = X_train[:n_val]
        train_sequences = X_train[n_val:]

        X_train_tensor = torch.tensor(np.array(train_sequences), dtype=torch.float32)
        X_val_tensor = torch.tensor(np.array(val_sequences), dtype=torch.float32)

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(train_sequences))
            epoch_loss = 0.0

            for i in range(0, len(train_sequences), batch_size):
                idx = perm[i:i + batch_size]
                batch = X_train_tensor[idx]

                # Forward pass
                reconstruction = self.model(batch)
                loss = nn.functional.mse_loss(reconstruction, batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_reconstruction = self.model(X_val_tensor)
                val_loss = nn.functional.mse_loss(val_reconstruction, X_val_tensor).item()
            self.model.train()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"[LSTM] Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

            if patience_counter >= patience:
                logger.info(f"[LSTM] Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])

        # Set threshold using VALIDATION set reconstruction errors
        self.model.eval()
        with torch.no_grad():
            val_reconstruction = self.model(X_val_tensor)
            val_errors = torch.mean((val_reconstruction - X_val_tensor) ** 2, dim=(1, 2)).numpy()
            self.threshold = float(np.percentile(val_errors, self.threshold_percentile))

        logger.info(
            f"[LSTM] Training complete. Best val loss: {best_val_loss:.6f}, "
            f"Threshold (95th percentile): {self.threshold:.6f}"
        )

    def save_model(self, path: str):
        """Save trained model to disk."""
        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("[LSTM] Cannot save: PyTorch not available")
            return

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'threshold': self.threshold,
            'config': {
                'sequence_length': self.sequence_length,
                'input_dim': self.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'n_layers': self.model.n_layers,
                'threshold_percentile': self.threshold_percentile,
            }
        }, path)
        logger.info(f"[LSTM] Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model from disk."""
        if not TORCH_AVAILABLE:
            logger.warning("[LSTM] Cannot load: PyTorch not available")
            return

        # PyTorch 2.6+ requires weights_only=False for numpy arrays
        checkpoint = torch.load(path, weights_only=False)
        
        # Restore config
        config = checkpoint['config']
        self.sequence_length = config['sequence_length']
        self.input_dim = config['input_dim']
        self.threshold_percentile = config.get('threshold_percentile', 95.0)

        # Recreate model with saved architecture
        self.model = LSTMAutoencoder(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Restore normalization and threshold
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        self.threshold = checkpoint['threshold']

        logger.info(f"[LSTM] Model loaded from {path}")

    def detect(self, sequence: np.ndarray) -> AnomalyResult:
        """
        Detect if a sequence is anomalous.

        Parameters
        ----------
        sequence : (seq_len, input_dim) array
        """
        if not TORCH_AVAILABLE or self.model is None or self.threshold is None:
            return AnomalyResult(
                is_anomalous=False,
                anomaly_score=0.0,
                threshold=1.0,
                timestamp=datetime.utcnow(),
                metric_name="lstm_reconstruction",
                actual_value=0.0,
            )

        # Normalize
        normalized = (sequence - self.scaler_mean) / self.scaler_std
        X = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

        # Reconstruct
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(X)
            error = torch.mean((reconstruction - X) ** 2).item()

        # Score (normalize by threshold)
        anomaly_score = min(1.0, error / (self.threshold + 1e-8))
        is_anomalous = error > self.threshold

        # Severity based on how far above threshold
        if error > self.threshold * 3:
            severity = "critical"
        elif error > self.threshold * 2:
            severity = "high"
        elif error > self.threshold * 1.5:
            severity = "medium"
        else:
            severity = "low"

        return AnomalyResult(
            is_anomalous=is_anomalous,
            anomaly_score=anomaly_score,
            threshold=self.threshold,
            timestamp=datetime.utcnow(),
            metric_name="lstm_reconstruction_error",
            actual_value=error,
            expected_value=self.threshold,
            reconstruction_error=error,
            severity=severity,
        )


# ═══════════════════════════════════════════════════════════════════════
#  2. Online Streaming Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════

class StreamingAnomalyDetector:
    """
    Online anomaly detection using Half-Space Trees (River library).

    Updates incrementally as new data points arrive.
    No batch retraining required — perfect for real-time monitoring.
    """

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 8,
        window_size: int = 100,
        seed: int = 42,
    ):
        self.n_trees = n_trees
        self.height = height
        self.window_size = window_size

        if RIVER_AVAILABLE:
            self.detector = river_anomaly.HalfSpaceTrees(
                n_trees=n_trees,
                height=height,
                window_size=window_size,
                seed=seed,
            )
            self.scaler = river_preprocessing.StandardScaler()
            self._fitted = False
        else:
            self.detector = None
            logger.warning("[StreamingDetector] River library not available")

        # Track history for threshold calibration
        self.score_history: deque = deque(maxlen=1000)
        self.threshold: Optional[float] = None

    def update_and_detect(
        self,
        features: Dict[str, float],
        metric_name: str = "online_anomaly",
    ) -> AnomalyResult:
        """
        Process one observation: score, then learn from it.

        Parameters
        ----------
        features : dict
            Feature values for this time point
            Example: {"impressions": 1200, "clicks": 45, "spend": 87.5, ...}
        """
        if not RIVER_AVAILABLE or self.detector is None:
            return AnomalyResult(
                is_anomalous=False,
                anomaly_score=0.0,
                threshold=1.0,
                timestamp=datetime.utcnow(),
                metric_name=metric_name,
                actual_value=0.0,
            )

        # Normalize features
        scaled_features = self.scaler.learn_one(features).transform_one(features)

        # Score BEFORE learning (this is the anomaly score)
        score = self.detector.score_one(scaled_features)

        # Learn from this observation
        self.detector.learn_one(scaled_features)
        self._fitted = True

        # Track scores for threshold calibration
        self.score_history.append(score)

        # Adaptive threshold (95th percentile of recent scores)
        if len(self.score_history) >= 100:
            self.threshold = float(np.percentile(list(self.score_history), 95))
        else:
            # Bootstrap threshold
            self.threshold = 0.6

        is_anomalous = score > self.threshold

        # Severity
        if score > self.threshold * 2:
            severity = "critical"
        elif score > self.threshold * 1.5:
            severity = "high"
        elif score > self.threshold * 1.2:
            severity = "medium"
        else:
            severity = "low"

        return AnomalyResult(
            is_anomalous=is_anomalous,
            anomaly_score=min(1.0, score),
            threshold=self.threshold,
            timestamp=datetime.utcnow(),
            metric_name=metric_name,
            actual_value=score,
            expected_value=self.threshold,
            severity=severity if is_anomalous else "low",
        )


# ═══════════════════════════════════════════════════════════════════════
#  3. Explainable Anomaly Attribution
# ═══════════════════════════════════════════════════════════════════════

class AnomalyExplainer:
    """
    Explains WHY a data point is anomalous using SHAP-style attribution.

    For each feature, computes its contribution to the anomaly score by
    measuring how the score changes when that feature is masked/perturbed.
    """

    def __init__(
        self,
        anomaly_scorer: callable,
        n_samples: int = 100,
    ):
        """
        Parameters
        ----------
        anomaly_scorer : callable
            Function that takes features dict and returns anomaly score
        n_samples : int
            Number of Monte Carlo samples for approximation
        """
        self.anomaly_scorer = anomaly_scorer
        self.n_samples = n_samples

    def explain(
        self,
        features: Dict[str, float],
        baseline: Optional[Dict[str, float]] = None,
    ) -> List[ExplanationFeature]:
        """
        Compute feature attributions for an anomalous observation.

        Returns ranked list of features by contribution to anomaly.
        """
        if baseline is None:
            # Use zeros or median as baseline
            baseline = {k: 0.0 for k in features.keys()}

        # Original anomaly score
        original_score = self.anomaly_scorer(features)

        attributions = []

        for feature_name in features.keys():
            # Marginal contribution: E[score(with feature) - score(without feature)]
            # Approximate using Monte Carlo sampling
            contributions = []

            for _ in range(self.n_samples):
                # Random subset of other features
                subset = {}
                for k, v in features.items():
                    if k == feature_name:
                        continue
                    # Include with 50% probability
                    if np.random.rand() > 0.5:
                        subset[k] = v
                    else:
                        subset[k] = baseline[k]

                # Score with feature included
                subset_with = subset.copy()
                subset_with[feature_name] = features[feature_name]
                score_with = self.anomaly_scorer(subset_with)

                # Score with feature excluded (baseline)
                subset_without = subset.copy()
                subset_without[feature_name] = baseline[feature_name]
                score_without = self.anomaly_scorer(subset_without)

                contributions.append(score_with - score_without)

            avg_contribution = float(np.mean(contributions))

            # Direction
            if avg_contribution > 0.01:
                direction = "increase"
            elif avg_contribution < -0.01:
                direction = "decrease"
            else:
                direction = "neutral"

            attributions.append(
                ExplanationFeature(
                    name=feature_name,
                    contribution=avg_contribution,
                    actual_value=features[feature_name],
                    baseline_value=baseline[feature_name],
                    direction=direction,
                )
            )

        # Sort by absolute contribution
        attributions.sort(key=lambda x: abs(x.contribution), reverse=True)

        return attributions


# ═══════════════════════════════════════════════════════════════════════
#  4. Root Cause Analysis
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TimestampedAnomaly:
    """Anomaly with timestamp for temporal analysis."""
    entity_id: str
    entity_type: str
    timestamp: datetime
    severity: str = "medium"


class RootCauseAnalyzer:
    """
    AWS Lookout-inspired root cause analysis with multi-hop traversal.

    Traces anomalies through dependency graph:
        Keyword → Campaign → Portfolio → Account

    Identifies whether anomaly is:
        - Isolated (single keyword)
        - Localized (affecting campaign)
        - Systemic (affecting portfolio/account)
    """

    def __init__(self):
        self.dependency_graph: Dict[str, List[Tuple[str, str]]] = {}  # key -> [(parent_id, parent_type)]
        self.entity_hierarchy = {
            "keyword": "campaign",
            "campaign": "portfolio",
            "portfolio": "account",
        }

    def add_dependency(self, child_id: str, parent_id: str, entity_type: str, parent_type: Optional[str] = None):
        """
        Register a dependency: child depends on parent.

        Example:
            add_dependency("keyword_123", "campaign_45", "keyword", "campaign")
            add_dependency("campaign_45", "portfolio_1", "campaign", "portfolio")
        """
        if parent_type is None:
            parent_type = self.entity_hierarchy.get(entity_type, "unknown")
        
        key = f"{entity_type}:{child_id}"
        if key not in self.dependency_graph:
            self.dependency_graph[key] = []
        self.dependency_graph[key].append((parent_id, parent_type))

    def _find_anomalous_ancestors(
        self,
        entity_id: str,
        entity_type: str,
        related_anomalies: List[Tuple[str, str]],
        max_depth: int = 3,
    ) -> List[Tuple[str, str, int]]:
        """Traverse up dependency graph to find anomalous ancestors."""
        visited = set()
        queue = [(entity_id, entity_type, 0)]
        anomalous_ancestors = []

        while queue:
            current_id, current_type, depth = queue.pop(0)

            if depth > max_depth:
                continue

            key = f"{current_type}:{current_id}"
            if key in visited:
                continue
            visited.add(key)

            # Check if current entity is anomalous
            if any(a_id == current_id and a_type == current_type for a_id, a_type in related_anomalies):
                anomalous_ancestors.append((current_id, current_type, depth))

            # Add parents to queue
            if key in self.dependency_graph:
                for parent_id, parent_type in self.dependency_graph[key]:
                    queue.append((parent_id, parent_type, depth + 1))

        return anomalous_ancestors

    def _count_total_siblings(self, entity_id: str, entity_type: str) -> int:
        """Count total sibling entities (approximation)."""
        # This would ideally query database for total count
        # For now, return conservative estimate
        return 100  # TODO: Implement actual sibling count

    def analyze(
        self,
        anomaly_entity_id: str,
        anomaly_entity_type: str,
        related_anomalies: List[Tuple[str, str]],
    ) -> List[str]:
        """
        Trace root cause of an anomaly with multi-hop traversal.

        Parameters
        ----------
        anomaly_entity_id : str
            Entity where anomaly was detected
        anomaly_entity_type : str
            Type of entity ("keyword", "campaign", "portfolio")
        related_anomalies : list of (entity_id, entity_type)
            Other anomalies detected around the same time

        Returns
        -------
        root_causes : list of str
            Ranked list of potential root causes
        """
        root_causes = []

        # Check for anomalous ancestors (multi-hop)
        ancestors = self._find_anomalous_ancestors(
            anomaly_entity_id,
            anomaly_entity_type,
            related_anomalies,
            max_depth=3,
        )

        # Filter out self
        ancestors = [
            (aid, atype, depth) for aid, atype, depth in ancestors
            if not (aid == anomaly_entity_id and atype == anomaly_entity_type)
        ]

        if ancestors:
            # Sort by depth (closest ancestor first is likely root cause)
            ancestors.sort(key=lambda x: x[2])
            closest_ancestor = ancestors[0]
            root_causes.append(
                f"Inherited from {closest_ancestor[1]} {closest_ancestor[0]} "
                f"({closest_ancestor[2]} levels up) - systemic issue"
            )

        # Check sibling entities with ratio
        sibling_count = sum(
            1 for rel_id, rel_type in related_anomalies
            if rel_type == anomaly_entity_type and rel_id != anomaly_entity_id
        )

        if sibling_count > 0:
            total_siblings = self._count_total_siblings(anomaly_entity_id, anomaly_entity_type)
            sibling_ratio = sibling_count / max(total_siblings, 1)

            if sibling_ratio > 0.3 or sibling_count > 5:
                root_causes.insert(
                    0,
                    f"Widespread issue: {sibling_count} sibling {anomaly_entity_type}s anomalous "
                    f"({sibling_ratio:.1%} of total)"
                )

        if not root_causes:
            root_causes.append(f"Isolated anomaly in {anomaly_entity_type} {anomaly_entity_id}")

        return root_causes[:3]  # Top 3 causes

    def analyze_temporal_causality(
        self,
        anomaly: TimestampedAnomaly,
        related_anomalies: List[TimestampedAnomaly],
        time_window_minutes: int = 30,
    ) -> List[str]:
        """Identify which anomaly occurred first (likely root cause)."""
        root_causes = []

        # Find anomalies that occurred before target anomaly
        earlier_anomalies = [
            a for a in related_anomalies
            if a.timestamp < anomaly.timestamp
            and (anomaly.timestamp - a.timestamp).total_seconds() < time_window_minutes * 60
        ]

        if not earlier_anomalies:
            return [f"{anomaly.entity_type} {anomaly.entity_id} was first anomaly detected"]

        # Sort by how much earlier + hierarchy level
        # Campaign anomaly before keyword is more likely to be root cause
        hierarchy_priority = {"account": 4, "portfolio": 3, "campaign": 2, "keyword": 1}

        def score_causality(a: TimestampedAnomaly) -> float:
            time_diff_minutes = (anomaly.timestamp - a.timestamp).total_seconds() / 60
            recency_score = 1.0 / (1.0 + time_diff_minutes)  # Closer in time = higher score
            hierarchy_score = hierarchy_priority.get(a.entity_type, 0) / 4.0
            severity_score = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}.get(a.severity, 0.4)
            return recency_score * 0.4 + hierarchy_score * 0.4 + severity_score * 0.2

        earlier_anomalies.sort(key=score_causality, reverse=True)

        # Top candidate
        top_candidate = earlier_anomalies[0]
        time_diff = (anomaly.timestamp - top_candidate.timestamp).total_seconds() / 60
        root_causes.append(
            f"Likely caused by {top_candidate.entity_type} {top_candidate.entity_id} "
            f"(occurred {time_diff:.1f} min earlier, severity: {top_candidate.severity})"
        )

        return root_causes


# ═══════════════════════════════════════════════════════════════════════
#  5. Integrated Ensemble Detector
# ═══════════════════════════════════════════════════════════════════════

class EnsembleAnomalyDetector:
    """
    Combines all Phase 6 detectors into a unified anomaly detection system.

    Uses weighted voting across:
        1. LSTM Autoencoder (sequence reconstruction)
        2. Streaming Half-Space Trees (online)
        3. Isolation Forest (batch, from Phase 0)

    Returns explainable anomaly results with root cause analysis.
    """

    def __init__(self):
        self.lstm_detector = TimeSeriesAnomalyDetector() if TORCH_AVAILABLE else None
        self.streaming_detector = StreamingAnomalyDetector() if RIVER_AVAILABLE else None
        self.root_cause_analyzer = RootCauseAnalyzer()

        # Isolation Forest (sklearn-based batch detector)
        try:
            from sklearn.ensemble import IsolationForest
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
            )
            self._isolation_forest_fitted = False
        except ImportError:
            self.isolation_forest = None
            logger.warning("[Ensemble] scikit-learn not available, IsolationForest disabled")

        # Weights for ensemble voting (sum to 1.0)
        self.weights = {
            "lstm": 0.5,
            "streaming": 0.3,
            "isolation_forest": 0.2,
        }

    def fit_isolation_forest(self, X: np.ndarray):
        """Fit Isolation Forest on batch training data."""
        if self.isolation_forest is None:
            return
        
        self.isolation_forest.fit(X)
        self._isolation_forest_fitted = True
        logger.info(f"[Ensemble] Isolation Forest fitted on {len(X)} samples")

    def _create_ensemble_scorer(self, entity_id: str) -> callable:
        """Create scorer that uses actual streaming detector."""
        def scorer(features_dict: Dict[str, float]) -> float:
            if self.streaming_detector is None:
                # Fallback: simple deviation score
                total = sum(abs(features_dict.get(k, 0)) for k in features_dict.keys())
                return min(1.0, total / max(len(features_dict), 1) / 100)
            
            # Use actual streaming detector (without updating it)
            if not hasattr(self.streaming_detector, '_temp_detector'):
                # Create temporary detector clone for scoring
                temp_detector = StreamingAnomalyDetector(
                    n_trees=self.streaming_detector.n_trees,
                    height=self.streaming_detector.height,
                    window_size=self.streaming_detector.window_size,
                )
                # Copy state
                if RIVER_AVAILABLE and self.streaming_detector._fitted:
                    temp_detector.detector = self.streaming_detector.detector
                    temp_detector.scaler = self.streaming_detector.scaler
                self.streaming_detector._temp_detector = temp_detector
            
            # Score without learning
            if RIVER_AVAILABLE:
                scaled = self.streaming_detector.scaler.transform_one(features_dict)
                score = self.streaming_detector.detector.score_one(scaled)
                return min(1.0, score)
            
            return 0.0
        
        return scorer

    def detect_with_explanation(
        self,
        sequence: np.ndarray,
        features: Dict[str, float],
        entity_id: str,
        entity_type: str,
        related_anomalies: List[Tuple[str, str]] = None,
    ) -> AnomalyResult:
        """
        Comprehensive anomaly detection with explanation.

        Parameters
        ----------
        sequence : (seq_len, n_features) array
            Time series sequence for LSTM
        features : dict
            Current feature values for streaming detector
        entity_id : str
        entity_type : str ("keyword", "campaign", etc.)
        related_anomalies : list of (id, type)
            Other recent anomalies for root cause analysis
        """
        scores = []
        results = []

        # 1. LSTM Autoencoder
        if self.lstm_detector is not None:
            lstm_result = self.lstm_detector.detect(sequence)
            scores.append(lstm_result.anomaly_score * self.weights["lstm"])
            results.append(("lstm", lstm_result))

        # 2. Streaming Detector
        if self.streaming_detector is not None:
            stream_result = self.streaming_detector.update_and_detect(
                features, metric_name=f"{entity_type}_{entity_id}"
            )
            scores.append(stream_result.anomaly_score * self.weights["streaming"])
            results.append(("streaming", stream_result))

        # 3. Isolation Forest (if fitted)
        if self.isolation_forest is not None and self._isolation_forest_fitted:
            # Convert features dict to array
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            iso_score = self.isolation_forest.score_samples(feature_array)[0]
            # Convert to [0, 1] where higher = more anomalous
            iso_anomaly_score = min(1.0, max(0.0, -iso_score / 2.0))
            scores.append(iso_anomaly_score * self.weights["isolation_forest"])
            results.append(("isolation_forest", iso_anomaly_score))

        # Ensemble score
        if not scores:
            return AnomalyResult(
                is_anomalous=False,
                anomaly_score=0.0,
                threshold=0.5,
                timestamp=datetime.utcnow(),
                metric_name="ensemble",
                actual_value=0.0,
                severity="low",
            )

        ensemble_score = sum(scores)
        threshold = 0.5  # Calibrate based on validation data

        is_anomalous = ensemble_score > threshold

        # Severity
        if ensemble_score > 0.8:
            severity = "critical"
        elif ensemble_score > 0.7:
            severity = "high"
        elif ensemble_score > 0.6:
            severity = "medium"
        else:
            severity = "low"

        # Root cause analysis
        root_causes = []
        if is_anomalous and related_anomalies is not None:
            root_causes = self.root_cause_analyzer.analyze(
                entity_id, entity_type, related_anomalies
            )

        # Explanation (use actual ensemble scorer)
        explanation = None
        if is_anomalous and features:
            # Create scorer using actual detector models
            scorer = self._create_ensemble_scorer(entity_id)
            
            explainer = AnomalyExplainer(scorer, n_samples=50)
            explanations = explainer.explain(features)
            explanation = {
                exp.name: exp.contribution
                for exp in explanations[:5]  # Top 5 features
            }

        return AnomalyResult(
            is_anomalous=is_anomalous,
            anomaly_score=ensemble_score,
            threshold=threshold,
            timestamp=datetime.utcnow(),
            metric_name="ensemble_anomaly",
            actual_value=ensemble_score,
            severity=severity if is_anomalous else "low",
            explanation=explanation,
            root_causes=root_causes,
        )
