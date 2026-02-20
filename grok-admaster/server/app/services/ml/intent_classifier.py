"""
Shopping Intent Classifier — Rufus & Cosmo Optimization Layer

Classifies Amazon search queries into shopping intent types to optimize
bidding, bleed detection, and opportunity discovery for conversational
AI traffic (Rufus) and semantic ranking (Cosmo).

Intent Types:
  - transactional: Direct purchase intent ("nike air max 90 mens size 11")
  - informational_rufus: Research/comparison queries from Rufus ("best wireless earbuds for running")
  - navigational: Brand-specific navigation ("anker powerbank")
  - discovery: Category exploration ("gifts for hikers under $50")
"""
import re
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ShoppingIntent(str, Enum):
    TRANSACTIONAL = "transactional"
    INFORMATIONAL_RUFUS = "informational_rufus"
    NAVIGATIONAL = "navigational"
    DISCOVERY = "discovery"


@dataclass
class IntentResult:
    """Classification result for a single query."""
    query: str
    intent: ShoppingIntent
    confidence: float
    scores: Dict[str, float]
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "intent": self.intent.value,
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "signals": self.signals,
        }


# --- Intent threshold config (used by bleed/opportunity callers) ---

INTENT_THRESHOLDS = {
    ShoppingIntent.TRANSACTIONAL: {
        "bleed_threshold": 0.35,
        "opportunity_floor": 0.75,
    },
    ShoppingIntent.INFORMATIONAL_RUFUS: {
        "bleed_threshold": 0.20,
        "opportunity_floor": 0.50,
    },
    ShoppingIntent.NAVIGATIONAL: {
        "bleed_threshold": 0.30,
        "opportunity_floor": 0.70,
    },
    ShoppingIntent.DISCOVERY: {
        "bleed_threshold": 0.15,
        "opportunity_floor": 0.45,
    },
}


def get_intent_thresholds(intent: ShoppingIntent) -> Dict[str, float]:
    """Return bleed/opportunity thresholds adjusted for the query intent."""
    return INTENT_THRESHOLDS.get(intent, INTENT_THRESHOLDS[ShoppingIntent.TRANSACTIONAL])


class IntentClassifier:
    """
    Hybrid intent classifier combining pattern-based rules with
    embedding similarity for robust shopping intent detection.

    Pattern layer catches explicit signals (question words, brand names,
    size/color specifiers, comparison phrases). Embedding layer measures
    distance to intent-prototype sentences to resolve ambiguous cases.
    """

    # --- Pattern Definitions ---

    # Rufus-style informational / research queries
    RUFUS_PATTERNS = [
        r"\b(?:what|which|how|why|can|does|do|is|are|should)\b",
        r"\bbest\s+\w+\s+(?:for|under|over|around|with)\b",
        r"\b(?:vs|versus|compared?\s+to|difference\s+between|or)\b",
        r"\b(?:review|recommend|rating|worth|alternative)\b",
        r"\b(?:pros?\s+and\s+cons?|top\s+\d+|guide|tips?)\b",
    ]

    # Transactional — ready to buy
    TRANSACTIONAL_PATTERNS = [
        r"\b(?:buy|order|purchase|add\s+to\s+cart|subscribe)\b",
        r"\b(?:pack\s+of|count|set\s+of)\s+\d+",
        r"\b(?:size|color|model)\s+\S+",
        r"\b\d+(?:oz|ml|lb|kg|inch|in|mm|cm|ct|pack)\b",
        r"\b[A-Z][A-Z0-9\-]{4,}\b",  # model numbers like B0DWK3C1R7
    ]

    # Navigational — brand-seeking
    NAVIGATIONAL_PATTERNS = [
        r"\b(?:brand|official|genuine|authentic|original)\b",
        r"\b(?:amazon\s+basics|amazon\s+choice|sponsored)\b",
    ]

    # Discovery — exploring categories
    DISCOVERY_PATTERNS = [
        r"\b(?:gift|gifts?)\s+(?:for|under|idea)\b",
        r"\b(?:things?\s+to|stuff\s+for|items?\s+for)\b",
        r"\b(?:under|around|less\s+than)\s+\$?\d+",
        r"\b(?:cool|unique|popular|trending|new)\s+\w+",
        r"\b(?:ideas?\s+for|essentials?\s+for|must\s+have)\b",
    ]

    # Prototype sentences for embedding-based intent matching
    INTENT_PROTOTYPES = {
        ShoppingIntent.TRANSACTIONAL: [
            "buy nike air max 90 mens size 11 black",
            "order 24 pack alkaline batteries",
            "anker powercore 20000mah portable charger",
            "samsung galaxy s24 ultra 256gb unlocked",
        ],
        ShoppingIntent.INFORMATIONAL_RUFUS: [
            "what is the best wireless earbuds for running",
            "which protein powder is best for weight loss",
            "how to choose a good laptop for college",
            "difference between air fryer and convection oven",
            "is the new kindle worth it compared to paperwhite",
        ],
        ShoppingIntent.NAVIGATIONAL: [
            "anker official store",
            "apple airpods pro",
            "dyson v15 vacuum",
            "yeti rambler tumbler",
        ],
        ShoppingIntent.DISCOVERY: [
            "gifts for hikers under 50 dollars",
            "cool kitchen gadgets",
            "unique gifts for dad",
            "trending home office accessories",
            "must have camping essentials",
        ],
    }

    def __init__(self, embedding_service=None):
        """
        Args:
            embedding_service: Optional embedding service instance.
                If provided, enables the embedding similarity layer.
                If None, classification uses patterns only.
        """
        self._embedding_service = embedding_service
        self._prototype_embeddings: Optional[Dict[ShoppingIntent, np.ndarray]] = None
        self._compiled_patterns = {
            ShoppingIntent.INFORMATIONAL_RUFUS: [re.compile(p, re.IGNORECASE) for p in self.RUFUS_PATTERNS],
            ShoppingIntent.TRANSACTIONAL: [re.compile(p, re.IGNORECASE) for p in self.TRANSACTIONAL_PATTERNS],
            ShoppingIntent.NAVIGATIONAL: [re.compile(p, re.IGNORECASE) for p in self.NAVIGATIONAL_PATTERNS],
            ShoppingIntent.DISCOVERY: [re.compile(p, re.IGNORECASE) for p in self.DISCOVERY_PATTERNS],
        }

    def _ensure_prototypes(self):
        """Lazily compute prototype embeddings on first use."""
        if self._prototype_embeddings is not None or self._embedding_service is None:
            return

        self._prototype_embeddings = {}
        for intent, sentences in self.INTENT_PROTOTYPES.items():
            vecs = [self._embedding_service.encode(s) for s in sentences]
            # Average prototype vector (centroid)
            self._prototype_embeddings[intent] = np.mean(vecs, axis=0)

    # --- Core API ---

    def classify(self, query: str) -> IntentResult:
        """Classify a single search query into a shopping intent type."""
        query_clean = query.strip().lower()
        if not query_clean:
            return IntentResult(
                query=query,
                intent=ShoppingIntent.TRANSACTIONAL,
                confidence=0.0,
                scores={i.value: 0.0 for i in ShoppingIntent},
                signals=["empty_query"],
            )

        # Layer 1: Pattern scoring
        pattern_scores, signals = self._score_patterns(query_clean)

        # Layer 2: Embedding similarity scoring
        embedding_scores = self._score_embeddings(query_clean)

        # Layer 3: Structural features
        structural_scores, structural_signals = self._score_structural(query_clean)
        signals.extend(structural_signals)

        # Combine layers (pattern=0.45, embedding=0.35, structural=0.20)
        combined = {}
        for intent in ShoppingIntent:
            p = pattern_scores.get(intent, 0.0)
            e = embedding_scores.get(intent, 0.0)
            s = structural_scores.get(intent, 0.0)
            combined[intent] = 0.45 * p + 0.35 * e + 0.20 * s

        # Pick winner
        best_intent = max(combined, key=combined.get)
        best_score = combined[best_intent]

        # If all scores are very low, default to transactional (most common on Amazon)
        if best_score < 0.10:
            best_intent = ShoppingIntent.TRANSACTIONAL
            best_score = 0.30
            signals.append("default_transactional")

        # Confidence = margin between top and second-best
        sorted_scores = sorted(combined.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        confidence = min(1.0, best_score + margin * 0.5)

        return IntentResult(
            query=query,
            intent=best_intent,
            confidence=confidence,
            scores={k.value: v for k, v in combined.items()},
            signals=signals,
        )

    def classify_batch(self, queries: List[str]) -> List[IntentResult]:
        """Classify a batch of search queries."""
        return [self.classify(q) for q in queries]

    # --- Scoring Layers ---

    def _score_patterns(self, query: str) -> Tuple[Dict[ShoppingIntent, float], List[str]]:
        """Score query against regex patterns for each intent."""
        scores: Dict[ShoppingIntent, float] = {}
        signals: List[str] = []

        for intent, patterns in self._compiled_patterns.items():
            matches = sum(1 for p in patterns if p.search(query))
            total = len(patterns)
            score = min(1.0, matches / max(total * 0.4, 1))  # Normalize; 40% match = 1.0
            scores[intent] = score
            if matches > 0:
                signals.append(f"pattern:{intent.value}:{matches}")

        return scores, signals

    def _score_embeddings(self, query: str) -> Dict[ShoppingIntent, float]:
        """Score query by cosine similarity to intent prototype centroids."""
        if self._embedding_service is None:
            return {i: 0.0 for i in ShoppingIntent}

        self._ensure_prototypes()
        if self._prototype_embeddings is None:
            return {i: 0.0 for i in ShoppingIntent}

        query_vec = self._embedding_service.encode(query)
        if np.all(query_vec == 0):
            return {i: 0.0 for i in ShoppingIntent}

        scores = {}
        for intent, proto_vec in self._prototype_embeddings.items():
            dot = float(np.dot(query_vec, proto_vec))
            norm = float(np.linalg.norm(query_vec) * np.linalg.norm(proto_vec))
            sim = dot / norm if norm > 0 else 0.0
            # Map similarity [0, 1] -> score [0, 1] with a floor at 0
            scores[intent] = max(0.0, sim)

        # Normalize to emphasize the winner
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def _score_structural(self, query: str) -> Tuple[Dict[ShoppingIntent, float], List[str]]:
        """Score based on structural features of the query."""
        tokens = query.split()
        word_count = len(tokens)
        signals: List[str] = []

        scores: Dict[ShoppingIntent, float] = {i: 0.0 for i in ShoppingIntent}

        # Word count heuristics
        if word_count <= 2:
            # Short queries are usually navigational or transactional
            scores[ShoppingIntent.NAVIGATIONAL] += 0.4
            scores[ShoppingIntent.TRANSACTIONAL] += 0.3
            signals.append("short_query")
        elif word_count <= 4:
            scores[ShoppingIntent.TRANSACTIONAL] += 0.4
            scores[ShoppingIntent.NAVIGATIONAL] += 0.2
        elif word_count <= 7:
            # Medium-long: could be anything, slight lean toward discovery/rufus
            scores[ShoppingIntent.DISCOVERY] += 0.3
            scores[ShoppingIntent.INFORMATIONAL_RUFUS] += 0.3
            signals.append("medium_long_query")
        else:
            # Long queries (8+) are very likely Rufus conversational
            scores[ShoppingIntent.INFORMATIONAL_RUFUS] += 0.6
            signals.append("long_query_rufus_likely")

        # Question marks
        if "?" in query:
            scores[ShoppingIntent.INFORMATIONAL_RUFUS] += 0.4
            signals.append("question_mark")

        # Contains digits (size, quantity, price) — transactional signal
        if re.search(r"\d", query):
            scores[ShoppingIntent.TRANSACTIONAL] += 0.2

        # Contains "for" followed by a person/use case — discovery signal
        if re.search(r"\bfor\s+(?:my|him|her|kids?|men|women|boys?|girls?)\b", query):
            scores[ShoppingIntent.DISCOVERY] += 0.3
            signals.append("persona_target")

        # Cap scores at 1.0
        scores = {k: min(1.0, v) for k, v in scores.items()}
        return scores, signals


# --- Module-level singleton ---

_classifier_instance: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the global IntentClassifier singleton."""
    global _classifier_instance
    if _classifier_instance is None:
        try:
            from app.services.ml.embedding_service import EmbeddingService
            emb = EmbeddingService.get_instance()
            _classifier_instance = IntentClassifier(embedding_service=emb)
        except Exception:
            logger.warning("Embedding service unavailable — using pattern-only intent classifier")
            _classifier_instance = IntentClassifier(embedding_service=None)
    return _classifier_instance
