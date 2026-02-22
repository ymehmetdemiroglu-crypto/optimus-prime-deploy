"""
Rich Product Embedding Builder — Cosmo Alignment Layer

Amazon's Cosmo ranking system evaluates query–product relevance using
structured product signals far beyond title + bullet points:

  • Category hierarchy (browse-node path)
  • Brand identity
  • Product attributes (size, color, material …)
  • Price tier
  • Review quality signals

This service constructs a *structured source text* that mirrors those
signals before embedding, so our 384-d vectors align with how Cosmo
actually scores relevance.  The richer the input, the higher the
cosmo_alignment_score (0–1 confidence that our vector matches Cosmo's
internal representation).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("rich_product_embeddings")

# ── Embedding version tag ────────────────────────────────────────────
EMBEDDING_VERSION_RICH = "v2_cosmo_rich"
EMBEDDING_VERSION_BASIC = "v1_title_only"

# ── Weight map for Cosmo alignment score ─────────────────────────────
# Each field contributes a fraction to the overall alignment confidence.
# Weights sum to 1.0.
_ALIGNMENT_WEIGHTS: Dict[str, float] = {
    "title":         0.25,
    "bullet_points": 0.15,
    "brand":         0.12,
    "category_path": 0.15,
    "product_type":  0.08,
    "attributes":    0.10,
    "price":         0.05,
    "review_score":  0.05,
    "review_count":  0.05,
}


@dataclass
class ProductMetadata:
    """All product fields that can contribute to a rich embedding."""

    asin: str
    title: str
    bullet_points: Optional[List[str]] = None
    brand: Optional[str] = None
    category_path: Optional[str] = None       # "Home & Kitchen > Kitchen > Coffee"
    product_type: Optional[str] = None        # "COFFEE_MAKER"
    attributes: Optional[Dict[str, str]] = None
    price: Optional[float] = None
    review_score: Optional[float] = None
    review_count: Optional[int] = None
    parent_asin: Optional[str] = None

    # Populated by the builder
    source_text: str = ""
    cosmo_alignment_score: float = 0.0
    embedding_version: str = EMBEDDING_VERSION_BASIC


def _price_tier(price: float) -> str:
    """Map a price to a human-readable tier bucket that Cosmo would understand."""
    if price < 10:
        return "budget"
    if price < 25:
        return "value"
    if price < 75:
        return "mid-range"
    if price < 200:
        return "premium"
    return "luxury"


def _review_tier(score: float, count: int) -> str:
    """Convert review signals into a textual quality tier."""
    if count < 5:
        return "new product"
    if score >= 4.5:
        return "top rated"
    if score >= 4.0:
        return "well reviewed"
    if score >= 3.0:
        return "average reviews"
    return "low rated"


class RichProductEmbeddingBuilder:
    """
    Builds structured source text for product embeddings and computes
    a cosmo_alignment_score that quantifies how much signal we captured.

    Usage::

        builder = RichProductEmbeddingBuilder()
        meta = ProductMetadata(asin="B09XYZ", title="...", brand="...", ...)
        meta = builder.build(meta)
        # meta.source_text  → structured text ready for embedding_service.embed_text()
        # meta.cosmo_alignment_score → 0.0 – 1.0
    """

    def build(self, meta: ProductMetadata) -> ProductMetadata:
        """Populate source_text, cosmo_alignment_score, and embedding_version."""
        segments: List[str] = []
        present: Dict[str, bool] = {}

        # 1. Title — always present
        segments.append(meta.title)
        present["title"] = True

        # 2. Brand
        if meta.brand:
            segments.append(f"Brand: {meta.brand}")
            present["brand"] = True

        # 3. Category path  (e.g. "Home & Kitchen > Kitchen > Coffee")
        if meta.category_path:
            segments.append(f"Category: {meta.category_path}")
            present["category_path"] = True

        # 4. Product type
        if meta.product_type:
            # Normalise underscored Amazon PT into readable text
            readable = meta.product_type.replace("_", " ").title()
            segments.append(f"Type: {readable}")
            present["product_type"] = True

        # 5. Bullet points
        if meta.bullet_points:
            # Keep first 5 bullets, truncate very long ones
            clean = [_truncate(b, 200) for b in meta.bullet_points[:5]]
            segments.append("Features: " + " | ".join(clean))
            present["bullet_points"] = True

        # 6. Attributes (key-value pairs)
        if meta.attributes:
            attr_parts = [f"{k}: {v}" for k, v in meta.attributes.items()]
            segments.append("Attributes: " + ", ".join(attr_parts[:10]))
            present["attributes"] = True

        # 7. Price tier
        if meta.price is not None and meta.price > 0:
            tier = _price_tier(meta.price)
            segments.append(f"Price: {tier}")
            present["price"] = True

        # 8. Review tier
        if meta.review_score is not None and meta.review_count is not None:
            tier = _review_tier(meta.review_score, meta.review_count)
            segments.append(f"Reviews: {tier}")
            present["review_score"] = True
            present["review_count"] = True

        # ── Assemble ────────────────────────────────────────────────
        meta.source_text = " . ".join(segments)
        meta.cosmo_alignment_score = self._compute_alignment(present)
        meta.embedding_version = (
            EMBEDDING_VERSION_RICH if meta.cosmo_alignment_score >= 0.40
            else EMBEDDING_VERSION_BASIC
        )

        logger.debug(
            "Built rich embedding for %s: alignment=%.2f, version=%s, fields=%s",
            meta.asin, meta.cosmo_alignment_score, meta.embedding_version,
            list(present.keys()),
        )
        return meta

    # ── Internal ────────────────────────────────────────────────────
    @staticmethod
    def _compute_alignment(present: Dict[str, bool]) -> float:
        """Sum weights for all fields that are present."""
        return round(
            sum(
                _ALIGNMENT_WEIGHTS.get(k, 0.0)
                for k, v in present.items() if v
            ),
            4,
        )


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters, preserving word boundaries."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut + "…"


# ── Module-level convenience ─────────────────────────────────────────
_builder: Optional[RichProductEmbeddingBuilder] = None


def get_rich_embedding_builder() -> RichProductEmbeddingBuilder:
    global _builder
    if _builder is None:
        _builder = RichProductEmbeddingBuilder()
    return _builder
