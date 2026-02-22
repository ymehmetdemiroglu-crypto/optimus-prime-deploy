"""
Tests for Rich Product Embeddings — Cosmo Alignment Layer.

Validates:
- RichProductEmbeddingBuilder produces correct structured source text
- cosmo_alignment_score weights are computed accurately
- Embedding version tagging (v1 basic vs v2 rich)
- Edge cases: minimal metadata, maximal metadata, missing fields
- ProductMetadata dataclass behavior
- Backward compatibility: title-only embeds still work
"""

import pytest
from app.services.ml.rich_product_embeddings import (
    RichProductEmbeddingBuilder,
    ProductMetadata,
    EMBEDDING_VERSION_BASIC,
    EMBEDDING_VERSION_RICH,
    _ALIGNMENT_WEIGHTS,
    _price_tier,
    _review_tier,
    _truncate,
    get_rich_embedding_builder,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def builder():
    return RichProductEmbeddingBuilder()


@pytest.fixture
def full_metadata():
    """A fully-populated product metadata object."""
    return ProductMetadata(
        asin="B09XYZ1234",
        title="Stainless Steel French Press Coffee Maker 34oz",
        bullet_points=[
            "Double-wall vacuum insulation keeps coffee hot for hours",
            "Premium 18/10 stainless steel construction",
            "4-level filtration system for smooth, grit-free coffee",
            "Dishwasher safe — easy cleanup",
        ],
        brand="BrewMaster",
        category_path="Home & Kitchen > Kitchen & Dining > Coffee > French Presses",
        product_type="FRENCH_PRESS",
        attributes={
            "material": "Stainless Steel",
            "capacity": "34 oz",
            "color": "Silver",
            "dishwasher_safe": "Yes",
        },
        price=34.99,
        review_score=4.6,
        review_count=1287,
        parent_asin="B09XYZ0000",
    )


@pytest.fixture
def minimal_metadata():
    """Title-only product — backward-compatible bare minimum."""
    return ProductMetadata(
        asin="B00MINIMAL",
        title="Basic Widget",
    )


# ── Alignment weight sanity ──────────────────────────────────────────

class TestAlignmentWeights:
    def test_weights_sum_to_one(self):
        """All alignment weights should sum to 1.0."""
        total = sum(_ALIGNMENT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_all_fields_have_weights(self):
        expected_fields = {
            "title", "bullet_points", "brand", "category_path",
            "product_type", "attributes", "price", "review_score", "review_count",
        }
        assert set(_ALIGNMENT_WEIGHTS.keys()) == expected_fields


# ── Price tier ───────────────────────────────────────────────────────

class TestPriceTier:
    @pytest.mark.parametrize("price,expected", [
        (5.99, "budget"),
        (9.99, "budget"),
        (15.00, "value"),
        (24.99, "value"),
        (50.00, "mid-range"),
        (74.99, "mid-range"),
        (100.00, "premium"),
        (199.99, "premium"),
        (250.00, "luxury"),
        (999.99, "luxury"),
    ])
    def test_price_tiers(self, price, expected):
        assert _price_tier(price) == expected


# ── Review tier ──────────────────────────────────────────────────────

class TestReviewTier:
    @pytest.mark.parametrize("score,count,expected", [
        (4.8, 500, "top rated"),
        (4.5, 100, "top rated"),
        (4.2, 50, "well reviewed"),
        (4.0, 10, "well reviewed"),
        (3.5, 20, "average reviews"),
        (2.5, 30, "low rated"),
        (4.9, 3, "new product"),      # high score but too few reviews
        (1.0, 2, "new product"),
    ])
    def test_review_tiers(self, score, count, expected):
        assert _review_tier(score, count) == expected


# ── Text truncation ──────────────────────────────────────────────────

class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello world", 100) == "hello world"

    def test_long_text_truncated_at_word_boundary(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = _truncate(text, 20)
        assert len(result) <= 21  # 20 + "…"
        assert result.endswith("…")

    def test_exact_length_unchanged(self):
        text = "exact"
        assert _truncate(text, 5) == "exact"


# ── Builder: full metadata ───────────────────────────────────────────

class TestBuilderFullMetadata:
    def test_source_text_contains_all_segments(self, builder, full_metadata):
        result = builder.build(full_metadata)

        assert "Stainless Steel French Press Coffee Maker 34oz" in result.source_text
        assert "Brand: BrewMaster" in result.source_text
        assert "Category: Home & Kitchen" in result.source_text
        assert "Type: French Press" in result.source_text  # underscores → title case
        assert "Features:" in result.source_text
        assert "vacuum insulation" in result.source_text
        assert "Attributes:" in result.source_text
        assert "material: Stainless Steel" in result.source_text
        assert "Price: mid-range" in result.source_text
        assert "Reviews: top rated" in result.source_text

    def test_full_alignment_score(self, builder, full_metadata):
        result = builder.build(full_metadata)
        # All fields present → score should equal sum of all weights = 1.0
        assert result.cosmo_alignment_score == pytest.approx(1.0, abs=0.001)

    def test_embedding_version_is_rich(self, builder, full_metadata):
        result = builder.build(full_metadata)
        assert result.embedding_version == EMBEDDING_VERSION_RICH

    def test_segments_separated_by_dot(self, builder, full_metadata):
        result = builder.build(full_metadata)
        assert " . " in result.source_text


# ── Builder: minimal metadata ────────────────────────────────────────

class TestBuilderMinimalMetadata:
    def test_source_text_is_just_title(self, builder, minimal_metadata):
        result = builder.build(minimal_metadata)
        assert result.source_text == "Basic Widget"

    def test_alignment_score_title_only(self, builder, minimal_metadata):
        result = builder.build(minimal_metadata)
        assert result.cosmo_alignment_score == pytest.approx(
            _ALIGNMENT_WEIGHTS["title"], abs=0.001
        )

    def test_embedding_version_is_basic(self, builder, minimal_metadata):
        result = builder.build(minimal_metadata)
        assert result.embedding_version == EMBEDDING_VERSION_BASIC


# ── Builder: partial metadata (brand + title) ────────────────────────

class TestBuilderPartialMetadata:
    def test_brand_and_title(self, builder):
        meta = ProductMetadata(
            asin="B00BRAND",
            title="Premium Dog Treats",
            brand="PawNatural",
        )
        result = builder.build(meta)

        expected_score = _ALIGNMENT_WEIGHTS["title"] + _ALIGNMENT_WEIGHTS["brand"]
        assert result.cosmo_alignment_score == pytest.approx(expected_score, abs=0.001)
        assert "Brand: PawNatural" in result.source_text
        assert "Premium Dog Treats" in result.source_text

    def test_title_bullets_brand_category(self, builder):
        meta = ProductMetadata(
            asin="B00COMBO",
            title="Wireless Bluetooth Earbuds",
            bullet_points=["Active noise cancellation", "30hr battery"],
            brand="SoundPeak",
            category_path="Electronics > Headphones > In-Ear",
        )
        result = builder.build(meta)

        expected_score = sum(
            _ALIGNMENT_WEIGHTS[k]
            for k in ("title", "bullet_points", "brand", "category_path")
        )
        assert result.cosmo_alignment_score == pytest.approx(expected_score, abs=0.001)
        # 0.25+0.15+0.12+0.15 = 0.67 → rich version
        assert result.embedding_version == EMBEDDING_VERSION_RICH

    def test_threshold_boundary(self, builder):
        """A score just below 0.40 should stay v1 basic."""
        meta = ProductMetadata(
            asin="B00EDGE",
            title="Widget",
            brand="Acme",
        )
        result = builder.build(meta)
        # title(0.25) + brand(0.12) = 0.37 < 0.40
        assert result.cosmo_alignment_score == pytest.approx(0.37, abs=0.001)
        assert result.embedding_version == EMBEDDING_VERSION_BASIC


# ── Builder: bullet point truncation ─────────────────────────────────

class TestBuilderBulletTruncation:
    def test_only_first_five_bullets_used(self, builder):
        meta = ProductMetadata(
            asin="B00MANY",
            title="Product",
            bullet_points=[f"Bullet {i}" for i in range(10)],
        )
        result = builder.build(meta)
        assert "Bullet 4" in result.source_text
        assert "Bullet 5" not in result.source_text

    def test_long_bullet_truncated(self, builder):
        long_bullet = "A" * 300
        meta = ProductMetadata(
            asin="B00LONG",
            title="Product",
            bullet_points=[long_bullet],
        )
        result = builder.build(meta)
        # The bullet should be truncated to ~200 chars + ellipsis
        assert len(result.source_text) < 250


# ── Builder: attribute limits ────────────────────────────────────────

class TestBuilderAttributes:
    def test_only_first_ten_attributes(self, builder):
        attrs = {f"attr_{i}": f"val_{i}" for i in range(15)}
        meta = ProductMetadata(
            asin="B00ATTRS",
            title="Product",
            attributes=attrs,
        )
        result = builder.build(meta)
        assert "attr_9" in result.source_text
        assert "attr_10" not in result.source_text


# ── Singleton accessor ───────────────────────────────────────────────

class TestSingleton:
    def test_get_rich_embedding_builder_returns_same_instance(self):
        a = get_rich_embedding_builder()
        b = get_rich_embedding_builder()
        assert a is b

    def test_builder_is_correct_type(self):
        assert isinstance(get_rich_embedding_builder(), RichProductEmbeddingBuilder)


# ── ProductMetadata dataclass ────────────────────────────────────────

class TestProductMetadata:
    def test_defaults(self):
        meta = ProductMetadata(asin="B00X", title="T")
        assert meta.bullet_points is None
        assert meta.brand is None
        assert meta.source_text == ""
        assert meta.cosmo_alignment_score == 0.0
        assert meta.embedding_version == EMBEDDING_VERSION_BASIC

    def test_all_fields_settable(self, full_metadata):
        assert full_metadata.price == 34.99
        assert full_metadata.review_count == 1287
        assert full_metadata.parent_asin == "B09XYZ0000"


# ── Product type normalisation ───────────────────────────────────────

class TestProductTypeNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("FRENCH_PRESS", "Type: French Press"),
        ("COFFEE_MAKER", "Type: Coffee Maker"),
        ("WIRELESS_EARBUDS", "Type: Wireless Earbuds"),
        ("DOG_TREAT", "Type: Dog Treat"),
    ])
    def test_underscore_to_title_case(self, builder, raw, expected):
        meta = ProductMetadata(asin="B00X", title="T", product_type=raw)
        result = builder.build(meta)
        assert expected in result.source_text
