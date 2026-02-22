"""
Tests for HNSW Vector Search Service (Priority #4).

Validates:
- HNSWConfig parameter validation and ranges
- ef_search resolution per SearchMode
- VectorSearchService query construction (mocked DB)
- 2-step KNN pattern: product lookup then neighbor scan
- Backward compatibility: all existing result fields preserved
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from app.services.ml.vector_search import (
    HNSWConfig,
    SearchMode,
    VectorSearchService,
    get_ef_search,
    get_hnsw_config,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_M_RANGE,
    HNSW_EF_CONSTRUCTION_RANGE,
    HNSW_EF_SEARCH_RANGE,
    EMBEDDING_DIM,
    _EF_SEARCH_DEFAULTS,
)


# ── HNSW Config ──────────────────────────────────────────────────────

class TestHNSWConfig:
    def test_default_config_is_valid(self):
        cfg = HNSWConfig()
        assert cfg.validate() is True

    def test_default_m(self):
        assert HNSWConfig().m == 16

    def test_default_ef_construction(self):
        assert HNSWConfig().ef_construction == 128

    def test_default_embedding_dim(self):
        assert HNSWConfig().embedding_dim == 384

    def test_m_below_range_invalid(self):
        cfg = HNSWConfig(m=1)
        assert cfg.validate() is False

    def test_m_above_range_invalid(self):
        cfg = HNSWConfig(m=101)
        assert cfg.validate() is False

    def test_ef_construction_below_range_invalid(self):
        cfg = HNSWConfig(ef_construction=3)
        assert cfg.validate() is False

    def test_ef_construction_above_range_invalid(self):
        cfg = HNSWConfig(ef_construction=1001)
        assert cfg.validate() is False

    def test_zero_dim_invalid(self):
        cfg = HNSWConfig(embedding_dim=0)
        assert cfg.validate() is False

    def test_boundary_values_valid(self):
        cfg = HNSWConfig(m=2, ef_construction=4, embedding_dim=1)
        assert cfg.validate() is True
        cfg = HNSWConfig(m=100, ef_construction=1000, embedding_dim=1536)
        assert cfg.validate() is True

    def test_config_immutable(self):
        cfg = HNSWConfig()
        with pytest.raises(AttributeError):
            cfg.m = 32

    def test_get_hnsw_config_returns_default(self):
        cfg = get_hnsw_config()
        assert cfg.m == HNSW_M
        assert cfg.ef_construction == HNSW_EF_CONSTRUCTION
        assert cfg.embedding_dim == EMBEDDING_DIM


# ── Parameter Ranges ─────────────────────────────────────────────────

class TestParameterRanges:
    def test_m_range_reasonable(self):
        low, high = HNSW_M_RANGE
        assert low >= 2, "m must be at least 2"
        assert high <= 100, "m above 100 is impractical"
        assert low <= HNSW_M <= high

    def test_ef_construction_range(self):
        low, high = HNSW_EF_CONSTRUCTION_RANGE
        assert low >= 4
        assert high <= 1000
        assert low <= HNSW_EF_CONSTRUCTION <= high

    def test_ef_search_range(self):
        low, high = HNSW_EF_SEARCH_RANGE
        assert low >= 1
        assert high <= 1000

    def test_default_m_matches_migration(self):
        """m=16 is what migration 08 uses."""
        assert HNSW_M == 16

    def test_default_ef_construction_matches_migration(self):
        """ef_construction=128 is what migration 08 uses."""
        assert HNSW_EF_CONSTRUCTION == 128


# ── ef_search Resolution ─────────────────────────────────────────────

class TestEfSearch:
    def test_bleed_mode_default(self):
        assert get_ef_search(SearchMode.BLEED) == 200

    def test_opportunity_mode_default(self):
        assert get_ef_search(SearchMode.OPPORTUNITY) == 100

    def test_patrol_mode_default(self):
        assert get_ef_search(SearchMode.PATROL) == 80

    def test_override_respected(self):
        assert get_ef_search(SearchMode.BLEED, override=50) == 50

    def test_override_clamped_low(self):
        assert get_ef_search(SearchMode.BLEED, override=0) == HNSW_EF_SEARCH_RANGE[0]

    def test_override_clamped_high(self):
        assert get_ef_search(SearchMode.BLEED, override=9999) == HNSW_EF_SEARCH_RANGE[1]

    def test_all_modes_have_defaults(self):
        for mode in SearchMode:
            assert mode in _EF_SEARCH_DEFAULTS
            assert get_ef_search(mode) > 0

    def test_bleed_higher_than_opportunity(self):
        """Bleed needs higher recall, so ef_search should be higher."""
        assert get_ef_search(SearchMode.BLEED) > get_ef_search(SearchMode.OPPORTUNITY)


# ── SearchMode Enum ──────────────────────────────────────────────────

class TestSearchMode:
    def test_bleed_value(self):
        assert SearchMode.BLEED.value == "bleed"

    def test_opportunity_value(self):
        assert SearchMode.OPPORTUNITY.value == "opportunity"

    def test_patrol_value(self):
        assert SearchMode.PATROL.value == "patrol"

    def test_all_modes_are_strings(self):
        for mode in SearchMode:
            assert isinstance(mode.value, str)


# ── VectorSearchService ──────────────────────────────────────────────

class TestVectorSearchService:
    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        return db

    @pytest.fixture
    def service(self, mock_db):
        return VectorSearchService(mock_db)

    @pytest.mark.asyncio
    async def test_set_ef_search(self, service, mock_db):
        await service._set_ef_search(200)
        mock_db.execute.assert_called_once()
        # Extract the TextClause's .text property
        sql_clause = mock_db.execute.call_args[0][0]
        assert "hnsw.ef_search" in sql_clause.text
        assert "200" in sql_clause.text

    @pytest.mark.asyncio
    async def test_set_ef_search_clamps_low(self, service, mock_db):
        await service._set_ef_search(-5)
        sql_clause = mock_db.execute.call_args[0][0]
        assert f"= {HNSW_EF_SEARCH_RANGE[0]}" in sql_clause.text

    @pytest.mark.asyncio
    async def test_set_ef_search_clamps_high(self, service, mock_db):
        await service._set_ef_search(9999)
        sql_clause = mock_db.execute.call_args[0][0]
        assert f"= {HNSW_EF_SEARCH_RANGE[1]}" in sql_clause.text

    @pytest.mark.asyncio
    async def test_get_product_vector_returns_none_when_missing(self, service, mock_db):
        result_mock = MagicMock()
        result_mock.fetchone.return_value = None
        mock_db.execute.return_value = result_mock

        result = await service.get_product_vector("B00MISSING", 1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_product_vector_returns_dict(self, service, mock_db):
        import uuid
        row_mock = MagicMock()
        row_mock.id = uuid.uuid4()
        row_mock.embedding = [0.1] * 384
        row_mock.embedding_version = "v2_cosmo_rich"
        row_mock.cosmo_alignment_score = Decimal("0.8500")

        result_mock = MagicMock()
        result_mock.fetchone.return_value = row_mock
        mock_db.execute.return_value = result_mock

        result = await service.get_product_vector("B09XYZ", 42)
        assert result is not None
        assert result["embedding"] == [0.1] * 384
        assert result["embedding_version"] == "v2_cosmo_rich"
        assert result["cosmo_alignment_score"] == Decimal("0.8500")
        assert "product_embedding_id" in result

    @pytest.mark.asyncio
    async def test_find_nearest_calls_set_ef_search(self, service, mock_db):
        # Mock return for both SET ef_search and the query
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_db.execute.return_value = result_mock

        await service.find_nearest(
            reference_vector=[0.1] * 384,
            account_id=1,
            k=10,
            mode=SearchMode.OPPORTUNITY,
        )

        # Should have 2 execute calls: SET ef_search + the query
        assert mock_db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_find_farthest_calls_set_ef_search(self, service, mock_db):
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_db.execute.return_value = result_mock

        await service.find_farthest(
            reference_vector=[0.1] * 384,
            account_id=1,
            k=10,
            mode=SearchMode.BLEED,
        )

        assert mock_db.execute.call_count == 2


# ── Integration: 2-Step Pattern ──────────────────────────────────────

class TestTwoStepPattern:
    """Validates that the 2-step KNN pattern works end-to-end (mocked)."""

    @pytest.mark.asyncio
    async def test_bleed_detector_uses_vector_search(self):
        """BleedDetector should use VectorSearchService, not CROSS JOIN."""
        from app.services.analytics.semantic_engine import BleedDetector

        db = AsyncMock()
        detector = BleedDetector(db)

        # Mock VectorSearchService
        with patch("app.services.analytics.semantic_engine.VectorSearchService") as MockVS:
            mock_vs = AsyncMock()
            MockVS.return_value = mock_vs

            # Product not found → should return empty
            mock_vs.get_product_vector.return_value = None
            result = await detector.detect_bleed("B00MISSING", 1)
            assert result == []
            mock_vs.get_product_vector.assert_called_once_with("B00MISSING", 1)

    @pytest.mark.asyncio
    async def test_opportunity_finder_uses_vector_search(self):
        """OpportunityFinder should use VectorSearchService, not CROSS JOIN."""
        from app.services.analytics.semantic_engine import OpportunityFinder

        db = AsyncMock()
        finder = OpportunityFinder(db)

        with patch("app.services.analytics.semantic_engine.VectorSearchService") as MockVS:
            mock_vs = AsyncMock()
            MockVS.return_value = mock_vs

            mock_vs.get_product_vector.return_value = None
            result = await finder.find_opportunities("B00MISSING", 1)
            assert result == []
            mock_vs.get_product_vector.assert_called_once_with("B00MISSING", 1)

    @pytest.mark.asyncio
    async def test_bleed_detector_full_flow(self):
        """BleedDetector returns results with all expected fields including hnsw_accelerated."""
        from app.services.analytics.semantic_engine import BleedDetector
        import uuid

        db = AsyncMock()
        detector = BleedDetector(db)

        product_id = uuid.uuid4()

        with patch("app.services.analytics.semantic_engine.VectorSearchService") as MockVS:
            mock_vs = AsyncMock()
            MockVS.return_value = mock_vs

            mock_vs.get_product_vector.return_value = {
                "product_embedding_id": product_id,
                "embedding": [0.1] * 384,
                "embedding_version": "v2_cosmo_rich",
                "cosmo_alignment_score": Decimal("0.85"),
            }
            mock_vs.find_farthest.return_value = [
                {
                    "embedding_id": uuid.uuid4(),
                    "term": "completely irrelevant term",
                    "similarity": Decimal("0.12"),
                    "spend": Decimal("15.50"),
                    "clicks": 20,
                    "impressions": 500,
                    "orders": 0,
                    "sales": Decimal("0"),
                    "acos": None,
                    "intent_type": "transactional",
                }
            ]

            results = await detector.detect_bleed(
                "B09XYZ", 42, similarity_threshold=0.40, intent_aware=False
            )

            assert len(results) == 1
            r = results[0]
            assert r["term"] == "completely irrelevant term"
            assert r["semantic_similarity"] == 0.12
            assert r["spend"] == 15.50
            assert r["hnsw_accelerated"] is True
            assert r["embedding_version"] == "v2_cosmo_rich"
            assert r["cosmo_alignment_score"] == 0.85
            assert r["recommendation"] == "ADD_NEGATIVE"
            assert r["urgency"] == "HIGH"  # spend > 10

    @pytest.mark.asyncio
    async def test_opportunity_finder_full_flow(self):
        """OpportunityFinder returns results with all expected fields including hnsw_accelerated."""
        from app.services.analytics.semantic_engine import OpportunityFinder
        import uuid

        db = AsyncMock()
        finder = OpportunityFinder(db)

        with patch("app.services.analytics.semantic_engine.VectorSearchService") as MockVS:
            mock_vs = AsyncMock()
            MockVS.return_value = mock_vs

            mock_vs.get_product_vector.return_value = {
                "product_embedding_id": uuid.uuid4(),
                "embedding": [0.1] * 384,
                "embedding_version": "v2_cosmo_rich",
                "cosmo_alignment_score": Decimal("0.90"),
            }
            mock_vs.find_nearest.return_value = [
                {
                    "embedding_id": uuid.uuid4(),
                    "term": "exact match keyword",
                    "similarity": Decimal("0.92"),
                    "spend": Decimal("5.00"),
                    "clicks": 10,
                    "impressions": 200,
                    "orders": 5,
                    "sales": Decimal("75.00"),
                    "acos": Decimal("6.67"),
                    "intent_type": "transactional",
                }
            ]

            results = await finder.find_opportunities(
                "B09XYZ", 42, similarity_floor=0.70, intent_aware=False
            )

            assert len(results) == 1
            r = results[0]
            assert r["term"] == "exact match keyword"
            assert r["semantic_similarity"] == 0.92
            assert r["hnsw_accelerated"] is True
            assert r["embedding_version"] == "v2_cosmo_rich"
            assert r["suggested_match_type"] == "exact"  # sim > 0.85
            assert r["confidence"] == "HIGH"  # sim > 0.85 and orders >= 3
            assert r["recommendation"] == "ADD_AS_TARGET"

    @pytest.mark.asyncio
    async def test_bleed_detector_respects_limit(self):
        """BleedDetector should stop collecting after hitting the limit."""
        from app.services.analytics.semantic_engine import BleedDetector
        import uuid

        db = AsyncMock()
        detector = BleedDetector(db)

        with patch("app.services.analytics.semantic_engine.VectorSearchService") as MockVS:
            mock_vs = AsyncMock()
            MockVS.return_value = mock_vs

            mock_vs.get_product_vector.return_value = {
                "product_embedding_id": uuid.uuid4(),
                "embedding": [0.1] * 384,
                "embedding_version": "v1_title_only",
                "cosmo_alignment_score": Decimal("0.25"),
            }
            # Return 10 candidates, all below threshold
            mock_vs.find_farthest.return_value = [
                {
                    "embedding_id": uuid.uuid4(),
                    "term": f"bad term {i}",
                    "similarity": Decimal("0.10"),
                    "spend": Decimal("5.00"),
                    "clicks": 3,
                    "impressions": 100,
                    "orders": 0,
                    "sales": Decimal("0"),
                    "acos": None,
                    "intent_type": "transactional",
                }
                for i in range(10)
            ]

            results = await detector.detect_bleed(
                "B09XYZ", 42, limit=3, intent_aware=False
            )
            assert len(results) == 3


# ── Migration SQL Validation ─────────────────────────────────────────

class TestMigrationSQL:
    """Validate migration 08 SQL is structurally correct."""

    @pytest.fixture
    def migration_sql(self):
        with open("migrations/08_hnsw_vector_indexes.sql") as f:
            return f.read()

    def test_creates_ste_hnsw_index(self, migration_sql):
        assert "idx_ste_embedding_hnsw" in migration_sql
        assert "search_term_embeddings" in migration_sql
        assert "hnsw" in migration_sql.lower()

    def test_creates_pe_hnsw_index(self, migration_sql):
        assert "idx_pe_embedding_hnsw" in migration_sql
        assert "product_embeddings" in migration_sql

    def test_uses_cosine_ops(self, migration_sql):
        assert "vector_cosine_ops" in migration_sql

    def test_m_parameter(self, migration_sql):
        assert "m = 16" in migration_sql

    def test_ef_construction_parameter(self, migration_sql):
        assert "ef_construction = 128" in migration_sql

    def test_idempotent_index_creation(self, migration_sql):
        assert "IF NOT EXISTS" in migration_sql

    def test_sets_search_path(self, migration_sql):
        assert "SET search_path = public, extensions" in migration_sql

    def test_sets_maintenance_work_mem(self, migration_sql):
        assert "maintenance_work_mem" in migration_sql

    def test_rewrites_bleed_function(self, migration_sql):
        assert "find_semantic_bleed" in migration_sql
        assert "ORDER BY ste.embedding <=> product_vec DESC" in migration_sql

    def test_rewrites_opportunity_function(self, migration_sql):
        assert "find_semantic_opportunities" in migration_sql
        assert "ORDER BY ste.embedding <=> product_vec ASC" in migration_sql

    def test_resets_maintenance_work_mem(self, migration_sql):
        assert "RESET maintenance_work_mem" in migration_sql
