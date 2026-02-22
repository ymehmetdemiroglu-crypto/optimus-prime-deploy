"""
Vector Search Service — HNSW-Accelerated Nearest Neighbor Queries

Provides configurable KNN search over pgvector HNSW indexes with
per-query ef_search tuning for accuracy/speed tradeoffs.

HNSW Index Parameters (set in migration 08):
  m = 16            — graph connectivity (standard for 384-dim)
  ef_construction = 128  — build-time quality

Query-Time Parameter:
  ef_search — controls the size of the search beam at query time.
  Higher = more accurate (better recall) but slower.
  Must be >= k (the number of results requested).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("vector_search")


# ── HNSW Configuration Constants ─────────────────────────────────────

# Index build parameters (must match migration 08)
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 128
EMBEDDING_DIM = 384

# Valid ranges for HNSW parameters (pgvector constraints)
HNSW_M_RANGE = (2, 100)
HNSW_EF_CONSTRUCTION_RANGE = (4, 1000)
HNSW_EF_SEARCH_RANGE = (1, 1000)


class SearchMode(str, Enum):
    """Query modes with different accuracy/speed profiles."""
    BLEED = "bleed"              # High recall — must catch all waste
    OPPORTUNITY = "opportunity"  # Balanced — speed matters more
    PATROL = "patrol"            # Background scan — can sacrifice some recall


# Default ef_search per mode
_EF_SEARCH_DEFAULTS: Dict[SearchMode, int] = {
    SearchMode.BLEED: 200,        # High recall to catch bleed
    SearchMode.OPPORTUNITY: 100,  # Balanced for opportunity scanning
    SearchMode.PATROL: 80,        # Background — speed preferred
}


@dataclass(frozen=True)
class HNSWConfig:
    """Immutable HNSW configuration snapshot."""
    m: int = HNSW_M
    ef_construction: int = HNSW_EF_CONSTRUCTION
    embedding_dim: int = EMBEDDING_DIM

    def validate(self) -> bool:
        """Check that parameters are within pgvector's valid ranges."""
        return (
            HNSW_M_RANGE[0] <= self.m <= HNSW_M_RANGE[1]
            and HNSW_EF_CONSTRUCTION_RANGE[0] <= self.ef_construction <= HNSW_EF_CONSTRUCTION_RANGE[1]
            and self.embedding_dim > 0
        )


def get_ef_search(mode: SearchMode, override: Optional[int] = None) -> int:
    """
    Resolve the ef_search value for a given search mode.

    Args:
        mode: The search mode (bleed, opportunity, patrol).
        override: Optional caller-specified override.

    Returns:
        ef_search value clamped to valid range.
    """
    value = override if override is not None else _EF_SEARCH_DEFAULTS[mode]
    return max(HNSW_EF_SEARCH_RANGE[0], min(value, HNSW_EF_SEARCH_RANGE[1]))


class VectorSearchService:
    """
    HNSW-accelerated vector search over pgvector tables.

    Uses a 2-step pattern:
      1. Fetch the reference vector (e.g. product embedding)
      2. ORDER BY embedding <=> reference_vec LIMIT k  (HNSW-accelerated)

    Callers set `ef_search` per-query to tune recall vs speed.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def _set_ef_search(self, ef_search: int) -> None:
        """Set the HNSW ef_search parameter for the current transaction."""
        clamped = max(HNSW_EF_SEARCH_RANGE[0], min(ef_search, HNSW_EF_SEARCH_RANGE[1]))
        await self.db.execute(text(f"SET LOCAL hnsw.ef_search = {clamped}"))

    async def get_product_vector(
        self, asin: str, account_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch the product embedding vector and metadata.

        Returns dict with keys: embedding, embedding_version,
        cosmo_alignment_score, product_embedding_id — or None if not found.
        """
        result = await self.db.execute(
            text("""
                SELECT id, embedding, embedding_version, cosmo_alignment_score
                FROM product_embeddings
                WHERE asin = :asin AND account_id = :account_id
                LIMIT 1
            """),
            {"asin": asin, "account_id": account_id},
        )
        row = result.fetchone()
        if row is None:
            return None
        return {
            "product_embedding_id": row.id,
            "embedding": row.embedding,
            "embedding_version": row.embedding_version,
            "cosmo_alignment_score": row.cosmo_alignment_score,
        }

    async def find_nearest(
        self,
        reference_vector: Any,
        account_id: int,
        k: int = 100,
        *,
        mode: SearchMode = SearchMode.OPPORTUNITY,
        ef_search_override: Optional[int] = None,
        min_orders: int = 0,
        min_spend: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find the k nearest search term embeddings to a reference vector.

        Uses ORDER BY embedding <=> vector ASC LIMIT k which is
        accelerated by the HNSW index.

        Returns rows ordered by ascending distance (nearest first).
        """
        ef = get_ef_search(mode, ef_search_override)
        await self._set_ef_search(ef)

        query = text("""
            SELECT
                ste.id AS embedding_id,
                ste.term,
                ROUND((1 - (ste.embedding <=> :ref_vec::vector))::NUMERIC, 4) AS similarity,
                ste.spend,
                ste.clicks,
                ste.impressions,
                ste.orders,
                ste.sales,
                ste.acos,
                ste.intent_type
            FROM search_term_embeddings ste
            WHERE ste.account_id = :account_id
              AND ste.orders >= :min_orders
              AND ste.spend >= :min_spend
            ORDER BY ste.embedding <=> :ref_vec::vector ASC
            LIMIT :k
        """)

        result = await self.db.execute(query, {
            "ref_vec": str(reference_vector),
            "account_id": account_id,
            "min_orders": min_orders,
            "min_spend": min_spend,
            "k": k,
        })
        return [dict(row._mapping) for row in result.fetchall()]

    async def find_farthest(
        self,
        reference_vector: Any,
        account_id: int,
        k: int = 200,
        *,
        mode: SearchMode = SearchMode.BLEED,
        ef_search_override: Optional[int] = None,
        min_spend: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Find the k farthest search term embeddings from a reference vector.

        Uses ORDER BY embedding <=> vector DESC LIMIT k.
        Note: pgvector HNSW indexes accelerate ASC (nearest) queries
        natively.  For DESC (farthest) queries the planner may fall back
        to a sequential scan, but the index on other WHERE columns
        (account_id) still helps.  At typical scale (<50k terms per
        account) this is performant.

        Returns rows ordered by descending distance (farthest first).
        """
        ef = get_ef_search(mode, ef_search_override)
        await self._set_ef_search(ef)

        query = text("""
            SELECT
                ste.id AS embedding_id,
                ste.term,
                ROUND((1 - (ste.embedding <=> :ref_vec::vector))::NUMERIC, 4) AS similarity,
                ste.spend,
                ste.clicks,
                ste.impressions,
                ste.orders,
                ste.sales,
                ste.acos,
                ste.intent_type
            FROM search_term_embeddings ste
            WHERE ste.account_id = :account_id
              AND ste.spend >= :min_spend
            ORDER BY ste.embedding <=> :ref_vec::vector DESC
            LIMIT :k
        """)

        result = await self.db.execute(query, {
            "ref_vec": str(reference_vector),
            "account_id": account_id,
            "min_spend": min_spend,
            "k": k,
        })
        return [dict(row._mapping) for row in result.fetchall()]


# ── Module-level convenience ─────────────────────────────────────────

def get_hnsw_config() -> HNSWConfig:
    """Return the current HNSW configuration (matches migration 08)."""
    return HNSWConfig()
