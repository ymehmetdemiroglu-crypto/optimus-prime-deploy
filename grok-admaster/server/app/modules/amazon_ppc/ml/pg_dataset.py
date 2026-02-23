"""
PostgreSQL-backed PyTorch Dataset for the RL/Bandit training loop.

Problem solved
--------------
Without this class, the main loop alternates between:
  1. Blocking DB query (fetches keyword rows + pgvector embeddings one-by-one)
  2. Model forward/backward pass

This serialises IO and compute, so the GPU (or CPU) sits idle while
the DB query runs and vice-versa.

Solution
--------
``KeywordPGDataset`` wraps a pre-fetched batch of keyword records as a
``torch.utils.data.Dataset``.  A ``DataLoader`` with ``num_workers > 0``
spawns background workers that call ``__getitem__`` concurrently, which
means the *next* batch is being built from raw DB rows while the model
processes the *current* batch.

Batch-fetching strategy
-----------------------
Rather than issuing one ``SELECT`` per keyword (N+1 problem), the
companion async helper ``fetch_keyword_batch`` pulls an entire batch in
a single query and converts the pgvector ``embedding`` column directly
to a pre-allocated NumPy array via ``torch.from_numpy()``, skipping an
intermediate Python-list conversion.

Usage
-----
::

    rows = await fetch_keyword_batch(db, keyword_ids)
    dataset = KeywordPGDataset(rows, feature_keys=FEATURE_KEYS)
    loader  = DataLoader(
        dataset,
        batch_size=64,
        num_workers=2,        # pre-fetch next batch while model runs
        pin_memory=True,      # faster host→GPU transfer
        persistent_workers=True,
    )

    for batch in loader:
        features, embeddings, keyword_ids = batch
        arm_id, multiplier, _ = await ctx_ts.select_arm(keyword_id)
        ...
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Scalar performance features extracted from each DB row
DEFAULT_FEATURE_KEYS: Tuple[str, ...] = (
    "acos_7d",
    "ctr_7d",
    "cvr_7d",
    "spend_velocity_7d",
    "sales_velocity_7d",
    "impression_share",
    "clicks_momentum",
    "orders_momentum",
    "avg_cpc_trend",
    "bid_to_cpc_ratio",
    "keyword_competition_score",
    "match_type_encoded",
)

EMBEDDING_DIM = 384  # sentence-transformers MiniLM-L6 / pgvector column width


class KeywordPGDataset(Dataset):
    """
    In-memory Dataset built from a pre-fetched list of keyword DB rows.

    Each item returns a tuple of:
      * ``features``    – FloatTensor (n_features,)   scalar ad metrics
      * ``embedding``   – FloatTensor (embedding_dim,) pgvector embedding
      * ``keyword_id``  – int

    The dataset is constructed **once per training step** from rows that
    were fetched in a single batched SQL query (see ``fetch_keyword_batch``).
    The ``DataLoader`` then handles shuffling and worker-based pre-fetching.

    Parameters
    ----------
    rows:
        List of dicts, one per keyword, as returned by ``fetch_keyword_batch``.
    feature_keys:
        Ordered list of scalar feature names to extract from each row.
    embedding_key:
        Column name that holds the pgvector embedding (list[float] or
        numpy array of shape (embedding_dim,)).
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
        embedding_key: str = "embedding",
    ) -> None:
        self.keyword_ids: List[int] = []
        feature_list: List[List[float]] = []
        embedding_list: List[np.ndarray] = []

        for row in rows:
            self.keyword_ids.append(int(row.get("keyword_id", row.get("id", 0))))

            # Scalar features — safe float extraction with 0-fill for NULLs
            feat = [float(row.get(k) or 0.0) for k in feature_keys]
            feature_list.append(feat)

            # pgvector embedding — may arrive as list[float] or ndarray
            emb = row.get(embedding_key)
            if emb is None:
                emb_arr = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            elif isinstance(emb, np.ndarray):
                emb_arr = emb.astype(np.float32)
            else:
                emb_arr = np.asarray(emb, dtype=np.float32)
            embedding_list.append(emb_arr)

        # Build contiguous numpy arrays first, then a single torch.from_numpy()
        # call — avoids O(N) individual tensor allocations and the extra copy
        # that torch.tensor(python_list) would perform.
        self._features: torch.Tensor = torch.from_numpy(
            np.array(feature_list, dtype=np.float32)
        )
        self._embeddings: torch.Tensor = torch.from_numpy(
            np.stack(embedding_list, axis=0)  # (N, embedding_dim)
        )

        self.feature_keys = list(feature_keys)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.keyword_ids)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return (
            self._features[idx],
            self._embeddings[idx],
            self.keyword_ids[idx],
        )


# ──────────────────────────────────────────────────────────────────────
#  Async batch-fetch helper
# ──────────────────────────────────────────────────────────────────────

async def fetch_keyword_batch(
    db,  # sqlalchemy AsyncSession
    keyword_ids: List[int],
    extra_columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch a batch of keyword rows + pgvector embeddings in **one query**.

    This replaces the pattern of looping over keyword_ids and issuing N
    individual ``SELECT`` statements.  A single ``WHERE id = ANY(:ids)``
    query has much lower round-trip overhead and allows PostgreSQL to use
    bitmap index scans across all requested rows.

    The embedding column (vector type) is cast to ``float[]`` so asyncpg
    returns it as a Python list that we can convert to numpy efficiently.

    Parameters
    ----------
    db:
        SQLAlchemy ``AsyncSession`` (asyncpg driver).
    keyword_ids:
        List of keyword primary-key IDs to fetch.
    extra_columns:
        Additional scalar column names to include beyond the defaults.
    """
    if not keyword_ids:
        return []

    from sqlalchemy import text

    # Build the column list — always include the pgvector embedding cast
    scalar_cols = list(DEFAULT_FEATURE_KEYS) + (extra_columns or [])
    cols_sql = ", ".join(scalar_cols)

    query = text(
        f"""
        SELECT
            id                AS keyword_id,
            {cols_sql},
            embedding::float[]  AS embedding
        FROM keywords
        WHERE id = ANY(:ids)
        """
    )

    try:
        result = await db.execute(query, {"ids": keyword_ids})
        rows = [dict(row._mapping) for row in result]
    except Exception as exc:
        logger.error("fetch_keyword_batch failed: %s", exc)
        rows = []

    return rows


# ──────────────────────────────────────────────────────────────────────
#  Convenience: build a DataLoader from DB rows
# ──────────────────────────────────────────────────────────────────────

def make_keyword_loader(
    rows: List[Dict[str, Any]],
    batch_size: int = 64,
    num_workers: int = 2,
    shuffle: bool = True,
    feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader from pre-fetched DB rows.

    ``num_workers > 0`` means the DataLoader spawns background processes
    that call ``__getitem__`` concurrently, pre-loading the *next* batch
    of tensors while the model runs on the *current* batch.

    ``pin_memory=True`` pages the output tensors into pinned (page-locked)
    CPU memory so the host→GPU DMA transfer can overlap with compute.
    """
    dataset = KeywordPGDataset(rows, feature_keys=feature_keys)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
