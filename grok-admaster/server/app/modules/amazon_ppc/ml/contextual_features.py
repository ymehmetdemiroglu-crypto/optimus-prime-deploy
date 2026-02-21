"""
Contextual Feature Extraction for Contextual Thompson Sampling.

Extracts rich feature vectors from keyword, campaign, and market context
to condition bandit decisions. Features include:
  - Temporal (hour, day, seasonality, paydays)
  - Performance trends (rolling ACoS, CTR, CVR, momentum)
  - Market signals (competitor price changes, avg CPC trend)
  - Keyword meta (match type, impression share)
"""
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
import logging
import json
import math

logger = logging.getLogger(__name__)

# ─── Feature definition ──────────────────────────────────────────────
CONTEXT_FEATURE_NAMES: List[str] = [
    # Temporal (8 features)
    "hour_sin", "hour_cos",           # Cyclical hour encoding
    "dow_sin", "dow_cos",             # Cyclical day-of-week encoding
    "is_weekend",                     # 0/1
    "day_of_month_norm",              # 0-1 (payday proxy)
    "month_sin", "month_cos",         # Cyclical month for seasonality

    # Performance trends (8 features)
    "acos_7d",                        # 7-day rolling ACoS
    "ctr_7d",                         # 7-day rolling CTR
    "cvr_7d",                         # 7-day rolling conversion rate
    "spend_velocity_7d",              # Spend trend slope
    "sales_velocity_7d",              # Sales trend slope
    "impression_share",               # Est. impression share
    "clicks_momentum",                # 3d vs 7d click ratio
    "orders_momentum",                # 3d vs 7d order ratio

    # Market context (4 features)
    "competitor_price_change_pct",    # Competitor price movement
    "avg_cpc_trend",                  # CPC trend (rising / falling)
    "bid_to_cpc_ratio",              # Current bid / avg CPC
    "keyword_competition_score",      # 0-1 competition density

    # Keyword meta (7 features)
    "match_type_encoded",             # exact=1, phrase=0.5, broad=0
    "log_impressions_total",          # log(1+total impressions)
    "log_spend_total",                # log(1+total spend)
    "rufus_intent_probability",       # Probability of conversational intent
    "query_length_norm",              # Normalized word count
    "question_tf_idf_weight",         # TF-IDF style question word weight
    "intent_confidence_score",        # Confidence of intent classification
]

CONTEXT_DIM = len(CONTEXT_FEATURE_NAMES)  # 23


class ContextFeatureExtractor:
    """
    Builds a fixed-length numeric context vector for a keyword at a point
    in time.  All features are normalised to roughly [0,1] or [-1,1] so
    that Bayesian linear regression posteriors converge quickly.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── public API ────────────────────────────────────────────────────
    async def extract(
        self,
        keyword_id: int,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """
        Return a 1-D numpy array of shape (CONTEXT_DIM,) for the given
        keyword at *timestamp* (defaults to now).
        """
        ts = timestamp or datetime.utcnow()

        # Gather raw signals in parallel-ish (sequential for safety
        # with a single session)
        temporal = self._temporal_features(ts)
        perf = await self._performance_features(keyword_id, ts)
        market = await self._market_features(keyword_id)
        meta = await self._keyword_meta_features(keyword_id)

        vec = np.concatenate([temporal, perf, market, meta])
        assert vec.shape == (CONTEXT_DIM,), (
            f"Expected {CONTEXT_DIM} features, got {vec.shape[0]}"
        )
        return vec

    async def extract_dict(
        self,
        keyword_id: int,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Same as extract() but returns a labelled dictionary."""
        vec = await self.extract(keyword_id, timestamp)
        return dict(zip(CONTEXT_FEATURE_NAMES, vec.tolist()))

    # ── temporal features (no DB needed) ──────────────────────────────
    @staticmethod
    def _temporal_features(ts: datetime) -> np.ndarray:
        hour = ts.hour
        dow = ts.weekday()        # Mon=0 … Sun=6
        dom = ts.day              # 1-31
        month = ts.month          # 1-12

        return np.array([
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            1.0 if dow >= 5 else 0.0,
            dom / 31.0,
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
        ], dtype=np.float64)

    # ── performance features (DB) ─────────────────────────────────────
    async def _performance_features(
        self, keyword_id: int, ts: datetime
    ) -> np.ndarray:
        """7-day rolling metrics from performance_records."""
        defaults = np.zeros(8, dtype=np.float64)

        try:
            result = await self.db.execute(text("""
                SELECT
                    COALESCE(SUM(impressions), 0)  AS imps,
                    COALESCE(SUM(clicks), 0)       AS clicks,
                    COALESCE(SUM(spend), 0)        AS spend,
                    COALESCE(SUM(sales), 0)        AS sales,
                    COALESCE(SUM(orders), 0)       AS orders
                FROM performance_records
                WHERE keyword_id = :kid
                  AND date >= :start
                  AND date <  :end
            """), {"kid": keyword_id, "start": ts - timedelta(days=7), "end": ts})
            row7 = result.mappings().first()

            # 3-day window for momentum
            result3 = await self.db.execute(text("""
                SELECT
                    COALESCE(SUM(clicks), 0)  AS clicks,
                    COALESCE(SUM(orders), 0)  AS orders
                FROM performance_records
                WHERE keyword_id = :kid
                  AND date >= :start
                  AND date <  :end
            """), {"kid": keyword_id, "start": ts - timedelta(days=3), "end": ts})
            row3 = result3.mappings().first()

            if row7 is None:
                return defaults

            imps7 = float(row7["imps"])
            clicks7 = float(row7["clicks"])
            spend7 = float(row7["spend"])
            sales7 = float(row7["sales"])
            orders7 = float(row7["orders"])

            acos_7d = spend7 / sales7 if sales7 > 0 else 1.0
            ctr_7d = clicks7 / imps7 if imps7 > 0 else 0.0
            cvr_7d = orders7 / clicks7 if clicks7 > 0 else 0.0

            # Velocity = simple slope proxy  (spend per day)
            spend_vel = spend7 / 7.0
            sales_vel = sales7 / 7.0

            # Impression share estimate (capped)
            imp_share = min(imps7 / max(imps7 * 1.5, 1), 1.0)

            # Momentum: ratio of 3d to 7d (scaled to daily)
            clicks3 = float(row3["clicks"]) if row3 else 0
            orders3 = float(row3["orders"]) if row3 else 0
            clicks_mom = (clicks3 / 3) / (clicks7 / 7 + 1e-9)
            orders_mom = (orders3 / 3) / (orders7 / 7 + 1e-9)

            return np.array([
                min(acos_7d, 2.0),        # cap extreme ACoS
                min(ctr_7d, 0.5),         # cap extreme CTR
                min(cvr_7d, 0.5),
                min(spend_vel / 100, 2.0),  # normalise by $100/day
                min(sales_vel / 500, 2.0),  # normalise by $500/day
                imp_share,
                min(clicks_mom, 3.0),      # cap momentum
                min(orders_mom, 3.0),
            ], dtype=np.float64)

        except Exception as e:
            logger.warning(f"Error extracting performance features for kw {keyword_id}: {e}")
            return defaults

    # ── market context (DB) ───────────────────────────────────────────
    async def _market_features(self, keyword_id: int) -> np.ndarray:
        """Competitor / market signals."""
        defaults = np.zeros(4, dtype=np.float64)

        try:
            # Latest competitor price change %
            price_result = await self.db.execute(text("""
                SELECT COALESCE(AVG(change_pct), 0) AS avg_change
                FROM price_changes
                WHERE detected_at >= NOW() - INTERVAL '7 days'
                LIMIT 1
            """))
            price_row = price_result.mappings().first()
            comp_price_change = float(price_row["avg_change"]) / 100 if price_row else 0.0

            # Avg CPC trend from keyword market volumes
            cpc_result = await self.db.execute(text("""
                SELECT COALESCE(AVG(cpc), 0) AS avg_cpc
                FROM market_keyword_volumes
                WHERE recorded_at >= NOW() - INTERVAL '7 days'
                LIMIT 1
            """))
            cpc_row = cpc_result.mappings().first()
            avg_cpc = float(cpc_row["avg_cpc"]) if cpc_row else 0.5

            # Bid-to-CPC ratio
            bid_result = await self.db.execute(text("""
                SELECT COALESCE(bid, 0.5) AS bid
                FROM ppc_keywords WHERE id = :kid
            """), {"kid": keyword_id})
            bid_row = bid_result.mappings().first()
            current_bid = float(bid_row["bid"]) if bid_row else 0.5

            bid_to_cpc = current_bid / (avg_cpc + 1e-9)

            # Competition score (from market_keyword_volumes)
            comp_result = await self.db.execute(text("""
                SELECT COALESCE(AVG(competition), 0.5) AS comp
                FROM market_keyword_volumes
                WHERE recorded_at >= NOW() - INTERVAL '30 days'
                LIMIT 1
            """))
            comp_row = comp_result.mappings().first()
            competition = float(comp_row["comp"]) if comp_row else 0.5

            return np.array([
                max(-1.0, min(1.0, comp_price_change)),
                max(-1.0, min(1.0, (avg_cpc - 0.5) / 2)),  # centre around $0.50
                min(bid_to_cpc, 5.0) / 5.0,                 # normalise
                min(competition, 1.0),
            ], dtype=np.float64)

        except Exception as e:
            logger.warning(f"Error extracting market features for kw {keyword_id}: {e}")
            return defaults

    # ── keyword meta features (DB) ────────────────────────────────────
    async def _keyword_meta_features(self, keyword_id: int) -> np.ndarray:
        """Static keyword properties."""
        defaults = np.zeros(7, dtype=np.float64)

        try:
            result = await self.db.execute(text("""
                SELECT keyword_text, match_type, impressions, spend
                FROM ppc_keywords
                WHERE id = :kid
            """), {"kid": keyword_id})
            row = result.mappings().first()

            if row is None:
                return defaults

            # Encode match type
            mt = str(row["match_type"]).upper()
            mt_map = {"EXACT": 1.0, "PHRASE": 0.5, "BROAD": 0.0}
            mt_encoded = mt_map.get(mt, 0.25)

            imps = float(row["impressions"] or 0)
            spend = float(row["spend"] or 0)

            kw_text = str(row.get("keyword_text") or "").lower()
            words = kw_text.split()
            word_count = len(words)
            
            # Query length norm (assuming typical transactional is 2-3, max around 10)
            query_length_norm = min(word_count / 10.0, 1.0)
            
            # Question TF-IDF weight proxy
            question_words = {'how', 'what', 'where', 'why', 'which', 'best', 'compare', 'vs', 'difference'}
            q_count = sum(1 for w in words if w in question_words)
            question_tf = (q_count / word_count) if word_count > 0 else 0.0
            question_tf_idf_weight = min(question_tf * 2.0, 1.0)
            
            # Rufus intent probability
            rufus_intent_prob = min((query_length_norm * 0.4) + (question_tf_idf_weight * 0.8), 1.0)
            
            # Intent confidence score proxy
            intent_confidence = 0.5 + min(abs(rufus_intent_prob - 0.5) * 2, 0.5)

            return np.array([
                mt_encoded,
                math.log1p(imps) / 15.0,   # normalise (log scale)
                math.log1p(spend) / 10.0,
                rufus_intent_prob,
                query_length_norm,
                question_tf_idf_weight,
                intent_confidence
            ], dtype=np.float64)

        except Exception as e:
            logger.warning(f"Error extracting keyword meta for kw {keyword_id}: {e}")
            return defaults


def context_to_json(vec: np.ndarray) -> str:
    """Serialise context vector to JSON for storage in DB column."""
    return json.dumps(dict(zip(CONTEXT_FEATURE_NAMES, vec.tolist())))


def context_from_json(json_str: str) -> np.ndarray:
    """Deserialise context vector from JSON string."""
    d = json.loads(json_str)
    return np.array([d.get(f, 0.0) for f in CONTEXT_FEATURE_NAMES])
