"""
Rufus Attribution Tracking — data foundation to measure
the influence of Amazon's Rufus AI assistant on PPC performance.

Rufus is Amazon's conversational shopping assistant. When a shopper
queries Rufus and our ASIN appears in the response, this module:

  1. Records the Rufus interaction event with rich context
  2. Links subsequent conversions back to the Rufus touchpoint
  3. Computes multi-channel attribution splits (Rufus / PPC / Organic)
  4. Generates a comparison report showing Rufus-driven ROAS vs. PPC ROAS

The attribution logic mirrors the existing AttributionEngine in
attribution.py but adds a Rufus-specific channel to the credit model.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Numeric, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, text

from app.core.database import Base

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  ORM Models
# ═══════════════════════════════════════════════════════════════

class RufusAttributionEvent(Base):
    """
    Single Rufus AI interaction event.

    Created whenever Amazon's Rufus assistant surfaces one of our
    managed ASINs in response to a shopper query.
    """
    __tablename__ = "rufus_attribution_events"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    profile_id              = Column(String, ForeignKey("profiles.profile_id", ondelete="CASCADE"), nullable=False)
    asin                    = Column(String(50), nullable=False)
    keyword_id              = Column(Integer, ForeignKey("ppc_keywords.id", ondelete="SET NULL"), nullable=True)
    campaign_id             = Column(Integer, ForeignKey("ppc_campaigns.id", ondelete="SET NULL"), nullable=True)

    rufus_query             = Column(String, nullable=False)
    query_intent            = Column(String(50), default="informational")

    rufus_rank              = Column(Integer, nullable=True)
    rufus_confidence        = Column(Numeric(5, 4), nullable=True)
    rufus_intent_probability = Column(Numeric(5, 4), nullable=True)

    context_snapshot        = Column(JSONB, nullable=True)

    attributed_order_id     = Column(String(255), nullable=True)
    attributed_revenue      = Column(Numeric(12, 2), nullable=True)
    converted               = Column(Boolean, default=False)
    conversion_delay_hours  = Column(Numeric(8, 2), nullable=True)

    rufus_credit            = Column(Numeric(5, 4), default=1.0)
    ppc_credit              = Column(Numeric(5, 4), default=0.0)
    organic_credit          = Column(Numeric(5, 4), default=0.0)

    event_at                = Column(DateTime(timezone=True), default=datetime.utcnow)
    created_at              = Column(DateTime(timezone=True), default=datetime.utcnow)


class RufusChannelComparison(Base):
    """Daily Rufus vs. PPC vs. Organic attribution snapshot per ASIN."""
    __tablename__ = "rufus_channel_comparison"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    profile_id              = Column(String, ForeignKey("profiles.profile_id", ondelete="CASCADE"), nullable=False)
    asin                    = Column(String(50), nullable=False)
    date                    = Column(DateTime(timezone=True), nullable=False)

    rufus_impressions       = Column(Integer, default=0)
    rufus_conversions       = Column(Integer, default=0)
    rufus_revenue           = Column(Numeric(12, 2), default=0)
    rufus_conversion_rate   = Column(Numeric(6, 4), default=0)
    avg_rufus_rank          = Column(Numeric(5, 2), nullable=True)
    avg_conversion_delay_h  = Column(Numeric(8, 2), nullable=True)

    ppc_clicks              = Column(Integer, default=0)
    ppc_conversions         = Column(Integer, default=0)
    ppc_revenue             = Column(Numeric(12, 2), default=0)
    ppc_spend               = Column(Numeric(12, 2), default=0)

    organic_sessions        = Column(Integer, default=0)
    organic_conversions     = Column(Integer, default=0)
    organic_revenue         = Column(Numeric(12, 2), default=0)

    total_attributed_revenue = Column(Numeric(12, 2), default=0)
    rufus_revenue_share     = Column(Numeric(5, 4), default=0)
    rufus_roas              = Column(Numeric(8, 4), nullable=True)
    rufus_incrementality    = Column(Numeric(5, 4), nullable=True)

    computed_at             = Column(DateTime(timezone=True), default=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════
#  Attribution Helpers
# ═══════════════════════════════════════════════════════════════

def _classify_intent(query: str) -> str:
    """
    Classify a Rufus shopper query into an intent bucket.

    Returns one of:
        'transactional'  – ready-to-buy signals
        'comparison'     – evaluating options
        'informational'  – researching / learning
        'navigational'   – looking for a specific product/brand
    """
    q = query.lower()
    words = set(q.split())

    transactional_signals = {"buy", "purchase", "order", "get", "add", "cart", "checkout", "deal", "sale"}
    comparison_signals    = {"vs", "versus", "compare", "difference", "better", "best", "which", "recommend"}
    navigational_signals  = {"asin", "brand", "model", "product", "specific", "exact"}

    if words & transactional_signals:
        return "transactional"
    if words & comparison_signals:
        return "comparison"
    if words & navigational_signals:
        return "navigational"
    return "informational"


def _estimate_rufus_intent_prob(query: str, rank: Optional[int]) -> float:
    """
    Heuristic estimate of the probability that a Rufus interaction
    leads to a purchase.

    Higher rank position → lower probability.
    Transactional / comparison intent → higher probability.
    """
    intent = _classify_intent(query)
    base = {"transactional": 0.45, "comparison": 0.28, "informational": 0.12, "navigational": 0.35}
    prob = base.get(intent, 0.15)

    # Decay by rank: #1 → multiplier=1.0, #3 → 0.7, #5 → 0.5
    if rank and rank > 1:
        prob *= max(0.3, 1.0 - (rank - 1) * 0.15)

    return round(min(prob, 1.0), 4)


def _calculate_credit_split(
    has_ppc_click: bool,
    ppc_click_hours_before: Optional[float],
    rufus_hours_before: Optional[float],
) -> Dict[str, float]:
    """
    Calculate Rufus / PPC / Organic credit fractions.

    Rules:
      - If only Rufus touchpoint: rufus=1.0, ppc=0.0, organic=0.0
      - If Rufus + PPC both present: position-based split
        (closer touch gets more credit; Rufus within 24h → +20% boost)
      - If neither (organic): organic=1.0
    """
    if not has_ppc_click:
        return {"rufus": 1.0, "ppc": 0.0, "organic": 0.0}

    # Both Rufus and PPC contributed
    # Time-decay based on proximity to conversion
    rufus_h  = rufus_hours_before  or 72.0
    ppc_h    = ppc_click_hours_before or 72.0

    # Exponential decay: credit ∝ exp(−λ * hours)
    decay = 0.04
    import math
    rufus_w = math.exp(-decay * rufus_h)
    ppc_w   = math.exp(-decay * ppc_h)
    total   = rufus_w + ppc_w

    if total < 1e-9:
        return {"rufus": 0.5, "ppc": 0.5, "organic": 0.0}

    return {
        "rufus":   round(rufus_w / total, 4),
        "ppc":     round(ppc_w   / total, 4),
        "organic": 0.0,
    }


# ═══════════════════════════════════════════════════════════════
#  Service
# ═══════════════════════════════════════════════════════════════

class RufusAttributionService:
    """
    Service layer for Rufus attribution tracking.

    Typical flow
    ------------
    1.  record_event()        — on each Rufus impression for our ASIN
    2.  attribute_conversion()— when Amazon reports a downstream sale
    3.  build_daily_snapshot()— nightly aggregation job
    4.  get_attribution_report() — API / dashboard query
    """

    # Conversion attribution window: if an order arrives within this
    # many hours of the Rufus event, we link them.
    ATTRIBUTION_WINDOW_HOURS: int = 336   # 14 days (Amazon's default)

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── 1. Record Event ───────────────────────────────────────────────

    async def record_event(
        self,
        profile_id: str,
        asin: str,
        rufus_query: str,
        rufus_rank: Optional[int] = None,
        keyword_id: Optional[int] = None,
        campaign_id: Optional[int] = None,
        rufus_confidence: Optional[float] = None,
        context_snapshot: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record a Rufus impression event.

        Returns the new event ID.
        """
        intent = _classify_intent(rufus_query)
        intent_prob = _estimate_rufus_intent_prob(rufus_query, rufus_rank)

        event = RufusAttributionEvent(
            profile_id=profile_id,
            asin=asin,
            keyword_id=keyword_id,
            campaign_id=campaign_id,
            rufus_query=rufus_query,
            query_intent=intent,
            rufus_rank=rufus_rank,
            rufus_confidence=rufus_confidence,
            rufus_intent_probability=intent_prob,
            context_snapshot=context_snapshot,
            converted=False,
            rufus_credit=1.0,
            ppc_credit=0.0,
            organic_credit=0.0,
            event_at=datetime.utcnow(),
        )
        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)
        logger.info(
            f"[Rufus] Recorded event id={event.id} asin={asin} "
            f"intent={intent} rank={rufus_rank}"
        )
        return event.id

    # ── 2. Attribute Conversion ───────────────────────────────────────

    async def attribute_conversion(
        self,
        profile_id: str,
        asin: str,
        order_id: str,
        revenue: float,
        converted_at: Optional[datetime] = None,
        ppc_click_hours_before: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Link an Amazon order to the most recent Rufus event for the ASIN.

        Finds the latest unattributed Rufus event within the attribution
        window, calculates the credit split, and persists the outcome.

        Returns attribution summary dict.
        """
        converted_at = converted_at or datetime.utcnow()
        window_start = converted_at - timedelta(hours=self.ATTRIBUTION_WINDOW_HOURS)

        # Find the most recent eligible Rufus event for this ASIN
        result = await self.db.execute(
            select(RufusAttributionEvent)
            .where(
                RufusAttributionEvent.profile_id == profile_id,
                RufusAttributionEvent.asin == asin,
                RufusAttributionEvent.converted == False,       # noqa: E712
                RufusAttributionEvent.event_at >= window_start,
                RufusAttributionEvent.event_at <= converted_at,
            )
            .order_by(RufusAttributionEvent.event_at.desc())
            .limit(1)
        )
        event = result.scalars().first()

        if event is None:
            logger.debug(
                f"[Rufus] No eligible event for asin={asin} order={order_id}"
            )
            return {"attributed": False, "reason": "no_matching_event"}

        rufus_hours_before = (
            converted_at - event.event_at
        ).total_seconds() / 3600

        has_ppc = ppc_click_hours_before is not None
        credits = _calculate_credit_split(
            has_ppc_click=has_ppc,
            ppc_click_hours_before=ppc_click_hours_before,
            rufus_hours_before=rufus_hours_before,
        )

        attributed_revenue = round(revenue * credits["rufus"], 2)

        # Persist outcome
        stmt = (
            update(RufusAttributionEvent)
            .where(RufusAttributionEvent.id == event.id)
            .values(
                converted=True,
                attributed_order_id=order_id,
                attributed_revenue=attributed_revenue,
                conversion_delay_hours=round(rufus_hours_before, 2),
                rufus_credit=credits["rufus"],
                ppc_credit=credits["ppc"],
                organic_credit=credits["organic"],
            )
        )
        await self.db.execute(stmt)
        await self.db.commit()

        logger.info(
            f"[Rufus] Attributed order={order_id} asin={asin} "
            f"rev={attributed_revenue:.2f} rufus_credit={credits['rufus']:.2%}"
        )
        return {
            "attributed": True,
            "event_id": event.id,
            "order_id": order_id,
            "total_revenue": revenue,
            "attributed_revenue": attributed_revenue,
            "credit_split": credits,
            "conversion_delay_hours": round(rufus_hours_before, 2),
        }

    # ── 3. Build Daily Snapshot ───────────────────────────────────────

    async def build_daily_snapshot(
        self,
        profile_id: str,
        date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate Rufus events for a given day and upsert into
        rufus_channel_comparison.

        Should be called nightly by the scheduler.
        Returns the list of snapshots written.
        """
        target_date = (date or datetime.utcnow()).date()
        day_start = datetime(target_date.year, target_date.month, target_date.day)
        day_end   = day_start + timedelta(days=1)

        # Aggregate events for the day
        rows = (
            await self.db.execute(
                text("""
                    SELECT
                        asin,
                        COUNT(*)                              AS impressions,
                        COUNT(*) FILTER (WHERE converted)    AS conversions,
                        COALESCE(SUM(attributed_revenue) FILTER (WHERE converted), 0)
                                                              AS revenue,
                        AVG(rufus_rank)                       AS avg_rank,
                        AVG(conversion_delay_hours) FILTER (WHERE converted)
                                                              AS avg_delay
                    FROM rufus_attribution_events
                    WHERE profile_id = :pid
                      AND event_at >= :start
                      AND event_at <  :end
                    GROUP BY asin
                """),
                {"pid": profile_id, "start": day_start, "end": day_end},
            )
        ).mappings().all()

        snapshots: List[Dict[str, Any]] = []
        for row in rows:
            asin        = row["asin"]
            impressions = int(row["impressions"] or 0)
            conversions = int(row["conversions"] or 0)
            revenue     = float(row["revenue"]   or 0)
            cvr         = conversions / impressions if impressions > 0 else 0.0

            # Fetch PPC metrics from performance_records for comparison
            ppc_row = (
                await self.db.execute(
                    text("""
                        SELECT
                            COALESCE(SUM(clicks),  0) AS clicks,
                            COALESCE(SUM(orders),  0) AS orders,
                            COALESCE(SUM(sales),   0) AS sales,
                            COALESCE(SUM(spend),   0) AS spend
                        FROM performance_records pr
                        JOIN ppc_campaigns c ON pr.campaign_id = c.id
                        WHERE c.profile_id = :pid
                          AND pr.date >= :start
                          AND pr.date <  :end
                    """),
                    {"pid": profile_id, "start": day_start, "end": day_end},
                )
            ).mappings().first()

            ppc_clicks = int(ppc_row["clicks"] or 0) if ppc_row else 0
            ppc_orders = int(ppc_row["orders"] or 0) if ppc_row else 0
            ppc_sales  = float(ppc_row["sales"] or 0) if ppc_row else 0.0
            ppc_spend  = float(ppc_row["spend"] or 0) if ppc_row else 0.0

            total_rev = revenue + ppc_sales
            rufus_share = revenue / total_rev if total_rev > 0 else 0.0
            rufus_roas  = revenue / ppc_spend if ppc_spend > 0 else None

            # Upsert snapshot
            await self.db.execute(
                text("""
                    INSERT INTO rufus_channel_comparison (
                        profile_id, asin, date,
                        rufus_impressions, rufus_conversions, rufus_revenue,
                        rufus_conversion_rate, avg_rufus_rank, avg_conversion_delay_h,
                        ppc_clicks, ppc_conversions, ppc_revenue, ppc_spend,
                        total_attributed_revenue, rufus_revenue_share, rufus_roas,
                        computed_at
                    ) VALUES (
                        :pid, :asin, :date,
                        :rufus_imps, :rufus_conv, :rufus_rev,
                        :rufus_cvr, :avg_rank, :avg_delay,
                        :ppc_clicks, :ppc_conv, :ppc_rev, :ppc_spend,
                        :total_rev, :rufus_share, :rufus_roas,
                        NOW()
                    )
                    ON CONFLICT (profile_id, asin, date) DO UPDATE SET
                        rufus_impressions     = EXCLUDED.rufus_impressions,
                        rufus_conversions     = EXCLUDED.rufus_conversions,
                        rufus_revenue         = EXCLUDED.rufus_revenue,
                        rufus_conversion_rate = EXCLUDED.rufus_conversion_rate,
                        avg_rufus_rank        = EXCLUDED.avg_rufus_rank,
                        avg_conversion_delay_h = EXCLUDED.avg_conversion_delay_h,
                        ppc_clicks            = EXCLUDED.ppc_clicks,
                        ppc_conversions       = EXCLUDED.ppc_conversions,
                        ppc_revenue           = EXCLUDED.ppc_revenue,
                        ppc_spend             = EXCLUDED.ppc_spend,
                        total_attributed_revenue = EXCLUDED.total_attributed_revenue,
                        rufus_revenue_share   = EXCLUDED.rufus_revenue_share,
                        rufus_roas            = EXCLUDED.rufus_roas,
                        computed_at           = NOW()
                """),
                {
                    "pid": profile_id, "asin": asin,
                    "date": day_start,
                    "rufus_imps":  impressions, "rufus_conv":  conversions,
                    "rufus_rev":   revenue,     "rufus_cvr":   round(cvr, 4),
                    "avg_rank":    row["avg_rank"], "avg_delay": row["avg_delay"],
                    "ppc_clicks":  ppc_clicks,  "ppc_conv":   ppc_orders,
                    "ppc_rev":     ppc_sales,   "ppc_spend":  ppc_spend,
                    "total_rev":   round(total_rev, 2),
                    "rufus_share": round(rufus_share, 4),
                    "rufus_roas":  round(rufus_roas, 4) if rufus_roas is not None else None,
                },
            )

            snapshots.append({
                "asin": asin,
                "date": str(target_date),
                "rufus_impressions": impressions,
                "rufus_conversions": conversions,
                "rufus_revenue": revenue,
                "rufus_conversion_rate": round(cvr, 4),
                "rufus_revenue_share": round(rufus_share, 4),
                "rufus_roas": round(rufus_roas, 4) if rufus_roas is not None else None,
            })

        await self.db.commit()
        logger.info(
            f"[Rufus] Built daily snapshot for {profile_id} / {target_date}: "
            f"{len(snapshots)} ASINs"
        )
        return snapshots

    # ── 4. Reports ────────────────────────────────────────────────────

    async def get_attribution_report(
        self,
        profile_id: str,
        days: int = 30,
        asin: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return a summary attribution report for the profile.

        Includes:
          - Top converting Rufus queries
          - Rufus vs. PPC revenue comparison
          - Intent breakdown
          - Average conversion delay
        """
        since = datetime.utcnow() - timedelta(days=days)

        filters = "profile_id = :pid AND event_at >= :since"
        params: Dict[str, Any] = {"pid": profile_id, "since": since}
        if asin:
            filters += " AND asin = :asin"
            params["asin"] = asin

        # Overall stats
        overall = (
            await self.db.execute(
                text(f"""
                    SELECT
                        COUNT(*)                              AS total_events,
                        COUNT(*) FILTER (WHERE converted)    AS total_conversions,
                        COALESCE(SUM(attributed_revenue) FILTER (WHERE converted), 0)
                                                              AS total_revenue,
                        AVG(rufus_intent_probability)         AS avg_intent_prob,
                        AVG(conversion_delay_hours) FILTER (WHERE converted)
                                                              AS avg_delay_hours
                    FROM rufus_attribution_events
                    WHERE {filters}
                """),
                params,
            )
        ).mappings().first()

        # Intent breakdown
        intent_rows = (
            await self.db.execute(
                text(f"""
                    SELECT
                        query_intent,
                        COUNT(*)                            AS events,
                        COUNT(*) FILTER (WHERE converted)  AS conversions,
                        COALESCE(SUM(attributed_revenue) FILTER (WHERE converted), 0)
                                                            AS revenue
                    FROM rufus_attribution_events
                    WHERE {filters}
                    GROUP BY query_intent
                    ORDER BY revenue DESC
                """),
                params,
            )
        ).mappings().all()

        # Top queries by attributed revenue
        top_queries = (
            await self.db.execute(
                text(f"""
                    SELECT
                        rufus_query,
                        COUNT(*) FILTER (WHERE converted)  AS conversions,
                        COALESCE(SUM(attributed_revenue) FILTER (WHERE converted), 0)
                                                            AS revenue,
                        AVG(rufus_intent_probability)       AS avg_intent_prob
                    FROM rufus_attribution_events
                    WHERE {filters}
                    GROUP BY rufus_query
                    HAVING COUNT(*) FILTER (WHERE converted) > 0
                    ORDER BY revenue DESC
                    LIMIT 10
                """),
                params,
            )
        ).mappings().all()

        total_events       = int(overall["total_events"] or 0)
        total_conversions  = int(overall["total_conversions"] or 0)
        total_revenue      = float(overall["total_revenue"] or 0)
        overall_cvr        = total_conversions / total_events if total_events > 0 else 0.0

        return {
            "profile_id": profile_id,
            "period_days": days,
            "asin_filter": asin,
            "summary": {
                "total_events":      total_events,
                "total_conversions": total_conversions,
                "conversion_rate":   round(overall_cvr, 4),
                "total_revenue":     total_revenue,
                "avg_intent_prob":   round(float(overall["avg_intent_prob"] or 0), 4),
                "avg_delay_hours":   round(float(overall["avg_delay_hours"] or 0), 2),
            },
            "intent_breakdown": [
                {
                    "intent":       r["query_intent"],
                    "events":       int(r["events"]),
                    "conversions":  int(r["conversions"]),
                    "revenue":      float(r["revenue"]),
                }
                for r in intent_rows
            ],
            "top_queries": [
                {
                    "query":          r["rufus_query"],
                    "conversions":    int(r["conversions"]),
                    "revenue":        float(r["revenue"]),
                    "avg_intent_prob": round(float(r["avg_intent_prob"] or 0), 4),
                }
                for r in top_queries
            ],
        }

    async def get_channel_comparison(
        self,
        profile_id: str,
        days: int = 30,
        asin: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return Rufus vs. PPC vs. Organic attribution comparison.
        Pulls from the pre-computed rufus_channel_comparison snapshots.
        """
        since = datetime.utcnow() - timedelta(days=days)

        filters = "profile_id = :pid AND date >= :since"
        params: Dict[str, Any] = {"pid": profile_id, "since": since}
        if asin:
            filters += " AND asin = :asin"
            params["asin"] = asin

        agg = (
            await self.db.execute(
                text(f"""
                    SELECT
                        SUM(rufus_impressions)        AS rufus_imps,
                        SUM(rufus_conversions)        AS rufus_conv,
                        SUM(rufus_revenue)            AS rufus_rev,
                        SUM(ppc_clicks)               AS ppc_clicks,
                        SUM(ppc_conversions)          AS ppc_conv,
                        SUM(ppc_revenue)              AS ppc_rev,
                        SUM(ppc_spend)                AS ppc_spend,
                        SUM(total_attributed_revenue) AS total_rev,
                        AVG(rufus_roas)               AS avg_rufus_roas
                    FROM rufus_channel_comparison
                    WHERE {filters}
                """),
                params,
            )
        ).mappings().first()

        rufus_rev = float(agg["rufus_rev"] or 0)
        ppc_rev   = float(agg["ppc_rev"]   or 0)
        ppc_spend = float(agg["ppc_spend"] or 0)
        total_rev = float(agg["total_rev"] or rufus_rev + ppc_rev)

        ppc_roas = ppc_rev / ppc_spend if ppc_spend > 0 else None

        return {
            "profile_id": profile_id,
            "period_days": days,
            "rufus": {
                "impressions":   int(agg["rufus_imps"] or 0),
                "conversions":   int(agg["rufus_conv"] or 0),
                "revenue":       rufus_rev,
                "revenue_share": round(rufus_rev / total_rev, 4) if total_rev > 0 else 0.0,
                "avg_roas":      round(float(agg["avg_rufus_roas"] or 0), 4),
            },
            "ppc": {
                "clicks":        int(agg["ppc_clicks"] or 0),
                "conversions":   int(agg["ppc_conv"] or 0),
                "revenue":       ppc_rev,
                "spend":         ppc_spend,
                "roas":          round(ppc_roas, 4) if ppc_roas else None,
                "revenue_share": round(ppc_rev / total_rev, 4) if total_rev > 0 else 0.0,
            },
            "total_attributed_revenue": total_rev,
            "rufus_halo_roas": (
                round(rufus_rev / ppc_spend, 4) if ppc_spend > 0 else None
            ),
        }
