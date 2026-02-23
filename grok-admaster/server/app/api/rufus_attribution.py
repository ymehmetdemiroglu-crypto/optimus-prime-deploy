"""
Rufus Attribution Tracking API

REST endpoints to record Amazon Rufus AI assistant interactions,
attribute conversions, and compare channel performance.

Routes
------
POST /api/v1/rufus/events                 – Record a Rufus impression
POST /api/v1/rufus/events/{id}/convert    – Attribute a conversion to an event
POST /api/v1/rufus/snapshots/{profile_id} – Build/refresh daily snapshots
GET  /api/v1/rufus/report/{profile_id}    – Full attribution report
GET  /api/v1/rufus/channels/{profile_id}  – Rufus vs PPC vs Organic comparison
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.amazon_ppc.ml.rufus_attribution import RufusAttributionService

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
#  Request / Response models
# ═══════════════════════════════════════════════════════════════

class RufusEventRequest(BaseModel):
    profile_id:        str
    asin:              str
    rufus_query:       str
    rufus_rank:        Optional[int]   = Field(None, ge=1, le=20)
    keyword_id:        Optional[int]   = None
    campaign_id:       Optional[int]   = None
    rufus_confidence:  Optional[float] = Field(None, ge=0.0, le=1.0)
    context_snapshot:  Optional[Dict[str, Any]] = None


class ConversionRequest(BaseModel):
    order_id:                 str
    revenue:                  float = Field(..., gt=0)
    converted_at:             Optional[datetime] = None
    ppc_click_hours_before:   Optional[float]   = Field(None, ge=0)


class SnapshotRequest(BaseModel):
    date: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════

@router.post("/events", summary="Record a Rufus AI impression event")
async def record_rufus_event(
    request: RufusEventRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Record a single Rufus impression — i.e. when Amazon's Rufus
    AI assistant surfaces our ASIN in response to a shopper query.

    This is the entry-point for the attribution data pipeline.
    Call this endpoint whenever your Rufus integration detects an
    impression for a managed ASIN.
    """
    svc = RufusAttributionService(db)
    event_id = await svc.record_event(
        profile_id=request.profile_id,
        asin=request.asin,
        rufus_query=request.rufus_query,
        rufus_rank=request.rufus_rank,
        keyword_id=request.keyword_id,
        campaign_id=request.campaign_id,
        rufus_confidence=request.rufus_confidence,
        context_snapshot=request.context_snapshot,
    )
    return {"event_id": event_id, "status": "recorded"}


@router.post(
    "/events/{event_id}/convert",
    summary="Attribute a conversion to a Rufus event"
)
async def attribute_conversion(
    event_id: int,
    request: ConversionRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Not used directly — prefer the profile-level /events/{asin}/convert
    which auto-selects the best matching event.

    This endpoint links a specific Rufus event to an Amazon order.
    """
    from sqlalchemy import select, update
    from app.modules.amazon_ppc.ml.rufus_attribution import (
        RufusAttributionEvent, _calculate_credit_split
    )

    result = await db.execute(
        select(RufusAttributionEvent).where(RufusAttributionEvent.id == event_id)
    )
    event = result.scalars().first()
    if not event:
        raise HTTPException(status_code=404, detail="Rufus event not found")

    if event.converted:
        raise HTTPException(
            status_code=409,
            detail=f"Event {event_id} is already attributed to order "
                   f"{event.attributed_order_id}"
        )

    converted_at = request.converted_at or datetime.utcnow()
    rufus_hours  = (converted_at - event.event_at).total_seconds() / 3600
    has_ppc      = request.ppc_click_hours_before is not None
    credits      = _calculate_credit_split(
        has_ppc_click=has_ppc,
        ppc_click_hours_before=request.ppc_click_hours_before,
        rufus_hours_before=rufus_hours,
    )
    attributed_revenue = round(request.revenue * credits["rufus"], 2)

    await db.execute(
        update(RufusAttributionEvent)
        .where(RufusAttributionEvent.id == event_id)
        .values(
            converted=True,
            attributed_order_id=request.order_id,
            attributed_revenue=attributed_revenue,
            conversion_delay_hours=round(rufus_hours, 2),
            rufus_credit=credits["rufus"],
            ppc_credit=credits["ppc"],
            organic_credit=credits["organic"],
        )
    )
    await db.commit()

    return {
        "event_id":            event_id,
        "order_id":            request.order_id,
        "total_revenue":       request.revenue,
        "attributed_revenue":  attributed_revenue,
        "credit_split":        credits,
        "conversion_delay_h":  round(rufus_hours, 2),
    }


@router.post(
    "/attribute/{profile_id}/{asin}",
    summary="Auto-attribute a conversion for an ASIN"
)
async def auto_attribute_conversion(
    profile_id: str,
    asin: str,
    request: ConversionRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Auto-select the most recent eligible Rufus event for this ASIN
    and attribute the order to it.

    This is the preferred endpoint for bulk attribution pipelines.
    """
    svc = RufusAttributionService(db)
    result = await svc.attribute_conversion(
        profile_id=profile_id,
        asin=asin,
        order_id=request.order_id,
        revenue=request.revenue,
        converted_at=request.converted_at,
        ppc_click_hours_before=request.ppc_click_hours_before,
    )
    return result


@router.post(
    "/snapshots/{profile_id}",
    summary="Build / refresh daily Rufus channel snapshot"
)
async def build_snapshot(
    profile_id: str,
    request: SnapshotRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Aggregate all Rufus events for the given date and upsert
    the per-ASIN channel comparison snapshot.

    Should be called nightly by the scheduler.
    Defaults to *today* if no date is supplied.
    """
    svc = RufusAttributionService(db)
    snapshots = await svc.build_daily_snapshot(
        profile_id=profile_id,
        date=request.date,
    )
    return {
        "profile_id":    profile_id,
        "date":          str((request.date or datetime.utcnow()).date()),
        "asins_updated": len(snapshots),
        "snapshots":     snapshots,
    }


@router.get(
    "/report/{profile_id}",
    summary="Full Rufus attribution report"
)
async def get_attribution_report(
    profile_id: str,
    days: int = Query(30, ge=1, le=365),
    asin: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns a comprehensive Rufus attribution report:

    - Summary metrics (events, conversions, revenue, CVR)
    - Intent breakdown (transactional / comparison / informational)
    - Top-10 converting Rufus queries
    - Average conversion delay

    Filter by `asin` for product-level attribution.
    """
    svc = RufusAttributionService(db)
    return await svc.get_attribution_report(
        profile_id=profile_id,
        days=days,
        asin=asin,
    )


@router.get(
    "/channels/{profile_id}",
    summary="Rufus vs. PPC vs. Organic channel comparison"
)
async def get_channel_comparison(
    profile_id: str,
    days: int = Query(30, ge=1, le=365),
    asin: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns the multi-channel attribution breakdown showing:

    - Rufus-attributed revenue and share
    - PPC-attributed revenue, spend, and ROAS
    - Rufus "halo ROAS" (Rufus revenue / PPC spend)

    Use this to quantify the value Rufus adds on top of paid channels.
    """
    svc = RufusAttributionService(db)
    return await svc.get_channel_comparison(
        profile_id=profile_id,
        days=days,
        asin=asin,
    )
