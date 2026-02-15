from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from typing import List

router = APIRouter()

@router.get("/audiences")
async def get_dsp_audiences(db: AsyncSession = Depends(get_db)):
    """Fetch AI-curated DSP audiences based on segment strategy."""
    # Mocking for now, but wired to the DSP route
    return [
        {"id": "seg_001", "name": "Past 30-Day Viewers", "type": "retargeting", "size": "45k", "relevance": 0.98},
        {"id": "seg_002", "name": "Anker Purchaser Lookalike", "type": "competitor_conquest", "size": "120k", "relevance": 0.85},
        {"id": "seg_003", "name": "High-CLTV Lifestyle", "type": "lifestyle", "size": "2.1M", "relevance": 0.72}
    ]

@router.get("/metrics/funnel")
async def get_full_funnel_metrics():
    """Returns the data for the 'Full-Funnel Domination' dashboard."""
    return {
        "top_funnel": {"reach": 1200000, "impressions": 4500000, "cpm": 0.65},
        "mid_funnel": {"views": 85000, "ctr": 0.012},
        "bottom_funnel": {"conversions": 1200, "roas": 4.5, "ntb_percent": 32}
    }
