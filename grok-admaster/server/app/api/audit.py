
from fastapi import APIRouter
from typing import List, Dict, Any
from pydantic import BaseModel

router = APIRouter()

# --- Models ---

class AuditScorecard(BaseModel):
    overall_score: int
    health_status: str # Excellent, Good, Fair, Poor
    summary: str
    
    # Key Stats
    active_campaigns: int
    active_campaigns_trend: int
    wasted_spend: float
    wasted_spend_trend: int
    ipi_score: int
    ipi_score_trend: int
    avg_rating: float
    avg_rating_trend: str # Stable, Up, Down

    # Breakdown
    bidding_health: Dict[str, Any]
    keyword_health: Dict[str, Any]
    inventory_health: Dict[str, Any]

# --- Mock Data ---

MOCK_AUDIT = AuditScorecard(
    overall_score=87,
    health_status="Excellent",
    summary="Your account is performing well. Optimization opportunities identified in Bidding.",
    active_campaigns=142,
    active_campaigns_trend=12,
    wasted_spend=1200.0,
    wasted_spend_trend=-8,
    ipi_score=745,
    ipi_score_trend=2,
    avg_rating=4.8,
    avg_rating_trend="Stable",
    bidding_health={
        "score": 72,
        "items": [
            {"title": "ACoS Targets Met", "status": "Fail", "desc": "Campaigns exceed target ACoS by >15%."},
            {"title": "Budget Utilization", "status": "Warning", "desc": "3 campaigns running out of budget before 6PM."},
            {"title": "Bid Optimization", "status": "Pass", "desc": "Bids are updated regularly via automation."}
        ]
    },
    keyword_health={
        "score": 94,
        "items": [
            {"title": "Branded Keyword Share", "status": "Pass", "desc": "Strong dominance on all branded terms."},
            {"title": "Negative Keywords", "status": "Pass", "desc": "Active negative targeting reducing waste."},
            {"title": "Search Term Isolation", "status": "Warning", "desc": "Some high-performing terms not isolated."}
        ]
    },
    inventory_health={
        "score": 88,
        "items": [
            {"title": "IPI Score Health", "status": "Pass", "desc": "Score is above 500 threshold."},
            {"title": "Stranded Inventory", "status": "Warning", "desc": "12 units stranded requiring action."}
        ]
    }
)

# --- Endpoints ---

@router.get("/scorecard", response_model=AuditScorecard)
async def get_audit_scorecard():
    return MOCK_AUDIT
