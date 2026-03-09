"""
Paid Ads Module — API Router

Exposes the multi-platform paid advertising strategy engine as REST endpoints.
"""

from fastapi import APIRouter, HTTPException
import logging

from .schemas import (
    PlatformRecommendationRequest, PlatformRecommendationResponse,
    CampaignStructureRequest, CampaignStructureResponse,
    AdCopyRequest, AdCopyResponse,
    BudgetAllocationRequest, BudgetAllocationResponse,
    RetargetingStrategyRequest, RetargetingStrategyResponse,
    CampaignOptimizationRequest, CampaignOptimizationResponse,
    SetupChecklistRequest, SetupChecklistResponse,
)
from .service import (
    recommend_platforms,
    generate_campaign_structure,
    generate_ad_copy,
    allocate_budget,
    build_retargeting_strategy,
    diagnose_campaign,
    get_setup_checklist,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ──────────────────────────────────────────────────────────
#  1. Platform Recommendation
# ──────────────────────────────────────────────────────────

@router.post(
    "/recommend-platforms",
    response_model=PlatformRecommendationResponse,
    summary="Recommend ad platforms",
    description="Score Google, Meta, LinkedIn, TikTok, and Amazon against your product profile.",
)
async def recommend_platforms_endpoint(req: PlatformRecommendationRequest):
    try:
        return recommend_platforms(req)
    except Exception as e:
        logger.error(f"Platform recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────
#  2. Campaign Structure
# ──────────────────────────────────────────────────────────

@router.post(
    "/campaign-structure",
    response_model=CampaignStructureResponse,
    summary="Generate campaign structure",
    description="Returns a full campaign → ad-set → ad hierarchy with naming conventions, budgets, and bid strategies.",
)
async def campaign_structure_endpoint(req: CampaignStructureRequest):
    try:
        return generate_campaign_structure(req)
    except Exception as e:
        logger.error(f"Campaign structure generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────
#  3. Ad Copy Generation
# ──────────────────────────────────────────────────────────

@router.post(
    "/ad-copy",
    response_model=AdCopyResponse,
    summary="Generate ad copy",
    description="Produce headline + primary text + CTA variations using PAS, BAB, Social Proof, or AIDA frameworks.",
)
async def ad_copy_endpoint(req: AdCopyRequest):
    try:
        return generate_ad_copy(req)
    except Exception as e:
        logger.error(f"Ad copy generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────
#  4. Budget Allocation
# ──────────────────────────────────────────────────────────

@router.post(
    "/budget-allocation",
    response_model=BudgetAllocationResponse,
    summary="Allocate budget across platforms",
    description="Split your budget across selected platforms using objective-weighted heuristics.",
)
async def budget_allocation_endpoint(req: BudgetAllocationRequest):
    try:
        return allocate_budget(req)
    except Exception as e:
        logger.error(f"Budget allocation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────
#  5. Retargeting Strategy
# ──────────────────────────────────────────────────────────

@router.post(
    "/retargeting-strategy",
    response_model=RetargetingStrategyResponse,
    summary="Build retargeting strategy",
    description="Generate a 3-layer funnel retargeting plan (TOF / MOF / BOF) with audiences, windows, and frequency caps.",
)
async def retargeting_strategy_endpoint(req: RetargetingStrategyRequest):
    try:
        return build_retargeting_strategy(req)
    except Exception as e:
        logger.error(f"Retargeting strategy generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────
#  6. Campaign Optimisation
# ──────────────────────────────────────────────────────────

@router.post(
    "/diagnose",
    response_model=CampaignOptimizationResponse,
    summary="Diagnose campaign performance",
    description="Analyse CPA, CTR, CPM, ROAS against targets and return prioritised fix actions.",
)
async def diagnose_campaign_endpoint(req: CampaignOptimizationRequest):
    try:
        return diagnose_campaign(req)
    except Exception as e:
        logger.error(f"Campaign diagnosis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────
#  7. Setup Checklist
# ──────────────────────────────────────────────────────────

@router.post(
    "/setup-checklist",
    response_model=SetupChecklistResponse,
    summary="Get platform setup checklist",
    description="Return a pre-launch checklist for the selected ad platform.",
)
async def setup_checklist_endpoint(req: SetupChecklistRequest):
    try:
        return get_setup_checklist(req)
    except Exception as e:
        logger.error(f"Setup checklist generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
