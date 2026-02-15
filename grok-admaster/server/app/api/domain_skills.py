"""
Domain Skills API Router
Exposes capabilities of Financial Analyst and Campaign Strategist.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, List
from pydantic import BaseModel

from app.modules.amazon_ppc.ingestion.client import AmazonAdsAPIClient
from app.services.domain_skills.financial_analyst.service import FinancialAnalystService
from app.services.domain_skills.campaign_strategist.service import CampaignStrategistService

# Dependency to get API Client (Stub - replace with actual auth/client factory)
# In a real app, this would use the current user's active profile credentials
async def get_api_client():
    # Placeholder: fetch credentials from DB or Environment
    return AmazonAdsAPIClient(
        client_id="stub_id",
        client_secret="stub_secret",
        refresh_token="stub_refresh"
    )

async def get_financial_service(client: AmazonAdsAPIClient = Depends(get_api_client)):
    return FinancialAnalystService(client)

async def get_strategist_service(client: AmazonAdsAPIClient = Depends(get_api_client)):
    return CampaignStrategistService(client)

router = APIRouter()

# --- Request Models ---

class ProfitabilityRequest(BaseModel):
    profile_id: str
    asin: str
    price: float
    cogs: float

class BudgetOptimizationRequest(BaseModel):
    profile_id: str
    total_budget: float

class LaunchPlanRequest(BaseModel):
    product_name: str
    asin: str
    launch_date: str
    total_budget: float
    aggressiveness: str = "balanced"

class DesignStructureRequest(BaseModel):
    product_name: str
    asin: str

# --- Endpoints ---

@router.post("/financial/profitability")
async def calculate_profitability(
    req: ProfitabilityRequest,
    service: FinancialAnalystService = Depends(get_financial_service)
):
    """
    Calculate True Profitability for a product using live Ad Data.
    """
    try:
        return await service.get_product_profitability(
            req.profile_id, req.asin, req.price, req.cogs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/financial/budget-optimization")
async def optimize_budget(
    req: BudgetOptimizationRequest,
    service: FinancialAnalystService = Depends(get_financial_service)
):
    """
    Generate optimal budget allocation based on marginal ROAS.
    """
    try:
        return await service.optimize_budget(req.profile_id, req.total_budget)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategist/launch-plan")
async def create_launch_plan(
    req: LaunchPlanRequest,
    service: CampaignStrategistService = Depends(get_strategist_service)
):
    """
    Generate a 60-day product launch roadmap.
    """
    return await service.generate_launch_plan(
        req.product_name, req.asin, req.launch_date, req.total_budget, req.aggressiveness
    )

@router.get("/strategist/audit/{profile_id}")
async def audit_account(
    profile_id: str,
    service: CampaignStrategistService = Depends(get_strategist_service)
):
    """
    Audit campaign structure for best practices.
    """
    try:
        return await service.audit_account_structure(profile_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategist/structure-design")
async def design_structure(
    req: DesignStructureRequest,
    service: CampaignStrategistService = Depends(get_strategist_service)
):
    """
    Design ideal camping structure for a product.
    """
    return await service.design_campaigns_for_product(req.product_name, req.asin)

@router.get("/audit/full-report/{profile_id}")
async def get_full_audit_report(
    profile_id: str,
    strat_service: CampaignStrategistService = Depends(get_strategist_service),
    fin_service: FinancialAnalystService = Depends(get_financial_service)
):
    """
    Generate a comprehensive Audit Report combining:
    1. Financial Health (Wasted Spend, Profitability)
    2. Structural Health (Naming, Campaign Types)
    """
    try:
        # Parallel execution for speed
        import asyncio
        structure_task = asyncio.create_task(strat_service.audit_account_structure(profile_id))
        financial_task = asyncio.create_task(fin_service.analyze_account(profile_id))
        
        structure_data = await structure_task
        financial_data = await financial_task
        
        # Calculate Overall Grade (Weighted)
        # Financial Health (60%) + Structure (40%)
        # Financial Score: ACoS deviation from ideal (30%) + Wasted Spend ratio
        
        fin_score = 100
        if financial_data['acos'] > 40: fin_score -= 20
        if financial_data['acos'] > 60: fin_score -= 20
        
        waste_ratio = financial_data['wasted_spend'] / financial_data['total_spend'] if financial_data['total_spend'] > 0 else 0
        if waste_ratio > 0.1: fin_score -= 10
        if waste_ratio > 0.3: fin_score -= 20
        
        struct_score = structure_data.get('score', 80)
        
        overall_score = int((fin_score * 0.6) + (struct_score * 0.4))
        
        # Determine Grade
        if overall_score >= 90: grade = "A"
        elif overall_score >= 80: grade = "B"
        elif overall_score >= 70: grade = "C"
        elif overall_score >= 60: grade = "D"
        else: grade = "F"
        
        return {
            "accountName": "Amazon Account", # Placeholder until Profile API fetch
            "auditDate": "Today",
            "overallScore": overall_score,
            "grade": grade,
            "totalWaste": financial_data['wasted_spend'],
            "potentialSavings": financial_data['potential_savings'],
            "metrics": [
                {"label": "Monthly Ad Spend", "value": f"${financial_data['total_spend']}", "trend": "neutral"},
                {"label": "Current ACoS", "value": f"{financial_data['acos']}%", "trend": "down" if financial_data['acos'] < 30 else "up", "severity": "success" if financial_data['acos'] < 30 else "warning"},
                {"label": "Wasted Spend", "value": f"${financial_data['wasted_spend']}", "severity": "critical" if financial_data['wasted_spend'] > 0 else "success"},
                {"label": "Profit Opportunity", "value": f"${financial_data['potential_savings']}", "severity": "success"}
            ],
            # Simplified Leakage buckets for demo
            "leakage": [
                {"category": "Zero-Sale Keywords", "amount": financial_data['wasted_spend'], "percentage": int(waste_ratio * 100), "icon": "🎯"},
                {"category": "Inefficient Campaigns", "amount": financial_data['potential_savings'] - financial_data['wasted_spend'], "percentage": 100 - int(waste_ratio * 100), "icon": "📉"}
            ],
            "issues": [
                {"id": str(i), "title": issue['type'], "description": issue['msg'], "severity": "warning", "potentialSavings": 0, "category": "Structure"} 
                for i, issue in enumerate(structure_data.get('issues', []))
            ],
            "structureBreakdown": {
                "targeting": fin_score, # Proxy
                "structure": struct_score,
                "profitability": fin_score,
                "efficiency": fin_score
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
