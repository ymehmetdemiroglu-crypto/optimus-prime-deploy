"""
Dashboard API endpoints with mock data. AI actions are driven by PPC optimizer.
"""
from fastapi import APIRouter, Query, Depends
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import random
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.amazon_ppc.models.ppc_data import PPCCampaign
from app.models.schemas import DashboardSummary, PerformanceMetric, TrendDirection, AIAction
from app.modules.amazon_ppc.optimization.engine import OptimizationEngine, OptimizationStrategy

router = APIRouter()


@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary():
    """Returns high-level KPI cards for the War Room."""
    return DashboardSummary(
        total_sales=15423.67,
        ad_spend=2347.82,
        acos=15.2,
        roas=6.57,
        velocity_trend=TrendDirection.UP
    )


@router.get("/chart-data", response_model=List[PerformanceMetric])
async def get_chart_data(time_range: str = Query("7d", pattern="^(7d|30d|ytd)$", alias="range")):
    """Returns time-series data for the main graph."""
    days = 7 if time_range == "7d" else 30 if time_range == "30d" else 90
    data = []
    
    base_date = datetime.now()
    for i in range(days, 0, -1):
        date = base_date - timedelta(days=i)
        # Generate realistic mock data with some variance
        base_organic = 1200 + (i * 15)
        base_ad = 800 + (i * 10)
        
        data.append(PerformanceMetric(
            timestamp=date.strftime("%Y-%m-%d"),
            organic_sales=round(base_organic + random.uniform(-100, 150), 2),
            ad_sales=round(base_ad + random.uniform(-50, 100), 2),
            spend=round((base_ad * 0.15) + random.uniform(-10, 20), 2),
            impressions=int(50000 + random.randint(-5000, 10000))
        ))
    
    return data


@router.get("/ai-actions", response_model=List[AIAction])
async def get_ai_actions(db: AsyncSession = Depends(get_db)):
    """Returns AI actions from PPC optimizer for active campaigns."""
    result = await db.execute(select(PPCCampaign).where(PPCCampaign.state == "enabled"))
    campaigns = result.scalars().all()
    
    engine = OptimizationEngine(db)
    all_actions = []
    
    # Strategy Mapping
    strategy_map = {
        "auto_pilot": OptimizationStrategy.BALANCED,
        "aggressive_growth": OptimizationStrategy.AGGRESSIVE,
        "profit_guard": OptimizationStrategy.PROFIT_FOCUSED,
        # Default fallbacks
        "manual": OptimizationStrategy.BALANCED, 
        "advanced": OptimizationStrategy.BALANCED,
        "autonomous": OptimizationStrategy.BALANCED
    }
    
    for c in campaigns:
        # Skip truly manual mode if we strictly follow logic, but dashboard usually shows "what if" insights
        # For now, let's generate insights for everyone except explicit Manual if desired, 
        # but legacy showed insights for all.
        
        mode_str = str(c.ai_mode.value if hasattr(c.ai_mode, "value") else c.ai_mode).lower()
        if mode_str == "manual": 
            continue # Prior logic only showed "insights" for manual in specific cases, mostly skip for feed clarity
            
        strategy = strategy_map.get(mode_str, OptimizationStrategy.BALANCED)
        
        # Generate Plan
        try:
            plan = await engine.generate_optimization_plan(
                campaign_id=c.id,
                strategy=strategy,
                target_acos=float(c.target_acos or 30),
                target_roas=float(c.target_roas or 3.0)
            )
            
            for action in plan.actions:
                # Map OptimizationAction -> AIAction
                all_actions.append(AIAction(
                    id=f"{c.id}_{action.entity_id}_{action.action_type}",
                    action_type=action.action_type.value,
                    description=action.reasoning,
                    timestamp=action.execution_time or datetime.now(), # Plan actions might not have time yet
                    campaign_name=c.name
                ))
        except Exception as e:
            print(f"Error generating plan for {c.id}: {e}")
            continue

    return all_actions


class ClientMatrixNode(BaseModel):
    id: str
    name: str
    logo: str
    status: str  # healthy, warning, critical
    sales: float
    spend: float
    acos: float
    acos_trend: float
    apis: List[str]

@router.get("/matrix", response_model=List[ClientMatrixNode])
async def get_client_matrix():
    """Returns data for the Global Client Matrix."""
    return [
        ClientMatrixNode(
            id="NX-4029",
            name="NEXUS CORP",
            logo="https://lh3.googleusercontent.com/aida-public/AB6AXuAINLXFQdyuHrqAAPobHbIEBcebYXIjtGWWfKb8WJEJPkZRF_OMwcQMlzrUDJ5BuH1KaNLBR_Q8Tk1rWnyaPQD8yOc1WWMatkXefrRixGRIMCnGp1kkMH_50apI024Gm0wbWcN4zWmXmjxT9D1GpOj4PxcqsLM5OEaKaPLNP4jYDy-p3ZNTtccT7PfObG1yv4FXapna8vSqN75CjPOZykCSL-PvoQCXvWJGgSF2eJ-ofBT96q-74EjTOUYIESzFTF5kfZU5bT1j-Ksk",
            status="healthy",
            sales=14205.00,
            spend=1890.00,
            acos=13.3,
            acos_trend=-12,
            apis=["ADS API", "SP-API"]
        ),
        ClientMatrixNode(
            id="ST-9921",
            name="STELLAR DYN",
            logo="https://lh3.googleusercontent.com/aida-public/AB6AXuAb0os4q4xIXAkssL2IWJt_5R4lRvEBBAyBvkoNYRrrdZQ9036L4puR-kikazPrs6nTB4OCnTUflb9k7Jh3tJ-QREh_X8yLuzit1hs-leQ_dKE5iwDuQ8WXXdGJkwDQabaCEYwvK6vrf7cYH1TcN4HEA8pwcrOhfAebnCGAja2WQTagBDJb6_W4-y0cKirZX3tNzGeH2Ke1cyU_9-6pUxPIAh9azRiHG_P-w8N5qttvgvsQacO7cQ9dTGVCFJ77KIohAa_ZgCAAfvGw",
            status="warning",
            sales=8100.00,
            spend=2268.00,
            acos=28.0,
            acos_trend=5,
            apis=["ADS API", "SYNCING"]
        ),
        ClientMatrixNode(
            id="CY-800",
            name="CYBERDYNE",
            logo="https://lh3.googleusercontent.com/aida-public/AB6AXuC9UuqevtrWLFpoxwUgOXjJi-u8tBN4zMNwQvj_x2jb-3a7azvL2S7KLLE33ZP9akg_o-bYUx5I32HAkVdUTK80H-yWZpfwlNZU8qfMJOYcOaynTYsaEDGbTW271N8wHh3NiyiWD4elj3NWuZv_o2vdk0kCXk2L1_YIIXIdiFIUKGBYKLq5Fy0jOyJ898XKx8hqm6NBdd2opL5m-uJK-M5Ifo5gzykoz0czAOpeWbfj5lgHzd8681cOmY49JX3zYjRnZakBOgrh7R5_",
            status="critical",
            sales=0.00,
            spend=0.00,
            acos=0.0,
            acos_trend=0,
            apis=["DISCONNECTED"]
        ),
        ClientMatrixNode(
            id="OC-1011",
            name="OMNI CONSUMER",
            logo="https://lh3.googleusercontent.com/aida-public/AB6AXuBJq8aKEc6E63sA5N8U4UV8z0mU4iCGIEeQU4-PQnUwf8UdLYKSgTX3mtfQKrtbfppMTnqOxjdeH1s8ONQcRkip_QR7RAusHIigY3HWxOEMAebzNcPMZWwCvSoIflHAo-BflRFdkzbar7ngKuXgfJAxQVaTf7pB7KvnAdbeEtms0ZIixuy9PE4-Xo4KR7QRf86Xu1oFkbpLofzpF-nXo0GxoxN53BDUp8XtlysWjQ5BCWAEJ564W70khjcPP9DlWXw5XnhRJvnWlAuF",
            status="healthy",
            sales=32450.00,
            spend=5840.00,
            acos=18.0,
            acos_trend=-3,
            apis=["ADS API", "SP-API"]
        ),
        ClientMatrixNode(
            id="WY-2093",
            name="WEYLAND YUT",
            logo="https://lh3.googleusercontent.com/aida-public/AB6AXuDpLOZYgbJih_ky-nOvzMGo6NlaF5pPII_hQyrPzKFCbkCeFfBsfKAktdSX7Awa3Bqzr5CRxAISxLCZStSylV8B4dboAn5cSF5gHgfXUigFV9t52Vrzi5Nef08wxSTgWDi9GGZovDiZcENnXVudnTnYHeXtNertitfOydbAHcT7JMBWW7UGLMkFMTBDeDkwqcHCE8tYPEV8jSsjKiOKZafB3pCpqa3Vb6I4YnxOPv2fbbZKuFMyRo4SbghuNlR2HHKICFS_zpvOD6NV",
            status="healthy",
            sales=56100.00,
            spend=5049.00,
            acos=9.0,
            acos_trend=-1.5,
            apis=["ADS API", "SP-API"]
        )
    ]
