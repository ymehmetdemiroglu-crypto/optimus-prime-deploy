from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
# Use Consolidated Model
from app.modules.amazon_ppc.models.ppc_data import PPCCampaign
from app.models.schemas import Campaign as SchemaCampaign, CampaignStrategyUpdate, CampaignStatus

router = APIRouter()

def map_campaign_to_schema(c: PPCCampaign) -> SchemaCampaign:
    """Helper to map DB model to Pydantic schema."""
    # Calculate ACoS
    spend = float(c.spend or 0)
    sales = float(c.sales or 0)
    acos = (spend / sales * 100) if sales > 0 else 0.0
    
    # Map State -> Status
    # Schema expects 'active', 'paused', 'archived'
    # DB has 'enabled', 'paused', 'archived' typically
    status_map = {
        "enabled": CampaignStatus.ACTIVE,
        "active": CampaignStatus.ACTIVE,
        "paused": CampaignStatus.PAUSED,
        "archived": CampaignStatus.ARCHIVED
    }
    status = status_map.get(str(c.state).lower(), CampaignStatus.PAUSED)
    
    return SchemaCampaign(
        id=str(c.id), # Schema expects str ID? Check schema.
        name=c.name or "Unknown",
        status=status,
        ai_mode=c.ai_mode, # Now compatible due to updated Enum
        daily_budget=float(c.daily_budget or 0),
        spend=spend,
        sales=sales,
        acos=round(acos, 2)
    )

@router.get("", response_model=List[SchemaCampaign])
async def get_campaigns(db: AsyncSession = Depends(get_db)):
    """List all campaigns from the unified ppc_campaigns table."""
    result = await db.execute(select(PPCCampaign))
    campaigns = result.scalars().all()
    
    return [map_campaign_to_schema(c) for c in campaigns]

@router.get("/{campaign_id}", response_model=SchemaCampaign)
async def get_campaign(campaign_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific campaign by ID."""
    result = await db.execute(select(PPCCampaign).where(PPCCampaign.id == campaign_id))
    campaign = result.scalar_one_or_none()
    
    if not campaign:
        # Try looking up by Amazon Campaign ID (string) if int lookup fails?
        # The route param is int, so we assume internal ID. 
        raise HTTPException(status_code=404, detail="Campaign not found")
        
    return map_campaign_to_schema(campaign)

@router.patch("/{campaign_id}/strategy", response_model=SchemaCampaign)
async def update_campaign_strategy(
    campaign_id: int, 
    update: CampaignStrategyUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update the AI strategy for a specific campaign."""
    result = await db.execute(select(PPCCampaign).where(PPCCampaign.id == campaign_id))
    campaign = result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Update AI Mode
    campaign.ai_mode = update.ai_mode
    
    # Reset target ACoS to default if moving to managed mode?
    # For now, just set mode.
    
    await db.commit()
    await db.refresh(campaign)
    
    return map_campaign_to_schema(campaign)


class AdvancedCampaign(BaseModel):
    id: int
    name: str
    status: str
    campaign_type: str
    targeting_type: str
    spend: float
    sales: float
    acos: float
    roas: float
    trend_percentage: float
    sparkline: List[float]  # Daily sales for last 7 days

@router.get("/advanced", response_model=List[AdvancedCampaign])
async def get_advanced_campaigns(db: AsyncSession = Depends(get_db)):
    """
    Returns enriched campaign data with 7-day performance history for the Advanced Manager.
    """
    # Fetch all enabled campaigns
    result = await db.execute(select(PPCCampaign).where(PPCCampaign.state != "archived"))
    campaigns = result.scalars().all()
    
    advanced_data = []
    
    # Date range for history (last 7 days)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    from app.modules.amazon_ppc.models.ppc_data import PerformanceRecord
    
    # Ideally we'd batch fetch performance records, but for prototype we iterate
    # Or fetch all relevant records in one go
    perf_result = await db.execute(
        select(PerformanceRecord)
        .where(PerformanceRecord.date >= start_date)
        .order_by(PerformanceRecord.date)
    )
    all_recs = perf_result.scalars().all()
    
    # Group records by campaign_id
    recs_by_campaign = {}
    for r in all_recs:
        cid = r.campaign_id
        if cid not in recs_by_campaign:
            recs_by_campaign[cid] = []
        recs_by_campaign[cid].append(r)
        
    for c in campaigns:
        recs = recs_by_campaign.get(c.id, [])
        
        # Calculate Sparkline (Daily Sales)
        # Ensure we have 7 points, fill missing days with 0
        sparkline = [0.0] * 7
        total_sales_7d = 0.0
        total_spend_7d = 0.0
        
        for r in recs:
            day_idx = (r.date.date() - start_date.date()).days
            if 0 <= day_idx < 7:
                sales = float(r.sales or 0)
                sparkline[day_idx] += sales
                total_sales_7d += sales
                total_spend_7d += float(r.spend or 0)
                
        # Calculate Metrics
        spend = float(c.spend or 0)
        sales = float(c.sales or 0)
        acos = (spend / sales * 100) if sales > 0 else 0.0
        roas = (sales / spend) if spend > 0 else 0.0
        
        # Trend (Simple comparison of last 3 days vs previous 3 days of sparkline)
        recent_3 = sum(sparkline[4:])
        prev_3 = sum(sparkline[1:4])
        if prev_3 > 0:
            trend_pct = ((recent_3 - prev_3) / prev_3) * 100
        else:
            trend_pct = 100.0 if recent_3 > 0 else 0.0
            
        advanced_data.append(AdvancedCampaign(
            id=c.id,
            name=c.name or "Unknown",
            status=str(c.state).lower(),
            campaign_type=c.campaign_type or "SP",
            targeting_type=c.targeting_type or "MANUAL",
            spend=spend,
            sales=sales,
            acos=round(acos, 2),
            roas=round(roas, 2),
            trend_percentage=round(trend_pct, 1),
            sparkline=sparkline
        ))
        
    return advanced_data
