from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
# Use Unified PPC Model
from app.modules.amazon_ppc.models.ppc_data import PPCCampaign, AIStrategyType

async def seed_campaigns_if_empty(session: AsyncSession) -> bool:
    """Seed campaigns only if none exist. Returns True if seed ran."""
    result = await session.execute(select(PPCCampaign).limit(1))
    if result.scalar_one_or_none() is not None:
        return False

    base_date = datetime.now(timezone.utc)
    campaigns = [
        PPCCampaign(
            campaign_id="camp_001", # Amazon ID
            name="Blue Widgets - Exact",
            campaign_type="sponsoredProducts",
            targeting_type="manual",
            state="enabled", # String in DB, or Enum if mapped. Model uses String column but Enum constraint? Model def says String.
            daily_budget=50.0,
            start_date=base_date,
            ai_mode=AIStrategyType.AUTO_PILOT,
            target_acos=18.0,
            spend=420.0,
            sales=2450.0,
            clicks=100, # Estimated
            impressions=5000,
            # current_acos is calculated property in API, not column here (except via snapshot update)
        ),
        PPCCampaign(
            campaign_id="camp_002",
            name="Widgets - Auto",
            campaign_type="sponsoredProducts",
            targeting_type="auto",
            state="enabled",
            daily_budget=30.0,
            start_date=base_date,
            ai_mode=AIStrategyType.AGGRESSIVE_GROWTH,
            target_acos=25.0,
            spend=180.0,
            sales=620.0,
            clicks=150,
            impressions=7000,
        ),
        PPCCampaign(
            campaign_id="camp_003",
            name="Gadgets Pro - Broad",
            campaign_type="sponsoredProducts",
            targeting_type="manual",
            state="enabled",
            daily_budget=80.0,
            start_date=base_date,
            ai_mode=AIStrategyType.PROFIT_GUARD,
            target_acos=15.0,
            spend=1100.0,
            sales=7200.0,
            clicks=900,
            impressions=40000,
        ),
        PPCCampaign(
            campaign_id="camp_004",
            name="Premium Line - Phrase",
            campaign_type="sponsoredProducts",
            targeting_type="manual",
            state="enabled",
            daily_budget=40.0,
            start_date=base_date,
            ai_mode=AIStrategyType.MANUAL,
            target_acos=20.0,
            spend=320.0,
            sales=1600.0,
            clicks=200,
            impressions=8000,
        ),
        PPCCampaign(
            campaign_id="camp_005",
            name="Summer Collection",
            campaign_type="sponsoredProducts",
            targeting_type="manual",
            state="paused",
            daily_budget=60.0,
            start_date=base_date,
            ai_mode=AIStrategyType.AUTO_PILOT,
            target_acos=22.0,
            spend=90.0,
            sales=380.0,
            clicks=45,
            impressions=1200,
        ),
    ]
    for c in campaigns:
        session.add(c)
    await session.commit()
    return True


async def run_seed():
    """Entry point: open session, seed if empty, close."""
    async with AsyncSessionLocal() as session:
        return await seed_campaigns_if_empty(session)
