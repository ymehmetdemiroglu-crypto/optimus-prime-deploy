
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from app.core.database import Base
from app.modules.amazon_ppc.models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord, AIStrategyType
from app.modules.amazon_ppc.optimization.engine import OptimizationEngine, OptimizationStrategy

# Use factories for easy object creation
from tests.factories import CampaignFactory, FeatureFactory

@pytest.mark.asyncio
async def test_optimization_loop_e2e():
    """
    E2E Test: Full loop from DB Data -> Features -> ML/Rules -> Plan -> Execution -> DB Update.
    Uses in-memory SQLite database.
    """
    # 1. Setup In-Memory DB
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    TestSession = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    
    async with TestSession() as session:
        # 2. Seed Data (High ACoS Scenario)
        # Campaign
        campaign = PPCCampaign(
            campaign_id="CAM_E2E_001",
            name="E2E Test Campaign",
            daily_budget=50.0,
            ai_mode=AIStrategyType.PROFIT_GUARD, # Should reduce bids on high ACoS
            target_acos=20.0,
            state="enabled"
        )
        session.add(campaign)
        await session.flush() # Get ID
        
        # Keyword (High Spend, Low Sales)
        keyword = PPCKeyword(
            keyword_id="KW_E2E_001",
            campaign_id=campaign.id,
            keyword_text="expensive widget",
            match_type="exact",
            state="enabled",
            bid=2.00
        )
        session.add(keyword)
        await session.flush()
        
        # Performance History (Last 30 days)
        # Spend $100, Sales $200 -> ACoS 50% (High Check). Target is 20%.
        # We add a record for yesterday so it falls in 7d/30d windows.
        record = PerformanceRecord(
            campaign_id=campaign.id,
            keyword_id=keyword.id,
            date=datetime.now() - timedelta(days=1),
            impressions=2000,
            clicks=100, # Increased for data maturity > 0.6
            spend=150.0, # ACoS 37.5% (High but < 2x Target of 20%, so Bid Decrease not Pause)
            sales=400.0, # ROAS 2.66
            orders=20
        )
        session.add(record)
        await session.commit()
        
        # 3. Initialize Engine with real session
        engine = OptimizationEngine(session)
        
        # 4. Generate Plan
        # Note: We are NOT mocking ML components here. 
        # BidOptimizer will fall back to Rule-Based because model is not trained/loaded.
        # Rule Based: High ACoS (50%) > Target (20%) -> Should Decrease Bid.
        
        plan = await engine.generate_optimization_plan(
            campaign_id=campaign.id,
            strategy=OptimizationStrategy.PROFIT_FOCUSED,
            target_acos=20.0
        )
        
        # Verify Plan
        assert len(plan.actions) > 0
        action = plan.actions[0]
        assert action.action_type.value == "bid_decrease"
        assert action.current_value == 2.00
        assert action.recommended_value < 2.00
        
        # Approve for execution
        action.approved = True
        
        # 5. Execute Plan
        result = await engine.execute_plan(plan, dry_run=False)
        assert result['summary']['executed'] == 1
        
        # 6. Verify DB Update
        # Need to start new transaction or refresh?
        # execute_optimization_plan commits.
        
        await session.refresh(keyword)
        
        print(f"Old Bid: 2.00, New Bid: {keyword.bid}")
        assert float(keyword.bid) < 2.00
        assert float(keyword.bid) == float(action.recommended_value)

    await test_engine.dispose()
