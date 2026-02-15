
import asyncio
import os
import sys
import random
import logging
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv

# Load .env explicitly
server_dir = os.path.dirname(os.path.abspath(__file__)).replace('scripts', '')
sys.path.append(server_dir)
load_dotenv(os.path.join(server_dir, '.env'))

from app.core.database import engine, AsyncSessionLocal
from app.modules.amazon_ppc.ml.hierarchical_rl import HierarchicalBudgetController
from app.modules.amazon_ppc.models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord
from app.modules.amazon_ppc.accounts.models import Profile # Import to register table for FK
from sqlalchemy import select, text, delete

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RLSimulator")

async def run_simulation(days=30, profile_id="SIM_PROFILE_001"):
    """
    Simulates a 2-week run of the Hierarchical RL Agent.
    
    1. Setup: Creates synthetic campaigns and keywords for a test profile.
    2. Loop (Day 1 to N):
       - Agent allocates budget.
       - Environment simulates market response (Random + Bias).
         * Campaign A is "High ROAS" (hidden ground truth).
         * Campaign B is "Low ROAS".
       - Agent sees yesterday's performance and Updates.
    3. Goal: Agent should shift budget from B to A over time.
    """

    print(f"\nStarting RL Budget Allocation Simulation for {days} days...")
    
    async with AsyncSessionLocal() as db:
        # 0. ENSURE PROFILE EXISTS
        from app.modules.amazon_ppc.accounts.models import Account, Profile
        
        result = await db.execute(select(Profile).where(Profile.profile_id == profile_id))
        existing_profile = result.scalar_one_or_none()
        
        if not existing_profile:
            print("   [Setup] Creating dummy profile...")
            # Check/Create dummy account
            res_acc = await db.execute(select(Account).where(Account.amazon_account_id == "SIM_ACC_001"))
            acc = res_acc.scalar_one_or_none()
            
            if not acc:
                acc = Account(
                    name="Simulation Corp",
                    amazon_account_id="SIM_ACC_001",
                    region="NA",
                    status="active"
                )
                db.add(acc)
                await db.commit()
                await db.refresh(acc)
            
            # Create dummy profile
            prof = Profile(
                profile_id=profile_id,
                account_id=acc.id,
                country_code="US",
                currency_code="USD",
                timezone="UTC",
                account_info_id="SIM_ENTITY"
            )
            db.add(prof)
            await db.commit()
        else:
            print(f"   [Setup] Using existing profile {profile_id}")

        # 1. CLEANUP & SETUP
        print("   [Setup] Cleaning previous simulation data...")
        # Clean RL state first (might reference profile? No, references profile_id)
        await db.execute(text("DELETE FROM rl_portfolio_state WHERE profile_id = :pid"), {"pid": profile_id})
        
        # Clean campaigns and related
        # Performance records
        await db.execute(text("DELETE FROM performance_records WHERE campaign_id IN (SELECT id FROM ppc_campaigns WHERE profile_id = :pid)"), {"pid": profile_id})
        # Keywords
        await db.execute(text("DELETE FROM ppc_keywords WHERE campaign_id IN (SELECT id FROM ppc_campaigns WHERE profile_id = :pid)"), {"pid": profile_id})
        # Campaigns
        await db.execute(text("DELETE FROM ppc_campaigns WHERE profile_id = :pid"), {"pid": profile_id})
        
        await db.commit()

        print("   [Setup] Creating synthetic campaigns...")
        # Campaign 1: High Potential (ROAS ~ 5.0)
        c1 = PPCCampaign(
            campaign_id="SIM_CAMP_A",
            profile_id=profile_id,
            name="Winner Campaign (High ROAS)",
            campaign_type="sponsoredProducts",
            # Let's check model... model says 'state' in ppc_data.py
            state="ENABLED",
            daily_budget=50.0,
            targeting_type="manual"
        )
        # Campaign 2: Low Potential (ROAS ~ 1.5)
        c2 = PPCCampaign(
            campaign_id="SIM_CAMP_B",
            profile_id=profile_id,
            name="Loser Campaign (Low ROAS)",
            campaign_type="sponsoredProducts",
            state="ENABLED",
            daily_budget=50.0,
            targeting_type="manual"
        )
        db.add_all([c1, c2])
        await db.commit()
        await db.refresh(c1)
        await db.refresh(c2)

        # Add some keywords
        k1 = PPCKeyword(keyword_id="K1", campaign_id=c1.id, keyword_text="win keyword", match_type="EXACT", state="ENABLED", bid=1.0)
        k2 = PPCKeyword(keyword_id="K2", campaign_id=c2.id, keyword_text="lose keyword", match_type="EXACT", state="ENABLED", bid=1.0)
        db.add_all([k1, k2])
        await db.commit()
        await db.refresh(k1)
        await db.refresh(k2)

        # 2. RUN SIMULATION LOOP
        controller = HierarchicalBudgetController(db, learning_rate=0.05, temperature=0.2)
        total_budget = 100.0
        
        # Pre-seed some initial history (Day 0)
        # We need this so the agent has *something* to look at for the first state
        today = datetime.utcnow().date()
        base_date = today - timedelta(days=days)
        
        # Helper function definition moved outside loop context if needed, but here we can define it inline or use lambda
        # _add_record helper is defined at module level
        
        print("   [Setup] Seeding initial history...")
        _add_record_to_db(db, c1.id, k1.id, base_date - timedelta(days=1), spend=10, sales=50) # ROAS 5
        _add_record_to_db(db, c2.id, k2.id, base_date - timedelta(days=1), spend=10, sales=15) # ROAS 1.5
        await db.commit()

        history = []

        for day in range(days):
            current_date_obj = base_date + timedelta(days=day)
            # Convert to datetime for SQL binding
            current_date = datetime.combine(current_date_obj, datetime.min.time())
            
            print(f"\nDay {day+1} ({current_date.date()})")

            # --- A. Agent Allocation ---
            # Run allocation logic
            plan = await controller.run_allocation(profile_id, total_budget, reference_date=current_date)
            
            # Extract allocated budgets
            alloc_a = float(plan['campaign_allocations'].get(str(c1.id), 0))
            alloc_b = float(plan['campaign_allocations'].get(str(c2.id), 0))
            
            print(f"   Allocation: Winner=${alloc_a:.1f} | Loser=${alloc_b:.1f}")

            # --- B. Simulate Market Response ---
            # Winner campaign returns ~5x ROAS, Loser returns ~1.5x ROAS
            # But varies randomly
            
            # Campaign A performance
            spend_a = alloc_a # Full spend simulated
            sales_a = spend_a * random.normalvariate(5.0, 1.0)
            sales_a = max(0, sales_a)
            
            # Campaign B performance
            spend_b = alloc_b
            sales_b = spend_b * random.normalvariate(1.5, 0.5)
            sales_b = max(0, sales_b)

            # Record to DB (as if it happened today)
            _add_record_to_db(db, c1.id, k1.id, current_date, spend_a, sales_a)
            _add_record_to_db(db, c2.id, k2.id, current_date, spend_b, sales_b)
            await db.commit()
            
            total_sales = sales_a + sales_b
            global_roas = total_sales / total_budget if total_budget > 0 else 0
            print(f"   Market: Sales=${total_sales:.1f}, ROAS={global_roas:.2f}")

            # --- C. Agent Learning ---
            # Agent looks at recent period. We want recent=current_date.
            # So pass tomorrow as ref_date.
            update_result = await controller.learn_from_outcomes(
                profile_id, 
                lookback_days=1,
                reference_date=current_date + timedelta(days=1)
            )
            
            if update_result.get('updated'):
                print(f"   Learning: Reward={update_result['reward']:.3f}")
            else:
                print(f"   Learning Skipped: {update_result.get('status')}")

            history.append({
                "day": day,
                "alloc_winner": alloc_a,
                "alloc_loser": alloc_b,
                "roas": global_roas
            })

    # 3. SUMMARY
    print("\nSimulation Summary")
    print("Day | Winner $$ | Loser $$ | Portfolio ROAS")
    print("----|-----------|----------|---------------")
    for h in history:
        print(f"{h['day']+1:3} | ${h['alloc_winner']:6.1f}   | ${h['alloc_loser']:6.1f}   | {h['roas']:.2f}")

    winner_final = history[-1]['alloc_winner']
    loser_final = history[-1]['alloc_loser']
    
    if winner_final > loser_final:
        print(f"\nSUCCESS: Agent correctly learned to prioritize the Winner campaign.")
    else:
        print(f"\nFAILURE: Agent failed to prioritize the Winner campaign.")

def _add_record_to_db(db, cid, kid, date, spend, sales):
    rec = PerformanceRecord(
        campaign_id=cid,
        keyword_id=kid,
        date=date,
        spend=spend,
        sales=sales,
        clicks=int(spend/1.0), # Assume $1 CPC
        orders=int(sales/20.0), # Assume $20 AOV
        impressions=int(spend/1.0)*10
    )
    db.add(rec)

if __name__ == "__main__":
    asyncio.run(run_simulation())
