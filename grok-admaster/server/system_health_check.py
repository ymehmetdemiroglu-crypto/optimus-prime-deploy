
import asyncio
import sys
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add parent dir to path so we can import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings

async def run_health_check():
    print("="*60)
    print("OPTIMUS PRYME SYSTEM HEALTH CHECK")
    print("="*60)
    
    # 1. Database Connection
    print("\n[1/4] Checking Database Connection...")
    db_url = settings.ASYNC_DATABASE_URL
    try:
        # Mask password
        safe_url = db_url.split('@')[-1]
        print(f"Connecting to: ...@{safe_url}")
        
        engine = create_async_engine(db_url)
        print("Engine created.")
        
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version();"))
            version = result.scalar()
            print(f"SUCCESS: Connected to {version}")
            
            # 2. Check Critical Tables
            print("\n[2/4] Verifying Critical Tables...")
            tables_to_check = [
                "accounts", "campaigns", "keywords", 
                "optimization_plans", "anomalies", "forecasts"
            ]
            
            for table in tables_to_check:
                # Check if table exists
                check_query = text(f"SELECT to_regclass('public.{table}');")
                res = await conn.execute(check_query)
                if res.scalar():
                    # Count rows
                    count_res = await conn.execute(text(f"SELECT count(*) FROM public.{table}"))
                    count = count_res.scalar()
                    print(f"  - {table}: OK (Rows: {count})")
                else:
                    print(f"  - {table}: MISSING!")
                    
            # 3. Check RLS Policies
            print("\n[3/4] Verifying Security Policies (RLS)...")
            
            # Query pg_policies to see if our strict policies exist
            policy_query = text("""
                SELECT tablename, policyname 
                FROM pg_policies 
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """)
            
            policies = await conn.execute(policy_query)
            policy_map = {}
            for row in policies:
                if row.tablename not in policy_map:
                    policy_map[row.tablename] = []
                policy_map[row.tablename].append(row.policyname)
                
            required_policies = {
                "accounts": "Users can only access their own accounts",
                "campaigns": "Users can access campaigns of their accounts",
                "keywords": "Users can access keywords of their campaigns"
            }
            
            for table, expected_policy in required_policies.items():
                current_policies = policy_map.get(table, [])
                if any(p == expected_policy for p in current_policies):
                     print(f"  - {table}: SECURE (Policy found: '{expected_policy}')")
                else:
                     print(f"  - {table}: WARNING! Policy '{expected_policy}' not found. Current: {current_policies}")

    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        return

    print("\n[4/4] Configuration Check...")
    print(f"  - Environment: {settings.ENV}")
    print(f"  - Project Name: {settings.PROJECT_NAME}")

    print("\n" + "="*60)
    print("Build Successful")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_health_check())
