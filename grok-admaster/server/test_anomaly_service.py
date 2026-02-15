
import asyncio
import sys
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import async_session_maker
from app.modules.amazon_ppc.accounts.models import Profile
from app.modules.amazon_ppc.anomaly.service import anomaly_service

async def test_service():
    print("Testing Anomaly Service...")
    
    async with async_session_maker() as db:
        try:
            print("Calling get_active_alerts with profile_id='1'...")
            alerts = await anomaly_service.get_active_alerts(db, profile_id="1")
            print(f"Success! Got {len(alerts)} alerts.")
        except Exception as e:
            print(f"Error calling get_active_alerts: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(test_service())
