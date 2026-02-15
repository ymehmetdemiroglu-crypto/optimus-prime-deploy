import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
from app.core.database import Base

# Import models to ensure they're registered
from app.modules.amazon_ppc.competitive_intel.models import (
    CompetitorPriceHistory, PriceChangeEvent, CompetitorForecast, 
    UndercutProbability, StrategicSimulation, KeywordCannibalization
)

async def apply_schema():
    print("[Migration] Connecting to database...")
    print(f"[Migration] Database: {settings.POSTGRES_DB}")
    
    engine = create_async_engine(
        settings.ASYNC_DATABASE_URL,
        echo=True,
    )
    
    try:
        async with engine.begin() as conn:
            print("[Migration] Creating tables for Phase 7 Competitive Intelligence...")
            # create_all only creates tables that don't exist
            await conn.run_sync(Base.metadata.create_all)
            print("[Migration] Tables checked/created successfully.")
            
        return True
    except Exception as e:
        print(f"[Migration] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.dispose()

if __name__ == "__main__":
    success = asyncio.run(apply_schema())
    sys.exit(0 if success else 1)
