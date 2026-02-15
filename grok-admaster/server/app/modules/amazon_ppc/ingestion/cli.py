"""
CLI utility for testing and running ingestion manually.
Usage: python -m app.modules.amazon_ppc.ingestion.cli
"""
import asyncio
import sys
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from .manager import IngestionManager

async def run_ingestion():
    """Run ingestion and print results."""
    # Create async engine
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        manager = IngestionManager(session)
        
        print("🚀 Starting ingestion for all accounts...")
        try:
            await manager.sync_all_accounts()
            print("✅ Ingestion completed successfully!")
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(run_ingestion())
