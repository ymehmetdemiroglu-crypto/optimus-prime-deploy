"""
Run anomaly detection database migration using SQLAlchemy.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.core.config import settings


async def run_migration():
    """Apply anomaly detection schema migration."""
    
    print("[Migration] Connecting to database...")
    print(f"[Migration] Database: {settings.POSTGRES_DB}")
    
    # Create engine
    engine = create_async_engine(
        settings.ASYNC_DATABASE_URL,
        echo=True,
    )
    
    try:
        # Read SQL migration file
        sql_file = Path(__file__).parent / "migrations" / "anomaly_detection.sql"
        
        if not sql_file.exists():
            print(f"[Migration] ❌ SQL file not found: {sql_file}")
            return False
        
        print(f"[Migration] Reading SQL from: {sql_file}")
        sql_content = sql_file.read_text()
        
        print(f"[Migration] Executing entire migration script...")
        
        async with engine.begin() as conn:
            # Execute the entire script at once
            await conn.execute(text(sql_content))
            print("[Migration] Migration executed successfully")
        
        print("\n[Migration] SUCCESS! Migration completed successfully!")
        print("\n[Migration] Created tables:")
        print("  - anomaly_alerts (real-time alerts, 90-day retention)")
        print("  - anomaly_history (historical tracking, indefinite)")
        print("  - anomaly_training_data (ML training data)")
        print("\n[Migration] Created indexes: 20+")
        print("[Migration] Created functions: archive_old_anomaly_alerts(), get_anomaly_stats()")
        
        return True
        
    except Exception as e:
        print(f"\n[Migration] ERROR: Migration failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await engine.dispose()


if __name__ == "__main__":
    success = asyncio.run(run_migration())
    sys.exit(0 if success else 1)
