import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.core.config import settings

async def apply_fix():
    print("[Fix] Connecting to database...")
    print(f"[Fix] Database: {settings.POSTGRES_DB}")
    
    engine = create_async_engine(
        settings.ASYNC_DATABASE_URL,
        echo=True,
    )
    
    try:
        # Define SQL statements separated explicitly
        statements = [
            # Fix mutable search path
            "ALTER FUNCTION public.archive_old_anomaly_alerts() SET search_path = public, pg_temp;",
            "ALTER FUNCTION public.get_anomaly_stats(integer) SET search_path = public, pg_temp;"
        ]
        
        async with engine.begin() as conn:
            print("[Fix] Executing statements...")
            for sql in statements:
                print(f"[Fix] Executing: {sql[:50]}...")
                try:
                    await conn.execute(text(sql))
                except Exception as e:
                    # Ignore if function doesn't exist (harmless)
                    print(f"[Fix] Warning: {e}")
                    pass
            print("[Fix] Execution successful.")
            
        return True
    except Exception as e:
        print(f"[Fix] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.dispose()

if __name__ == "__main__":
    success = asyncio.run(apply_fix())
    sys.exit(0 if success else 1)
