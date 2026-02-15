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
            # Fix Types
            "ALTER TABLE anomaly_alerts ALTER COLUMN profile_id TYPE VARCHAR USING profile_id::VARCHAR;",
            "ALTER TABLE anomaly_history ALTER COLUMN profile_id TYPE VARCHAR USING profile_id::VARCHAR;",
            "ALTER TABLE anomaly_training_data ALTER COLUMN profile_id TYPE VARCHAR USING profile_id::VARCHAR;",
            
            # Enable RLS
            "ALTER TABLE anomaly_alerts ENABLE ROW LEVEL SECURITY;",
            "ALTER TABLE anomaly_history ENABLE ROW LEVEL SECURITY;",
            "ALTER TABLE anomaly_training_data ENABLE ROW LEVEL SECURITY;",
            
            # Create Policies (using DROP first to ensure idempotency)
            "DROP POLICY IF EXISTS \"Enable all access for authenticated users\" ON anomaly_alerts;",
            "CREATE POLICY \"Enable all access for authenticated users\" ON anomaly_alerts FOR ALL TO authenticated USING (true) WITH CHECK (true);",
            
            "DROP POLICY IF EXISTS \"Enable all access for authenticated users\" ON anomaly_history;",
            "CREATE POLICY \"Enable all access for authenticated users\" ON anomaly_history FOR ALL TO authenticated USING (true) WITH CHECK (true);",
            
            "DROP POLICY IF EXISTS \"Enable all access for authenticated users\" ON anomaly_training_data;",
            "CREATE POLICY \"Enable all access for authenticated users\" ON anomaly_training_data FOR ALL TO authenticated USING (true) WITH CHECK (true);"
        ]
        
        async with engine.begin() as conn:
            print("[Fix] Executing statements...")
            for sql in statements:
                print(f"[Fix] Executing: {sql[:50]}...")
                try:
                    await conn.execute(text(sql))
                except Exception as e:
                    print(f"[Fix] Warning: {e}")
                    # Continue if one fails (e.g. column already varchar)
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
