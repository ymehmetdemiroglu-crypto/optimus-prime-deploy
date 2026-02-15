"""
Apply AI Tables Migration
Run this script to add the new ML persistence tables to your database.
"""
import asyncio
import asyncpg
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

async def apply_migration():
    """Apply the 01_ai_tables.sql migration."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in .env")
        return False
    
    # Read SQL file
    sql_file = Path(__file__).parent / "updates" / "01_ai_tables.sql"
    if not sql_file.exists():
        print(f"ERROR: Migration file not found: {sql_file}")
        return False
    
    sql_content = sql_file.read_text(encoding='utf-8')
    
    print(f"Reading migration: {sql_file.name}")
    print(f"Connecting to database...")
    
    try:
        conn = await asyncpg.connect(db_url)
        print("SUCCESS: Connected successfully")
        
        print("Applying migration...")
        
        # Execute the entire SQL file as a single transaction
        # Use execute() instead of fetch() for DDL statements
        await conn.execute(sql_content)
        
        # Verify tables were created
        print("\nVerifying tables...")
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('model_registry', 'bandit_arms', 'prediction_logs', 'training_jobs')
            ORDER BY table_name
        """)
        
        print(f"  Found {len(tables)} new tables:")
        for table in tables:
            print(f"    - {table['table_name']}")
        
        await conn.close()
        print("\nSUCCESS: Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nERROR: Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(apply_migration())
    exit(0 if success else 1)
