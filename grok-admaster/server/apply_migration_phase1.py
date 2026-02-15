
import asyncio
import os
import sys

# Add server directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import engine
from sqlalchemy import text

async def migrate():
    print("Starting migration: Adding AI columns to ppc_campaigns...")
    async with engine.begin() as conn:
        # Check if columns exist to avoid errors
        # This is postgres specific syntax usually, but let's try generic SQL or check dialect
        # For simplicity, we wrap in try-catch blocks or use generic ALTER TABLE
        
        commands = [
            "ALTER TABLE ppc_campaigns ADD COLUMN IF NOT EXISTS ai_mode VARCHAR",
            "ALTER TABLE ppc_campaigns ADD COLUMN IF NOT EXISTS target_acos FLOAT DEFAULT 30.0",
            "ALTER TABLE ppc_campaigns ADD COLUMN IF NOT EXISTS target_roas FLOAT DEFAULT 3.0"
        ]
        
        for cmd in commands:
            try:
                print(f"Executing: {cmd}")
                await conn.execute(text(cmd))
            except Exception as e:
                print(f"Error executing {cmd}: {e}")
                
    print("Migration complete.")

if __name__ == "__main__":
    asyncio.run(migrate())
