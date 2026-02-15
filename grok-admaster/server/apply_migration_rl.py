
import asyncio
import os
import sys

# Add server directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import engine
from sqlalchemy import text

async def migrate():
    print("Starting migration: Creating RL Budget Allocation tables...")
    
    # Read the SQL file
    sql_path = os.path.join(os.path.dirname(__file__), 'updates', '07_rl_budget_tables.sql')
    with open(sql_path, 'r') as f:
        sql_commands = f.read().split(';')
    
    async with engine.begin() as conn:
        for cmd in sql_commands:
            if cmd.strip():
                try:
                    print(f"Executing: {cmd[:50]}...")
                    await conn.execute(text(cmd))
                except Exception as e:
                    print(f"Error executing command: {e}")
                    # Continue even if error (e.g. table already exists)
                    raise e
                
    print("Migration complete.")

if __name__ == "__main__":
    asyncio.run(migrate())
