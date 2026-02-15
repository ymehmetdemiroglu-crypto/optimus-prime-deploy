
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import engine
from sqlalchemy import text

async def drop_table():
    print("Dropping legacy 'campaigns' table...")
    async with engine.begin() as conn:
        try:
            await conn.execute(text("DROP TABLE IF EXISTS campaigns CASCADE"))
            print("Table 'campaigns' dropped.")
        except Exception as e:
            print(f"Error dropping table: {e}")

if __name__ == "__main__":
    asyncio.run(drop_table())
