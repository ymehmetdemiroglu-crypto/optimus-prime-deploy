import asyncio
import os
import re
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Manually load .env
def load_env_manual(filepath=".env"):
    if not os.path.exists(filepath):
        # Look in parent dir if running from scripts/
        if os.path.exists(os.path.join("..", filepath)):
            filepath = os.path.join("..", filepath)
        else:
            return {}
            
    env_vars = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip()
    return env_vars

env = load_env_manual()
DATABASE_URL = env.get("DATABASE_URL")

if DATABASE_URL and "postgresql://" in DATABASE_URL and "asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

async def apply_migration():
    if not DATABASE_URL:
        print("Error: DATABASE_URL not found in .env")
        return

    print(f"Connecting to database...")
    engine = create_async_engine(DATABASE_URL, echo=False)
    
    # Adjust path if running from scripts/
    migration_file = "migrations/06_semantic_schema.sql"
    if not os.path.exists(migration_file):
        if os.path.exists(os.path.join("..", migration_file)):
            migration_file = os.path.join("..", migration_file)
        else:
            print(f"Error: Migration file {migration_file} not found.")
            return

    with open(migration_file, "r") as f:
        sql_content = f.read()

    start_match = re.search(r'-- 1\.', sql_content)
    if not start_match:
        print("Could not find start of migration commands (-- 1.)")
        return

    body = sql_content[start_match.start():]
    chunks = re.split(r'\n(?=-- \d+\.)', body)
    
    print(f"Executing migration in {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
            
        print(f"Executing Chunk {i+1}...")
        try:
            async with engine.begin() as conn:
                # Set search path explicitly for pgvector
                await conn.execute(text("SET search_path = public, extensions;"))
                await conn.execute(text(chunk))
            print(f"Chunk {i+1} committed.")
        except Exception as e:
            print(f"Error in chunk {i+1}: {e}")
            # Do not raise, print error and exit cleanly (or continue if partial failure is okay? No, usually stop)
            # But script is meant for manual run now.
            return

    await engine.dispose()
    print("All chunks executed successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(apply_migration())
    except Exception as e:
        print(f"Fatal execution error: {e}")
