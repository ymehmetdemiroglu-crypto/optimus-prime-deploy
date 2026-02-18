#!/usr/bin/env python3
"""
Database management script for Optimus Prime.

Provides commands for:
- Running migrations
- Creating new migrations
- Resetting database (dev only)
- Checking migration status

Usage:
    python manage_db.py migrate       # Run pending migrations
    python manage_db.py rollback      # Rollback last migration
    python manage_db.py current       # Show current revision
    python manage_db.py history       # Show migration history
    python manage_db.py reset         # Reset database (dev only)
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.core.database import engine, Base
from alembic.config import Config
from alembic import command


def get_alembic_config() -> Config:
    """Get Alembic configuration."""
    alembic_cfg = Config("alembic.ini")
    return alembic_cfg


def migrate():
    """Run all pending migrations."""
    print("🔄 Running database migrations...")
    alembic_cfg = get_alembic_config()
    command.upgrade(alembic_cfg, "head")
    print("✅ Migrations completed successfully")


def rollback():
    """Rollback the last migration."""
    print("⏪ Rolling back last migration...")
    alembic_cfg = get_alembic_config()
    command.downgrade(alembic_cfg, "-1")
    print("✅ Rollback completed successfully")


def current():
    """Show current database revision."""
    print("📍 Current database revision:")
    alembic_cfg = get_alembic_config()
    command.current(alembic_cfg, verbose=True)


def history():
    """Show migration history."""
    print("📜 Migration history:")
    alembic_cfg = get_alembic_config()
    command.history(alembic_cfg, verbose=True)


def create_migration(message: str):
    """Create a new migration with autogenerate."""
    print(f"📝 Creating new migration: {message}")
    alembic_cfg = get_alembic_config()
    command.revision(alembic_cfg, message=message, autogenerate=True)
    print("✅ Migration created successfully")


async def reset_database():
    """Reset database (DROP ALL and recreate). DEV ONLY!"""
    if settings.ENV.lower() == "production":
        print("❌ ERROR: Cannot reset database in production!")
        sys.exit(1)

    print("⚠️  WARNING: This will DELETE ALL DATA!")
    response = input("Type 'yes' to confirm: ")
    if response.lower() != "yes":
        print("❌ Aborted")
        return

    print("🗑️  Dropping all tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    print("📦 Creating fresh database schema...")
    # Run migrations to create tables
    migrate()

    print("✅ Database reset completed")


def show_help():
    """Show help message."""
    print(__doc__)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)

    command_name = sys.argv[1].lower()

    commands = {
        "migrate": migrate,
        "rollback": rollback,
        "current": current,
        "history": history,
        "reset": lambda: asyncio.run(reset_database()),
        "help": show_help,
    }

    if command_name == "create":
        if len(sys.argv) < 3:
            print("❌ ERROR: Migration message required")
            print("Usage: python manage_db.py create 'migration message'")
            sys.exit(1)
        create_migration(" ".join(sys.argv[2:]))
    elif command_name in commands:
        commands[command_name]()
    else:
        print(f"❌ ERROR: Unknown command '{command_name}'")
        show_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
