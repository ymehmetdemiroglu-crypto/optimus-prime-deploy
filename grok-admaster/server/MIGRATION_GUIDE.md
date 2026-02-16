# Database Migration Guide - Grok-AdMaster

This guide covers database schema management using Alembic for the Grok-AdMaster application.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Migration Commands](#migration-commands)
3. [Creating Migrations](#creating-migrations)
4. [Upgrading Existing Databases](#upgrading-existing-databases)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

---

## 🚀 Quick Start

### First-Time Setup

```bash
# 1. Install dependencies (including Alembic)
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your database credentials

# 3. Run migrations to create database schema
python manage_db.py migrate
```

### Verify Setup

```bash
# Check current migration status
python manage_db.py current

# Should show: 20260215_0001 (head)
```

---

## 🛠️ Migration Commands

### `manage_db.py` Script

The `manage_db.py` script provides convenient commands for database management:

```bash
# Run all pending migrations
python manage_db.py migrate

# Rollback the last migration
python manage_db.py rollback

# Show current database revision
python manage_db.py current

# Show full migration history
python manage_db.py history

# Create a new migration with autogenerate
python manage_db.py create "description of changes"

# Reset database (DEV ONLY - deletes all data!)
python manage_db.py reset

# Show help
python manage_db.py help
```

### Direct Alembic Commands

You can also use Alembic directly:

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade 20260215_0001

# Downgrade one revision
alembic downgrade -1

# Show current revision
alembic current

# Show history
alembic history --verbose

# Generate new migration
alembic revision --autogenerate -m "description"
```

---

## 📝 Creating Migrations

### Automatic Generation (Recommended)

When you make changes to SQLAlchemy models, Alembic can auto-generate migrations:

```bash
# 1. Modify your models in app/modules/amazon_ppc/*/models.py
# Example: Add a new column to Account model

# 2. Generate migration
python manage_db.py create "add status column to accounts"

# 3. Review the generated migration in alembic/versions/
# Edit if needed to add data migrations or custom logic

# 4. Run the migration
python manage_db.py migrate
```

### Manual Migration Creation

For complex changes (data migrations, conditional logic):

```bash
# Create empty migration template
alembic revision -m "migrate existing credentials to encrypted format"

# Edit the generated file in alembic/versions/
# Add custom upgrade/downgrade logic
```

**Example: Data Migration**

```python
"""Encrypt existing plaintext credentials

Revision ID: 20260215_0002
Revises: 20260215_0001
"""
from alembic import op
import sqlalchemy as sa
from app.core.encryption import encrypt_value

def upgrade() -> None:
    # Get database connection
    connection = op.get_bind()

    # Fetch all credentials
    result = connection.execute(
        sa.text("SELECT id, client_id, client_secret, refresh_token FROM credentials")
    )

    # Encrypt each credential
    for row in result:
        encrypted_client_id = encrypt_value(row.client_id)
        encrypted_client_secret = encrypt_value(row.client_secret)
        encrypted_refresh_token = encrypt_value(row.refresh_token)

        connection.execute(
            sa.text("""
                UPDATE credentials
                SET client_id = :cid,
                    client_secret = :cs,
                    refresh_token = :rt
                WHERE id = :id
            """),
            {
                "cid": encrypted_client_id,
                "cs": encrypted_client_secret,
                "rt": encrypted_refresh_token,
                "id": row.id
            }
        )

def downgrade() -> None:
    # Downgrade not supported for encryption change
    raise NotImplementedError("Cannot downgrade encrypted credentials to plaintext")
```

---

## 🔄 Upgrading Existing Databases

### Scenario 1: Fresh Development Database

If you have no data to preserve:

```bash
# Option A: Reset and migrate (DESTROYS ALL DATA)
python manage_db.py reset

# Option B: Manual reset
python -c "from app.core.database import Base, engine; import asyncio; asyncio.run(Base.metadata.drop_all(engine))"
python manage_db.py migrate
```

### Scenario 2: Production Database with Existing Data

**⚠️ CRITICAL: Always backup before migrating production!**

```bash
# 1. BACKUP YOUR DATABASE
pg_dump -h host -U user -d database > backup_$(date +%Y%m%d_%H%M%S).sql

# 2. Test migration on staging first
# Deploy to staging environment
python manage_db.py current  # Check current state
python manage_db.py migrate  # Run migrations

# 3. Verify application works correctly

# 4. Deploy to production
# SSH to production server
python manage_db.py current
python manage_db.py migrate

# 5. Verify production health
curl http://your-api/health
```

### Scenario 3: Migrating from Old Schema (Pre-Alembic)

If you have an existing database without Alembic version tracking:

```bash
# 1. Backup database
pg_dump your_database > backup.sql

# 2. Check if tables match the initial migration
# Compare your database with alembic/versions/20260215_0001_initial_schema.py

# 3. If tables match, stamp the database with initial revision
alembic stamp 20260215_0001

# 4. Run any new migrations
python manage_db.py migrate

# If tables don't match, you'll need to create custom migrations
# to transform your schema to match the initial migration
```

---

## 🔍 Troubleshooting

### Error: "Can't locate revision identified by '...'"

**Cause:** Alembic can't find the migration history table or a referenced migration.

**Solution:**
```bash
# Initialize version table
alembic stamp head

# Or start fresh (dev only)
python manage_db.py reset
```

### Error: "Target database is not up to date"

**Cause:** Database has migrations not in your codebase.

**Solution:**
```bash
# Check current state
python manage_db.py current

# Downgrade to known good state
alembic downgrade <known_revision>

# Or reset (dev only)
python manage_db.py reset
```

### Error: "Column already exists"

**Cause:** Migration tried to add a column that already exists.

**Solution:**
```bash
# Check database state vs migrations
python manage_db.py current

# If out of sync, manually fix or stamp correct version
alembic stamp <correct_revision>
```

### Error: "Encryption failed" when starting app

**Cause:** Trying to decrypt credentials encrypted with a different `SECRET_KEY`.

**Solution:**
```bash
# If SECRET_KEY changed, you need to re-encrypt all credentials
# 1. Export credentials before changing SECRET_KEY
# 2. Change SECRET_KEY
# 3. Re-import and re-encrypt credentials

# Or reset database (dev only)
python manage_db.py reset
```

### Error: "Database not initialized"

**Cause:** Starting app without running migrations first.

**Solution:**
```bash
# Run migrations
python manage_db.py migrate

# App will now start successfully
```

---

## 📚 Best Practices

### 1. Always Review Auto-Generated Migrations

```bash
# After generating
python manage_db.py create "add new column"

# ALWAYS review the file before running
cat alembic/versions/2026*_add_new_column.py

# Check for:
# - Correct column types
# - Proper constraints
# - Missing indexes
# - Data migration needs
```

### 2. Write Reversible Migrations

Always implement both `upgrade()` and `downgrade()`:

```python
def upgrade() -> None:
    op.add_column('accounts', sa.Column('email', sa.String(255)))

def downgrade() -> None:
    op.drop_column('accounts', 'email')
```

### 3. Use Transactions

Wrap risky operations in transactions:

```python
def upgrade() -> None:
    connection = op.get_bind()

    # Transaction is automatic, but be explicit for complex operations
    with connection.begin():
        # Multiple operations here
        op.add_column(...)
        op.create_index(...)
```

### 4. Test Migrations Before Production

```bash
# 1. Create migration in dev
python manage_db.py create "add feature X"

# 2. Test upgrade
python manage_db.py migrate

# 3. Test downgrade
python manage_db.py rollback

# 4. Test upgrade again
python manage_db.py migrate

# 5. Deploy to staging
# 6. Only then deploy to production
```

### 5. Document Complex Migrations

Add detailed docstrings:

```python
"""Add search term embeddings for semantic analysis

This migration:
1. Creates search_term_embeddings table
2. Adds pgvector extension if not exists
3. Migrates existing search terms to new table
4. Creates vector similarity indexes

Estimated time on large database: ~5 minutes per 100k records
"""

def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ... rest of migration
```

### 6. Handle Production Data Carefully

```python
def upgrade() -> None:
    # Good: Preserve existing data
    op.add_column('accounts',
        sa.Column('status', sa.String(),
                  server_default='active',  # Default for new rows
                  nullable=False)
    )

    # Update existing rows
    op.execute("UPDATE accounts SET status = 'legacy' WHERE status IS NULL")

def upgrade_bad() -> None:
    # Bad: Could lose data
    op.drop_column('accounts', 'status')  # No backup!
```

### 7. Version Control Your Migrations

```bash
# Always commit migrations with code changes
git add alembic/versions/
git commit -m "Add migration for feature X"

# Tag releases with migration info
git tag -a v1.2.0 -m "Requires migration: 20260215_0003"
```

---

## 🔐 Security Considerations

### Credential Migration

When migrating to encrypted credentials:

1. **Backup unencrypted credentials** before migration
2. **Store backup securely** (encrypted, access-controlled)
3. **Test decryption** after migration
4. **Rotate credentials** after successful migration
5. **Delete plaintext backups** securely

### Migration Secrets

Never hardcode secrets in migrations:

```python
# Bad
def upgrade():
    op.execute("INSERT INTO api_keys VALUES ('sk-secret123')")

# Good
def upgrade():
    # Credentials managed via environment or separate secure process
    pass
```

---

## 📊 Migration Checklist

Before deploying migrations to production:

- [ ] Migration tested in development
- [ ] Migration tested in staging (if available)
- [ ] Database backup created
- [ ] Downgrade path tested (if applicable)
- [ ] Team notified of deployment
- [ ] Estimated downtime communicated (if any)
- [ ] Rollback plan documented
- [ ] Post-migration verification queries prepared
- [ ] Monitoring alerts configured

---

## 🆘 Emergency Rollback

If a migration causes issues in production:

```bash
# 1. IMMEDIATELY stop the application
systemctl stop grok-admaster

# 2. Rollback the migration
python manage_db.py rollback

# 3. Verify database state
python manage_db.py current

# 4. Deploy previous application version
git checkout <previous-version>

# 5. Restart application
systemctl start grok-admaster

# 6. Investigate issue
# Check logs, review migration, test in staging
```

---

## 📞 Support

For migration issues:

1. Check this guide first
2. Review Alembic documentation: https://alembic.sqlalchemy.org/
3. Check application logs
4. Create issue in project repository

---

**Last Updated:** 2026-02-15
**Alembic Version:** 1.13.0+
**SQLAlchemy Version:** 2.0.0+
