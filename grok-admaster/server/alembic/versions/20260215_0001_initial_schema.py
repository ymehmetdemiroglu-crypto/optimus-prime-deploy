"""Initial schema with encrypted credentials

Revision ID: 20260215_0001
Revises:
Create Date: 2026-02-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260215_0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables with encrypted credential fields."""

    # Accounts table
    op.create_table(
        'accounts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('amazon_account_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('region', sa.String(), nullable=False, server_default='NA'),
        sa.Column('status', sa.String(), server_default='onboarding'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('amazon_account_id')
    )
    op.create_index('ix_accounts_id', 'accounts', ['id'])
    op.create_index('ix_accounts_name', 'accounts', ['name'])

    # Profiles table
    op.create_table(
        'profiles',
        sa.Column('profile_id', sa.String(), nullable=False),
        sa.Column('account_id', sa.Integer(), nullable=True),
        sa.Column('country_code', sa.String(), nullable=True),
        sa.Column('currency_code', sa.String(), nullable=True),
        sa.Column('timezone', sa.String(), nullable=True),
        sa.Column('account_info_id', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.ForeignKeyConstraint(['account_id'], ['accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('profile_id')
    )
    op.create_index('ix_profiles_profile_id', 'profiles', ['profile_id'])

    # Credentials table - with encrypted fields (stored as larger VARCHAR)
    op.create_table(
        'credentials',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('account_id', sa.Integer(), nullable=True),
        sa.Column('client_id', sa.String(512), nullable=False),  # Encrypted
        sa.Column('client_secret', sa.String(512), nullable=False),  # Encrypted
        sa.Column('refresh_token', sa.String(512), nullable=False),  # Encrypted
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()')),
        sa.ForeignKeyConstraint(['account_id'], ['accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_credentials_id', 'credentials', ['id'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_index('ix_credentials_id', 'credentials')
    op.drop_table('credentials')
    op.drop_index('ix_profiles_profile_id', 'profiles')
    op.drop_table('profiles')
    op.drop_index('ix_accounts_name', 'accounts')
    op.drop_index('ix_accounts_id', 'accounts')
    op.drop_table('accounts')
