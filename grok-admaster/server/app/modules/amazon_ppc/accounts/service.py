from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from .models import Account, Credential, Profile
from .schemas import AccountCreate, CredentialCreate


class AccountService:
    async def create_account(self, db: AsyncSession, account_in: AccountCreate) -> Account:
        account = Account(
            company_name=account_in.company_name,
            primary_contact_email=account_in.primary_contact_email
        )
        db.add(account)
        try:
            await db.commit()
            await db.refresh(account)
        except Exception:
            await db.rollback()
            raise
        return account

    async def add_credential(self, db: AsyncSession, cred_in: CredentialCreate) -> Credential:
        # NOTE: sensitive fields should be encrypted before storage (see core/encryption.py)
        cred = Credential(
            account_id=cred_in.account_id,
            client_id=cred_in.client_id,
            client_secret=cred_in.client_secret,
            refresh_token=cred_in.refresh_token
        )
        db.add(cred)
        try:
            await db.commit()
            await db.refresh(cred)
        except Exception:
            await db.rollback()
            raise
        return cred

    async def get_accounts(self, db: AsyncSession) -> List[Account]:
        query = select(Account).options(selectinload(Account.profiles))
        result = await db.execute(query)
        return result.scalars().all()

    async def get_account(self, db: AsyncSession, account_id: int) -> Optional[Account]:
        query = select(Account).where(Account.id == account_id).options(selectinload(Account.profiles))
        result = await db.execute(query)
        return result.scalars().first()

    # Placeholder for syncing profiles from Amazon API
    async def sync_profiles(self, db: AsyncSession, account_id: int):
        # This will need the Amazon Ads API Client to fetch profiles
        pass


account_service = AccountService()
