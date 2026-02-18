from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from .models import Account, Credential, Profile
from .schemas import AccountCreate, CredentialCreate

class AccountService:
    async def create_account(self, db: AsyncSession, account_in: AccountCreate) -> Account:
        account = Account(
            name=account_in.name,
            amazon_account_id=account_in.amazon_account_id,
            region=account_in.region,
        )
        db.add(account)
        await db.commit()
        await db.refresh(account)
        return account

    async def add_credential(self, db: AsyncSession, cred_in: CredentialCreate) -> Credential:
        # NOTE: logic to encrypt sensitive fields would go here or in a helper
        cred = Credential(
            account_id=cred_in.account_id,
            client_id=cred_in.client_id,
            client_secret=cred_in.client_secret,
            refresh_token=cred_in.refresh_token
        )
        db.add(cred)
        await db.commit()
        await db.refresh(cred)
        return cred

    async def get_accounts(self, db: AsyncSession) -> list[Account]:
        query = select(Account).options(selectinload(Account.profiles))
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_account(self, db: AsyncSession, account_id: int) -> Account | None:
        query = select(Account).where(Account.id == account_id).options(selectinload(Account.profiles))
        result = await db.execute(query)
        return result.scalars().first()

    # Placeholder for syncing profiles from Amazon API
    async def sync_profiles(self, db: AsyncSession, account_id: int):
        # This will need the Amazon Ads API Client to fetch profiles
        pass

account_service = AccountService()
