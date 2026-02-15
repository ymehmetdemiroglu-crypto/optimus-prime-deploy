from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.modules.amazon_ppc.accounts.models import Account, Credential

class CredentialManager:
    """Manages multi-tenant Amazon API credentials injected on-the-fly."""
    
    @staticmethod
    async def get_client_credentials(db: AsyncSession, account_id: str) -> Dict[str, Any]:
        """Fetch SP-API and Ads API credentials for a specific account."""
        
        # This assumes your database schema has an Account linked to Credentials
        result = await db.execute(
            select(Account, Credential).join(Credential).where(Account.id == account_id)
        )
        row = result.first()
        
        if not row:
            raise ValueError(f"No credentials found for account {account_id}")
            
        account, creds = row
        
        return {
            "refresh_token": creds.refresh_token,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "marketplace_id": account.marketplace_id,
            "region": account.region or "us-east-1"
        }

    @staticmethod
    def inject_into_sp_api(creds: Dict[str, Any]) -> Any:
        """Initialize a direct SP-API client with injected credentials."""
        # This is where you'd initialize 'python-amazon-sp-api' or your custom wrapper
        # For now, we return the config dict that your services will use
        return {
            "refresh_token": creds["refresh_token"],
            "lwa_client_id": creds["client_id"],
            "lwa_client_secret": creds["client_secret"],
            "role_arn": None, # If using IAM user direct
        }
