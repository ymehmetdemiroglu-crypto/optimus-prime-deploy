from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.modules.amazon_ppc.accounts.models import Account, Credential
import logging

logger = logging.getLogger(__name__)


class SecureCredential:
    """Wrapper for credentials that prevents accidental exposure."""

    def __init__(self, credentials: Dict[str, Any]):
        self._credentials = credentials

    def get_for_api(self) -> Dict[str, Any]:
        """Get credentials for API use. WARNING: Contains sensitive data."""
        return self._credentials.copy()

    def __repr__(self) -> str:
        """Safe representation that doesn't expose secrets."""
        return f"<SecureCredential account={self._credentials.get('marketplace_id', 'unknown')}>"

    def __str__(self) -> str:
        """Safe string representation."""
        return self.__repr__()

    def redacted_dict(self) -> Dict[str, Any]:
        """Return a redacted version safe for logging."""
        return {
            "marketplace_id": self._credentials.get("marketplace_id"),
            "region": self._credentials.get("region"),
            "client_id": self._mask_secret(self._credentials.get("client_id", "")),
            "refresh_token": "***REDACTED***",
            "client_secret": "***REDACTED***"
        }

    @staticmethod
    def _mask_secret(value: str, visible_chars: int = 4) -> str:
        """Mask a secret showing only first few characters."""
        if not value or len(value) <= visible_chars:
            return "***"
        return f"{value[:visible_chars]}...***"

class CredentialManager:
    """Manages multi-tenant Amazon API credentials with secure handling.

    SECURITY WARNINGS:
    - Credentials should be encrypted at rest in the database
    - Never log credential values directly
    - Use SecureCredential wrapper for safe handling
    - Rotate credentials periodically
    """

    @staticmethod
    async def get_client_credentials(db: AsyncSession, account_id: str) -> SecureCredential:
        """Fetch SP-API and Ads API credentials for a specific account.

        Returns:
            SecureCredential: Wrapped credentials with safe handling methods

        Raises:
            ValueError: If no credentials found for account
        """

        # This assumes your database schema has an Account linked to Credentials
        result = await db.execute(
            select(Account, Credential).join(Credential).where(Account.id == account_id)
        )
        row = result.first()

        if not row:
            logger.warning(f"Credential fetch failed: account {account_id} not found")
            raise ValueError(f"No credentials found for account {account_id}")

        account, creds = row

        # Log access without exposing credentials
        logger.info(f"Retrieved credentials for account {account_id}, marketplace {account.marketplace_id}")

        # Wrap in SecureCredential to prevent accidental exposure
        credential_dict = {
            "refresh_token": creds.refresh_token,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "marketplace_id": account.marketplace_id,
            "region": account.region or "us-east-1"
        }

        return SecureCredential(credential_dict)

    @staticmethod
    def inject_into_sp_api(creds: SecureCredential) -> Dict[str, Any]:
        """Initialize SP-API client configuration with credentials.

        Args:
            creds: SecureCredential wrapper containing API credentials

        Returns:
            Configuration dict for SP-API client initialization

        WARNING: The returned dict contains sensitive credentials.
        Use only for direct API client initialization and never log or expose.
        """
        # Get credentials from secure wrapper
        credential_dict = creds.get_for_api()

        # Return SP-API client configuration
        # This is where you'd initialize 'python-amazon-sp-api' or your custom wrapper
        api_config = {
            "refresh_token": credential_dict["refresh_token"],
            "lwa_client_id": credential_dict["client_id"],
            "lwa_client_secret": credential_dict["client_secret"],
            "role_arn": None,  # If using IAM user direct
        }

        logger.debug(f"SP-API config prepared for marketplace {credential_dict.get('marketplace_id')}")

        return api_config
