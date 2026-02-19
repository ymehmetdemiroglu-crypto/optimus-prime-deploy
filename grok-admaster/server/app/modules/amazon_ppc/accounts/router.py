import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.core.database import get_db
from .schemas import AccountCreate, AccountRead, CredentialCreate
from .service import account_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=AccountRead)
async def create_account(
    account_in: AccountCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new Client Account."""
    try:
        return await account_service.create_account(db, account_in)
    except Exception as e:
        logger.error(f"Failed to create account: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create account")


@router.get("/", response_model=List[AccountRead])
async def list_accounts(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=100, ge=1, le=500, description="Max records to return"),
    db: AsyncSession = Depends(get_db),
):
    """List Client Accounts with pagination."""
    try:
        return await account_service.get_accounts(db, skip=skip, limit=limit)
    except Exception as e:
        logger.error(f"Failed to list accounts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve accounts")


@router.post("/{account_id}/credentials")
async def add_credential(
    account_id: int,
    cred_in: CredentialCreate,
    db: AsyncSession = Depends(get_db)
):
    """Add Amazon Ads API Credentials to an Account."""
    if cred_in.account_id != account_id:
        raise HTTPException(status_code=400, detail="Account ID mismatch")

    account = await account_service.get_account(db, account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    try:
        await account_service.add_credential(db, cred_in)
        return {"status": "success", "message": "Credentials added"}
    except Exception as e:
        logger.error(f"Failed to add credentials for account {account_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store credentials")
