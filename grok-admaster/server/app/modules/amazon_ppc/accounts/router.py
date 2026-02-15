from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.core.database import get_db
from .schemas import AccountCreate, AccountRead, CredentialCreate
from .service import account_service

router = APIRouter()

@router.post("/", response_model=AccountRead)
async def create_account(
    account_in: AccountCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new Client Account.
    """
    return await account_service.create_account(db, account_in)

@router.get("/", response_model=List[AccountRead])
async def list_accounts(
    db: AsyncSession = Depends(get_db)
):
    """
    List all Client Accounts.
    """
    return await account_service.get_accounts(db)

@router.post("/{account_id}/credentials")
async def add_credential(
    account_id: int,
    cred_in: CredentialCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Add Amazon Ads API Credentials to an Account.
    """
    if cred_in.account_id != account_id:
         raise HTTPException(status_code=400, detail="Account ID mismatch")
    
    await account_service.add_credential(db, cred_in)
    return {"status": "success", "message": "Credentials added"}
