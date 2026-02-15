"""
API endpoints for data ingestion operations.
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from ..ingestion.manager import IngestionManager

router = APIRouter()

@router.post("/sync-all")
async def trigger_full_sync(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger a full data sync for all active accounts.
    Runs in the background.
    """
    async def sync_task():
        manager = IngestionManager(db)
        await manager.sync_all_accounts()
    
    background_tasks.add_task(sync_task)
    
    return {
        "status": "started",
        "message": "Data sync initiated in the background"
    }

@router.post("/sync-account/{account_id}")
async def trigger_account_sync(
    account_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger data sync for a specific account.
    """
    from ..accounts.service import account_service
    
    account = await account_service.get_account(db, account_id)
    if not account:
        return {"error": "Account not found"}, 404
    
    async def sync_task():
        manager = IngestionManager(db)
        await manager.sync_account(account)
    
    background_tasks.add_task(sync_task)
    
    return {
        "status": "started",
        "message": f"Data sync initiated for account {account_id}"
    }
