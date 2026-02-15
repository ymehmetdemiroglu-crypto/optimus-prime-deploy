
"""
API Endpoints for Hierarchical Reinforcement Learning Budget Allocation.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
from pydantic import BaseModel

from app.core.database import get_db
from app.modules.amazon_ppc.ml.hierarchical_rl import HierarchicalBudgetController

router = APIRouter()

class AllocationRequest(BaseModel):
    profile_id: str
    total_budget: float
    dry_run: bool = True

class ModelUpdateRequest(BaseModel):
    profile_id: str
    lookback_days: int = 1

@router.post("/allocate")
async def run_budget_allocation(
    request: AllocationRequest,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Trigger the hierarchical budget allocation process for a profile.
    
    - **profile_id**: The Amazon profile ID.
    - **total_budget**: Total daily budget to distribute.
    - **dry_run**: If True, only simulates allocations without saving actions as pending.
    """
    controller = HierarchicalBudgetController(db)
    try:
        result = await controller.run_allocation(
            profile_id=request.profile_id,
            total_budget=request.total_budget,
            dry_run=request.dry_run
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Allocation failed: {str(e)}")

@router.post("/learn")
async def trigger_learning_update(
    request: ModelUpdateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Trigger the learning update step (policy gradient) based on recent outcomes.
    This can be expensive, so it runs in the background.
    """
    
    async def _run_learning(profile_id: str, lookback: int):
        # We need a new session for background task if the request session closes
        # But here we are passing logic to controller which uses session.
        # Actually, background tasks should manage their own sessions or usage.
        # For simplicity in this architecture, we await it if it's fast, or use a task runner.
        # Since learn_from_outcomes is DB-intensive but not CP-intensive heavy computation (like training deep nets),
        # we can await it or run it.
        # The controller expects an active session.
        pass

    # For now, run synchronously to ensure we can return the result status.
    # Policy gradient update on small matrices is fast.
    controller = HierarchicalBudgetController(db)
    try:
        result = await controller.learn_from_outcomes(
            profile_id=request.profile_id,
            lookback_days=request.lookback_days
        )
        return result
    except Exception as e:
        # Log error instead of crashing if possible, but here we raise
        raise HTTPException(status_code=500, detail=f"Learning update failed: {str(e)}")

@router.get("/state/{profile_id}")
async def get_latest_portfolio_state(
    profile_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the most recent portfolio state vector and agent parameters.
    """
    from sqlalchemy import text
    try:
        result = await db.execute(
            text("""
                SELECT id, state_vector, total_budget, budget_remaining, timestamp
                FROM rl_portfolio_state
                WHERE profile_id = :pid
                ORDER BY timestamp DESC
                LIMIT 1
            """),
            {"pid": profile_id}
        )
        row = result.mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="No state found")
        
        return {
            "id": row.id,
            "total_budget": float(row.total_budget),
            "timestamp": row.timestamp,
            "state_vector": row.state_vector # JSONB is auto-converted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
