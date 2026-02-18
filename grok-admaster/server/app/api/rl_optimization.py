
"""
API Endpoints for Hierarchical Reinforcement Learning Budget Allocation.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.core.database import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.amazon_ppc.ml.hierarchical_rl import HierarchicalBudgetController

# All RL budget endpoints require a valid JWT — they can trigger live budget
# reallocations across client ad accounts.
router = APIRouter(dependencies=[Depends(get_current_user)])


class AllocationRequest(BaseModel):
    profile_id: str
    total_budget: float
    dry_run: bool = True


class ModelUpdateRequest(BaseModel):
    profile_id: str
    lookback_days: int = 1


class EnsembleOutcome(BaseModel):
    """A single observed outcome used to update ensemble model weights."""
    model_predictions: Dict[str, float]  # {model_name: predicted_bid}
    actual_optimal_bid: float            # observed ideal bid in retrospect


class EnsembleOutcomesRequest(BaseModel):
    outcomes: List[EnsembleOutcome]


@router.post("/allocate")
async def run_budget_allocation(
    request: AllocationRequest,
    db: AsyncSession = Depends(get_db),
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
            dry_run=request.dry_run,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Allocation failed: {str(e)}")


@router.post("/learn")
async def trigger_learning_update(
    request: ModelUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Trigger the REINFORCE policy-gradient update for the portfolio agent based
    on recent budget allocation outcomes. Runs synchronously — the policy matrix
    update is fast (small linear algebra, no deep network training).
    """
    controller = HierarchicalBudgetController(db)
    try:
        result = await controller.learn_from_outcomes(
            profile_id=request.profile_id,
            lookback_days=request.lookback_days,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning update failed: {str(e)}")


@router.post("/ensemble/update-weights")
async def update_ensemble_weights(
    request: EnsembleOutcomesRequest,
) -> Dict[str, Any]:
    """
    Update and persist ModelEnsemble adaptive weights based on observed outcomes.

    Send model predictions alongside the retrospectively-known optimal bid for
    each keyword decision. The ensemble adjusts weights (higher for models that
    were closer to optimal) and persists them to disk so they survive restarts.

    Example payload::

        {
          "outcomes": [
            {
              "model_predictions": {"gradient_boost": 1.20, "deep_nn": 1.15, "rl_agent": 1.18, "bandit": 1.22},
              "actual_optimal_bid": 1.17
            }
          ]
        }
    """
    from app.modules.amazon_ppc.ml.ensemble import ModelEnsemble
    ensemble = ModelEnsemble()
    outcome_dicts = [o.model_dump() for o in request.outcomes]
    ensemble.update_weights(outcome_dicts)
    return {
        "status": "weights_updated",
        "new_weights": ensemble.model_weights,
        "outcomes_processed": len(outcome_dicts),
    }

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
