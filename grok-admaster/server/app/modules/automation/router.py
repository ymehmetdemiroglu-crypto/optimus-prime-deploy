"""
Automation Agent — FastAPI Router

REST endpoints for the automation agent: dashboard, script CRUD,
trigger execution, approval queue, logs, and skills.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query

from app.core.dependencies import require_auth
from .schemas import (
    ScriptConfig,
    ScriptConfigCreate,
    TriggerPayload,
    ExecutionResult,
    ApprovalItem,
    ApprovalAction,
    ApprovalDecision,
    AutomationDashboardResponse,
    AutomationLogEntry,
    SkillDefinition,
)
from .engine import engine
from .triggers import trigger_handler
from .skills import skill_registry, register_builtin_skills

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/automation", tags=["Automation Agent"])

# Register built-in skills on import
register_builtin_skills()


# ──────────────────────────────────────────────────────────
#  Dashboard
# ──────────────────────────────────────────────────────────

@router.get(
    "/dashboard",
    response_model=AutomationDashboardResponse,
    summary="Agent dashboard overview",
)
async def get_dashboard(user: dict = Depends(require_auth)):
    return engine.get_dashboard()


# ──────────────────────────────────────────────────────────
#  Script CRUD
# ──────────────────────────────────────────────────────────

@router.get("/scripts", response_model=List[ScriptConfig], summary="List all scripts")
async def list_scripts(user: dict = Depends(require_auth)):
    return engine.list_scripts()


@router.post("/scripts", response_model=ScriptConfig, summary="Create a script")
async def create_script(
    body: ScriptConfigCreate,
    user: dict = Depends(require_auth),
):
    return engine.register_script(body)


@router.get("/scripts/{script_id}", response_model=ScriptConfig, summary="Get script details")
async def get_script(script_id: str, user: dict = Depends(require_auth)):
    script = engine.get_script(script_id)
    if not script:
        raise HTTPException(status_code=404, detail="Script not found")
    return script


@router.delete("/scripts/{script_id}", summary="Delete a script")
async def delete_script(script_id: str, user: dict = Depends(require_auth)):
    if not engine.delete_script(script_id):
        raise HTTPException(status_code=404, detail="Script not found")
    return {"status": "deleted", "script_id": script_id}


# ──────────────────────────────────────────────────────────
#  Trigger Execution
# ──────────────────────────────────────────────────────────

@router.post("/trigger", response_model=ExecutionResult, summary="Execute or queue a script")
async def trigger_script(
    payload: TriggerPayload,
    user: dict = Depends(require_auth),
):
    result = await trigger_handler.handle_api_trigger(payload, user_id=user.get("id"))
    return result


@router.get(
    "/executions/{execution_id}",
    response_model=ExecutionResult,
    summary="Get execution result",
)
async def get_execution(execution_id: str, user: dict = Depends(require_auth)):
    result = engine.get_execution(execution_id)
    if not result:
        raise HTTPException(status_code=404, detail="Execution not found")
    return result


# ──────────────────────────────────────────────────────────
#  Approval Queue
# ──────────────────────────────────────────────────────────

@router.get(
    "/approvals",
    response_model=List[ApprovalItem],
    summary="List pending approvals",
)
async def list_approvals(user: dict = Depends(require_auth)):
    return engine.get_pending_approvals()


@router.post(
    "/approvals/{execution_id}",
    response_model=ExecutionResult,
    summary="Approve or reject an execution",
)
async def process_approval(
    execution_id: str,
    action: ApprovalAction,
    user: dict = Depends(require_auth),
):
    if action.decision == ApprovalDecision.APPROVED:
        result = await engine.approve(execution_id)
    else:
        result = await engine.reject(execution_id)

    if not result:
        raise HTTPException(status_code=404, detail="Approval item not found")
    return result


# ──────────────────────────────────────────────────────────
#  Logs
# ──────────────────────────────────────────────────────────

@router.get(
    "/logs",
    response_model=List[AutomationLogEntry],
    summary="Recent execution logs",
)
async def get_logs(
    limit: int = Query(default=20, ge=1, le=200),
    user: dict = Depends(require_auth),
):
    return engine.get_recent_logs(limit)


# ──────────────────────────────────────────────────────────
#  Skills
# ──────────────────────────────────────────────────────────

@router.get(
    "/skills",
    response_model=List[SkillDefinition],
    summary="List registered agent skills",
)
async def list_skills(user: dict = Depends(require_auth)):
    return skill_registry.list_skills()
