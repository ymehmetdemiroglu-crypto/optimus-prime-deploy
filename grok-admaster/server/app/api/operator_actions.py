"""
Operator Action Review API

Exposes the action_review_queue for human operators to inspect,
approve, or reject autonomous recommendations before they execute.

Endpoints:
- GET  /operator-actions/pending         → List pending recommendations
- POST /operator-actions/{id}/approve    → Approve a single recommendation
- POST /operator-actions/{id}/reject     → Reject a single recommendation
- POST /operator-actions/bulk-approve    → Bulk-approve by account/asin/type
- GET  /operator-actions/history         → Reviewed actions (audit trail)
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.core.database import get_db
from app.models.semantic import ActionReviewQueue
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import User

router = APIRouter()


# ── Request / Response Models ────────────────────────────────────────────────

class ReviewDecision(BaseModel):
    note: Optional[str] = None


class BulkApproveRequest(BaseModel):
    account_id: Optional[int] = None
    asin: Optional[str] = None
    action_type: Optional[str] = None   # 'add_negative' | 'add_target'
    urgency: Optional[str] = None       # 'HIGH' | 'MEDIUM'
    note: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/pending")
async def list_pending_actions(
    account_id: Optional[int] = None,
    asin: Optional[str] = None,
    action_type: Optional[str] = None,
    urgency: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return all recommendations awaiting human review.
    Filterable by account, ASIN, action type, and urgency level.
    """
    stmt = (
        select(ActionReviewQueue)
        .where(ActionReviewQueue.status == "pending_review")
        .order_by(ActionReviewQueue.created_at.desc())
        .limit(limit)
    )
    if account_id is not None:
        stmt = stmt.where(ActionReviewQueue.account_id == account_id)
    if asin:
        stmt = stmt.where(ActionReviewQueue.asin == asin)
    if action_type:
        stmt = stmt.where(ActionReviewQueue.action_type == action_type)
    if urgency:
        stmt = stmt.where(ActionReviewQueue.urgency == urgency)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    return {
        "count": len(rows),
        "items": [_serialize(r) for r in rows],
    }


@router.post("/{action_id}/approve")
async def approve_action(
    action_id: str,
    body: ReviewDecision = ReviewDecision(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Approve a single pending recommendation — marks it ready for execution."""
    action = await _get_pending(db, action_id)
    action.status = "approved"
    action.reviewed_by = current_user.email
    action.reviewed_at = datetime.now(timezone.utc)
    action.review_note = body.note
    await db.commit()
    return {"status": "approved", "id": action_id}


@router.post("/{action_id}/reject")
async def reject_action(
    action_id: str,
    body: ReviewDecision = ReviewDecision(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Reject a pending recommendation — it will not be executed."""
    action = await _get_pending(db, action_id)
    action.status = "rejected"
    action.reviewed_by = current_user.email
    action.reviewed_at = datetime.now(timezone.utc)
    action.review_note = body.note
    await db.commit()
    return {"status": "rejected", "id": action_id}


@router.post("/bulk-approve")
async def bulk_approve_actions(
    body: BulkApproveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Approve all pending recommendations matching the supplied filters.
    At least one filter must be provided to prevent accidental mass-approval.
    """
    if not any([body.account_id, body.asin, body.action_type, body.urgency]):
        raise HTTPException(
            status_code=400,
            detail="At least one filter (account_id, asin, action_type, urgency) is required.",
        )

    stmt = (
        select(ActionReviewQueue)
        .where(ActionReviewQueue.status == "pending_review")
    )
    if body.account_id is not None:
        stmt = stmt.where(ActionReviewQueue.account_id == body.account_id)
    if body.asin:
        stmt = stmt.where(ActionReviewQueue.asin == body.asin)
    if body.action_type:
        stmt = stmt.where(ActionReviewQueue.action_type == body.action_type)
    if body.urgency:
        stmt = stmt.where(ActionReviewQueue.urgency == body.urgency)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    now = datetime.now(timezone.utc)

    for row in rows:
        row.status = "approved"
        row.reviewed_by = current_user.email
        row.reviewed_at = now
        row.review_note = body.note

    await db.commit()
    return {"approved": len(rows)}


@router.get("/history")
async def get_reviewed_actions(
    account_id: Optional[int] = None,
    status: Optional[str] = None,  # 'approved' | 'rejected' | 'executed'
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return recently reviewed actions for audit trail."""
    stmt = (
        select(ActionReviewQueue)
        .where(ActionReviewQueue.status != "pending_review")
        .order_by(ActionReviewQueue.reviewed_at.desc())
        .limit(limit)
    )
    if account_id is not None:
        stmt = stmt.where(ActionReviewQueue.account_id == account_id)
    if status:
        stmt = stmt.where(ActionReviewQueue.status == status)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    return {"count": len(rows), "items": [_serialize(r) for r in rows]}


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _get_pending(db: AsyncSession, action_id: str) -> ActionReviewQueue:
    try:
        uid = uuid.UUID(action_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Invalid action ID: {action_id!r}")
    result = await db.execute(
        select(ActionReviewQueue).where(
            ActionReviewQueue.id == uid,
            ActionReviewQueue.status == "pending_review",
        )
    )
    action = result.scalar_one_or_none()
    if not action:
        raise HTTPException(
            status_code=404,
            detail=f"Pending action {action_id!r} not found (may already be reviewed).",
        )
    return action


def _serialize(r: ActionReviewQueue) -> dict:
    return {
        "id": str(r.id),
        "patrol_cycle": r.patrol_cycle,
        "account_id": r.account_id,
        "asin": r.asin,
        "action_type": r.action_type,
        "term": r.term,
        "semantic_similarity": float(r.semantic_similarity) if r.semantic_similarity else None,
        "spend_at_detection": float(r.spend_at_detection) if r.spend_at_detection else None,
        "suggested_bid": float(r.suggested_bid) if r.suggested_bid else None,
        "suggested_match_type": r.suggested_match_type,
        "urgency": r.urgency,
        "status": r.status,
        "reviewed_by": r.reviewed_by,
        "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
        "review_note": r.review_note,
        "details": r.details,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }
