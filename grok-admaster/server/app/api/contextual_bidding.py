"""
Contextual Bidding Features — ML self-optimization layer API

Exposes the Contextual Thompson Sampling bid optimizer via REST endpoints.
The optimizer uses a 23-feature context vector (temporal, performance,
market, keyword-meta) to condition per-arm Bayesian linear regression
posteriors, enabling faster convergence and more accurate bids than
static Beta-distribution bandits.

Routes
------
POST /api/v1/contextual-bidding/{profile_id}/optimize
     Run a full contextual TS optimization cycle for all AI-enabled
     keywords in the profile.

POST /api/v1/contextual-bidding/{profile_id}/learn
     Evaluate recent bid changes and update arm posteriors (learning step).

GET  /api/v1/contextual-bidding/keywords/{keyword_id}/arms
     Retrieve per-arm posterior statistics for a keyword.

GET  /api/v1/contextual-bidding/keywords/{keyword_id}/features
     Extract and return the current context feature vector.

GET  /api/v1/contextual-bidding/keywords/{keyword_id}/feature-importance
     Return posterior-mean weight vectors showing feature influence.

POST /api/v1/contextual-bidding/keywords/{keyword_id}/select-arm
     Select the best bid multiplier arm for a keyword right now.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.amazon_ppc.ml.bid_optimizer_service import BidOptimizerService
from app.modules.amazon_ppc.ml.thompson_sampling_db import ContextualThompsonSamplingDB
from app.modules.amazon_ppc.ml.contextual_features import (
    ContextFeatureExtractor,
    CONTEXT_FEATURE_NAMES,
    CONTEXT_DIM,
    context_to_json,
)

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
#  Request models
# ═══════════════════════════════════════════════════════════════

class OptimizeRequest(BaseModel):
    dry_run: bool = Field(
        False,
        description="When True, plan changes but do not apply them to the database."
    )
    mode: str = Field(
        "contextual",
        description="Optimizer mode: 'contextual' (Bayesian LR) or 'static' (Beta TS)."
    )


class LearnRequest(BaseModel):
    lookback_days: int = Field(
        7,
        ge=1, le=30,
        description="Number of days of executed bid changes to evaluate."
    )
    mode: str = Field(
        "contextual",
        description="Must match the mode used during optimization."
    )


class SelectArmRequest(BaseModel):
    timestamp: Optional[datetime] = Field(
        None,
        description="Override timestamp for feature extraction (defaults to now)."
    )


# ═══════════════════════════════════════════════════════════════
#  Profile-level endpoints
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/{profile_id}/optimize",
    summary="Run contextual Thompson Sampling optimization for a profile"
)
async def optimize_profile(
    profile_id: str,
    request: OptimizeRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Plan and (optionally) execute bid adjustments for every AI-enabled
    keyword in *profile_id* using the Contextual Thompson Sampling optimizer.

    Each keyword's bid multiplier is selected by sampling from the arm
    with the highest posterior-mean reward conditioned on the current
    23-dimensional context vector (temporal + performance + market + keyword meta).

    Returns a summary of all planned/executed bid changes.
    """
    svc = BidOptimizerService(db, mode=request.mode)
    result = await svc.optimize_profile(
        profile_id=profile_id,
        dry_run=request.dry_run,
    )
    return result


@router.post(
    "/{profile_id}/learn",
    summary="Evaluate executed bids and update arm posteriors"
)
async def learn_from_outcomes(
    profile_id: str,
    request: LearnRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Close the explore → exploit feedback loop.

    Fetches executed bid actions from the past *lookback_days*, compares
    pre- and post-execution ACoS, computes rewards, and performs Bayesian
    posterior updates on each arm.

    Should be called daily after the Amazon Ads API performance data has
    been synced (i.e. after the ingestion pipeline runs).
    """
    svc = BidOptimizerService(db, mode=request.mode)
    result = await svc.evaluate_and_learn(
        profile_id=profile_id,
        lookback_days=request.lookback_days,
    )
    return result


# ═══════════════════════════════════════════════════════════════
#  Keyword-level endpoints
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/keywords/{keyword_id}/arms",
    summary="Get per-arm posterior statistics for a keyword"
)
async def get_arm_statistics(
    keyword_id: int,
    with_context: bool = Query(
        True,
        description="Include posterior-mean expected reward for the *current* context."
    ),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns the full state of every bid-multiplier arm for the keyword.

    Each arm record includes:
    - `multiplier`            — the bid multiplier this arm represents
    - `pulls`                 — number of times this arm has been selected
    - `alpha` / `beta`        — Beta distribution parameters (classic TS)
    - `expected_value_beta`   — Beta-distribution expected reward
    - `expected_value_contextual` — Bayesian LR expected reward for current context
    - `posterior_observations` — number of (context, reward) pairs observed
    """
    ctx_ts = ContextualThompsonSamplingDB(db)
    stats  = await ctx_ts.get_arm_statistics(
        keyword_id=keyword_id,
        context=None if not with_context else None,  # extractor runs inside
    )
    return {
        "keyword_id": keyword_id,
        "arm_count":  len(stats),
        "arms":       stats,
    }


@router.get(
    "/keywords/{keyword_id}/features",
    summary="Extract the current context feature vector for a keyword"
)
async def get_context_features(
    keyword_id: int,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Extracts and returns the 23-dimensional context feature vector that
    the Contextual Thompson Sampler uses to condition arm selection.

    Feature groups:
    - **Temporal** (8): cyclical hour/day/month + weekend flag + payday proxy
    - **Performance** (8): 7-day rolling ACoS, CTR, CVR, velocity, momentum
    - **Market** (4): competitor price change, CPC trend, bid/CPC ratio, competition
    - **Keyword meta** (7): match type, impressions, spend, Rufus intent probability,
      query length, question weight, intent confidence
    """
    extractor = ContextFeatureExtractor(db)
    try:
        features_dict = await extractor.extract_dict(keyword_id)
    except Exception as exc:
        return {
            "keyword_id": keyword_id,
            "error": str(exc),
            "features": {},
        }

    return {
        "keyword_id":   keyword_id,
        "feature_dim":  CONTEXT_DIM,
        "feature_names": CONTEXT_FEATURE_NAMES,
        "features":     features_dict,
    }


@router.get(
    "/keywords/{keyword_id}/feature-importance",
    summary="Return posterior-mean weight vectors showing feature influence"
)
async def get_feature_importance(
    keyword_id: int,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns the posterior-mean weight vector **for each arm**.

    Positive weights indicate features that push toward higher reward
    when selecting that multiplier; negative weights indicate features
    that reduce expected reward.

    Use this to understand *why* the model favours certain multipliers
    under specific market conditions (e.g. high ACoS → lower multipliers
    get positive weights).
    """
    ctx_ts = ContextualThompsonSamplingDB(db)
    importance = await ctx_ts.get_feature_importance(keyword_id)

    return {
        "keyword_id":     keyword_id,
        "feature_labels": ["bias"] + list(CONTEXT_FEATURE_NAMES),
        "arm_weights":    importance,
    }


@router.post(
    "/keywords/{keyword_id}/select-arm",
    summary="Select the best bid-multiplier arm for a keyword right now"
)
async def select_arm(
    keyword_id: int,
    request: SelectArmRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Thompson-sample the best bid multiplier for the keyword at this moment.

    The contextual posterior is conditioned on the live feature vector
    extracted from the database (or the supplied *timestamp* override).

    Useful for:
    - Manual bid exploration
    - Debugging what the model would recommend right now
    - Simulating recommendations at specific timestamps

    **Does not** write any bid change to the database.
    """
    ctx_ts = ContextualThompsonSamplingDB(db)
    arm_id, multiplier, ts_sample = await ctx_ts.select_arm(
        keyword_id=keyword_id,
        context=None,
        timestamp=request.timestamp,
    )

    # Also fetch the current bid for context
    from sqlalchemy import select as sa_select, text
    bid_row = (
        await db.execute(
            text("SELECT bid, keyword_text FROM ppc_keywords WHERE id = :kid"),
            {"kid": keyword_id},
        )
    ).mappings().first()

    current_bid = float(bid_row["bid"]) if bid_row else None
    proposed_bid = (
        round(current_bid * multiplier, 2) if current_bid else None
    )

    return {
        "keyword_id":     keyword_id,
        "keyword_text":   bid_row["keyword_text"] if bid_row else None,
        "selected_arm_id": arm_id,
        "multiplier":     multiplier,
        "ts_sample":      round(ts_sample, 6),
        "current_bid":    current_bid,
        "proposed_bid":   proposed_bid,
        "timestamp":      (request.timestamp or datetime.utcnow()).isoformat(),
    }


@router.get(
    "/status",
    summary="Contextual bidding optimizer status"
)
async def get_optimizer_status(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Returns high-level status of the contextual bidding optimizer.

    Includes:
    - Total keywords with initialized arms
    - Total arm observations (learning events)
    - Feature dimension
    - Available multiplier arms
    """
    from sqlalchemy import text as sa_text

    try:
        row = (
            await db.execute(
                sa_text("""
                    SELECT
                        COUNT(DISTINCT keyword_id)  AS keywords_tracked,
                        SUM(pulls)                  AS total_pulls,
                        MAX(last_updated)           AS last_updated
                    FROM bandit_arms
                """)
            )
        ).mappings().first()

        ctx_ts = ContextualThompsonSamplingDB(db)

        return {
            "status":           "operational",
            "mode":             "contextual_thompson_sampling",
            "feature_dim":      CONTEXT_DIM,
            "feature_names":    CONTEXT_FEATURE_NAMES,
            "default_arms":     ctx_ts.multipliers,
            "keywords_tracked": int(row["keywords_tracked"] or 0) if row else 0,
            "total_arm_pulls":  int(row["total_pulls"]      or 0) if row else 0,
            "last_updated":     str(row["last_updated"])           if row and row["last_updated"] else None,
        }
    except Exception as exc:
        return {
            "status":        "degraded",
            "error":         str(exc),
            "feature_dim":   CONTEXT_DIM,
        }
