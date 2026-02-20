"""
Semantic Analytics API — Exposes the Semantic Engine to the frontend and MCP.

Endpoints:
- POST /ingest      → Trigger embedding generation for an account
- POST /embed-product → Embed a single product (ASIN + title)
- POST /bleed       → Run bleed detection for a product
- POST /opportunities → Run opportunity discovery for a product
- POST /classify-intent → Classify search terms by shopping intent (Rufus/Cosmo)
- GET  /patrol-log  → View recent autonomous patrol activity
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.core.database import get_db
from app.services.analytics.semantic_engine import SemanticIngestor, BleedDetector, OpportunityFinder

router = APIRouter()


# --- Request Models ---

class IngestRequest(BaseModel):
    account_id: int
    limit: int = 500

class EmbedProductRequest(BaseModel):
    asin: str
    title: str
    bullet_points: Optional[List[str]] = None
    account_id: Optional[int] = None

class BleedRequest(BaseModel):
    asin: str
    account_id: int
    similarity_threshold: float = 0.40
    min_spend: float = 1.00
    intent_aware: bool = True

class OpportunityRequest(BaseModel):
    asin: str
    account_id: int
    similarity_floor: float = 0.70
    min_orders: int = 1
    intent_aware: bool = True

class IntentClassifyRequest(BaseModel):
    queries: List[str]


# --- Endpoints ---

@router.get("/health")
async def semantic_health():
    """Check if semantic engine is operational."""
    try:
        from app.services.embeddings import embedding_service
        # Quick test embed
        test_vec = embedding_service.embed_text("health check")
        return {
            "status": "operational",
            "embedding_model": embedding_service.MODEL_NAME,
            "embedding_dim": len(test_vec),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Semantic engine error: {e}")


@router.post("/ingest")
async def ingest_search_terms(request: IngestRequest, db: AsyncSession = Depends(get_db)):
    """Trigger embedding generation + intent classification for unprocessed search terms."""
    ingestor = SemanticIngestor(db)
    count = await ingestor.ingest_search_terms(request.account_id, request.limit)
    return {
        "status": "completed",
        "terms_embedded": count,
        "account_id": request.account_id
    }


@router.post("/embed-product")
async def embed_product(request: EmbedProductRequest, db: AsyncSession = Depends(get_db)):
    """Generate and store embedding for a product."""
    ingestor = SemanticIngestor(db)
    product = await ingestor.embed_product(
        asin=request.asin,
        title=request.title,
        bullet_points=request.bullet_points,
        account_id=request.account_id
    )
    return {
        "status": "embedded",
        "asin": request.asin,
        "embedding_id": str(product.id)
    }


@router.post("/bleed")
async def detect_bleed(request: BleedRequest, db: AsyncSession = Depends(get_db)):
    """Run semantic bleed detection with intent-aware thresholds."""
    detector = BleedDetector(db)
    results = await detector.detect_bleed(
        asin=request.asin,
        account_id=request.account_id,
        similarity_threshold=request.similarity_threshold,
        min_spend=request.min_spend,
        intent_aware=request.intent_aware
    )

    total_waste = sum(r["spend"] for r in results)
    return {
        "status": "completed",
        "asin": request.asin,
        "bleed_count": len(results),
        "total_wasted_spend": round(total_waste, 2),
        "intent_aware": request.intent_aware,
        "results": results
    }


@router.post("/opportunities")
async def find_opportunities(request: OpportunityRequest, db: AsyncSession = Depends(get_db)):
    """Run semantic opportunity discovery with intent-aware floors."""
    finder = OpportunityFinder(db)
    results = await finder.find_opportunities(
        asin=request.asin,
        account_id=request.account_id,
        similarity_floor=request.similarity_floor,
        min_orders=request.min_orders,
        intent_aware=request.intent_aware
    )

    total_potential = sum(r["sales"] for r in results)
    return {
        "status": "completed",
        "asin": request.asin,
        "opportunity_count": len(results),
        "total_revenue_potential": round(total_potential, 2),
        "intent_aware": request.intent_aware,
        "results": results
    }


@router.post("/classify-intent")
async def classify_intent(request: IntentClassifyRequest):
    """
    Classify search queries by shopping intent type.

    Intent types:
    - transactional: Direct purchase intent
    - informational_rufus: Research/comparison queries (Rufus AI traffic)
    - navigational: Brand-specific navigation
    - discovery: Category exploration

    Returns per-query intent, confidence, and scoring breakdown.
    """
    if not request.queries:
        raise HTTPException(status_code=400, detail="queries list must not be empty")
    if len(request.queries) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 queries per request")

    from app.services.ml.intent_classifier import get_intent_classifier, INTENT_THRESHOLDS, ShoppingIntent
    classifier = get_intent_classifier()
    results = classifier.classify_batch(request.queries)

    return {
        "status": "completed",
        "count": len(results),
        "results": [r.to_dict() for r in results],
        "intent_thresholds": {
            k.value: v for k, v in INTENT_THRESHOLDS.items()
        }
    }


class ThresholdOverrideRequest(BaseModel):
    account_id: int
    intent: str  # transactional, informational_rufus, navigational, discovery
    overrides: Dict[str, float]


@router.get("/thresholds")
async def get_thresholds(account_id: Optional[int] = None):
    """
    View the effective threshold profile for all intents.

    If account_id is provided, returns thresholds with any per-account
    overrides applied.  Otherwise returns the global defaults.
    """
    from app.services.ml.adaptive_thresholds import get_threshold_manager
    manager = get_threshold_manager()
    return {
        "account_id": account_id,
        "effective_thresholds": manager.get_full_profile(account_id),
        "account_overrides": manager.get_account_overrides(account_id) if account_id else {},
    }


@router.put("/thresholds")
async def set_thresholds(request: ThresholdOverrideRequest):
    """
    Set per-account threshold overrides for a specific intent type.

    Example body:
    {
        "account_id": 42,
        "intent": "informational_rufus",
        "overrides": {
            "bleed_threshold": 0.25,
            "min_orders_to_graduate": 7
        }
    }
    """
    from app.services.ml.adaptive_thresholds import get_threshold_manager
    from app.services.ml.intent_classifier import ShoppingIntent

    try:
        intent_enum = ShoppingIntent(request.intent)
    except ValueError:
        valid = [i.value for i in ShoppingIntent]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid intent '{request.intent}'. Valid: {valid}"
        )

    manager = get_threshold_manager()
    try:
        manager.set_overrides_bulk(request.account_id, intent_enum, request.overrides)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "updated",
        "account_id": request.account_id,
        "intent": request.intent,
        "effective_thresholds": manager.get_thresholds(intent_enum, request.account_id),
    }


@router.delete("/thresholds/{account_id}")
async def reset_thresholds(account_id: int):
    """Reset all threshold overrides for an account (revert to global defaults)."""
    from app.services.ml.adaptive_thresholds import get_threshold_manager
    manager = get_threshold_manager()
    manager.reset_account(account_id)
    return {
        "status": "reset",
        "account_id": account_id,
        "effective_thresholds": manager.get_full_profile(account_id),
    }


@router.get("/patrol-log")
async def get_patrol_log(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """View recent autonomous patrol activity."""
    result = await db.execute(text(
        "SELECT * FROM autonomous_patrol_log ORDER BY executed_at DESC LIMIT :limit"
    ), {"limit": limit})
    rows = result.fetchall()

    return {
        "total": len(rows),
        "logs": [
            {
                "id": str(r.id),
                "cycle": r.patrol_cycle,
                "action": r.action_type,
                "target": r.target_entity,
                "details": r.details,
                "status": r.status,
                "time": r.executed_at.isoformat() if r.executed_at else None
            }
            for r in rows
        ]
    }
