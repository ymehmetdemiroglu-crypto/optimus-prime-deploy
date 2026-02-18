"""
Semantic Analytics API — Exposes the Semantic Engine to the frontend and MCP.

Endpoints:
- POST /ingest      → Trigger embedding generation for an account
- POST /embed-product → Embed a single product (ASIN + title)
- POST /bleed       → Run bleed detection for a product
- POST /opportunities → Run opportunity discovery for a product
- GET  /patrol-log  → View recent autonomous patrol activity
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.core.database import get_db
from app.modules.auth.dependencies import get_current_user
from app.services.analytics.semantic_engine import SemanticIngestor, BleedDetector, OpportunityFinder

# All semantic endpoints require a valid JWT — they trigger data ingestion and
# expose account-level financial data, so unauthenticated access is unacceptable.
router = APIRouter(dependencies=[Depends(get_current_user)])


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
    product_category: Optional[str] = None  # e.g. 'electronics', 'grocery', 'apparel'
    similarity_threshold: Optional[float] = None  # None → auto-resolve from category
    min_spend: float = 1.00

class OpportunityRequest(BaseModel):
    asin: str
    account_id: int
    similarity_floor: float = 0.70
    min_orders: int = 1


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
    """Trigger embedding generation for search terms that haven't been processed yet."""
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
    """Run semantic bleed detection for a product."""
    detector = BleedDetector(db)
    results = await detector.detect_bleed(
        asin=request.asin,
        account_id=request.account_id,
        similarity_threshold=request.similarity_threshold,  # None → resolved from category
        product_category=request.product_category,
        min_spend=request.min_spend,
    )
    
    total_waste = sum(r["spend"] for r in results)
    return {
        "status": "completed",
        "asin": request.asin,
        "bleed_count": len(results),
        "total_wasted_spend": round(total_waste, 2),
        "results": results
    }


@router.post("/opportunities")
async def find_opportunities(request: OpportunityRequest, db: AsyncSession = Depends(get_db)):
    """Run semantic opportunity discovery for a product."""
    finder = OpportunityFinder(db)
    results = await finder.find_opportunities(
        asin=request.asin,
        account_id=request.account_id,
        similarity_floor=request.similarity_floor,
        min_orders=request.min_orders
    )
    
    total_potential = sum(r["sales"] for r in results)
    return {
        "status": "completed",
        "asin": request.asin,
        "opportunity_count": len(results),
        "total_revenue_potential": round(total_potential, 2),
        "results": results
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
