"""
Market Intelligence API Endpoints
Provides access to persisted competitor data, price history, and rankings.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.core.database import get_db
from app.services.market_intelligence_ingester import market_ingester
from app.services.researcher import amazon_search

router = APIRouter()


# ========================
# Pydantic Schemas
# ========================

class ProductSearchRequest(BaseModel):
    keyword: str
    location_code: int = 2840
    persist: bool = True


class ProductSearchResponse(BaseModel):
    keyword: str
    results_count: int
    products: List[dict]
    persisted: bool


class PriceHistoryResponse(BaseModel):
    asin: str
    price_history: List[dict]


class KeywordRankingsResponse(BaseModel):
    keyword: str
    rankings: List[dict]


class IngestionSummary(BaseModel):
    keyword: str
    products_ingested: int
    products_updated: int
    prices_recorded: int
    rankings_recorded: int
    timestamp: str


class RivalProduct(BaseModel):
    asin: str
    name: str
    brand: str
    price: float
    image: str
    rank: int
    rank_trend: str
    price_trend: float
    trend_percentage: float
    sov_score: int

class MarketIntelligenceSummary(BaseModel):
    est_market_share: float
    market_share_trend: float
    active_threats: int
    threats_trend: int
    avg_rival_price: float
    rival_price_trend: float
    category_rank: int
    rank_change: int
    rivals: List[RivalProduct]


# ========================
# Endpoints
# ========================

@router.post("/search", response_model=ProductSearchResponse, summary="Search Amazon & Persist")
async def search_and_persist(request: ProductSearchRequest):
    """
    Search Amazon for products matching a keyword.
    Results are automatically saved to the database for historical tracking.
    """
    results = await amazon_search(
        query=request.keyword,
        location_code=request.location_code,
        persist=request.persist
    )
    
    # Handle error string from amazon_search
    if isinstance(results, str):
        raise HTTPException(status_code=500, detail=results)
    
    return ProductSearchResponse(
        keyword=request.keyword,
        results_count=len(results),
        products=results,
        persisted=request.persist
    )


@router.get("/price-history/{asin}", response_model=PriceHistoryResponse, summary="Get Price History")
async def get_price_history(
    asin: str,
    days: int = Query(default=30, ge=1, le=365)
):
    """
    Get historical price data for a specific ASIN.
    Enables price trend analysis and competitor monitoring.
    """
    history = await market_ingester.get_price_history(asin, days=days)
    
    return PriceHistoryResponse(
        asin=asin,
        price_history=history
    )


@router.get("/keyword-rankings/{keyword}", response_model=KeywordRankingsResponse, summary="Get Keyword Rankings")
async def get_keyword_rankings(
    keyword: str,
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Get current product rankings for a specific keyword.
    Shows which products rank highest for a given search term.
    """
    rankings = await market_ingester.get_keyword_rankings(keyword, limit=limit)
    
    return KeywordRankingsResponse(
        keyword=keyword,
        rankings=rankings
    )


@router.get("/competitors", summary="List Tracked Competitors")
async def list_competitors(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200)
):
    """
    List all products marked as competitors.
    """
    from sqlalchemy import select
    from app.models.market_intelligence import MarketProduct
    
    result = await db.execute(
        select(MarketProduct)
        .where(MarketProduct.is_competitor == True)
        .order_by(MarketProduct.last_updated_at.desc())
        .limit(limit)
    )
    products = result.scalars().all()
    
    return {
        "count": len(products),
        "competitors": [
            {
                "asin": p.asin,
                "title": p.title,
                "brand": p.brand,
                "first_seen": p.first_seen_at.isoformat() if p.first_seen_at else None,
                "last_updated": p.last_updated_at.isoformat() if p.last_updated_at else None
            }
            for p in products
        ]
    }


@router.post("/track-product", summary="Mark Product for Tracking")
async def track_product(
    asin: str,
    is_competitor: bool = True,
    is_our_product: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Manually mark a product for tracking.
    Use this to add your own products or specific competitor ASINs.
    """
    from sqlalchemy import select
    from app.models.market_intelligence import MarketProduct
    
    result = await db.execute(
        select(MarketProduct).where(MarketProduct.asin == asin)
    )
    product = result.scalar_one_or_none()
    
    if product:
        product.is_competitor = is_competitor
        product.is_our_product = is_our_product
        await db.commit()
        return {"status": "updated", "asin": asin}
    else:
        # Create a placeholder - will be filled on next search
        new_product = MarketProduct(
            asin=asin,
            is_competitor=is_competitor,
            is_our_product=is_our_product
        )
        db.add(new_product)
        await db.commit()
        return {"status": "created", "asin": asin, "note": "Product will be enriched on next search"}


@router.get("/summary", response_model=MarketIntelligenceSummary, summary="Get Dashboard Summary")
async def get_market_intelligence_summary():
    # Mock data based on the stitch template
    rivals = [
        RivalProduct(
            asin="B08XJ92L1",
            name="Ultra SmartWatch 4",
            brand="Anker Soundcore",
            price=49.99,
            image="https://lh3.googleusercontent.com/aida-public/AB6AXuASUfY8kNEchcIlkN_506CDnHn1x0wLaW2JET-UMouGy6I03PM8Xm9bpEZs2OPOwtgY5SewpaTKFVyD09YlTi6BiiQi1D0ko3VZoMJZqfe3F5Az10PYnR53puo3LmXSJoqecQAwFJFwM6OTdeI4-Nqk3F7TQEv20_hTyuBQsCdsQUp6V7OCOetUEZ6JwsbO0MZysTPLgNwZSe4sPS3Ibnb9K7_bQ0c_KC0fcC8QL7S56fCpuMX494GQoaYcfGydy2dMU59F9CCQSPkL",
            rank=45,
            rank_trend="up",
            price_trend=-5.0,
            trend_percentage=70.0,
            sov_score=65
        ),
        RivalProduct(
            asin="B09Y2K9M2",
            name="PodLuxe Air",
            brand="LuxeAudio",
            price=129.00,
            image="https://lh3.googleusercontent.com/aida-public/AB6AXuAgApykwZy8lV-Q11-prelb87fMbyPHn84xPykaai4WG2K6ZXirUTCY1P2MYqR5i1DJkGgt3FXYTotoYLV-u0SA-x56GlqM65UFHHFJzT0A-smqu_R0CFNOO4IcbnT60qmh6CLRLTLczAfydAmCKNppoKCN6WT73hqTkA-xTQRjgMA9nY7M2SMXUQCiwHzPec536yQAeuxYASOs7jJ_PxDPMhXaMzmZqHWn64pV3vhToonNmQQ052lUzEJkjSwd_H-_77gEJK3bRvvg",
            rank=12,
            rank_trend="down",
            price_trend=0.0,
            trend_percentage=55.0,
            sov_score=40
        ),
        RivalProduct(
            asin="B07Z88H1Q",
            name="SportFlex 300",
            brand="FitGear Inc.",
            price=34.95,
            image="https://lh3.googleusercontent.com/aida-public/AB6AXuAS2yb_BrWvp3pZtq6PXge_IzBrgwL9HPhKXRUeqaR3CN7ptrGs8Dz_r1D_xu9reBK9uMcOP5-INx-QGEzeHBjzhSA5IZ8rFaPf88vQ7SyQwmjPFglw7KR8hEO1CBI5lOyj2pwPUH0J7jUre0QGMun9uaYYW21IFGdKYev3QGbDAYEX6d-CQxcpyE2CKFYWwpBmODOk4kJR3B0dxXlbpeBcWufiYM--qrToNU-psIoouMaLjmxzAqpFvTs48o571GZblzrlJ_-c-LgY",
            rank=89,
            rank_trend="up",
            price_trend=2.5,
            trend_percentage=90.0,
            sov_score=85
        )
    ]

    return MarketIntelligenceSummary(
        est_market_share=24.5,
        market_share_trend=2.1,
        active_threats=12,
        threats_trend=3,
        avg_rival_price=34.99,
        rival_price_trend=-5.4,
        category_rank=42,
        rank_change=2,
        rivals=rivals
    )
