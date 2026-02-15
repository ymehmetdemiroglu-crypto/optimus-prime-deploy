"""
Market Intelligence Ingestion Service
Persists data from DataForSEO into the database for long-term analysis.
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.market_intelligence import (
    MarketProduct,
    CompetitorPrice,
    KeywordRanking,
    MarketKeywordVolume
)
from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


class MarketIntelligenceIngester:
    """
    Service to ingest and persist market intelligence data.
    """
    
    async def ingest_amazon_products(
        self, 
        keyword: str, 
        products: List[Dict],
        mark_as_competitors: bool = True
    ) -> Dict:
        """
        Ingest products from an Amazon search result.
        
        Args:
            keyword: The search keyword used
            products: List of product dicts from DataForSEO
            mark_as_competitors: Whether to flag these as competitor products
            
        Returns:
            Summary of ingestion results
        """
        async with AsyncSessionLocal() as session:
            # OPTIMIZATION: Process all items in a single transaction block
            # This ensures swift execution and minimizes database round-trips.
            ingested = 0
            updated = 0
            prices_recorded = 0
            rankings_recorded = 0
            
            for idx, product_data in enumerate(products):
                asin = product_data.get("asin")
                if not asin:
                    continue
                
                # Upsert product
                existing = await session.execute(
                    select(MarketProduct).where(MarketProduct.asin == asin)
                )
                existing_product = existing.scalar_one_or_none()
                
                if existing_product:
                    # Update existing
                    existing_product.title = product_data.get("title") or existing_product.title
                    existing_product.last_updated_at = datetime.utcnow()
                    if mark_as_competitors:
                        existing_product.is_competitor = True
                    product_record = existing_product
                    updated += 1
                else:
                    # Create new
                    product_record = MarketProduct(
                        asin=asin,
                        title=product_data.get("title"),
                        brand=product_data.get("brand"),
                        product_url=product_data.get("url"),
                        is_competitor=mark_as_competitors
                    )
                    session.add(product_record)
                    await session.flush()  # Get the ID
                    ingested += 1
                
                # Record current price
                price = product_data.get("price")
                if price is not None:
                    price_record = CompetitorPrice(
                        product_id=product_record.id,
                        price=float(price),
                        currency="USD",
                        in_stock=True
                    )
                    session.add(price_record)
                    prices_recorded += 1
                
                # Record keyword ranking
                ranking_record = KeywordRanking(
                    product_id=product_record.id,
                    keyword=keyword,
                    rank_position=idx + 1,  # 1-indexed position
                    rank_page=1,  # Assuming first page for now
                    rating=product_data.get("rating"),
                    reviews_count=product_data.get("reviews_count")
                )
                session.add(ranking_record)
                rankings_recorded += 1
            
            await session.commit()
            
            summary = {
                "keyword": keyword,
                "products_ingested": ingested,
                "products_updated": updated,
                "prices_recorded": prices_recorded,
                "rankings_recorded": rankings_recorded,
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info(f"Market intelligence ingested: {summary}")
            return summary
    
    async def ingest_keyword_volume(
        self,
        keyword: str,
        volume_data: Dict,
        location_code: int = 2840
    ) -> Dict:
        """
        Ingest keyword volume data from DataForSEO.
        
        Args:
            keyword: The keyword
            volume_data: Dict with 'volume', 'cpc', 'competition' keys
            location_code: Location code (default US)
            
        Returns:
            Summary of ingestion
        """
        async with AsyncSessionLocal() as session:
            record = MarketKeywordVolume(
                keyword=keyword,
                search_volume=volume_data.get("volume"),
                cpc=volume_data.get("cpc"),
                competition=volume_data.get("competition"),
                location_code=location_code
            )
            session.add(record)
            await session.commit()
            
            summary = {
                "keyword": keyword,
                "volume": volume_data.get("volume"),
                "cpc": volume_data.get("cpc"),
                "recorded": True
            }
            logger.info(f"Keyword volume ingested: {summary}")
            return summary
    
    async def get_price_history(
        self,
        asin: str,
        days: int = 30
    ) -> List[Dict]:
        """
        Get price history for a product.
        
        Args:
            asin: The ASIN to query
            days: Number of days of history
            
        Returns:
            List of price records
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CompetitorPrice)
                .join(MarketProduct)
                .where(MarketProduct.asin == asin)
                .order_by(CompetitorPrice.recorded_at.desc())
                .limit(days * 24)  # Assuming hourly max
            )
            records = result.scalars().all()
            
            return [
                {
                    "price": r.price,
                    "currency": r.currency,
                    "is_deal": r.is_deal,
                    "in_stock": r.in_stock,
                    "recorded_at": r.recorded_at.isoformat()
                }
                for r in records
            ]
    
    async def get_keyword_rankings(
        self,
        keyword: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get current rankings for a keyword.
        
        Args:
            keyword: The keyword to query
            limit: Max results
            
        Returns:
            List of ranking records with product info
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(KeywordRanking, MarketProduct)
                .join(MarketProduct)
                .where(KeywordRanking.keyword == keyword)
                .order_by(KeywordRanking.recorded_at.desc())
                .limit(limit)
            )
            records = result.all()
            
            return [
                {
                    "asin": product.asin,
                    "title": product.title,
                    "rank_position": ranking.rank_position,
                    "rating": ranking.rating,
                    "reviews_count": ranking.reviews_count,
                    "recorded_at": ranking.recorded_at.isoformat()
                }
                for ranking, product in records
            ]


# Singleton instance
market_ingester = MarketIntelligenceIngester()
