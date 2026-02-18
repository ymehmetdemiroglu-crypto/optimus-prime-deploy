"""
Semantic Analytics Engine — The Intelligence Layer

Contains:
- SemanticIngestor: Pulls search terms from existing tables, generates embeddings, stores them.
- BleedDetector: Finds search terms that are semantically distant from the product (wasted spend).
- OpportunityFinder: Finds high-similarity terms with conversions that aren't being targeted yet.
"""
import logging
from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embeddings import embedding_service
from app.models.semantic import (
    SearchTermEmbedding, ProductEmbedding,
    SemanticBleedLog, SemanticOpportunityLog
)

logger = logging.getLogger("semantic_engine")


class SemanticIngestor:
    """
    Pulls raw search term data from the existing `search_term_reports` table,
    generates vector embeddings, and stores them in `search_term_embeddings`.
    
    This is the data pipeline that feeds all semantic analysis.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def ingest_search_terms(
        self,
        account_id: int,
        limit: int = 500
    ) -> int:
        """
        Process un-embedded search terms for an account.
        
        Steps:
        1. Query search_term_reports for terms not yet in search_term_embeddings
        2. Generate embeddings in batch
        3. Insert into search_term_embeddings
        
        Returns: Number of terms processed.
        """
        # Find terms not yet embedded
        query = text("""
            SELECT 
                str.search_term,
                str.campaign_id,
                SUM(str.impressions) as total_impressions,
                SUM(str.clicks) as total_clicks,
                SUM(str.cost) as total_spend,
                SUM(str.sales) as total_sales,
                SUM(str.orders) as total_orders,
                CASE WHEN SUM(str.sales) > 0 
                     THEN ROUND(SUM(str.cost) / SUM(str.sales) * 100, 2)
                     ELSE NULL END as acos
            FROM search_term_reports str
            JOIN ppc_campaigns c ON str.campaign_id = c.id
            JOIN profiles p ON c.profile_id = p.profile_id
            WHERE p.account_id = :account_id
              AND str.search_term NOT IN (
                  SELECT term FROM search_term_embeddings WHERE account_id = :account_id
              )
            GROUP BY str.search_term, str.campaign_id
            ORDER BY total_spend DESC
            LIMIT :limit
        """)
        
        result = await self.db.execute(query, {
            "account_id": account_id,
            "limit": limit
        })
        rows = result.fetchall()
        
        if not rows:
            logger.info(f"No new search terms to process for account {account_id}")
            return 0
        
        # Extract texts and generate embeddings in batch
        terms = [row.search_term for row in rows]
        logger.info(f"Generating embeddings for {len(terms)} search terms...")
        embeddings = embedding_service.embed_batch(terms)
        
        # Insert into search_term_embeddings
        objects = []
        for row, emb in zip(rows, embeddings):
            obj = SearchTermEmbedding(
                term=row.search_term,
                embedding=emb,
                account_id=account_id,
                campaign_id=row.campaign_id,
                impressions=row.total_impressions or 0,
                clicks=row.total_clicks or 0,
                spend=row.total_spend or 0,
                sales=row.total_sales or 0,
                orders=row.total_orders or 0,
                acos=row.acos,
            )
            objects.append(obj)
        
        self.db.add_all(objects)
        await self.db.commit()
        
        logger.info(f"Ingested {len(objects)} search term embeddings for account {account_id}")
        return len(objects)
    
    async def embed_product(
        self,
        asin: str,
        title: str,
        bullet_points: Optional[List[str]] = None,
        account_id: Optional[int] = None
    ) -> ProductEmbedding:
        """
        Generate and store the semantic identity of a product.
        The source text is the title + bullet points concatenated.
        """
        source_text = title
        if bullet_points:
            source_text += " " + " ".join(bullet_points)
        
        embedding = embedding_service.embed_text(source_text)
        
        # Upsert: update if exists, insert if not
        existing = await self.db.execute(
            select(ProductEmbedding).where(
                ProductEmbedding.asin == asin,
                ProductEmbedding.account_id == account_id
            )
        )
        existing_row = existing.scalars().first()
        
        if existing_row:
            existing_row.title = title
            existing_row.source_text = source_text
            existing_row.embedding = embedding
            existing_row.updated_at = datetime.now(timezone.utc)
            await self.db.commit()
            logger.info(f"Updated product embedding for {asin}")
            return existing_row
        else:
            product_emb = ProductEmbedding(
                asin=asin,
                title=title,
                source_text=source_text,
                embedding=embedding,
                account_id=account_id
            )
            self.db.add(product_emb)
            await self.db.commit()
            logger.info(f"Created product embedding for {asin}")
            return product_emb


class BleedDetector:
    """
    Detects Search Term Bleed — search terms that Amazon's broad/auto matching
    is triggering your ads for, but which are semantically distant from your product.

    "organic dog treats" → "organic dog shampoo" = BLEED (semantically distant)

    Category-aware thresholds
    -------------------------
    A single global cosine-similarity threshold (0.40) is too coarse across
    product categories with very different semantic density:
    - Electronics / software: narrow vocabulary, high inter-product similarity.
      A strict threshold (0.55) catches truly irrelevant traffic.
    - Grocery / food: broad vocabulary, loose matching is expected at 0.30.
    - Apparel / fashion: style descriptors overlap heavily across products (0.35).
    - Books / media: titles and genres are semantically close (0.30).
    - Default (all others): 0.40 — original conservative baseline.

    Callers can pass `product_category` to automatically select the right threshold,
    or continue to pass `similarity_threshold` explicitly to override per-call.
    """

    # Per-category default cosine-similarity thresholds.
    # Terms with similarity BELOW the threshold are flagged as bleed.
    # Higher threshold = stricter (flags more terms as bleed).
    CATEGORY_THRESHOLDS: Dict[str, float] = {
        "electronics":    0.55,
        "software":       0.55,
        "computers":      0.50,
        "appliances":     0.48,
        "tools":          0.45,
        "sports":         0.42,
        "toys":           0.42,
        "health":         0.40,
        "beauty":         0.38,
        "home":           0.38,
        "garden":         0.38,
        "office":         0.38,
        "pet_supplies":   0.40,
        "automotive":     0.45,
        "apparel":        0.35,
        "shoes":          0.35,
        "jewelry":        0.35,
        "grocery":        0.30,
        "food":           0.30,
        "beverages":      0.30,
        "books":          0.30,
        "movies":         0.28,
        "music":          0.28,
        "video_games":    0.38,
        # Default for unknown/uncategorised products
        "default":        0.40,
    }

    def get_threshold_for_category(
        self,
        product_category: Optional[str],
        override: Optional[float],
    ) -> float:
        """Resolve effective bleed threshold.

        Priority:
          1. Explicit ``override`` value (caller-supplied)
          2. Category-specific lookup from CATEGORY_THRESHOLDS
          3. Default (0.40)
        """
        if override is not None:
            return override
        if product_category:
            key = product_category.lower().replace(" ", "_").replace("-", "_")
            return self.CATEGORY_THRESHOLDS.get(key, self.CATEGORY_THRESHOLDS["default"])
        return self.CATEGORY_THRESHOLDS["default"]

    def __init__(self, db: AsyncSession):
        self.db = db

    async def detect_bleed(
        self,
        asin: str,
        account_id: int,
        similarity_threshold: Optional[float] = None,
        product_category: Optional[str] = None,
        min_spend: float = 1.00,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Find search terms that are wasting budget.

        Args:
            asin: The product ASIN to check bleed against.
            account_id: Client account ID.
            similarity_threshold: Explicit threshold override. If None, the threshold
                is derived from ``product_category`` via CATEGORY_THRESHOLDS.
            product_category: Product category string (e.g. 'electronics', 'grocery').
                Used to pick the appropriate default threshold when no explicit
                ``similarity_threshold`` is provided.
            min_spend: Minimum cumulative spend to include a term (filters noise).
            limit: Max results.

        Returns:
            List of bleed candidates with term, similarity, spend, urgency, etc.
        """
        effective_threshold = self.get_threshold_for_category(product_category, similarity_threshold)

        query = text("""
            SELECT
                ste.id as embedding_id,
                ste.term,
                ROUND((1 - (ste.embedding <=> pe.embedding))::NUMERIC, 4) AS similarity,
                ste.spend,
                ste.clicks,
                ste.impressions,
                ste.acos,
                pe.id as product_embedding_id
            FROM search_term_embeddings ste
            CROSS JOIN product_embeddings pe
            WHERE pe.asin = :asin
              AND pe.account_id = :account_id
              AND ste.account_id = :account_id
              AND ste.spend >= :min_spend
              AND (1 - (ste.embedding <=> pe.embedding)) < :threshold
            ORDER BY ste.spend DESC
            LIMIT :limit
        """)

        result = await self.db.execute(query, {
            "asin": asin,
            "account_id": account_id,
            "threshold": effective_threshold,
            "min_spend": min_spend,
            "limit": limit,
        })
        rows = result.fetchall()

        # Relative urgency: use total waste as denominator so that
        # a $5 term in a $20 account is HIGH; the same $5 in a $5000 account is LOW.
        # Absolute floors prevent all terms in low-spend / single-result sets
        # from being classified HIGH regardless of actual dollar impact.
        total_waste = sum(float(r.spend) for r in rows)
        high_cutoff = max(total_waste * 0.20, 50.0)   # ≥20% of waste AND ≥$50 → HIGH
        medium_cutoff = max(total_waste * 0.05, 10.0)  # ≥5% of waste AND ≥$10 → MEDIUM

        bleed_items = []
        for row in rows:
            spend = float(row.spend)
            if spend >= high_cutoff:
                urgency = "HIGH"
            elif spend >= medium_cutoff:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"

            bleed_items.append({
                "embedding_id": str(row.embedding_id),
                "product_embedding_id": str(row.product_embedding_id),
                "term": row.term,
                "semantic_similarity": float(row.similarity),
                "spend": spend,
                "clicks": row.clicks,
                "impressions": row.impressions,
                "acos": float(row.acos) if row.acos else None,
                "recommendation": "ADD_NEGATIVE",
                "urgency": urgency,
                "effective_threshold": effective_threshold,
            })

        logger.info(
            f"Bleed scan for {asin}: found {len(bleed_items)} bleeding terms "
            f"(category={product_category!r}, threshold={effective_threshold}, min_spend=${min_spend})"
        )
        return bleed_items
    
    async def log_bleed_action(
        self,
        search_term_embedding_id: str,
        product_embedding_id: str,
        semantic_distance: float,
        spend: float,
        action: str = "negative_added",
        operator: str = "autonomous"
    ):
        """Record a bleed action in the audit log."""
        import uuid
        log_entry = SemanticBleedLog(
            search_term_embedding_id=uuid.UUID(search_term_embedding_id),
            product_embedding_id=uuid.UUID(product_embedding_id),
            semantic_distance=Decimal(str(semantic_distance)),
            spend_at_detection=Decimal(str(spend)),
            action_taken=action,
            operator=operator
        )
        self.db.add(log_entry)
        await self.db.commit()


class OpportunityFinder:
    """
    Finds untapped semantic opportunities — search terms that are
    semantically close to your product AND converting, but you're not
    explicitly targeting them yet.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def find_opportunities(
        self,
        asin: str,
        account_id: int,
        similarity_floor: float = 0.70,
        min_orders: int = 1,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Find high-value semantic clusters you're missing.
        
        Args:
            asin: The product ASIN.
            account_id: Client account ID.
            similarity_floor: Minimum similarity to be considered relevant.
            min_orders: Minimum conversions to prove demand.
            limit: Max results.
            
        Returns:
            List of opportunity candidates.
        """
        query = text("""
            SELECT 
                ste.term,
                ROUND((1 - (ste.embedding <=> pe.embedding))::NUMERIC, 4) AS similarity,
                ste.impressions,
                ste.clicks,
                ste.sales,
                ste.orders,
                ste.acos
            FROM search_term_embeddings ste
            CROSS JOIN product_embeddings pe
            WHERE pe.asin = :asin
              AND pe.account_id = :account_id
              AND ste.account_id = :account_id
              AND (1 - (ste.embedding <=> pe.embedding)) >= :sim_floor
              AND ste.orders >= :min_orders
            ORDER BY similarity DESC, ste.sales DESC
            LIMIT :limit
        """)
        
        result = await self.db.execute(query, {
            "asin": asin,
            "account_id": account_id,
            "sim_floor": similarity_floor,
            "min_orders": min_orders,
            "limit": limit
        })
        rows = result.fetchall()
        
        opportunities = []
        for row in rows:
            suggested_bid = self._calculate_suggested_bid(
                float(row.sales or 0), row.clicks or 1, float(row.acos or 30)
            )
            opportunities.append({
                "term": row.term,
                "semantic_similarity": float(row.similarity),
                "impressions": row.impressions,
                "clicks": row.clicks,
                "sales": float(row.sales) if row.sales else 0,
                "orders": row.orders,
                "acos": float(row.acos) if row.acos else None,
                "suggested_match_type": "exact" if float(row.similarity) > 0.85 else "phrase",
                "suggested_bid": suggested_bid,
                "recommendation": "ADD_AS_TARGET",
                "confidence": "HIGH" if float(row.similarity) > 0.85 and row.orders >= 3 else "MEDIUM"
            })
        
        logger.info(
            f"Opportunity scan for {asin}: found {len(opportunities)} targets "
            f"(similarity>={similarity_floor}, min_orders={min_orders})"
        )
        return opportunities
    
    def _calculate_suggested_bid(self, sales: float, clicks: int, target_acos: float) -> float:
        """Calculate a suggested bid based on historical performance."""
        if clicks == 0 or sales == 0:
            return 0.50  # Conservative default
        
        cpc = sales / clicks  # Revenue per click
        suggested = cpc * (target_acos / 100.0)
        return round(max(0.10, min(suggested, 5.00)), 2)  # Clamp to [0.10, 5.00]
    
    async def log_opportunity(
        self,
        term: str,
        asin: str,
        similarity: float,
        match_type: str = "exact",
        bid: float = 0.50
    ):
        """Record a discovered opportunity."""
        entry = SemanticOpportunityLog(
            term=term,
            closest_product_asin=asin,
            semantic_similarity=Decimal(str(similarity)),
            suggested_match_type=match_type,
            suggested_bid=Decimal(str(bid)),
            status="discovered"
        )
        self.db.add(entry)
        await self.db.commit()
