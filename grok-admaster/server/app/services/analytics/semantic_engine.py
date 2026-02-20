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
from app.services.ml.intent_classifier import (
    get_intent_classifier, get_intent_thresholds, ShoppingIntent
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
        
        # Classify intent for each search term
        classifier = get_intent_classifier()
        intent_results = classifier.classify_batch(terms)

        # Insert into search_term_embeddings
        objects = []
        for row, emb, intent_result in zip(rows, embeddings, intent_results):
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
                intent_type=intent_result.intent.value,
                intent_confidence=Decimal(str(round(intent_result.confidence, 4))),
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
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def detect_bleed(
        self,
        asin: str,
        account_id: int,
        similarity_threshold: float = 0.40,
        min_spend: float = 1.00,
        limit: int = 50,
        intent_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find search terms that are wasting budget.

        When intent_aware=True, each result is re-evaluated against
        intent-specific thresholds so Rufus/discovery traffic isn't
        incorrectly flagged as bleed.

        Args:
            asin: The product ASIN to check bleed against.
            account_id: Client account ID.
            similarity_threshold: Default threshold (used as ceiling for intent-aware mode).
            min_spend: Minimum spend to consider (filters noise).
            limit: Max results.
            intent_aware: Apply Rufus/Cosmo-aware threshold adjustments.

        Returns:
            List of bleed candidates with term, similarity, spend, etc.
        """
        # Use the most lenient threshold for the SQL query to cast a wide net;
        # intent-specific filtering happens post-query.
        sql_threshold = similarity_threshold
        if intent_aware:
            sql_threshold = max(
                similarity_threshold,
                max(t["bleed_threshold"] for t in get_intent_thresholds.__func__.__defaults__ or [similarity_threshold])  # type: ignore[union-attr]
            ) if False else similarity_threshold  # keep original for SQL; filter below

        query = text("""
            SELECT
                ste.id as embedding_id,
                ste.term,
                ROUND((1 - (ste.embedding <=> pe.embedding))::NUMERIC, 4) AS similarity,
                ste.spend,
                ste.clicks,
                ste.impressions,
                ste.acos,
                ste.intent_type,
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
            "threshold": similarity_threshold,
            "min_spend": min_spend,
            "limit": limit
        })
        rows = result.fetchall()

        classifier = get_intent_classifier() if intent_aware else None

        bleed_items = []
        for row in rows:
            sim = float(row.similarity)
            intent_str = getattr(row, "intent_type", None) or "unclassified"

            # Intent-aware filtering: reclassify if needed, then check threshold
            if intent_aware and classifier:
                if intent_str == "unclassified":
                    intent_result = classifier.classify(row.term)
                    intent_str = intent_result.intent.value
                try:
                    intent_enum = ShoppingIntent(intent_str)
                except ValueError:
                    intent_enum = ShoppingIntent.TRANSACTIONAL
                intent_thresh = get_intent_thresholds(intent_enum)["bleed_threshold"]
                # If similarity is above the intent-adjusted threshold, it's NOT bleed
                if sim >= intent_thresh:
                    continue

            bleed_items.append({
                "embedding_id": str(row.embedding_id),
                "product_embedding_id": str(row.product_embedding_id),
                "term": row.term,
                "semantic_similarity": sim,
                "spend": float(row.spend),
                "clicks": row.clicks,
                "impressions": row.impressions,
                "acos": float(row.acos) if row.acos else None,
                "intent_type": intent_str,
                "recommendation": "ADD_NEGATIVE",
                "urgency": "HIGH" if float(row.spend) > 10 else "MEDIUM"
            })

        logger.info(
            f"Bleed scan for {asin}: found {len(bleed_items)} bleeding terms "
            f"(threshold={similarity_threshold}, intent_aware={intent_aware}, min_spend=${min_spend})"
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
        limit: int = 30,
        intent_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find high-value semantic clusters you're missing.

        When intent_aware=True, Rufus/discovery terms are included at a
        lower similarity floor since conversational queries naturally
        have lower cosine similarity to product embeddings.

        Args:
            asin: The product ASIN.
            account_id: Client account ID.
            similarity_floor: Default similarity floor for transactional terms.
            min_orders: Minimum conversions to prove demand.
            limit: Max results.
            intent_aware: Apply Rufus/Cosmo-aware floor adjustments.

        Returns:
            List of opportunity candidates.
        """
        # Use the lowest intent-aware floor in SQL to capture more candidates
        sql_floor = similarity_floor
        if intent_aware:
            lowest_floor = min(
                t["opportunity_floor"]
                for t in [get_intent_thresholds(i) for i in ShoppingIntent]
            )
            sql_floor = min(similarity_floor, lowest_floor)

        query = text("""
            SELECT
                ste.term,
                ROUND((1 - (ste.embedding <=> pe.embedding))::NUMERIC, 4) AS similarity,
                ste.impressions,
                ste.clicks,
                ste.sales,
                ste.orders,
                ste.acos,
                ste.intent_type
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
            "sim_floor": sql_floor,
            "min_orders": min_orders,
            "limit": limit
        })
        rows = result.fetchall()

        classifier = get_intent_classifier() if intent_aware else None

        opportunities = []
        for row in rows:
            sim = float(row.similarity)
            intent_str = getattr(row, "intent_type", None) or "unclassified"

            # Intent-aware floor check
            if intent_aware and classifier:
                if intent_str == "unclassified":
                    intent_result = classifier.classify(row.term)
                    intent_str = intent_result.intent.value
                try:
                    intent_enum = ShoppingIntent(intent_str)
                except ValueError:
                    intent_enum = ShoppingIntent.TRANSACTIONAL
                intent_floor = get_intent_thresholds(intent_enum)["opportunity_floor"]
                if sim < intent_floor:
                    continue

            suggested_bid = self._calculate_suggested_bid(
                float(row.sales or 0), row.clicks or 1, float(row.acos or 30)
            )
            opportunities.append({
                "term": row.term,
                "semantic_similarity": sim,
                "impressions": row.impressions,
                "clicks": row.clicks,
                "sales": float(row.sales) if row.sales else 0,
                "orders": row.orders,
                "acos": float(row.acos) if row.acos else None,
                "intent_type": intent_str,
                "suggested_match_type": "exact" if sim > 0.85 else "phrase",
                "suggested_bid": suggested_bid,
                "recommendation": "ADD_AS_TARGET",
                "confidence": "HIGH" if sim > 0.85 and row.orders >= 3 else "MEDIUM"
            })

        logger.info(
            f"Opportunity scan for {asin}: found {len(opportunities)} targets "
            f"(floor={sql_floor}, intent_aware={intent_aware}, min_orders={min_orders})"
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
