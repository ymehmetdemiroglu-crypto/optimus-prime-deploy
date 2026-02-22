"""
SQLAlchemy models for the Semantic Intelligence Layer.
These map to the tables in migrations/06_semantic_schema.sql.
"""
from sqlalchemy import Column, String, Integer, Numeric, Text, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pgvector.sqlalchemy import Vector
from app.core.database import Base
import uuid
from datetime import datetime, timezone


class SearchTermEmbedding(Base):
    __tablename__ = "search_term_embeddings"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    term = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # MiniLM-L6-v2 output dim
    account_id = Column(Integer, ForeignKey("accounts.id"))
    campaign_id = Column(Integer, ForeignKey("ppc_campaigns.id"))
    source = Column(String(50), default="search_query_report")
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    spend = Column(Numeric(15, 2), default=0)
    sales = Column(Numeric(15, 2), default=0)
    orders = Column(Integer, default=0)
    acos = Column(Numeric(8, 4))
    # Rufus/Cosmo intent classification
    intent_type = Column(String(30), default="unclassified")   # transactional, informational_rufus, navigational, discovery
    query_source = Column(String(20), default="organic")       # organic, rufus, cosmo_matched
    intent_confidence = Column(Numeric(6, 4))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ProductEmbedding(Base):
    __tablename__ = "product_embeddings"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asin = Column(String(20), nullable=False)
    title = Column(Text)
    source_text = Column(Text)
    embedding = Column(Vector(384))
    account_id = Column(Integer, ForeignKey("accounts.id"))
    # --- Rich Product Metadata (Cosmo Alignment) ---
    brand = Column(String(255))
    category_path = Column(Text)                       # "Home & Kitchen > Kitchen > Coffee"
    product_type = Column(String(100))                 # Amazon product type / browse node label
    bullet_points = Column(JSON)                       # preserved as list for downstream use
    attributes = Column(JSON)                          # {"size": "12oz", "material": "stainless steel", ...}
    price = Column(Numeric(10, 2))
    review_score = Column(Numeric(3, 2))               # 1.00 – 5.00
    review_count = Column(Integer, default=0)
    parent_asin = Column(String(20))                   # variation parent ASIN
    embedding_version = Column(String(30), default="v1_title_only")  # tracks which enrichment produced the vector
    cosmo_alignment_score = Column(Numeric(6, 4))      # self-computed alignment confidence
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class SemanticBleedLog(Base):
    __tablename__ = "semantic_bleed_log"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    search_term_embedding_id = Column(PG_UUID(as_uuid=True), ForeignKey("search_term_embeddings.id"))
    product_embedding_id = Column(PG_UUID(as_uuid=True), ForeignKey("product_embeddings.id"))
    semantic_distance = Column(Numeric(6, 4), nullable=False)
    spend_at_detection = Column(Numeric(15, 2), default=0)
    action_taken = Column(String(50), default="flagged")
    operator = Column(String(50), default="autonomous")
    detected_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SemanticOpportunityLog(Base):
    __tablename__ = "semantic_opportunity_log"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    term = Column(Text, nullable=False)
    closest_product_asin = Column(String(20))
    semantic_similarity = Column(Numeric(6, 4), nullable=False)
    estimated_monthly_volume = Column(Integer)
    suggested_match_type = Column(String(20), default="exact")
    suggested_bid = Column(Numeric(10, 2))
    status = Column(String(50), default="discovered")
    discovered_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AutonomousPatrolLog(Base):
    __tablename__ = "autonomous_patrol_log"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patrol_cycle = Column(Integer, nullable=False)
    action_type = Column(String(50), nullable=False)
    target_entity = Column(String(255))
    details = Column(JSON)
    status = Column(String(50), default="success")
    executed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
