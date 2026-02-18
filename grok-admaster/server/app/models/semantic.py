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


class ActionReviewQueue(Base):
    """
    Staging table for autonomous recommendations pending human approval.

    Every bleed / opportunity action is written here with
    status='pending_review' before it is applied downstream.
    An admin approves or rejects via the /operator-actions API.

    Status transitions:
        pending_review → approved  (cleared for execution)
        pending_review → rejected  (discarded)
        approved       → executed  (applied to Amazon / downstream system)
    """
    __tablename__ = "action_review_queue"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patrol_cycle = Column(Integer, nullable=False)
    account_id = Column(Integer)
    asin = Column(String(20))
    action_type = Column(String(50), nullable=False)   # 'add_negative' | 'add_target'
    term = Column(Text, nullable=False)
    semantic_similarity = Column(Numeric(6, 4))
    spend_at_detection = Column(Numeric(15, 2))
    suggested_bid = Column(Numeric(10, 2))
    suggested_match_type = Column(String(20))
    urgency = Column(String(20), default="MEDIUM")
    status = Column(String(30), default="pending_review")
    reviewed_by = Column(String(255))
    reviewed_at = Column(DateTime(timezone=True))
    review_note = Column(Text)
    details = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
