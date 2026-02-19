"""
Database models for PPC Campaign Performance Data.
Extended to support multi-account architecture.
"""
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Enum, Numeric, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.core.database import Base

class MatchType(str, enum.Enum):
    EXACT = "exact"
    PHRASE = "phrase"
    BROAD = "broad"
from pgvector.sqlalchemy import Vector


class KeywordState(str, enum.Enum):
    ENABLED = "enabled"
    PAUSED = "paused"
    ARCHIVED = "archived"

class AIStrategyType(str, enum.Enum):
    """
    Control modes for the AI Optimizer.
    """
    MANUAL = "manual"
    AUTO_PILOT = "auto_pilot"           # Legacy Rule-based
    AGGRESSIVE_GROWTH = "aggressive_growth"
    PROFIT_GUARD = "profit_guard"
    ADVANCED = "advanced"              # New ML-driven mode
    AUTONOMOUS = "autonomous"          # Fully self-driving

class PPCCampaign(Base):
    """
    Campaign data ingested from Amazon Ads API.
    Linked to a specific Profile (which belongs to an Account).
    """
    __tablename__ = "ppc_campaigns"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    campaign_id = Column(String, unique=True, index=True)  # Amazon Campaign ID
    profile_id = Column(String, ForeignKey("profiles.profile_id"), index=True)
    
    name = Column(String, index=True)
    campaign_type = Column(String)  # sponsoredProducts, sponsoredBrands, etc.
    targeting_type = Column(String)  # manual / auto
    state = Column(String)
    daily_budget = Column(Numeric(10, 2))
    start_date = Column(DateTime(timezone=True))
    portfolio_id = Column(String, nullable=True, index=True)  # Amazon Portfolio ID
    
    # --- Optimus Intelligence Fields (Consolidated) ---
    ai_mode = Column(Enum(AIStrategyType), default=AIStrategyType.MANUAL)
    target_acos = Column(Float, nullable=True, default=30.0)
    target_roas = Column(Float, nullable=True, default=3.0) # Added for ROAS-based strategies
    
    # Denormalized metrics snapshot (updated periodically)
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    spend = Column(Numeric(10, 2), default=0)
    sales = Column(Numeric(10, 2), default=0)
    orders = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    keywords = relationship("PPCKeyword", back_populates="campaign", cascade="all, delete-orphan")
    performance_records = relationship("PerformanceRecord", back_populates="campaign", cascade="all, delete-orphan")

class PPCKeyword(Base):
    """
    Keyword-level data for PPC campaigns.
    """
    __tablename__ = "ppc_keywords"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    keyword_id = Column(String, unique=True, index=True)  # Amazon Keyword ID
    campaign_id = Column(Integer, ForeignKey("ppc_campaigns.id"))
    
    keyword_text = Column(String, index=True)
    match_type = Column(Enum(MatchType))
    state = Column(Enum(KeywordState))
    bid = Column(Numeric(10, 2))
    
    # Denormalized metrics
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    spend = Column(Numeric(10, 2), default=0)
    sales = Column(Numeric(10, 2), default=0)
    orders = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    campaign = relationship("PPCCampaign", back_populates="keywords")
    vector = relationship("KeywordVector", uselist=False, back_populates="keyword", cascade="all, delete-orphan")

class KeywordVector(Base):
    """
    Vector embeddings for keywords.
    Separate table to keep main table light.
    """
    __tablename__ = "keyword_vectors"

    keyword_id = Column(Integer, ForeignKey("ppc_keywords.id"), primary_key=True)
    embedding = Column(Vector(384))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    keyword = relationship("PPCKeyword", back_populates="vector")


class PerformanceRecord(Base):
    """
    Time-series performance data (daily snapshots).
    """
    __tablename__ = "performance_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    campaign_id = Column(Integer, ForeignKey("ppc_campaigns.id"), index=True)
    keyword_id = Column(Integer, ForeignKey("ppc_keywords.id"), nullable=True, index=True)
    
    date = Column(DateTime(timezone=True), index=True)
    
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    spend = Column(Numeric(10, 2), default=0)
    sales = Column(Numeric(10, 2), default=0)
    orders = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    campaign = relationship("PPCCampaign", back_populates="performance_records")
