"""
Market Intelligence Models
Stores competitor data, price history, and keyword rankings from DataForSEO
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class MarketProduct(Base):
    """
    Stores discovered products from Amazon searches.
    This is the master record for a tracked ASIN.
    """
    __tablename__ = "market_products"

    id = Column(Integer, primary_key=True, index=True)
    asin = Column(String(20), unique=True, nullable=False, index=True)
    title = Column(Text)
    brand = Column(String(255))
    category = Column(String(255))
    image_url = Column(Text)
    product_url = Column(Text)
    
    # Relationship flags
    is_competitor = Column(Boolean, default=False)
    is_our_product = Column(Boolean, default=False)
    
    # Timestamps
    first_seen_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    price_history = relationship("CompetitorPrice", back_populates="product", cascade="all, delete-orphan")
    keyword_rankings = relationship("KeywordRanking", back_populates="product", cascade="all, delete-orphan")


class CompetitorPrice(Base):
    """
    Time-series of price changes for tracked ASINs.
    Enables price war detection and historical analysis.
    """
    __tablename__ = "competitor_prices"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("market_products.id"), nullable=False)
    
    # Price data
    price = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    
    # Deal/Promotion flags
    is_deal = Column(Boolean, default=False)
    deal_type = Column(String(50))  # 'lightning', 'coupon', 'prime_exclusive', etc.
    discount_percent = Column(Float)
    
    # Stock status
    in_stock = Column(Boolean, default=True)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationship
    product = relationship("MarketProduct", back_populates="price_history")


class KeywordRanking(Base):
    """
    Tracks where products rank for specific keywords over time.
    Essential for SEO and organic visibility analysis.
    """
    __tablename__ = "keyword_rankings"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("market_products.id"), nullable=False)
    
    # Keyword info
    keyword = Column(String(500), nullable=False, index=True)
    
    # Ranking data
    rank_position = Column(Integer)  # Position in search results (1-indexed)
    rank_page = Column(Integer)  # Which page of results
    
    # Visibility metrics from that search
    rating = Column(Float)
    reviews_count = Column(Integer)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationship
    product = relationship("MarketProduct", back_populates="keyword_rankings")


class MarketKeywordVolume(Base):
    """
    Stores keyword search volume and CPC data from DataForSEO.
    Used for market sizing and opportunity analysis.
    """
    __tablename__ = "market_keyword_volumes"

    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String(500), nullable=False, index=True)
    
    # Volume and competition data
    search_volume = Column(Integer)
    cpc = Column(Float)
    competition = Column(Float)  # 0-1 scale
    
    # Location context
    location_code = Column(Integer, default=2840)  # US default
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())


# Indexes for efficient querying
Index('idx_competitor_prices_product_date', CompetitorPrice.product_id, CompetitorPrice.recorded_at)
Index('idx_keyword_rankings_keyword_date', KeywordRanking.keyword, KeywordRanking.recorded_at)
Index('idx_market_keyword_volumes_keyword', MarketKeywordVolume.keyword)
