from sqlalchemy import Column, String, Float, DateTime, Enum, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum

class DSPAudienceType(str, enum.Enum):
    RETARGETING = "retargeting"
    LOOKALIKE = "lookalike"
    COMPETITOR_CONQUEST = "competitor_conquest"
    LIFESTYLE = "lifestyle"

class DSPCampaign(Base):
    __tablename__ = "dsp_campaigns"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    status = Column(String, default="active")
    
    # DSP Specific Metrics
    budget = Column(Float)
    cpm = Column(Float, default=0.73) # Standard from your strategy
    reach = Column(Integer := 0)
    frequency = Column(Float, default=1.0)
    ntb_sales_percent = Column(Float, default=0.0)
    
    # Strategy Context
    strategy_type = Column(Enum(DSPAudienceType))
    target_asins = Column(JSON, nullable=True) # For Conquest/Halo Hijack
    
    last_optimized = Column(DateTime, nullable=True)

class DSPAudience(Base):
    __tablename__ = "dsp_audiences"

    id = Column(String, primary_key=True)
    name = Column(String)
    audience_type = Column(Enum(DSPAudienceType))
    size_estimate = Column(String)
    relevance_score = Column(Float) # AI dynamic score
