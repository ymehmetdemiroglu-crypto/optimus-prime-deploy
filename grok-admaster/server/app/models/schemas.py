"""
Pydantic models for Optimus Prime API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class AIStrategy(str, Enum):
    MANUAL = "manual"
    AUTO_PILOT = "auto_pilot"
    AGGRESSIVE = "aggressive_growth"
    PROFIT = "profit_guard"
    ADVANCED = "advanced"
    AUTONOMOUS = "autonomous"


class CampaignStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


# Dashboard Models
class DashboardSummary(BaseModel):
    total_sales: float
    ad_spend: float
    acos: float
    roas: float
    velocity_trend: TrendDirection


class PerformanceMetric(BaseModel):
    timestamp: str
    organic_sales: float
    ad_sales: float
    spend: float
    impressions: int


# Campaign Models
class Campaign(BaseModel):
    id: str
    name: str
    status: CampaignStatus
    campaign_type: Optional[str] = None
    targeting_type: Optional[str] = None
    ai_mode: AIStrategy
    daily_budget: float
    spend: float
    sales: float
    acos: float
    target_acos: Optional[float] = None
    target_roas: Optional[float] = None
    clicks: int = 0
    impressions: int = 0
    orders: int = 0


class CampaignStrategyUpdate(BaseModel):
    ai_mode: AIStrategy


# Chat Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    # ASIN format: 10 alphanumeric characters (e.g. B0DWK3C1R7)
    context_asin: Optional[str] = Field(
        default=None, pattern=r"^[A-Z0-9]{10}$"
    )


class ChatResponse(BaseModel):
    id: str
    sender: str = "optimus"
    content: str
    timestamp: datetime
    action_suggestion: Optional[dict] = None


# AI Action Feed
class AIAction(BaseModel):
    id: str
    action_type: str
    description: str
    timestamp: datetime
    campaign_name: Optional[str] = None
