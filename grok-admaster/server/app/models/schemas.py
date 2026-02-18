"""
Pydantic models for Optimus Prime API.
"""
from pydantic import BaseModel
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
    ai_mode: AIStrategy
    daily_budget: float
    spend: float
    sales: float
    acos: float


class CampaignStrategyUpdate(BaseModel):
    ai_mode: AIStrategy


# Chat Models
class ChatRequest(BaseModel):
    message: str
    context_asin: Optional[str] = None


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
