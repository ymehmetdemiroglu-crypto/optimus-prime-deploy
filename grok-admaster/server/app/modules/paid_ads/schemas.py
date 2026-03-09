"""
Paid Ads Module — Schemas

Pydantic models for the multi-platform paid advertising strategy engine.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ──────────────────────────────────────────────────────────
#  Enumerations
# ──────────────────────────────────────────────────────────

class AdPlatform(str, Enum):
    GOOGLE = "google"
    META = "meta"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"
    TWITTER_X = "twitter_x"
    AMAZON = "amazon"


class CampaignObjective(str, Enum):
    AWARENESS = "awareness"
    TRAFFIC = "traffic"
    LEADS = "leads"
    SALES = "sales"
    APP_INSTALLS = "app_installs"


class BidStrategyType(str, Enum):
    MANUAL = "manual"
    TARGET_CPA = "target_cpa"
    TARGET_ROAS = "target_roas"
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    MAXIMIZE_CLICKS = "maximize_clicks"


class FunnelStage(str, Enum):
    TOP = "top_of_funnel"
    MIDDLE = "middle_of_funnel"
    BOTTOM = "bottom_of_funnel"


class AdCopyFramework(str, Enum):
    PAS = "problem_agitate_solve"
    BAB = "before_after_bridge"
    SOCIAL_PROOF = "social_proof_lead"
    AIDA = "aida"


# ──────────────────────────────────────────────────────────
#  Request Models
# ──────────────────────────────────────────────────────────

class PlatformRecommendationRequest(BaseModel):
    """Ask the engine which ad platform(s) fit best."""
    product_description: str = Field(..., min_length=10, max_length=2000)
    target_audience: str = Field(..., min_length=5)
    monthly_budget: float = Field(..., gt=0)
    objective: CampaignObjective
    has_visual_assets: bool = False
    b2b: bool = False


class CampaignStructureRequest(BaseModel):
    """Generate a recommended campaign structure."""
    platform: AdPlatform
    product_name: str
    objective: CampaignObjective
    monthly_budget: float = Field(..., gt=0)
    target_audience: str
    keywords: Optional[List[str]] = None
    landing_page_url: Optional[str] = None


class AdCopyRequest(BaseModel):
    """Generate ad copy using a specific framework."""
    platform: AdPlatform
    framework: AdCopyFramework = AdCopyFramework.PAS
    product_name: str
    unique_selling_points: List[str] = Field(..., min_length=1)
    target_audience: str
    pain_points: Optional[List[str]] = None
    tone: str = "professional"
    num_variations: int = Field(default=3, ge=1, le=10)


class BudgetAllocationRequest(BaseModel):
    """Allocate budget across campaigns or platforms."""
    total_monthly_budget: float = Field(..., gt=0)
    platforms: List[AdPlatform]
    objective: CampaignObjective
    is_testing_phase: bool = True


class RetargetingStrategyRequest(BaseModel):
    """Build a funnel-based retargeting plan."""
    platform: AdPlatform
    product_name: str
    has_pixel: bool = True
    has_customer_list: bool = False
    average_order_value: Optional[float] = None


class CampaignOptimizationRequest(BaseModel):
    """Diagnose and fix underperforming campaigns."""
    platform: AdPlatform
    current_cpa: Optional[float] = None
    target_cpa: Optional[float] = None
    current_roas: Optional[float] = None
    target_roas: Optional[float] = None
    current_ctr: Optional[float] = None
    current_cpm: Optional[float] = None
    daily_budget: Optional[float] = None
    campaign_age_days: Optional[int] = None


class SetupChecklistRequest(BaseModel):
    """Get a platform-specific setup checklist."""
    platform: AdPlatform


# ──────────────────────────────────────────────────────────
#  Response Models
# ──────────────────────────────────────────────────────────

class PlatformScore(BaseModel):
    platform: AdPlatform
    score: int = Field(..., ge=0, le=100)
    reasoning: str
    recommended_campaign_types: List[str]
    estimated_cpc_range: str


class PlatformRecommendationResponse(BaseModel):
    recommended_platforms: List[PlatformScore]
    budget_split: Dict[str, float]
    notes: List[str]


class AdSetStructure(BaseModel):
    name: str
    targeting_description: str
    budget_percentage: float
    ad_count: int


class CampaignBlueprint(BaseModel):
    campaign_name: str
    objective: str
    ad_sets: List[AdSetStructure]
    naming_convention: str
    bid_strategy: str
    notes: List[str]


class CampaignStructureResponse(BaseModel):
    platform: AdPlatform
    campaigns: List[CampaignBlueprint]
    naming_convention_template: str
    budget_allocation_notes: str


class AdCopyVariation(BaseModel):
    headline: str
    primary_text: str
    description: Optional[str] = None
    cta: str
    framework_used: str


class AdCopyResponse(BaseModel):
    platform: AdPlatform
    variations: List[AdCopyVariation]
    testing_tips: List[str]


class BudgetSplit(BaseModel):
    platform: AdPlatform
    amount: float
    percentage: float
    rationale: str


class BudgetAllocationResponse(BaseModel):
    total_budget: float
    splits: List[BudgetSplit]
    testing_reserve_percentage: float
    scaling_guidelines: List[str]


class RetargetingLayer(BaseModel):
    funnel_stage: FunnelStage
    audience_description: str
    window_days: str
    frequency_cap: str
    message_theme: str
    exclusions: List[str]


class RetargetingStrategyResponse(BaseModel):
    platform: AdPlatform
    layers: List[RetargetingLayer]
    exclusion_rules: List[str]
    general_notes: List[str]


class OptimizationAction(BaseModel):
    priority: int
    area: str
    diagnosis: str
    action: str
    expected_impact: str


class CampaignOptimizationResponse(BaseModel):
    platform: AdPlatform
    health_score: int = Field(..., ge=0, le=100)
    diagnosis_summary: str
    actions: List[OptimizationAction]
    bid_strategy_recommendation: str


class ChecklistItem(BaseModel):
    task: str
    done: bool = False
    critical: bool = False


class SetupChecklistResponse(BaseModel):
    platform: AdPlatform
    checklist: List[ChecklistItem]
