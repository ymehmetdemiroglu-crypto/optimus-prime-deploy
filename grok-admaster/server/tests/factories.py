
import random
from datetime import datetime, timedelta
from typing import Optional

# Mocking factory_boy behavior if not installed, or use standard class methods
# Since I can't guarantee factory_boy is in requirements.txt, I'll write effective python classes
# that act as factories.

from app.modules.amazon_ppc.models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord, KeywordState, AIStrategyType

class CampaignFactory:
    @staticmethod
    def create(
        id: int = None,
        campaign_id: str = None,
        name: str = "Test Campaign",
        daily_budget: float = 50.0,
        ai_mode: str = AIStrategyType.MANUAL,
        target_acos: float = 30.0,
        state: str = "enabled"
    ) -> PPCCampaign:
        return PPCCampaign(
            id=id or random.randint(1, 10000),
            campaign_id=campaign_id or f"camp_{random.randint(1000, 9999)}",
            name=name,
            daily_budget=daily_budget,
            ai_mode=ai_mode,
            target_acos=target_acos,
            state=state,
            start_date=datetime.utcnow()
        )

class KeywordFactory:
    @staticmethod
    def create(
        id: int = None,
        keyword_id: str = None,
        campaign_id: int = 1,
        keyword_text: str = "test keyword",
        bid: float = 1.0,
        state: str = KeywordState.ENABLED
    ) -> PPCKeyword:
        return PPCKeyword(
            id=id or random.randint(1, 100000),
            keyword_id=keyword_id or f"kw_{random.randint(10000, 99999)}",
            campaign_id=campaign_id,
            keyword_text=keyword_text,
            match_type="exact",
            state=state,
            bid=bid,
            serving_status="RUNNING",
            creation_date=datetime.utcnow()
        )

class FeatureFactory:
    """Generates feature dictionaries for unit testing optimization logic."""
    @staticmethod
    def create_keyword_features(
        keyword_id: int,
        bid: float = 1.0,
        acos: float = 25.0,
        clicks: int = 50,
        impressions: int = 1000,
        spend: float = 50.0,
        sales: float = 200.0,
        data_maturity: float = 1.0
    ) -> dict:
        return {
            "keyword_id": keyword_id,
            "current_bid": bid,
            "acos": acos,
            "acos_7d": acos, # Simplified
            "clicks": clicks,
            "impressions": impressions,
            "spend": spend,
            "sales": sales,
            "ctr": clicks / impressions if impressions > 0 else 0,
            "cpc": spend / clicks if clicks > 0 else 0,
            "data_maturity": data_maturity,
            "spend_trend": 1.0,
            "sales_trend": 1.0,
            "conversion_rate_30d": (sales / 20.0) / clicks if clicks > 0 else 0.1 # Approx
        }
