"""
ETL (Extract, Transform, Load) pipelines for Amazon Ads data.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
import logging

from ..models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord
from ..accounts.models import Profile
from .schemas import CampaignResponse, KeywordResponse, CampaignMetrics, KeywordMetrics

logger = logging.getLogger(__name__)

class AmazonAdsETL:
    """
    Transforms and loads Amazon Ads API data into the database.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def load_campaigns(self, profile_id: str, campaigns_data: List[Dict[str, Any]]) -> int:
        """
        Load campaign data into the database.
        Returns the number of campaigns processed.
        """
        count = 0
        
        for campaign_raw in campaigns_data:
            try:
                campaign_id = campaign_raw.get("campaignId")
                
                # Check if campaign exists
                query = select(PPCCampaign).where(PPCCampaign.campaign_id == campaign_id)
                result = await self.db.execute(query)
                existing = result.scalars().first()
                
                if existing:
                    # Update existing
                    existing.name = campaign_raw.get("name", existing.name)
                    existing.state = campaign_raw.get("state", existing.state)
                    existing.daily_budget = Decimal(str(campaign_raw.get("dailyBudget", 0)))
                    existing.updated_at = datetime.now()
                else:
                    # Create new
                    campaign = PPCCampaign(
                        campaign_id=campaign_id,
                        profile_id=profile_id,
                        name=campaign_raw.get("name", ""),
                        campaign_type=campaign_raw.get("campaignType", "sponsoredProducts"),
                        targeting_type=campaign_raw.get("targetingType", "manual"),
                        state=campaign_raw.get("state", "paused"),
                        daily_budget=Decimal(str(campaign_raw.get("dailyBudget", 0))),
                        start_date=self._parse_date(campaign_raw.get("startDate"))
                    )
                    self.db.add(campaign)
                
                count += 1
            except Exception as e:
                logger.error(f"Failed to load campaign {campaign_raw.get('campaignId')}: {e}")
                continue
        
        await self.db.commit()
        logger.info(f"Loaded {count} campaigns for profile {profile_id}")
        return count
    
    async def load_keywords(self, keywords_data: List[Dict[str, Any]]) -> int:
        """
        Load keyword data into the database.
        Returns the number of keywords processed.
        """
        count = 0
        
        for keyword_raw in keywords_data:
            try:
                keyword_id = keyword_raw.get("keywordId")
                campaign_id_str = keyword_raw.get("campaignId")
                
                # Find the internal campaign ID
                query = select(PPCCampaign).where(PPCCampaign.campaign_id == campaign_id_str)
                result = await self.db.execute(query)
                campaign = result.scalars().first()
                
                if not campaign:
                    logger.warning(f"Campaign {campaign_id_str} not found for keyword {keyword_id}")
                    continue
                
                # Check if keyword exists
                query = select(PPCKeyword).where(PPCKeyword.keyword_id == keyword_id)
                result = await self.db.execute(query)
                existing = result.scalars().first()
                
                if existing:
                    # Update existing
                    existing.state = keyword_raw.get("state", existing.state)
                    existing.bid = Decimal(str(keyword_raw.get("bid", 0)))
                    existing.updated_at = datetime.now()
                else:
                    # Create new
                    keyword = PPCKeyword(
                        keyword_id=keyword_id,
                        campaign_id=campaign.id,
                        keyword_text=keyword_raw.get("keywordText", ""),
                        match_type=keyword_raw.get("matchType", "exact"),
                        state=keyword_raw.get("state", "paused"),
                        bid=Decimal(str(keyword_raw.get("bid", 0)))
                    )
                    self.db.add(keyword)
                
                count += 1
            except Exception as e:
                logger.error(f"Failed to load keyword {keyword_raw.get('keywordId')}: {e}")
                continue
        
        await self.db.commit()
        logger.info(f"Loaded {count} keywords")
        return count
    
    async def load_campaign_performance(
        self, 
        metrics_data: List[Dict[str, Any]], 
        report_date: date
    ) -> int:
        """
        Load campaign performance metrics.
        """
        count = 0
        
        for metric in metrics_data:
            try:
                campaign_id_str = metric.get("campaignId")
                
                # Find the internal campaign
                query = select(PPCCampaign).where(PPCCampaign.campaign_id == campaign_id_str)
                result = await self.db.execute(query)
                campaign = result.scalars().first()
                
                if not campaign:
                    continue
                
                # Create performance record
                perf = PerformanceRecord(
                    campaign_id=campaign.id,
                    date=datetime.combine(report_date, datetime.min.time()),
                    impressions=metric.get("impressions", 0),
                    clicks=metric.get("clicks", 0),
                    spend=Decimal(str(metric.get("cost", 0))),
                    sales=Decimal(str(metric.get("attributedSales14d", 0))),
                    orders=metric.get("attributedConversions14d", 0)
                )
                self.db.add(perf)
                
                # Update denormalized metrics on campaign
                campaign.impressions += perf.impressions
                campaign.clicks += perf.clicks
                campaign.spend += perf.spend
                campaign.sales += perf.sales
                campaign.orders += perf.orders
                
                count += 1
            except Exception as e:
                logger.error(f"Failed to load performance for campaign {metric.get('campaignId')}: {e}")
                continue
        
        await self.db.commit()
        logger.info(f"Loaded {count} performance records for {report_date}")
        return count
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse Amazon API date strings."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except:
            try:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except:
                return None
