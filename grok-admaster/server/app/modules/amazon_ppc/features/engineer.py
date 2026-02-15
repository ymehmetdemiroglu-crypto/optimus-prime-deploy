"""
Feature Engineering Module for PPC Optimization.
Transforms raw performance data into ML-ready features.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from decimal import Decimal
import logging
import math

from ..models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Computes derived features from raw PPC performance data.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # ==================== ROLLING AVERAGES ====================
    
    async def compute_rolling_metrics(
        self, 
        campaign_id: int,
        windows: List[int] = [7, 14, 30]
    ) -> Dict[str, float]:
        """
        Compute rolling averages for key metrics over multiple time windows.
        
        Returns dict with keys like:
        - ctr_7d, ctr_14d, ctr_30d
        - conversion_rate_7d, conversion_rate_14d, conversion_rate_30d
        - acos_7d, acos_14d, acos_30d
        - roas_7d, roas_14d, roas_30d
        """
        features = {}
        
        for window in windows:
            cutoff = datetime.now() - timedelta(days=window)
            
            query = select(
                func.sum(PerformanceRecord.impressions).label('impressions'),
                func.sum(PerformanceRecord.clicks).label('clicks'),
                func.sum(PerformanceRecord.spend).label('spend'),
                func.sum(PerformanceRecord.sales).label('sales'),
                func.sum(PerformanceRecord.orders).label('orders')
            ).where(
                and_(
                    PerformanceRecord.campaign_id == campaign_id,
                    PerformanceRecord.date >= cutoff
                )
            )
            
            result = await self.db.execute(query)
            row = result.first()
            
            if row and row.impressions:
                impressions = float(row.impressions or 0)
                clicks = float(row.clicks or 0)
                spend = float(row.spend or 0)
                sales = float(row.sales or 0)
                orders = int(row.orders or 0)
                
                # Click-Through Rate
                ctr = (clicks / impressions * 100) if impressions > 0 else 0
                features[f'ctr_{window}d'] = round(ctr, 4)
                
                # Conversion Rate
                conv_rate = (orders / clicks * 100) if clicks > 0 else 0
                features[f'conversion_rate_{window}d'] = round(conv_rate, 4)
                
                # ACoS (Advertising Cost of Sales)
                acos = (spend / sales * 100) if sales > 0 else 0
                features[f'acos_{window}d'] = round(acos, 2)
                
                # ROAS (Return on Ad Spend)
                roas = (sales / spend) if spend > 0 else 0
                features[f'roas_{window}d'] = round(roas, 2)
                
                # Cost Per Click
                cpc = spend / clicks if clicks > 0 else 0
                features[f'cpc_{window}d'] = round(cpc, 2)
                
                # Cost Per Acquisition
                cpa = spend / orders if orders > 0 else 0
                features[f'cpa_{window}d'] = round(cpa, 2)
            else:
                # No data for this window
                for metric in ['ctr', 'conversion_rate', 'acos', 'roas', 'cpc', 'cpa']:
                    features[f'{metric}_{window}d'] = 0.0
        
        return features
    
    # ==================== SEASONALITY FEATURES ====================
    
    def compute_seasonality_features(self, target_date: date = None) -> Dict[str, Any]:
        """
        Compute seasonality indicators for a given date.
        
        Returns:
        - day_of_week (0=Monday, 6=Sunday)
        - is_weekend (bool)
        - month (1-12)
        - quarter (1-4)
        - week_of_year (1-52)
        - is_month_start, is_month_end
        - is_prime_day, is_black_friday, is_cyber_monday, is_holiday_season
        """
        if target_date is None:
            target_date = date.today()
        
        features = {
            'day_of_week': target_date.weekday(),
            'is_weekend': target_date.weekday() >= 5,
            'month': target_date.month,
            'quarter': (target_date.month - 1) // 3 + 1,
            'week_of_year': target_date.isocalendar()[1],
            'day_of_month': target_date.day,
            'is_month_start': target_date.day <= 3,
            'is_month_end': target_date.day >= 28,
        }
        
        # Major Shopping Events
        year = target_date.year
        
        # Prime Day (typically mid-July)
        prime_day_start = date(year, 7, 11)
        prime_day_end = date(year, 7, 12)
        features['is_prime_day'] = prime_day_start <= target_date <= prime_day_end
        
        # Black Friday (4th Thursday of November + day after)
        november_first = date(year, 11, 1)
        days_to_thursday = (3 - november_first.weekday()) % 7
        thanksgiving = november_first + timedelta(days=days_to_thursday + 21)
        black_friday = thanksgiving + timedelta(days=1)
        features['is_black_friday'] = target_date == black_friday
        
        # Cyber Monday
        cyber_monday = black_friday + timedelta(days=3)
        features['is_cyber_monday'] = target_date == cyber_monday
        
        # Holiday Season (Nov 15 - Dec 31)
        holiday_start = date(year, 11, 15)
        holiday_end = date(year, 12, 31)
        features['is_holiday_season'] = holiday_start <= target_date <= holiday_end
        
        # Q4 (important for retail)
        features['is_q4'] = target_date.month >= 10
        
        # Back to School (Aug 1 - Sep 15)
        bts_start = date(year, 8, 1)
        bts_end = date(year, 9, 15)
        features['is_back_to_school'] = bts_start <= target_date <= bts_end
        
        return features
    
    # ==================== TREND FEATURES ====================
    
    async def compute_trend_features(
        self, 
        campaign_id: int,
        short_window: int = 7,
        long_window: int = 30
    ) -> Dict[str, float]:
        """
        Compute trend indicators comparing short-term vs long-term performance.
        
        Returns:
        - spend_trend: ratio of 7d avg to 30d avg
        - sales_trend: ratio of 7d avg to 30d avg
        - ctr_trend: direction of CTR change
        - momentum: overall performance momentum
        """
        features = {}
        
        # Get short window metrics
        short_cutoff = datetime.now() - timedelta(days=short_window)
        long_cutoff = datetime.now() - timedelta(days=long_window)
        
        # Short window
        short_query = select(
            func.avg(PerformanceRecord.spend).label('avg_spend'),
            func.avg(PerformanceRecord.sales).label('avg_sales'),
            func.sum(PerformanceRecord.clicks).label('clicks'),
            func.sum(PerformanceRecord.impressions).label('impressions')
        ).where(
            and_(
                PerformanceRecord.campaign_id == campaign_id,
                PerformanceRecord.date >= short_cutoff
            )
        )
        
        # Long window
        long_query = select(
            func.avg(PerformanceRecord.spend).label('avg_spend'),
            func.avg(PerformanceRecord.sales).label('avg_sales'),
            func.sum(PerformanceRecord.clicks).label('clicks'),
            func.sum(PerformanceRecord.impressions).label('impressions')
        ).where(
            and_(
                PerformanceRecord.campaign_id == campaign_id,
                PerformanceRecord.date >= long_cutoff
            )
        )
        
        short_result = await self.db.execute(short_query)
        long_result = await self.db.execute(long_query)
        
        short = short_result.first()
        long = long_result.first()
        
        if short and long:
            # Spend Trend
            if long.avg_spend and long.avg_spend > 0:
                features['spend_trend'] = round(float(short.avg_spend or 0) / float(long.avg_spend), 3)
            else:
                features['spend_trend'] = 1.0
            
            # Sales Trend
            if long.avg_sales and long.avg_sales > 0:
                features['sales_trend'] = round(float(short.avg_sales or 0) / float(long.avg_sales), 3)
            else:
                features['sales_trend'] = 1.0
            
            # CTR Trend
            short_ctr = (short.clicks / short.impressions) if short.impressions else 0
            long_ctr = (long.clicks / long.impressions) if long.impressions else 0
            if long_ctr > 0:
                features['ctr_trend'] = round(short_ctr / long_ctr, 3)
            else:
                features['ctr_trend'] = 1.0
            
            # Momentum Score (-1 to +1)
            # Positive = improving, Negative = declining
            momentum = (
                (features['sales_trend'] - 1) * 0.4 +
                (features['ctr_trend'] - 1) * 0.3 +
                (1 - features['spend_trend']) * 0.3  # Lower spend is good if sales stable
            )
            features['momentum'] = round(max(-1, min(1, momentum)), 3)
        else:
            features['spend_trend'] = 1.0
            features['sales_trend'] = 1.0
            features['ctr_trend'] = 1.0
            features['momentum'] = 0.0
        
        return features
    
    # ==================== COMPETITION FEATURES ====================
    
    async def compute_competition_features(self, campaign_id: int) -> Dict[str, float]:
        """
        Compute features that may indicate competitive pressure.
        
        Returns:
        - impression_share_proxy: relative impressions vs historical
        - cpc_volatility: standard deviation of CPC
        - bid_competition_index: derived competition metric
        """
        features = {}
        
        # Get last 30 days of data for volatility calculation
        cutoff = datetime.now() - timedelta(days=30)
        
        query = select(
            PerformanceRecord.spend,
            PerformanceRecord.clicks,
            PerformanceRecord.impressions
        ).where(
            and_(
                PerformanceRecord.campaign_id == campaign_id,
                PerformanceRecord.date >= cutoff
            )
        )
        
        result = await self.db.execute(query)
        records = result.all()
        
        if records:
            cpcs = []
            for r in records:
                if r.clicks and r.clicks > 0:
                    cpcs.append(float(r.spend) / r.clicks)
            
            if cpcs:
                # CPC Volatility (coefficient of variation)
                mean_cpc = sum(cpcs) / len(cpcs)
                variance = sum((x - mean_cpc) ** 2 for x in cpcs) / len(cpcs)
                std_cpc = math.sqrt(variance)
                cv = std_cpc / mean_cpc if mean_cpc > 0 else 0
                features['cpc_volatility'] = round(cv, 3)
                features['avg_cpc'] = round(mean_cpc, 2)
            else:
                features['cpc_volatility'] = 0.0
                features['avg_cpc'] = 0.0
            
            # Impression volatility (indicates auction stability)
            impressions = [r.impressions for r in records if r.impressions]
            if impressions:
                mean_imp = sum(impressions) / len(impressions)
                variance = sum((x - mean_imp) ** 2 for x in impressions) / len(impressions)
                std_imp = math.sqrt(variance)
                features['impression_volatility'] = round(std_imp / mean_imp if mean_imp > 0 else 0, 3)
            else:
                features['impression_volatility'] = 0.0
        else:
            features['cpc_volatility'] = 0.0
            features['avg_cpc'] = 0.0
            features['impression_volatility'] = 0.0
        
        return features
    
    # ==================== FULL FEATURE VECTOR ====================
    
    async def compute_full_feature_vector(
        self, 
        campaign_id: int,
        target_date: date = None
    ) -> Dict[str, Any]:
        """
        Compute the complete feature vector for a campaign.
        Combines all feature categories into a single dict.
        """
        features = {'campaign_id': campaign_id}
        
        # Rolling metrics
        rolling = await self.compute_rolling_metrics(campaign_id)
        features.update(rolling)
        
        # Seasonality
        seasonality = self.compute_seasonality_features(target_date)
        features.update(seasonality)
        
        # Trends
        trends = await self.compute_trend_features(campaign_id)
        features.update(trends)
        
        # Competition proxy features
        competition = await self.compute_competition_features(campaign_id)
        features.update(competition)
        
        features['computed_at'] = datetime.now().isoformat()
        
        return features
    
    async def compute_features_for_all_campaigns(self) -> List[Dict[str, Any]]:
        """
        Compute features for all active campaigns.
        Returns a list of feature vectors.
        """
        query = select(PPCCampaign.id).where(PPCCampaign.state == 'enabled')
        result = await self.db.execute(query)
        campaign_ids = [row[0] for row in result.all()]
        
        all_features = []
        for campaign_id in campaign_ids:
            try:
                features = await self.compute_full_feature_vector(campaign_id)
                all_features.append(features)
            except Exception as e:
                logger.error(f"Failed to compute features for campaign {campaign_id}: {e}")
        
        logger.info(f"Computed features for {len(all_features)} campaigns")
        return all_features
