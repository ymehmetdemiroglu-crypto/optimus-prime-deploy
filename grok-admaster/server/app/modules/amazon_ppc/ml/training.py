"""
Training Pipeline for ML Models.
Handles data preparation, training, and evaluation.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from fastapi.concurrency import run_in_threadpool

from ..models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord
from ..features import FeatureEngineer, KeywordFeatureEngineer
from .bid_optimizer import BidOptimizer
from .rl_agent import PPCRLAgent
from .forecaster import PerformanceForecaster

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates model training from database data.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.feature_engineer = FeatureEngineer(db)
        self.keyword_engineer = KeywordFeatureEngineer(db)
    
    async def prepare_training_data(
        self,
        min_clicks: int = 20,
        lookback_days: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Prepare training dataset from historical performance.
        
        Identifies keywords with sufficient data and calculates
        what the "optimal" bid would have been based on actual results.
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        # Get keywords with enough data
        query = select(PPCKeyword).where(PPCKeyword.clicks >= min_clicks)
        result = await self.db.execute(query)
        keywords = result.scalars().all()
        
        training_data = []
        
        for kw in keywords:
            features = await self.keyword_engineer.compute_keyword_features(kw.id)
            
            if features.get('clicks', 0) < min_clicks:
                continue
            
            # Calculate what "optimal" bid would have been
            # Based on achieving target ACoS of 25%
            rpc = features.get('revenue_per_click', 0)
            if rpc > 0:
                optimal_bid = rpc * 0.25  # Target 25% ACoS
            else:
                # No sales - lower bid
                optimal_bid = features.get('current_bid', 1.0) * 0.75
            
            features['optimal_bid'] = optimal_bid
            training_data.append(features)
        
        logger.info(f"Prepared {len(training_data)} training samples")
        return training_data
    
    async def train_bid_optimizer(
        self,
        min_clicks: int = 20
    ) -> Dict[str, Any]:
        """
        Train the gradient boosting bid optimizer.
        """
        logger.info("Starting bid optimizer training...")
        
        training_data = await self.prepare_training_data(min_clicks)
        
        if len(training_data) < 50:
            return {
                'status': 'insufficient_data',
                'samples': len(training_data),
                'required': 50
            }
        
        optimizer = BidOptimizer()
        result = await run_in_threadpool(optimizer.train, training_data)
        
        return {
            'status': 'trained',
            **result
        }
    
    async def train_rl_agent(
        self,
        min_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Train the RL agent from historical bid change outcomes.
        """
        logger.info("Starting RL agent training...")
        
        # Get historical performance transitions
        query = (
            select(PerformanceRecord)
            .order_by(PerformanceRecord.campaign_id, PerformanceRecord.date)
            .limit(5000)
        )
        result = await self.db.execute(query)
        records = result.scalars().all()
        
        if len(records) < min_samples:
            return {
                'status': 'insufficient_data',
                'samples': len(records),
                'required': min_samples
            }
        
        # Group by campaign and create transitions
        history = []
        campaign_records = {}
        
        for r in records:
            if r.campaign_id not in campaign_records:
                campaign_records[r.campaign_id] = []
            campaign_records[r.campaign_id].append(r)
        
        for campaign_id, recs in campaign_records.items():
            for i in range(len(recs) - 1):
                before = {
                    'acos_7d': float(recs[i].spend / max(1, recs[i].sales) * 100) if recs[i].sales else 50,
                    'sales': float(recs[i].sales),
                    'spend': float(recs[i].spend),
                    'momentum': 0,
                    'spend_trend': 1.0,
                    'cpc_volatility': 0.1
                }
                after = {
                    'acos_7d': float(recs[i+1].spend / max(1, recs[i+1].sales) * 100) if recs[i+1].sales else 50,
                    'sales': float(recs[i+1].sales),
                    'spend': float(recs[i+1].spend),
                    'momentum': 0,
                    'spend_trend': 1.0,
                    'cpc_volatility': 0.1
                }
                
                # Estimate bid change from spend change
                if recs[i].clicks and recs[i+1].clicks:
                    cpc_before = float(recs[i].spend) / recs[i].clicks
                    cpc_after = float(recs[i+1].spend) / recs[i+1].clicks
                    bid_change = cpc_after / cpc_before if cpc_before > 0 else 1.0
                else:
                    bid_change = 1.0
                
                history.append({
                    'before_features': before,
                    'after_features': after,
                    'bid_change': bid_change
                })
        
        if not history:
            return {'status': 'no_transitions', 'samples': 0}
        
        agent = PPCRLAgent()
        result = await run_in_threadpool(agent.train_from_history, history)
        
        return {
            'status': 'trained',
            **result
        }
    
    async def evaluate_models(
        self,
        campaign_id: int
    ) -> Dict[str, Any]:
        """
        Evaluate all models on a specific campaign.
        """
        # Get campaign features
        campaign_features = await self.feature_engineer.compute_full_feature_vector(campaign_id)
        
        # Get keyword features
        keyword_features = await self.keyword_engineer.bulk_compute_features(campaign_id)
        
        results = {
            'campaign_id': campaign_id,
            'keywords_analyzed': len(keyword_features),
            'recommendations': []
        }
        
        # Bid optimizer predictions
        optimizer = BidOptimizer()
        rl_agent = PPCRLAgent()
        
        for kw in keyword_features[:10]:  # Limit for demo
            # Merge campaign and keyword features
            merged = {**campaign_features, **kw}
            
            gb_pred = optimizer.predict_bid(merged)
            rl_rec = rl_agent.get_bid_recommendation(
                merged,
                kw.get('current_bid', 1.0)
            )
            
            results['recommendations'].append({
                'keyword_id': kw.get('keyword_id'),
                'current_bid': kw.get('current_bid'),
                'gb_recommendation': {
                    'bid': gb_pred.predicted_bid,
                    'confidence': gb_pred.confidence,
                    'reasoning': gb_pred.reasoning
                },
                'rl_recommendation': rl_rec
            })
        
        return results
    
    async def get_campaign_forecast(
        self,
        campaign_id: int,
        horizon: int = 7
    ) -> Dict[str, Any]:
        """
        Get performance forecast for a campaign.
        """
        # Get historical data
        cutoff = datetime.now() - timedelta(days=60)
        
        query = (
            select(PerformanceRecord)
            .where(
                and_(
                    PerformanceRecord.campaign_id == campaign_id,
                    PerformanceRecord.date >= cutoff
                )
            )
            .order_by(PerformanceRecord.date)
        )
        
        result = await self.db.execute(query)
        records = result.scalars().all()
        
        if not records:
            return {'error': 'No historical data available'}
        
        historical = [
            {
                'date': r.date.isoformat(),
                'impressions': r.impressions,
                'clicks': r.clicks,
                'spend': float(r.spend),
                'sales': float(r.sales),
                'orders': r.orders,
                'acos': float(r.spend / max(1, r.sales) * 100) if r.sales else 0
            }
            for r in records
        ]
        
        forecaster = PerformanceForecaster()
        forecasts = forecaster.forecast_campaign(historical, horizon)
        
        # Convert dataclass to dict
        return {
            'campaign_id': campaign_id,
            'historical_days': len(records),
            'forecast_horizon': horizon,
            'metrics': {
                name: {
                    'current': f.current_value,
                    'forecast': f.forecasted_values,
                    'dates': f.dates,
                    'trend': f.trend
                }
                for name, f in forecasts.items()
            }
        }
