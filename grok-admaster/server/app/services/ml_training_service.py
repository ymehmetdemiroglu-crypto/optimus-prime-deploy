"""
Model Training Service
Connects the Database (source) to the ML Engines (sink).
Feeds data into the BidOptimizer and persists the result.
"""
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy.future import select
from sqlalchemy import func, desc, case
from datetime import datetime, timedelta

from app.core.database import AsyncSessionLocal
from app.modules.amazon_ppc.models.ppc_data import PPCKeyword, PerformanceRecord, KeywordState
from app.modules.amazon_ppc.ml.bid_optimizer import BidOptimizer
from app.core.model_store import get_model_store, ModelStore
from app.modules.amazon_ppc.strategies.config import BidStrategyConfig
import asyncio

logger = logging.getLogger(__name__)

class ModelTrainingService:
    def __init__(self, model_store: Optional[ModelStore] = None):
        self.model_store = model_store or get_model_store()
        self.config = BidStrategyConfig()
        
        # Try to load existing model
        loaded_model = self.model_store.load("bid_optimizer_v1")
        self.bid_optimizer = BidOptimizer(model_artifact=loaded_model, config=self.config)

    async def train_bid_model(self, lookback_days: int = 60) -> Dict[str, Any]:
        """
        Fetch historical performance data with DAILY granularity and rolling metrics.
        Enables TimeSeriesSplit in BidOptimizer.
        """
        logger.info("Starting nightly model training (Daily Granularity)...")
        
        effective_lookback = lookback_days if lookback_days else self.config.lookback_window_days

        async with AsyncSessionLocal() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=effective_lookback)
            
            # Subquery: Daily Performance with Rolling 30d Window
            # defined via SQLAlchemy window functions
            
            w_30d = func.sum(PerformanceRecord.clicks).over(
                partition_by=PerformanceRecord.keyword_id,
                order_by=PerformanceRecord.date,
                rows=(29, 0) # 29 Preceding + Current = 30 days
            )
            
            w_30d_imps = func.sum(PerformanceRecord.impressions).over(
                partition_by=PerformanceRecord.keyword_id,
                order_by=PerformanceRecord.date,
                rows=(29, 0)
            )
            
            w_30d_spend = func.sum(PerformanceRecord.spend).over(
                partition_by=PerformanceRecord.keyword_id,
                order_by=PerformanceRecord.date,
                rows=(29, 0)
            )
            
            w_30d_sales = func.sum(PerformanceRecord.sales).over(
                partition_by=PerformanceRecord.keyword_id,
                order_by=PerformanceRecord.date,
                rows=(29, 0)
            )
            
            stmt = (
                select(
                    PerformanceRecord.date,
                    PerformanceRecord.keyword_id,
                    PerformanceRecord.clicks.label("daily_clicks"),
                    PerformanceRecord.orders.label("daily_orders"),
                    PerformanceRecord.sales.label("daily_sales"),
                    # Rolling features
                    w_30d.label("rolling_clicks"),
                    w_30d_imps.label("rolling_imps"),
                    w_30d_spend.label("rolling_spend"),
                    w_30d_sales.label("rolling_sales"),
                    PPCKeyword.bid.label("current_bid_setting") # Current bid setting (static for now, ideally history)
                )
                .join(PPCKeyword, PPCKeyword.id == PerformanceRecord.keyword_id)
                .where(
                    PerformanceRecord.date >= cutoff_date,
                    PPCKeyword.state == KeywordState.ENABLED
                )
                .order_by(PerformanceRecord.date)
                .limit(50000) # Safety limit for daily rows
            )
            
            result = await session.execute(stmt)
            rows = result.all()
            
            training_samples = []
            
            for row in rows:
                # Need at least some valid history
                if (row.rolling_clicks or 0) < 5:
                    continue

                clicks_30 = row.rolling_clicks or 0
                spend_30 = float(row.rolling_spend or 0)
                sales_30 = float(row.rolling_sales or 0)
                imps_30 = row.rolling_imps or 0
                
                # Derived Features tailored to mimic previous logic but with rolling data
                ctr = float(clicks_30) / float(imps_30) if imps_30 > 0 else 0
                cpc = float(spend_30) / float(clicks_30) if clicks_30 > 0 else 0
                acos = (spend_30 / sales_30 * 100) if sales_30 > 0 else 0
                
                # Current Bid (Fallback to CPC if unknown, or the static current setting)
                # Note: In a real system, we'd need the historical bid setting. 
                # For now, we approximate 'current_bid' as the CPC + small buffer or the static setting
                current_bid = float(row.current_bid_setting or cpc)

                features = {
                    "keyword_id": row.keyword_id,
                    "date": row.date.timestamp(), # Crucial for TimeSeriesSplit
                    "ctr_30d": ctr,
                    "acos_30d": acos,
                    "current_bid": current_bid,
                    "revenue_per_click": (sales_30 / clicks_30) if clicks_30 > 0 else 0,
                    "cpc_30d": cpc,
                    "spend_trend": 0.0,
                    "data_maturity": min(clicks_30 / 20.0, 1.0)
                }
                
                # Target Generation (Same Logic, applied to Rolling Stat context)
                optimal_bid = current_bid
                target_acos = self.config.target_acos
                
                if sales_30 > 0:
                    rpc = sales_30 / clicks_30
                    optimal_bid = rpc * (target_acos / 100.0)
                else:
                    optimal_bid = current_bid * 0.8
                    
                sample = features.copy()
                sample['optimal_bid'] = optimal_bid
                training_samples.append(sample)
                
            if len(training_samples) < self.config.min_training_samples:
                logger.warning(f"Insufficient data for training (<{self.config.min_training_samples} samples)")
                return {"status": "skipped", "reason": "Insufficient data"}
                
            # 2. Train (Non-blocking)
            loop = asyncio.get_running_loop()
            metrics = await loop.run_in_executor(
                None, 
                self.bid_optimizer.train, 
                training_samples, 
                'optimal_bid'
            )
            
            # 3. Save
            if metrics:
                success = self.model_store.save(self.bid_optimizer.model, "bid_optimizer_v1")
                metrics['model_persisted'] = success
                return {"status": "success", "metrics": metrics}
            
            return {"status": "failure", "reason": "Training returned no metrics"}
