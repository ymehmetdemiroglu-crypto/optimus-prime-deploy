"""
Keyword-level feature engineering for bid optimization.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import Dict, Any, List
from datetime import datetime, timedelta
from decimal import Decimal
import math
import logging
import asyncio

from ..models.ppc_data import PPCKeyword, PerformanceRecord, KeywordVector
from app.services.ml.embedding_service import embedding_service
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)

class KeywordFeatureEngineer:
    """
    Computes keyword-specific features for bid optimization.
    Optimized for bulk processing and scalability.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    def _calculate_features(
        self, 
        keyword: PPCKeyword, 
        perf: Any, 
        embedding: List[float], 
        lookback_days: int
    ) -> Dict[str, Any]:
        """
        Synchronous helper to calculate features from loaded data.
        """
        features = {'keyword_id': keyword.id}
        
        # Safe access for Enum or String types
        features['match_type'] = getattr(keyword.match_type, 'value', keyword.match_type) if keyword.match_type else 'unknown'
        features['current_bid'] = float(keyword.bid or 0)
        features['state'] = getattr(keyword.state, 'value', keyword.state) if keyword.state else 'unknown'
        
        features['embedding'] = embedding
        
        if perf and perf.impressions:
            impressions = float(perf.impressions)
            clicks = float(perf.clicks or 0)
            spend = float(perf.spend or 0)
            sales = float(perf.sales or 0)
            orders = int(perf.orders or 0)
            days = int(perf.days_active or 1)
            
            # Core metrics
            features['impressions'] = impressions
            features['clicks'] = clicks
            features['spend'] = spend
            features['sales'] = sales
            features['orders'] = orders
            
            # Derived metrics
            features['ctr'] = round(clicks / impressions * 100, 4) if impressions > 0 else 0
            features['conversion_rate'] = round(orders / clicks * 100, 4) if clicks > 0 else 0
            features['acos'] = round(spend / sales * 100, 2) if sales > 0 else 999
            features['roas'] = round(sales / spend, 2) if spend > 0 else 0
            features['cpc'] = round(spend / clicks, 2) if clicks > 0 else 0
            features['cpa'] = round(spend / orders, 2) if orders > 0 else 0
            
            # Daily averages
            features['daily_impressions'] = round(impressions / days, 2)
            features['daily_clicks'] = round(clicks / days, 2)
            features['daily_spend'] = round(spend / days, 2)
            features['daily_sales'] = round(sales / days, 2)
            
            # Revenue per click
            features['revenue_per_click'] = round(sales / clicks, 2) if clicks > 0 else 0
            
            # Profit metrics (assuming 30% margin)
            margin = 0.30
            features['estimated_profit'] = round(sales * margin - spend, 2)
            features['profit_per_click'] = round((sales * margin - spend) / clicks, 2) if clicks > 0 else 0
            
            # Data maturity score
            features['data_maturity'] = min(1.0, clicks / 100)
            
        else:
            # Zero-fill if no performance data
            features.update({
                'impressions': 0, 'clicks': 0, 'spend': 0, 
                'sales': 0, 'orders': 0, 'ctr': 0, 
                'conversion_rate': 0, 'acos': 0, 'roas': 0, 
                'cpc': 0, 'cpa': 0, 'daily_impressions': 0, 
                'daily_clicks': 0, 'daily_spend': 0, 'daily_sales': 0, 
                'revenue_per_click': 0, 'estimated_profit': 0, 
                'profit_per_click': 0, 'data_maturity': 0
            })
            
        return features

    async def compute_keyword_features(
        self, 
        keyword_id: int,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compute comprehensive feature set for a single keyword.
        """
        # Get keyword info
        kw_query = (
            select(PPCKeyword)
            .where(PPCKeyword.id == keyword_id)
            .options(selectinload(PPCKeyword.vector))
        )
        result = await self.db.execute(kw_query)
        keyword = result.scalars().first()
        
        if not keyword:
            return {'keyword_id': keyword_id}
            
        # Get embedding (compute if missing)
        embedding = []
        if keyword.vector and keyword.vector.embedding is not None:
             embedding = keyword.vector.embedding
        else:
             # Offload synchronous encoding to executor to avoid blocking loop
             loop = asyncio.get_running_loop()
             emb_array = await loop.run_in_executor(
                 None, 
                 embedding_service.encode, 
                 keyword.keyword_text
             )
             embedding = emb_array.tolist()

        # Get performance data
        cutoff = datetime.now() - timedelta(days=lookback_days)
        perf_query = select(
            func.sum(PerformanceRecord.impressions).label('impressions'),
            func.sum(PerformanceRecord.clicks).label('clicks'),
            func.sum(PerformanceRecord.spend).label('spend'),
            func.sum(PerformanceRecord.sales).label('sales'),
            func.sum(PerformanceRecord.orders).label('orders'),
            func.count(PerformanceRecord.id).label('days_active')
        ).where(
            and_(
                PerformanceRecord.keyword_id == keyword_id,
                PerformanceRecord.date >= cutoff
            )
        )
        
        result = await self.db.execute(perf_query)
        perf = result.first()
        
        return self._calculate_features(keyword, perf, embedding, lookback_days)
    
    async def compute_bid_recommendations(
        self,
        keyword_id: int,
        target_acos: float = 25.0,
        target_roas: float = 4.0
    ) -> Dict[str, Any]:
        """
        Compute bid recommendations based on performance data.
        """
        features = await self.compute_keyword_features(keyword_id)
        
        recommendations = {
            'keyword_id': keyword_id,
            'current_bid': features.get('current_bid', 0),
            'strategies': {}
        }
        
        rpc = features.get('revenue_per_click', 0)
        current_acos = features.get('acos', 0)
        clicks = features.get('clicks', 0)
        current_bid = features.get('current_bid', 0)
        
        # Strategy 1: Target ACoS
        if rpc > 0:
            acos_bid = rpc * (target_acos / 100)
            recommendations['strategies']['target_acos'] = {
                'bid': round(acos_bid, 2),
                'rationale': f'Bid to achieve {target_acos}% ACoS based on RPC ${rpc:.2f}'
            }
        
        # Strategy 2: Target ROAS
        if rpc > 0:
            roas_bid = rpc / target_roas
            recommendations['strategies']['target_roas'] = {
                'bid': round(roas_bid, 2),
                'rationale': f'Bid to achieve {target_roas}x ROAS'
            }
        
        # Strategy 3: Conservative
        if current_acos > target_acos and current_bid > 0:
            reduction = min(0.3, (current_acos - target_acos) / 100)
            conservative_bid = current_bid * (1 - reduction)
            recommendations['strategies']['conservative'] = {
                'bid': round(max(0.10, conservative_bid), 2),
                'rationale': f'Reduce bid by {reduction*100:.0f}% to lower ACoS from {current_acos:.1f}%'
            }
        
        # Strategy 4: Aggressive
        if current_acos < target_acos * 0.7 and clicks >= 10:
            increase = min(0.25, (target_acos - current_acos) / 100)
            aggressive_bid = current_bid * (1 + increase)
            recommendations['strategies']['aggressive'] = {
                'bid': round(aggressive_bid, 2),
                'rationale': f'Increase bid by {increase*100:.0f}% - keyword is outperforming at {current_acos:.1f}% ACoS'
            }
        
        # Confidence score
        if clicks >= 50:
            recommendations['confidence'] = 'high'
        elif clicks >= 20:
            recommendations['confidence'] = 'medium'
        else:
            recommendations['confidence'] = 'low'
        
        return recommendations
    
    async def bulk_compute_features(
        self, 
        campaign_id: int,
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Compute features for all keywords in a campaign using optimized batched queries.
        Reduces N+1 queries to 2 main queries + 1 batch embedding generation.
        Optimized to load only necessary columns (Lightweight Rows vs Full ORM Objects).
        """
        # 1. Fetch keywords with lightweight query (explicit columns + outer join)
        kw_query = (
            select(
                PPCKeyword.id,
                PPCKeyword.keyword_text,
                PPCKeyword.match_type,
                PPCKeyword.state,
                PPCKeyword.bid,
                KeywordVector.embedding
            )
            .outerjoin(KeywordVector, PPCKeyword.id == KeywordVector.keyword_id)
            .where(PPCKeyword.campaign_id == campaign_id)
        )
        result = await self.db.execute(kw_query)
        keywords_rows = result.all()
        
        if not keywords_rows:
            return []
            
        keyword_map = {row.id: row for row in keywords_rows}
        keyword_ids = list(keyword_map.keys())
        
        # 2. Bulk fetch performance data
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        # Optimize: Filter by campaign_id using index instead of large IN clause
        perf_query = (
            select(
                PerformanceRecord.keyword_id,
                func.sum(PerformanceRecord.impressions).label('impressions'),
                func.sum(PerformanceRecord.clicks).label('clicks'),
                func.sum(PerformanceRecord.spend).label('spend'),
                func.sum(PerformanceRecord.sales).label('sales'),
                func.sum(PerformanceRecord.orders).label('orders'),
                func.count(PerformanceRecord.id).label('days_active')
            )
            .where(
                and_(
                    PerformanceRecord.campaign_id == campaign_id,
                    PerformanceRecord.date >= cutoff
                )
            )
            .group_by(PerformanceRecord.keyword_id)
        )
        
        perf_result = await self.db.execute(perf_query)
        perf_map = {row.keyword_id: row for row in perf_result.all()}
        
        # 3. Handle missing embeddings with batch processing
        missing_embedding_rows = []
        for row in keywords_rows:
            if row.embedding is None:
                missing_embedding_rows.append(row)
        
        computed_embeddings = {}
        if missing_embedding_rows:
            texts = [row.keyword_text for row in missing_embedding_rows]
            ids = [row.id for row in missing_embedding_rows]
            
            # Batch size for embedding generation to prevent OOM/CPU spikes
            BATCH_SIZE = 500
            
            try:
                loop = asyncio.get_running_loop()
                
                # Process in chunks
                for i in range(0, len(texts), BATCH_SIZE):
                    batch_texts = texts[i : i + BATCH_SIZE]
                    batch_ids = ids[i : i + BATCH_SIZE]
                    
                    vectors = await loop.run_in_executor(
                        None,
                        embedding_service.encode_batch,
                        batch_texts
                    )
                    
                    # Map back to IDs
                    for j, vector in enumerate(vectors):
                        if j < len(batch_ids):
                            computed_embeddings[batch_ids[j]] = vector.tolist()
                        
            except Exception as e:
                logger.error(f"Failed to batch compute embeddings: {e}")

        # 4. Assemble Results
        all_features = []
        for kw_id, kw_row in keyword_map.items():
            perf = perf_map.get(kw_id)
            
            # Determine embedding
            if kw_id in computed_embeddings:
                embedding = computed_embeddings[kw_id]
            elif kw_row.embedding is not None:
                embedding = kw_row.embedding
            else:
                embedding = [] # Fallback
                
            features = self._calculate_features(kw_row, perf, embedding, lookback_days)
            all_features.append(features)
        
        return all_features
