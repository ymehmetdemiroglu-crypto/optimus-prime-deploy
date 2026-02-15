"""
API endpoints for ML model operations.
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from pydantic import BaseModel

from app.core.database import get_db
from ..ml import (
    BidOptimizer, PPCRLAgent, PerformanceForecaster,
    DeepBidOptimizer, BidBanditOptimizer, 
    LSTMForecaster, SeasonalDecomposer,
    BayesianBudgetOptimizer, SpendPacer,
    ModelEnsemble, StackingEnsemble, VotingEnsemble
)
from app.core.ml_models import model_cache
from ..ml.training import TrainingPipeline
from ..features import FeatureEngineer, KeywordFeatureEngineer

router = APIRouter()


# ==================== REQUEST MODELS ====================

class BudgetOptimizationRequest(BaseModel):
    campaigns: List[dict]
    total_budget: float


class HourlyPerformanceRequest(BaseModel):
    campaign_id: int
    hourly_data: List[dict]


# ==================== TRAINING ENDPOINTS ====================

@router.post("/train/bid-optimizer")
async def train_bid_optimizer(
    min_clicks: int = 20,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """Train the gradient boosting bid optimizer model."""
    pipeline = TrainingPipeline(db)
    result = await pipeline.train_bid_optimizer(min_clicks)
    return result


@router.post("/train/rl-agent")
async def train_rl_agent(
    min_samples: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Train the reinforcement learning agent."""
    pipeline = TrainingPipeline(db)
    result = await pipeline.train_rl_agent(min_samples)
    return result


@router.post("/train/deep-optimizer")
async def train_deep_optimizer(
    epochs: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Train the deep neural network optimizer."""
    pipeline = TrainingPipeline(db)
    training_data = await pipeline.prepare_training_data()
    
    if len(training_data) < 50:
        return {'status': 'insufficient_data', 'samples': len(training_data)}
    
    deep_opt = model_cache.get_deep_optimizer()
    result = deep_opt.train(training_data, epochs=epochs)
    return result


# ==================== PREDICTION ENDPOINTS ====================

@router.get("/predict/bid/{keyword_id}")
async def predict_bid(
    keyword_id: int,
    target_acos: float = 25.0,
    target_roas: float = 4.0,
    db: AsyncSession = Depends(get_db)
):
    """Get bid prediction for a specific keyword from all models."""
    keyword_engineer = KeywordFeatureEngineer(db)
    features = await keyword_engineer.compute_keyword_features(keyword_id)
    
    if not features or features.get('keyword_id') == 0:
        return {"error": "Keyword not found"}
    
    campaign_id = features.get('campaign_id')
    if campaign_id:
        engineer = FeatureEngineer(db)
        campaign_features = await engineer.compute_full_feature_vector(campaign_id)
        features.update(campaign_features)
    
    # Gradient Boosting prediction
    optimizer = model_cache.get_bid_optimizer()
    gb_prediction = await run_in_threadpool(optimizer.predict_bid, features, target_acos, target_roas)
    
    # RL Agent recommendation
    rl_agent = model_cache.get_rl_agent()
    rl_recommendation = await run_in_threadpool(
        rl_agent.get_bid_recommendation,
        features, features.get('current_bid', 1.0), target_acos
    )
    
    return {
        'keyword_id': keyword_id,
        'current_bid': features.get('current_bid'),
        'gradient_boosting': {
            'predicted_bid': gb_prediction.predicted_bid,
            'confidence': gb_prediction.confidence,
            'expected_acos': gb_prediction.expected_acos,
            'expected_roas': gb_prediction.expected_roas,
            'reasoning': gb_prediction.reasoning
        },
        'reinforcement_learning': rl_recommendation,
        'recommended_bid': (gb_prediction.predicted_bid + rl_recommendation['recommended_bid']) / 2
    }


@router.get("/predict/ensemble/{keyword_id}")
async def predict_ensemble(
    keyword_id: int,
    target_acos: float = 25.0,
    db: AsyncSession = Depends(get_db)
):
    """Get ensemble prediction combining all models."""
    keyword_engineer = KeywordFeatureEngineer(db)
    features = await keyword_engineer.compute_keyword_features(keyword_id)
    
    if not features or features.get('keyword_id') == 0:
        return {"error": "Keyword not found"}
    
    campaign_id = features.get('campaign_id')
    if campaign_id:
        engineer = FeatureEngineer(db)
        campaign_features = await engineer.compute_full_feature_vector(campaign_id)
        features.update(campaign_features)
    
    # Get ensemble prediction
    ensemble = model_cache.get_model_ensemble()
    prediction = await run_in_threadpool(ensemble.predict, features, target_acos)
    
    return {
        'keyword_id': keyword_id,
        'current_bid': features.get('current_bid'),
        'ensemble_prediction': prediction.final_bid,
        'confidence': prediction.confidence,
        'model_predictions': prediction.model_predictions,
        'model_weights': prediction.model_weights,
        'reasoning': prediction.reasoning
    }


@router.get("/predict/voting/{keyword_id}")
async def predict_voting(
    keyword_id: int,
    target_acos: float = 25.0,
    db: AsyncSession = Depends(get_db)
):
    """Get voting ensemble prediction."""
    keyword_engineer = KeywordFeatureEngineer(db)
    features = await keyword_engineer.compute_keyword_features(keyword_id)
    
    if not features:
        return {"error": "Keyword not found"}
    
    campaign_id = features.get('campaign_id')
    if campaign_id:
        engineer = FeatureEngineer(db)
        campaign_features = await engineer.compute_full_feature_vector(campaign_id)
        features.update(campaign_features)
    
    voting = VotingEnsemble()
    result = await run_in_threadpool(voting.vote, features, target_acos)
    
    return {
        'keyword_id': keyword_id,
        **result
    }


@router.get("/predict/bandit/{keyword_id}")
async def predict_bandit(
    keyword_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get multi-armed bandit recommendation."""
    keyword_engineer = KeywordFeatureEngineer(db)
    features = await keyword_engineer.compute_keyword_features(keyword_id)
    
    if not features:
        return {"error": "Keyword not found"}
    
    bandit = model_cache.get_bandit_optimizer()
    recommendation = await run_in_threadpool(bandit.select_bid_multiplier, features, keyword_id)
    
    current_bid = features.get('current_bid', 1.0)
    
    return {
        'keyword_id': keyword_id,
        'current_bid': current_bid,
        'recommended_bid': round(current_bid * recommendation['ensemble_multiplier'], 2),
        **recommendation
    }


@router.get("/predict/campaign/{campaign_id}/keywords")
async def predict_campaign_bids(
    campaign_id: int,
    target_acos: float = 25.0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get bid predictions for all keywords in a campaign."""
    keyword_engineer = KeywordFeatureEngineer(db)
    all_features = await keyword_engineer.bulk_compute_features(campaign_id)
    
    optimizer = model_cache.get_bid_optimizer()
    predictions = []
    
    for features in all_features[:limit]:
        pred = await run_in_threadpool(optimizer.predict_bid, features, target_acos)
        predictions.append({
            'keyword_id': features.get('keyword_id'),
            'current_bid': features.get('current_bid'),
            'predicted_bid': pred.predicted_bid,
            'bid_change': round(pred.predicted_bid - pred.current_bid, 2),
            'confidence': pred.confidence,
            'expected_acos': pred.expected_acos
        })
    
    predictions.sort(key=lambda x: abs(x['bid_change']), reverse=True)
    
    return {
        'campaign_id': campaign_id,
        'total_keywords': len(all_features),
        'predictions': predictions
    }


# ==================== FORECASTING ENDPOINTS ====================

@router.get("/forecast/campaign/{campaign_id}")
async def forecast_campaign(
    campaign_id: int,
    horizon: int = 7,
    db: AsyncSession = Depends(get_db)
):
    """Get performance forecast for a campaign."""
    pipeline = TrainingPipeline(db)
    forecast = await pipeline.get_campaign_forecast(campaign_id, horizon)
    return forecast


@router.get("/forecast/lstm/{campaign_id}")
async def forecast_lstm(
    campaign_id: int,
    horizon: int = 7,
    db: AsyncSession = Depends(get_db)
):
    """Get LSTM-based forecast for a campaign."""
    from datetime import datetime, timedelta
    from sqlalchemy import select, and_
    from ..models.ppc_data import PerformanceRecord
    
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
    
    result = await db.execute(query)
    records = result.scalars().all()
    
    if not records:
        return {'error': 'No historical data available'}
    
    historical = [
        {
            'impressions': r.impressions,
            'clicks': r.clicks,
            'spend': float(r.spend),
            'sales': float(r.sales),
            'orders': r.orders
        }
        for r in records
    ]
    
    lstm = LSTMForecaster()
    forecast = await run_in_threadpool(lstm.forecast, historical, horizon)
    
    return {
        'campaign_id': campaign_id,
        'model': 'lstm',
        'historical_days': len(records),
        **forecast
    }


@router.get("/decompose/{campaign_id}/{metric}")
async def decompose_time_series(
    campaign_id: int,
    metric: str = "sales",
    period: int = 7,
    db: AsyncSession = Depends(get_db)
):
    """Decompose metric into trend, seasonality, and residual."""
    from datetime import datetime, timedelta
    from sqlalchemy import select, and_
    from ..models.ppc_data import PerformanceRecord
    
    cutoff = datetime.now() - timedelta(days=90)
    
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
    
    result = await db.execute(query)
    records = result.scalars().all()
    
    if not records:
        return {'error': 'No historical data available'}
    
    # Extract metric values
    values = []
    for r in records:
        if metric == 'sales':
            values.append(float(r.sales))
        elif metric == 'spend':
            values.append(float(r.spend))
        elif metric == 'clicks':
            values.append(r.clicks)
        elif metric == 'impressions':
            values.append(r.impressions)
        else:
            values.append(float(r.sales))
    
    decomposer = SeasonalDecomposer(period=period)
    decomposition = await run_in_threadpool(decomposer.decompose, values)
    
    return {
        'campaign_id': campaign_id,
        'metric': metric,
        **decomposition
    }


# ==================== BUDGET OPTIMIZATION ====================

@router.post("/optimize/budget")
async def optimize_budget_portfolio(request: BudgetOptimizationRequest):
    """Optimize budget allocation across campaigns."""
    optimizer = BayesianBudgetOptimizer()
    allocations = await run_in_threadpool(optimizer.optimize_portfolio, request.campaigns, request.total_budget)
    
    return {
        'total_budget': request.total_budget,
        'allocations': [
            {
                'campaign_id': a.campaign_id,
                'current_budget': a.current_budget,
                'recommended_budget': a.recommended_budget,
                'expected_roi': a.expected_roi,
                'confidence': a.confidence,
                'reasoning': a.reasoning
            }
            for a in allocations
        ]
    }


@router.get("/optimize/budget/{campaign_id}")
async def suggest_campaign_budget(
    campaign_id: int,
    current_budget: float,
    min_budget: Optional[float] = None,
    max_budget: Optional[float] = None
):
    """Get budget suggestion for a single campaign."""
    optimizer = BayesianBudgetOptimizer()
    
    budget_range = None
    if min_budget and max_budget:
        budget_range = (min_budget, max_budget)
    
    suggestion = await run_in_threadpool(optimizer.suggest_budget, campaign_id, current_budget, budget_range)
    
    return {
        'campaign_id': suggestion.campaign_id,
        'current_budget': suggestion.current_budget,
        'recommended_budget': suggestion.recommended_budget,
        'expected_roi': suggestion.expected_roi,
        'confidence': suggestion.confidence,
        'reasoning': suggestion.reasoning
    }


@router.post("/dayparting/learn")
async def learn_dayparting(request: HourlyPerformanceRequest):
    """Learn hourly performance patterns for dayparting."""
    pacer = SpendPacer()
    await run_in_threadpool(pacer.learn_patterns, request.campaign_id, request.hourly_data)
    
    schedule = pacer.get_pacing_schedule(request.campaign_id, 100)  # Example $100 budget
    
    return {
        'campaign_id': request.campaign_id,
        'schedule': schedule
    }


@router.get("/dayparting/{campaign_id}")
async def get_dayparting_schedule(campaign_id: int, daily_budget: float = 100.0):
    """Get recommended dayparting schedule."""
    pacer = SpendPacer()
    schedule = pacer.get_pacing_schedule(campaign_id, daily_budget)
    
    return {
        'campaign_id': campaign_id,
        'daily_budget': daily_budget,
        'schedule': schedule
    }


# ==================== MODEL STATUS ====================

@router.get("/evaluate/{campaign_id}")
async def evaluate_models(
    campaign_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Evaluate all models on a specific campaign."""
    pipeline = TrainingPipeline(db)
    evaluation = await pipeline.evaluate_models(campaign_id)
    return evaluation


@router.get("/model-status")
async def get_model_status():
    """Get status of all ML models."""
    optimizer = model_cache.get_bid_optimizer()
    rl_agent = model_cache.get_rl_agent()
    deep_opt = model_cache.get_deep_optimizer()
    bandit = model_cache.get_bandit_optimizer()
    ensemble = model_cache.get_model_ensemble()
    
    return {
        'bid_optimizer': {
            'is_trained': optimizer.is_trained,
            'model_path': optimizer.model_path,
            'feature_count': len(optimizer.FEATURE_COLS)
        },
        'rl_agent': {
            'states_explored': len(rl_agent.q_table),
            'actions': len(rl_agent.ACTIONS),
            'model_path': rl_agent.model_path
        },
        'deep_optimizer': {
            'is_trained': deep_opt.is_trained,
            'architecture': '128->64->32->16->1',
            'model_path': deep_opt.model_path
        },
        'bandit': {
            'thompson_arms': len(bandit.thompson.arms),
            'ucb_arms': len(bandit.ucb.arms),
            'model_path': bandit.model_path
        },
        'ensemble': ensemble.get_model_status()
    }


@router.get("/feature-importance")
async def get_feature_importance():
    """
    Get feature importance from trained bid optimizer.
    """
    optimizer = model_cache.get_bid_optimizer()
    
    if not optimizer.is_trained:
        return {"error": "Model not trained yet"}
    
    importance = optimizer.get_feature_importance()
    
    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'features': [{'name': k, 'importance': round(v, 4)} for k, v in sorted_importance]
    }
