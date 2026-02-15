"""
AI Integration Service
----------------------
This service acts as the bridge between the core PPC logic and the Advanced ML models.
It handles feature extraction, model initialization, and recommendation generation.
INCLUDES: Database persistence for predictions and bandit states.
"""
from typing import Any, List, Dict, Optional
from datetime import datetime, timezone
import logging
import json
import asyncpg

# Import Advanced ML Models
from app.modules.amazon_ppc.ml import (
    ModelEnsemble,
    BidBanditOptimizer,
    LSTMForecaster,
    BayesianBudgetOptimizer,
    EnsemblePrediction
)

logger = logging.getLogger(__name__)

# Models are initialized as global singletons (consider dependency injection in production)
_ensemble_model = None
_bandit_model = None
_budget_optimizer = None
_forecaster = None

def _initialize_models():
    """Lazy initialize ML models."""
    global _ensemble_model, _bandit_model, _budget_optimizer, _forecaster
    if _ensemble_model is None:
        _ensemble_model = ModelEnsemble()
        _bandit_model = BidBanditOptimizer()
        _budget_optimizer = BayesianBudgetOptimizer()
        _forecaster = LSTMForecaster()
        logger.info("AI models initialized")

async def load_bandit_state_from_db(db_conn: asyncpg.Connection, keyword_id: Optional[int] = None):
    """Load bandit arm states from database."""
    _initialize_models()
    
    try:
        arms = await db_conn.fetch("""
            SELECT arm_id, multiplier, alpha, beta, pulls, total_reward 
            FROM bandit_arms 
            WHERE keyword_id IS NULL OR keyword_id = $1
        """, keyword_id)
        
        if arms:
            logger.info(f"Loaded {len(arms)} bandit arms from database")
            # Note: In production, you'd update the bandit model's internal state here
            # For now, the model uses its default initialization
        
        return arms
    except Exception as e:
        logger.warning(f"Could not load bandit state: {e}")
        return []

async def save_bandit_arm_to_db(
    db_conn: asyncpg.Connection,
    keyword_id: Optional[int],
    arm_id: int,
    multiplier: float,
    alpha: float,
    beta: float,
    pulls: int,
    total_reward: float
):
    """Save bandit arm state to database."""
    try:
        await db_conn.execute("""
            INSERT INTO bandit_arms (keyword_id, arm_id, multiplier, alpha, beta, pulls, total_reward, last_updated)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (keyword_id, arm_id) 
            DO UPDATE SET alpha = $4, beta = $5, pulls = $6, total_reward = $7, last_updated = NOW()
        """, keyword_id, arm_id, multiplier, alpha, beta, pulls, total_reward)
    except Exception as e:
        logger.error(f"Failed to save bandit arm: {e}")

async def log_prediction_to_db(
    db_conn: asyncpg.Connection,
    model_name: str,
    keyword_id: int,
    campaign_id: int,
    input_features: Dict[str, Any],
    predicted_bid: float,
    confidence: float,
    reasoning: str,
    model_predictions: Optional[Dict] = None
):
    """Log AI prediction to database for audit and future training."""
    try:
        await db_conn.execute("""
            INSERT INTO prediction_logs 
            (model_name, keyword_id, campaign_id, input_features, predicted_bid, 
             confidence_score, reasoning_text, model_predictions, model_version, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'v1.0', NOW())
        """, 
            model_name,
            keyword_id,
            campaign_id,
            json.dumps(input_features),
            predicted_bid,
            confidence,
            reasoning,
            json.dumps(model_predictions) if model_predictions else None
        )
        logger.debug(f"Logged prediction for keyword {keyword_id}")
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

async def get_ai_bid_recommendation(
    keyword_data: Dict[str, Any],
    target_acos: float = 25.0,
    db_conn: Optional[asyncpg.Connection] = None
) -> Dict[str, Any]:
    """
    Get a precise bid recommendation using the AI Ensemble.
    Decides between Bandit (exploration/new) and Ensemble (proven).
    
    Args:
        keyword_data: Dictionary with keyword metrics (impressions, clicks, spend, etc.)
        target_acos: Target advertising cost of sale percentage
        db_conn: Optional database connection for logging
    """
    _initialize_models()
    
    impressions = keyword_data.get('impressions', 0)
    keyword_id = keyword_data.get('id')
    campaign_id = keyword_data.get('campaign_id')
    
    # STRATEGY: Use Bandits for Exploration (New/Low Data)
    if impressions < 1000:
        # Load bandit state from DB if available
        if db_conn:
            await load_bandit_state_from_db(db_conn, keyword_id)
        
        bandit_result = _bandit_model.select_bid_multiplier(
            features=keyword_data,
            keyword_id=keyword_id
        )
        multiplier = bandit_result['ensemble_multiplier']
        current_bid = keyword_data.get('current_bid', 1.0)
        recommended_bid = current_bid * multiplier
        
        result = {
            "recommended_bid": round(recommended_bid, 2),
            "confidence": 0.6,  # Bandits are experimental by nature
            "reasoning": f"Exploration (Bandit): Testing multiplier {multiplier}x on new keyword",
            "model": "MultiArmedBandit",
            "bandit_details": bandit_result
        }
        
        # Log to database
        if db_conn and keyword_id:
            await log_prediction_to_db(
                db_conn, "MultiArmedBandit", keyword_id, campaign_id,
                keyword_data, recommended_bid, 0.6,
                result["reasoning"], bandit_result
            )
        
        return result
    
    # STRATEGY: Use Ensemble for Exploitation (Mature Data)
    else:
        prediction: EnsemblePrediction = _ensemble_model.predict(
            features=keyword_data,
            target_acos=target_acos
        )
        
        result = {
            "recommended_bid": prediction.final_bid,
            "confidence": prediction.confidence,
            "reasoning": prediction.reasoning,
            "model": "ModelEnsemble",
            "model_predictions": prediction.model_predictions
        }
        
        # Log to database
        if db_conn and keyword_id:
            await log_prediction_to_db(
                db_conn, "ModelEnsemble", keyword_id, campaign_id,
                keyword_data, prediction.final_bid, prediction.confidence,
                prediction.reasoning, prediction.model_predictions
            )
        
        return result

async def get_budget_forecast(
    campaign_history: List[Dict[str, Any]],
    horizon_days: int = 7,
    db_conn: Optional[asyncpg.Connection] = None
) -> Dict[str, Any]:
    """
    Predict future performance to assist with budget planning.
    Uses LSTM for time series forecasting.
    """
    _initialize_models()
    
    if len(campaign_history) < 14:
        return {"error": "Insufficient history for LSTM forecasting (need 14+ days)"}
        
    forecast_result = _forecaster.forecast(campaign_history, horizon=horizon_days)
    
    # Optionally save forecast to database
    # if db_conn and campaign_history:
    #     campaign_id = campaign_history[0].get('campaign_id')
    #     # Save to forecasts table
    
    return forecast_result

async def optimize_budget_allocation(
    campaign_id: str,
    current_budget: float,
    performance_metrics: Dict[str, float],
    db_conn: Optional[asyncpg.Connection] = None
) -> Dict[str, Any]:
    """
    Use Bayesian Optimization to suggest the ideal daily budget.
    """
    _initialize_models()
    
    return _budget_optimizer.suggest_budget(
        campaign_id=campaign_id,
        current_budget=current_budget,
        metrics=performance_metrics
    )
