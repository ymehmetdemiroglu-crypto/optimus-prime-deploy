
import pytest
from unittest.mock import MagicMock, AsyncMock
from app.modules.amazon_ppc.features.engineer import FeatureEngineer
from app.modules.amazon_ppc.ml.bid_optimizer import BidOptimizer
from app.modules.amazon_ppc.strategies.config import BidStrategyConfig

@pytest.mark.asyncio
async def test_feature_to_model_wiring():
    """
    Integration Test: Verifies that FeatureEngine output can be legally consumed by BidOptimizer.
    Ensures column names match and types are correct.
    """
    # 1. Setup Feature Engineer with Mock DB
    mock_db = AsyncMock()
    engineer = FeatureEngineer(mock_db)
    
    # Mock Data for Rolling Metrics
    mock_row = MagicMock()
    mock_row.impressions = 1000
    mock_row.clicks = 50
    mock_row.spend = 50.0
    mock_row.sales = 200.0
    mock_row.orders = 5
    
    mock_result_metrics = MagicMock()
    mock_result_metrics.first.return_value = mock_row
    
    # Mock Data for Competition
    mock_comp_records = [
        MagicMock(spend=1.0, clicks=1, impressions=100),
        MagicMock(spend=1.2, clicks=1, impressions=100)
    ]
    mock_result_comp = MagicMock()
    mock_result_comp.all.return_value = mock_comp_records
    
    # Mock Data for Trends (Short & Long)
    mock_trend = MagicMock()
    mock_trend.avg_spend = 50.0
    mock_trend.avg_sales = 200.0
    mock_trend.clicks = 50
    mock_trend.impressions = 1000

    mock_result_trend = MagicMock()
    mock_result_trend.first.return_value = mock_trend

    # Side Effect sequence: 
    # 1-3. Rolling metrics (7, 14, 30 days) -> 3 calls
    # 4-5. Trend (Short, Long) -> 2 calls
    # 6. Competition -> 1 call
    # Order inside compute_full_feature_vector: Rolling, Trend, Competition (Wait, let's check order in code)
    # Code: Rolling -> Seasonality (no db) -> Trends -> Competition.
    
    # Rolling (loop 3 times)
    side_effect = [mock_result_metrics, mock_result_metrics, mock_result_metrics] 
    # Trends (Short, Long)
    side_effect.extend([mock_result_trend, mock_result_trend])
    # Competition
    side_effect.extend([mock_result_comp])
    
    mock_db.execute.side_effect = side_effect
    
    # 2. Generate Features
    features = await engineer.compute_full_feature_vector(campaign_id=123)
    
    # Verify features exist
    assert 'ctr_30d' in features
    assert 'sales_trend' in features
    assert 'cpc_volatility' in features
    
    # 3. Feed to Bid Optimizer
    # We use an untramed optimizer for check, or mocking the internal model if needed.
    # But BidOptimizer handles rule-based fallback if model not trained.
    # This checks if valid features explode the predictor (e.g. type errors).
    
    optimizer = BidOptimizer()
    
    # Inject current_bid and valid context usually found in kwargs or extra features
    features['current_bid'] = 1.0
    features['keyword_id'] = 999
    
    config = BidStrategyConfig(target_acos=30.0)
    
    try:
        prediction = optimizer.predict_bid(features, config=config)
        assert prediction.predicted_bid > 0
        assert prediction.keyword_id == 999
    except Exception as e:
        pytest.fail(f"Wiring failed: Feature output crashed Optimizer: {e}")
