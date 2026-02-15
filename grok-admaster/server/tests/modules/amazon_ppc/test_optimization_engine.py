
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from app.modules.amazon_ppc.optimization.engine import OptimizationEngine, OptimizationStrategy, ActionType
from app.modules.amazon_ppc.models.ppc_data import PPCCampaign

# Use our new factories
from tests.factories import CampaignFactory, FeatureFactory

@pytest.mark.asyncio
class TestOptimizationEngine:
    
    @pytest.fixture
    def mock_db(self):
        return AsyncMock()
    
    @pytest.fixture
    def engine(self, mock_db):
        engine = OptimizationEngine(mock_db)
        # Mock internal components to isolate Engine logic
        engine.feature_engineer = AsyncMock()
        engine.keyword_engineer = AsyncMock()
        engine.bid_optimizer = MagicMock()
        engine.rl_agent = MagicMock()
        return engine

    async def test_aggressive_strategy_increases_bid(self, engine, mock_db):
        # Setup
        campaign = CampaignFactory.create(ai_mode="aggressive_growth")
        
        # Setup DB Mock correctly for AsyncSession
        # scalars() is synchronous, first() is synchronous
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = campaign
        mock_db.execute.return_value = mock_result # await db.execute returns mock_result
        
        # Features: High performance, room to grow
        engine.feature_engineer.compute_full_feature_vector.return_value = {"sales_trend": 1.2}
        kw_features = FeatureFactory.create_keyword_features(
            keyword_id=101, bid=1.0, acos=15.0, data_maturity=1.0
        )
        engine.keyword_engineer.bulk_compute_features.return_value = [kw_features]
        
        # Mock ML predictions to support increase
        # predicted_bid must be float
        prediction_mock = MagicMock()
        prediction_mock.predicted_bid = 1.3
        prediction_mock.confidence = 0.9
        prediction_mock.reasoning = "Good"
        engine.bid_optimizer.predict_bid.return_value = prediction_mock
        
        engine.rl_agent.get_bid_recommendation.return_value = {
            'recommended_bid': 1.3, 'confidence': 0.8
        }

        # Execute
        plan = await engine.generate_optimization_plan(
            campaign_id=campaign.id, 
            strategy=OptimizationStrategy.AGGRESSIVE,
            target_acos=30.0
        )
        
        # Assert
        assert plan is not None
        assert len(plan.actions) >= 1, f"Expected actions, got none. Data maturity: {kw_features['data_maturity']}"
        action = plan.actions[0]
        assert action.action_type == ActionType.BID_INCREASE
        assert action.change_percent > 0
        # Aggressive allows up to 30% increase
        assert action.recommended_value <= 1.0 * 1.30 

    async def test_profit_strategy_decreases_bid_on_high_acos(self, engine, mock_db):
        # Setup
        campaign = CampaignFactory.create(ai_mode="profit_guard")
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = campaign
        mock_db.execute.return_value = mock_result
        
        # FIX: Set campaign features to empty dict
        engine.feature_engineer.compute_full_feature_vector.return_value = {}
        
        # Features: Poor performance
        kw_features = FeatureFactory.create_keyword_features(
            keyword_id=202, bid=2.0, acos=45.0, data_maturity=1.0
        )
        engine.keyword_engineer.bulk_compute_features.return_value = [kw_features]
        
        # Mock ML predictions to support decrease
        pred_mock = MagicMock()
        pred_mock.predicted_bid = 1.5
        pred_mock.confidence = 0.9
        pred_mock.reasoning = "Bad"
        engine.bid_optimizer.predict_bid.return_value = pred_mock
        
        engine.rl_agent.get_bid_recommendation.return_value = {
            'recommended_bid': 1.4, 'confidence': 0.8
        }

        # Execute
        plan = await engine.generate_optimization_plan(
            campaign_id=campaign.id, 
            strategy=OptimizationStrategy.PROFIT_FOCUSED,
            target_acos=25.0 # Target 25, Actual 45
        )
        
        # Assert (Wait, params for Profit: max_decrease=0.30)
        # 1.45 avg bid. 2.0 -> 1.45 is -27.5%. Allowed.
        assert len(plan.actions) >= 1
        action = plan.actions[0]
        assert action.action_type == ActionType.BID_DECREASE
        assert action.current_value == 2.0
        assert action.recommended_value < 2.0

    async def test_skips_low_data_maturity(self, engine, mock_db):
        campaign = CampaignFactory.create()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = campaign
        mock_db.execute.return_value = mock_result
        
        engine.feature_engineer.compute_full_feature_vector.return_value = {}
        
        # Features: New keyword
        kw_features = FeatureFactory.create_keyword_features(
            keyword_id=303, bid=1.0, acos=0.0, clicks=2, data_maturity=0.1
        )
        engine.keyword_engineer.bulk_compute_features.return_value = [kw_features]
        
        plan = await engine.generate_optimization_plan(campaign.id)
        
        assert len(plan.actions) == 0 # Should skip

    async def test_pauses_extremely_poor_performer(self, engine, mock_db):
        campaign = CampaignFactory.create()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = campaign
        mock_db.execute.return_value = mock_result
        
        engine.feature_engineer.compute_full_feature_vector.return_value = {}
        
        # Features: Terrible performance (ACoS 100% vs Target 25%)
        kw_features = FeatureFactory.create_keyword_features(
            keyword_id=404, bid=1.0, acos=100.0, clicks=60, data_maturity=1.0
        )
        engine.keyword_engineer.bulk_compute_features.return_value = [kw_features]
             
        # Predictions (even if ML is gentle, heuristics might override)
        pred_mock = MagicMock()
        pred_mock.predicted_bid = 0.5
        pred_mock.confidence = 0.5
        pred_mock.reasoning = "Bad"
        engine.bid_optimizer.predict_bid.return_value = pred_mock
        
        engine.rl_agent.get_bid_recommendation.return_value = {
            'recommended_bid': 0.5, 'confidence': 0.5
        }

        plan = await engine.generate_optimization_plan(
            campaign.id, target_acos=25.0
        )
        
        assert len(plan.actions) >= 1
        action = plan.actions[0]
        assert action.action_type == ActionType.PAUSE_KEYWORD
        assert action.priority == 10 # High priority
