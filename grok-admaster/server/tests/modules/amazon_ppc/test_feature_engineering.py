
import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from app.modules.amazon_ppc.features.engineer import FeatureEngineer
from app.modules.amazon_ppc.models.ppc_data import PerformanceRecord

class TestFeatureEngineer:
    
    @pytest.fixture
    def engineer(self):
        # We mock DB for init, but seasonality doesn't use it
        return FeatureEngineer(AsyncMock())

    # ==================== SEASONALITY TESTS ====================
    
    def test_seasonality_detects_weekend(self, engineer):
        # Saturday
        dt = date(2023, 10, 7) 
        features = engineer.compute_seasonality_features(dt)
        assert features['is_weekend'] is True
        assert features['day_of_week'] == 5 # Sat

        # Monday
        dt = date(2023, 10, 9)
        features = engineer.compute_seasonality_features(dt)
        assert features['is_weekend'] is False

    def test_seasonality_detects_prime_day(self, engineer):
        # Prime Day 2023: July 11-12 (Hardcoded in logic? Yes, 7, 11)
        dt = date(2023, 7, 11) 
        features = engineer.compute_seasonality_features(dt)
        assert features['is_prime_day'] is True
        
        dt = date(2023, 7, 13)
        features = engineer.compute_seasonality_features(dt)
        assert features['is_prime_day'] is False

    def test_seasonality_detects_black_friday(self, engineer):
        # Black Friday 2023: Nov 24
        # 1st Nov 2023 is Wednesday. 
        # Thanksgiving is 4th Thursday.
        # Nov 1 (Wed) -> 1st Thu (2) -> 4th Thu (23). BF = 24.
        dt = date(2023, 11, 24)
        features = engineer.compute_seasonality_features(dt)
        assert features['is_black_friday'] is True
        
        dt = date(2023, 11, 23) # Thanksgiving
        features = engineer.compute_seasonality_features(dt)
        assert features['is_black_friday'] is False

    # ==================== ROLLING METRICS TESTS (Math & Zero Handling) ====================
    
    @pytest.mark.asyncio
    async def test_rolling_metrics_handles_zero_division(self, engineer):
        # Mock DB Result: 100 Impressions, 0 Clicks (CTR 0, CPC 0)
        mock_row = MagicMock()
        mock_row.impressions = 100
        mock_row.clicks = 0
        mock_row.spend = 0
        mock_row.sales = 0
        mock_row.orders = 0
        
        mock_result = MagicMock()
        mock_result.first.return_value = mock_row
        engineer.db.execute.return_value = mock_result
        
        features = await engineer.compute_rolling_metrics(campaign_id=1, windows=[7])
        
        assert features['ctr_7d'] == 0.0
        assert features['cpc_7d'] == 0.0
        assert features['roas_7d'] == 0.0
        assert features['acos_7d'] == 0.0 # Should be 0, not Inf

    @pytest.mark.asyncio
    async def test_rolling_metrics_calculates_correctly(self, engineer):
        # Mock DB Result: Standard data
        mock_row = MagicMock()
        mock_row.impressions = 1000
        mock_row.clicks = 50
        mock_row.spend = 50.0 # CPC $1
        mock_row.sales = 200.0 # ROAS 4.0
        mock_row.orders = 5
        
        mock_result = MagicMock()
        mock_result.first.return_value = mock_row
        engineer.db.execute.return_value = mock_result
        
        features = await engineer.compute_rolling_metrics(campaign_id=1, windows=[30])
        
        # CTR: 50/1000 = 5%
        assert features['ctr_30d'] == 5.0
        
        # ACoS: 50 / 200 = 25%
        assert features['acos_30d'] == 25.0
        
        # ROAS: 200 / 50 = 4.0
        assert features['roas_30d'] == 4.0
        
        # CPA: 50 / 5 = 10.0
        assert features['cpa_30d'] == 10.0

    # ==================== TREND FEATURES (Detailed Logic) ====================
    
    @pytest.mark.asyncio
    async def test_trend_handles_declining_momentum(self, engineer):
        # Short Term: Bad (Low CTR, Low Sales)
        short = MagicMock()
        short.avg_spend = 100
        short.avg_sales = 100
        short.clicks = 10
        short.impressions = 1000 # CTR 1%
        
        # Long Term: Good (High CTR, High Sales)
        long = MagicMock()
        long.avg_spend = 100
        long.avg_sales = 200
        long.clicks = 40
        long.impressions = 2000 # CTR 2%
        
        # Mock 2 DB calls (short result, long result)
        res_short = MagicMock()
        res_short.first.return_value = short
        
        res_long = MagicMock()
        res_long.first.return_value = long
        
        engineer.db.execute.side_effect = [res_short, res_long]
        
        features = await engineer.compute_trend_features(campaign_id=1)
        
        # Spend Trend: 100/100 = 1.0. (1 - 1.0)*0.3 = 0 contribution
        # Sales Trend: 100/200 = 0.5. (0.5 - 1)*0.4 = -0.2
        # CTR Trend: 0.01 / 0.02 = 0.5. (0.5 - 1)*0.3 = -0.15
        # Momentum: -0.2 - 0.15 + 0 = -0.35
        
        assert features['sales_trend'] == 0.5
        assert features['ctr_trend'] == 0.5
        assert -0.4 <= features['momentum'] <= -0.3 # Approx check
