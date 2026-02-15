"""
Performance Forecasting Model.
Predicts future campaign metrics using time series analysis.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, date
import logging

logger = logging.getLogger(__name__)

@dataclass
class Forecast:
    """Forecast result for a metric."""
    metric: str
    current_value: float
    forecasted_values: List[float]
    dates: List[str]
    confidence_lower: List[float]
    confidence_upper: List[float]
    trend: str  # 'up', 'down', 'stable'


class PerformanceForecaster:
    """
    Time series forecasting for PPC metrics.
    Uses exponential smoothing and trend analysis.
    """
    
    def __init__(self):
        self.smoothing_factor = 0.3  # Alpha for exponential smoothing
        self.trend_factor = 0.1     # Beta for trend smoothing
    
    def forecast_metric(
        self,
        historical_values: List[float],
        horizon: int = 7,
        confidence: float = 0.95
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Forecast a single metric using Holt's linear trend method.
        
        Returns: (forecast, lower_bound, upper_bound)
        """
        if len(historical_values) < 3:
            # Not enough data - return flat forecast
            last_val = historical_values[-1] if historical_values else 0
            forecast = [last_val] * horizon
            return forecast, forecast, forecast
        
        values = np.array(historical_values)
        n = len(values)
        
        # Initialize level and trend
        level = values[0]
        trend = (values[-1] - values[0]) / n
        
        # Smooth through historical data
        smoothed_level = level
        smoothed_trend = trend
        residuals = []
        
        for i in range(1, n):
            prev_level = smoothed_level
            smoothed_level = self.smoothing_factor * values[i] + (1 - self.smoothing_factor) * (smoothed_level + smoothed_trend)
            smoothed_trend = self.trend_factor * (smoothed_level - prev_level) + (1 - self.trend_factor) * smoothed_trend
            
            # Track residuals for confidence intervals
            forecast_val = prev_level + smoothed_trend
            residuals.append(values[i] - forecast_val)
        
        # Calculate forecast
        forecast = []
        for h in range(1, horizon + 1):
            forecast.append(smoothed_level + h * smoothed_trend)
        
        # Calculate confidence intervals
        std_residual = np.std(residuals) if residuals else values.std()
        z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90% CI
        
        lower = [max(0, f - z * std_residual * np.sqrt(h)) for h, f in enumerate(forecast, 1)]
        upper = [f + z * std_residual * np.sqrt(h) for h, f in enumerate(forecast, 1)]
        
        return forecast, lower, upper
    
    def detect_trend(self, values: List[float]) -> str:
        """Detect if metric is trending up, down, or stable."""
        if len(values) < 3:
            return 'stable'
        
        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize by mean
        mean_val = np.mean(values)
        if mean_val == 0:
            return 'stable'
        
        relative_slope = slope / mean_val
        
        if relative_slope > 0.02:
            return 'up'
        elif relative_slope < -0.02:
            return 'down'
        return 'stable'
    
    def forecast_campaign(
        self,
        historical_data: List[Dict[str, Any]],
        horizon: int = 7
    ) -> Dict[str, Forecast]:
        """
        Forecast all key metrics for a campaign.
        
        historical_data: List of daily performance records
        """
        metrics_to_forecast = ['impressions', 'clicks', 'spend', 'sales', 'orders']
        forecasts = {}
        
        start_date = date.today() + timedelta(days=1)
        dates = [(start_date + timedelta(days=i)).isoformat() for i in range(horizon)]
        
        for metric in metrics_to_forecast:
            values = [float(d.get(metric, 0)) for d in historical_data]
            
            if not values:
                continue
            
            forecast_vals, lower, upper = self.forecast_metric(values, horizon)
            trend = self.detect_trend(values)
            
            forecasts[metric] = Forecast(
                metric=metric,
                current_value=values[-1] if values else 0,
                forecasted_values=[round(v, 2) for v in forecast_vals],
                dates=dates,
                confidence_lower=[round(v, 2) for v in lower],
                confidence_upper=[round(v, 2) for v in upper],
                trend=trend
            )
        
        # Derived metrics
        if 'spend' in forecasts and 'sales' in forecasts:
            spend_forecast = forecasts['spend'].forecasted_values
            sales_forecast = forecasts['sales'].forecasted_values
            
            acos_forecast = []
            for s, rev in zip(spend_forecast, sales_forecast):
                acos = (s / rev * 100) if rev > 0 else 0
                acos_forecast.append(round(acos, 2))
            
            forecasts['acos'] = Forecast(
                metric='acos',
                current_value=round((forecasts['spend'].current_value / max(1, forecasts['sales'].current_value)) * 100, 2),
                forecasted_values=acos_forecast,
                dates=dates,
                confidence_lower=acos_forecast,  # Simplified
                confidence_upper=acos_forecast,
                trend=self.detect_trend([d.get('acos', 0) for d in historical_data[-14:]])
            )
        
        return forecasts
    
    def forecast_budget_pacing(
        self,
        daily_spend: List[float],
        daily_budget: float,
        days_remaining: int
    ) -> Dict[str, Any]:
        """
        Forecast budget pacing and expected spend.
        """
        if not daily_spend:
            return {'status': 'no_data'}
        
        avg_daily_spend = np.mean(daily_spend[-7:])  # Last 7 days
        total_spent = sum(daily_spend)
        
        # Forecast remaining spend
        forecast_spend, _, _ = self.forecast_metric(daily_spend, days_remaining)
        projected_total = total_spent + sum(forecast_spend)
        
        # Budget for period
        total_budget = daily_budget * (len(daily_spend) + days_remaining)
        
        # Pacing
        expected_daily = total_budget / (len(daily_spend) + days_remaining)
        pacing_ratio = avg_daily_spend / expected_daily if expected_daily > 0 else 1
        
        if pacing_ratio < 0.8:
            pacing_status = 'underspent'
            recommendation = f"Increase daily spend by ${expected_daily - avg_daily_spend:.2f}"
        elif pacing_ratio > 1.2:
            pacing_status = 'overspent'
            recommendation = f"Reduce daily spend by ${avg_daily_spend - expected_daily:.2f}"
        else:
            pacing_status = 'on_pace'
            recommendation = "Maintain current pacing"
        
        return {
            'avg_daily_spend': round(avg_daily_spend, 2),
            'daily_budget': daily_budget,
            'pacing_ratio': round(pacing_ratio, 2),
            'pacing_status': pacing_status,
            'total_spent': round(total_spent, 2),
            'projected_total': round(projected_total, 2),
            'total_budget': round(total_budget, 2),
            'recommendation': recommendation
        }
    
    def detect_anomalies(
        self,
        values: List[float],
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metric values using z-score method.
        """
        if len(values) < 7:
            return []
        
        anomalies = []
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        for i, val in enumerate(values):
            z_score = abs(val - mean) / std
            if z_score > threshold:
                anomalies.append({
                    'index': i,
                    'value': val,
                    'z_score': round(z_score, 2),
                    'type': 'spike' if val > mean else 'drop'
                })
        
        return anomalies
