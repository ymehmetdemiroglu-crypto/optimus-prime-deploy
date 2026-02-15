"""
Keyword Health and Churn Prediction.
Predicts which keywords are likely to underperform or become inactive.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class KeywordHealth(str, Enum):
    """Keyword health status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AT_RISK = "at_risk"
    DECLINING = "declining"
    CRITICAL = "critical"


@dataclass
class KeywordHealthReport:
    """Keyword health assessment."""
    keyword_id: int
    keyword_text: str
    health_status: KeywordHealth
    health_score: float  # 0-100
    risk_factors: List[str]
    recommendations: List[str]
    predicted_days_to_decline: Optional[int]
    improvement_potential: float


class KeywordHealthAnalyzer:
    """
    Analyzes keyword health and predicts decline.
    """
    
    # Health thresholds
    THRESHOLDS = {
        'excellent': 85,
        'good': 70,
        'at_risk': 50,
        'declining': 30,
        'critical': 0
    }
    
    def analyze_keyword_health(
        self,
        keyword_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None,
        target_acos: float = 25.0
    ) -> KeywordHealthReport:
        """
        Comprehensive health analysis for a keyword.
        """
        keyword_id = keyword_data.get('keyword_id', 0)
        keyword_text = keyword_data.get('keyword_text', '')
        
        # Calculate individual health factors
        performance_score = self._calculate_performance_score(keyword_data, target_acos)
        trend_score = self._calculate_trend_score(historical_data) if historical_data else 50
        efficiency_score = self._calculate_efficiency_score(keyword_data)
        engagement_score = self._calculate_engagement_score(keyword_data)
        
        # Overall health score (weighted average)
        health_score = (
            performance_score * 0.35 +
            trend_score * 0.25 +
            efficiency_score * 0.25 +
            engagement_score * 0.15
        )
        
        # Determine health status
        health_status = self._determine_health_status(health_score)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            keyword_data, historical_data, target_acos
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            keyword_data, health_status, risk_factors, target_acos
        )
        
        # Predict days to decline
        days_to_decline = self._predict_decline(historical_data) if historical_data else None
        
        # Calculate improvement potential
        improvement_potential = self._calculate_improvement_potential(
            keyword_data, health_score, target_acos
        )
        
        return KeywordHealthReport(
            keyword_id=keyword_id,
            keyword_text=keyword_text,
            health_status=health_status,
            health_score=round(health_score, 1),
            risk_factors=risk_factors,
            recommendations=recommendations,
            predicted_days_to_decline=days_to_decline,
            improvement_potential=round(improvement_potential, 2)
        )
    
    def _calculate_performance_score(
        self,
        data: Dict[str, Any],
        target_acos: float
    ) -> float:
        """Calculate performance score based on ACoS and ROAS."""
        acos = data.get('acos', 100)
        roas = data.get('roas', 0)
        
        # ACoS score (lower is better)
        if acos == 0:
            acos_score = 0  # No sales
        elif acos <= target_acos * 0.5:
            acos_score = 100
        elif acos <= target_acos:
            acos_score = 80 + (target_acos - acos) / target_acos * 20
        elif acos <= target_acos * 1.5:
            acos_score = 50 + (target_acos * 1.5 - acos) / (target_acos * 0.5) * 30
        else:
            acos_score = max(0, 50 - (acos - target_acos * 1.5) / target_acos * 50)
        
        # ROAS score
        roas_score = min(100, roas / 5 * 100)
        
        return acos_score * 0.6 + roas_score * 0.4
    
    def _calculate_trend_score(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate score based on performance trends."""
        if not historical_data or len(historical_data) < 7:
            return 50  # Neutral
        
        # Calculate ROAS trend
        roas_values = [d.get('roas', 0) for d in historical_data]
        
        if len(roas_values) < 2:
            return 50
        
        # Linear regression slope
        x = np.arange(len(roas_values))
        slope = np.polyfit(x, roas_values, 1)[0]
        
        avg_roas = np.mean(roas_values)
        if avg_roas > 0:
            relative_slope = slope / avg_roas
        else:
            relative_slope = 0
        
        # Convert to score
        if relative_slope > 0.05:
            return 90  # Strong upward trend
        elif relative_slope > 0.01:
            return 70  # Slight upward
        elif relative_slope > -0.01:
            return 50  # Stable
        elif relative_slope > -0.05:
            return 30  # Slight decline
        else:
            return 10  # Strong decline
    
    def _calculate_efficiency_score(self, data: Dict[str, Any]) -> float:
        """Calculate efficiency score based on spend and conversions."""
        spend = data.get('spend', 0)
        orders = data.get('orders', 0)
        clicks = data.get('clicks', 0)
        
        if spend == 0:
            return 50  # No data
        
        # Cost per order
        cpo = spend / orders if orders > 0 else spend
        
        # Click to conversion rate
        cvr = orders / clicks if clicks > 0 else 0
        
        # Score based on CVR
        cvr_score = min(100, cvr * 10 * 100)  # 10% CVR = 100
        
        return cvr_score
    
    def _calculate_engagement_score(self, data: Dict[str, Any]) -> float:
        """Calculate engagement score based on CTR."""
        ctr = data.get('ctr', 0)
        impressions = data.get('impressions', 0)
        
        if impressions < 100:
            return 30  # Low data reliability
        
        # CTR score
        if ctr >= 1.0:
            return 100
        elif ctr >= 0.5:
            return 70 + (ctr - 0.5) / 0.5 * 30
        elif ctr >= 0.2:
            return 40 + (ctr - 0.2) / 0.3 * 30
        else:
            return ctr / 0.2 * 40
    
    def _determine_health_status(self, score: float) -> KeywordHealth:
        """Determine health status from score."""
        if score >= self.THRESHOLDS['excellent']:
            return KeywordHealth.EXCELLENT
        elif score >= self.THRESHOLDS['good']:
            return KeywordHealth.GOOD
        elif score >= self.THRESHOLDS['at_risk']:
            return KeywordHealth.AT_RISK
        elif score >= self.THRESHOLDS['declining']:
            return KeywordHealth.DECLINING
        else:
            return KeywordHealth.CRITICAL
    
    def _identify_risk_factors(
        self,
        data: Dict[str, Any],
        historical: Optional[List[Dict[str, Any]]],
        target_acos: float
    ) -> List[str]:
        """Identify risk factors for the keyword."""
        risks = []
        
        acos = data.get('acos', 0)
        ctr = data.get('ctr', 0)
        impressions = data.get('impressions', 0)
        clicks = data.get('clicks', 0)
        orders = data.get('orders', 0)
        spend = data.get('spend', 0)
        
        # High ACoS
        if acos > target_acos * 1.5:
            risks.append(f"ACoS ({acos:.1f}%) significantly above target ({target_acos}%)")
        
        # Low CTR
        if ctr < 0.3 and impressions >= 1000:
            risks.append(f"Very low CTR ({ctr:.2f}%)")
        
        # No conversions with spend
        if orders == 0 and spend > 20:
            risks.append(f"No conversions despite ${spend:.2f} spend")
        
        # Low impressions
        if impressions < 100 and impressions > 0:
            risks.append("Low impression volume - limited data")
        
        # Declining trend
        if historical and len(historical) >= 14:
            recent = historical[-7:]
            previous = historical[-14:-7]
            
            recent_roas = np.mean([d.get('roas', 0) for d in recent])
            previous_roas = np.mean([d.get('roas', 0) for d in previous])
            
            if previous_roas > 0 and recent_roas < previous_roas * 0.7:
                risks.append("ROAS declined 30%+ in last week")
        
        return risks
    
    def _generate_recommendations(
        self,
        data: Dict[str, Any],
        health_status: KeywordHealth,
        risk_factors: List[str],
        target_acos: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        acos = data.get('acos', 0)
        ctr = data.get('ctr', 0)
        bid = data.get('bid', 1.0)
        impressions = data.get('impressions', 0)
        
        if health_status == KeywordHealth.EXCELLENT:
            recommendations.append("Maintain current strategy - keyword performing well")
            if acos < target_acos * 0.5:
                recommendations.append(f"Consider increasing bid to scale volume")
        
        elif health_status == KeywordHealth.GOOD:
            recommendations.append("Minor optimizations recommended")
            if ctr < 0.5:
                recommendations.append("Consider improving ad relevance to boost CTR")
        
        elif health_status == KeywordHealth.AT_RISK:
            recommendations.append("Attention needed - performance declining")
            if acos > target_acos:
                recommendations.append(f"Reduce bid by 10-15% to improve efficiency")
            if ctr < 0.3:
                recommendations.append("Review keyword relevance and ad copy")
        
        elif health_status == KeywordHealth.DECLINING:
            recommendations.append("Urgent action required")
            recommendations.append("Consider pausing keyword temporarily")
            recommendations.append("Analyze search terms for negative keyword opportunities")
        
        else:  # CRITICAL
            recommendations.append("Immediate intervention needed")
            recommendations.append("Pause keyword and review targeting")
            if acos > target_acos * 2:
                recommendations.append("Wasted spend detected - pause immediately")
        
        return recommendations
    
    def _predict_decline(
        self,
        historical: List[Dict[str, Any]]
    ) -> Optional[int]:
        """Predict days until keyword becomes unprofitable."""
        if not historical or len(historical) < 14:
            return None
        
        roas_values = [d.get('roas', 0) for d in historical]
        
        if len(roas_values) < 2:
            return None
        
        # Calculate trend
        x = np.arange(len(roas_values))
        slope, intercept = np.polyfit(x, roas_values, 1)
        
        if slope >= 0:
            return None  # Not declining
        
        # Predict when ROAS reaches 1.0 (breakeven)
        current_roas = roas_values[-1]
        
        if current_roas <= 1:
            return 0  # Already unprofitable
        
        days_to_breakeven = (1 - current_roas) / slope
        
        return max(0, int(days_to_breakeven))
    
    def _calculate_improvement_potential(
        self,
        data: Dict[str, Any],
        current_score: float,
        target_acos: float
    ) -> float:
        """Calculate potential improvement if optimized."""
        # Max possible score is 100
        potential = 100 - current_score
        
        # Adjust based on current metrics
        acos = data.get('acos', 100)
        ctr = data.get('ctr', 0)
        
        # If ACoS is way over target, harder to improve
        if acos > target_acos * 3:
            potential *= 0.3
        elif acos > target_acos * 2:
            potential *= 0.5
        
        # Low CTR means room for improvement
        if ctr < 0.3:
            potential *= 1.2
        
        return min(100 - current_score, potential)


class KeywordLifecyclePredictor:
    """
    Predicts keyword lifecycle stage and future performance.
    """
    
    LIFECYCLE_STAGES = {
        'new': {'min_days': 0, 'max_days': 14},
        'growing': {'min_days': 14, 'max_days': 60},
        'mature': {'min_days': 60, 'max_days': 180},
        'declining': {'min_days': 180, 'max_days': 365},
        'end_of_life': {'min_days': 365, 'max_days': float('inf')}
    }
    
    def predict_lifecycle_stage(
        self,
        keyword_data: Dict[str, Any],
        days_active: int,
        performance_trend: str  # 'up', 'stable', 'down'
    ) -> Dict[str, Any]:
        """
        Predict current lifecycle stage and future trajectory.
        """
        # Determine stage based on age
        stage = 'new'
        for stage_name, bounds in self.LIFECYCLE_STAGES.items():
            if bounds['min_days'] <= days_active < bounds['max_days']:
                stage = stage_name
                break
        
        # Adjust based on performance trend
        if stage in ['growing', 'mature'] and performance_trend == 'down':
            stage = 'declining'
        elif stage == 'new' and performance_trend == 'up':
            stage = 'growing'
        
        # Predict future
        predictions = self._predict_future(stage, performance_trend, keyword_data)
        
        return {
            'current_stage': stage,
            'days_active': days_active,
            'performance_trend': performance_trend,
            'predictions': predictions,
            'recommendations': self._stage_recommendations(stage)
        }
    
    def _predict_future(
        self,
        stage: str,
        trend: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make predictions about future performance."""
        roas = data.get('roas', 1.0)
        
        if stage == 'new':
            if trend == 'up':
                return {
                    'likely_stage_in_30_days': 'growing',
                    'roas_prediction': round(roas * 1.2, 2),
                    'outlook': 'positive'
                }
            return {
                'likely_stage_in_30_days': 'growing',
                'roas_prediction': round(roas, 2),
                'outlook': 'neutral'
            }
        
        elif stage == 'growing':
            return {
                'likely_stage_in_30_days': 'mature' if trend != 'down' else 'declining',
                'roas_prediction': round(roas * (1.1 if trend == 'up' else 0.9), 2),
                'outlook': 'positive' if trend == 'up' else 'caution'
            }
        
        elif stage == 'mature':
            return {
                'likely_stage_in_30_days': 'mature' if trend != 'down' else 'declining',
                'roas_prediction': round(roas * (1.0 if trend != 'down' else 0.85), 2),
                'outlook': 'stable' if trend != 'down' else 'warning'
            }
        
        elif stage == 'declining':
            return {
                'likely_stage_in_30_days': 'end_of_life' if trend == 'down' else 'mature',
                'roas_prediction': round(roas * (0.7 if trend == 'down' else 1.0), 2),
                'outlook': 'negative' if trend == 'down' else 'recovery_possible'
            }
        
        else:  # end_of_life
            return {
                'likely_stage_in_30_days': 'end_of_life',
                'roas_prediction': round(roas * 0.5, 2),
                'outlook': 'negative'
            }
    
    def _stage_recommendations(self, stage: str) -> List[str]:
        """Get recommendations for lifecycle stage."""
        recommendations = {
            'new': [
                "Allow 2 weeks for data gathering",
                "Monitor closely but avoid premature optimization",
                "Consider increasing bids to accelerate learning"
            ],
            'growing': [
                "Scale successful campaigns",
                "Optimize bids based on performance",
                "Expand to similar keywords"
            ],
            'mature': [
                "Focus on maintaining profitability",
                "Fine-tune bids for efficiency",
                "Monitor for signs of decline"
            ],
            'declining': [
                "Reduce bids to maintain profitability",
                "Analyze for external factors",
                "Consider refreshing creative"
            ],
            'end_of_life': [
                "Evaluate if keyword still relevant",
                "Consider pausing or removing",
                "Reallocate budget to performing keywords"
            ]
        }
        
        return recommendations.get(stage, [])
