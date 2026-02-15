"""
Advanced Rule Engine - ML-enhanced rule evaluation.
Integrates anomaly detection, keyword health, and predictive triggers.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np

from ..models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord
from ..ml import PPCAnomalyDetector, KeywordHealthAnalyzer, PerformanceForecaster
from .engine import OptimizationAction, ActionType

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TriggerType(str, Enum):
    """Types of triggers."""
    # Rule-based
    THRESHOLD = "threshold"
    TREND = "trend"
    
    # ML-based
    ANOMALY = "anomaly"
    HEALTH_DECLINE = "health_decline"
    FORECAST_ALERT = "forecast_alert"
    PATTERN = "pattern"


@dataclass
class SmartAlert:
    """ML-enhanced alert."""
    trigger_type: TriggerType
    severity: AlertSeverity
    entity_type: str
    entity_id: int
    title: str
    message: str
    metrics: Dict[str, Any]
    recommended_actions: List[Dict[str, Any]]
    ml_confidence: float
    triggered_at: datetime
    auto_resolve: bool = False
    resolution_eta: Optional[str] = None


@dataclass
class RuleConfig:
    """Configuration for a smart rule."""
    name: str
    trigger_type: TriggerType
    enabled: bool = True
    severity: AlertSeverity = AlertSeverity.WARNING
    
    # Threshold settings
    metric: Optional[str] = None
    threshold_value: Optional[float] = None
    threshold_direction: str = 'above'  # 'above' or 'below'
    
    # Time settings
    lookback_days: int = 7
    cooldown_hours: int = 24
    
    # ML settings
    ml_confidence_threshold: float = 0.6
    use_anomaly_detection: bool = False
    use_forecasting: bool = False
    
    # Action settings
    auto_action: Optional[ActionType] = None
    action_magnitude: float = 0.10  # 10% change


class AdvancedRuleEngine:
    """
    ML-enhanced rule engine with intelligent alerting.
    """
    
    DEFAULT_RULES = [
        # === THRESHOLD RULES ===
        RuleConfig(
            name="High ACoS Alert",
            trigger_type=TriggerType.THRESHOLD,
            metric="acos",
            threshold_value=50.0,
            threshold_direction='above',
            severity=AlertSeverity.CRITICAL,
            auto_action=ActionType.BID_DECREASE,
            action_magnitude=0.15
        ),
        RuleConfig(
            name="Low ROAS Warning",
            trigger_type=TriggerType.THRESHOLD,
            metric="roas",
            threshold_value=2.0,
            threshold_direction='below',
            severity=AlertSeverity.WARNING,
            auto_action=ActionType.BID_DECREASE,
            action_magnitude=0.10
        ),
        RuleConfig(
            name="Budget Utilization Alert",
            trigger_type=TriggerType.THRESHOLD,
            metric="budget_utilization",
            threshold_value=95.0,
            threshold_direction='above',
            severity=AlertSeverity.INFO,
            auto_action=ActionType.BUDGET_INCREASE,
            action_magnitude=0.20
        ),
        
        # === ML-BASED RULES ===
        RuleConfig(
            name="Spend Anomaly Detection",
            trigger_type=TriggerType.ANOMALY,
            metric="spend",
            severity=AlertSeverity.CRITICAL,
            use_anomaly_detection=True,
            ml_confidence_threshold=0.7
        ),
        RuleConfig(
            name="Performance Anomaly",
            trigger_type=TriggerType.ANOMALY,
            metric="overall",
            severity=AlertSeverity.WARNING,
            use_anomaly_detection=True,
            ml_confidence_threshold=0.65
        ),
        RuleConfig(
            name="Keyword Health Decline",
            trigger_type=TriggerType.HEALTH_DECLINE,
            severity=AlertSeverity.WARNING,
            ml_confidence_threshold=0.6
        ),
        RuleConfig(
            name="Negative Performance Forecast",
            trigger_type=TriggerType.FORECAST_ALERT,
            severity=AlertSeverity.INFO,
            use_forecasting=True,
            ml_confidence_threshold=0.7
        ),
        
        # === TREND RULES ===
        RuleConfig(
            name="Declining CTR Trend",
            trigger_type=TriggerType.TREND,
            metric="ctr",
            threshold_value=-0.20,  # 20% decline
            threshold_direction='below',
            severity=AlertSeverity.WARNING,
            lookback_days=14
        ),
        RuleConfig(
            name="Rising CPC Trend",
            trigger_type=TriggerType.TREND,
            metric="cpc",
            threshold_value=0.30,  # 30% increase
            threshold_direction='above',
            severity=AlertSeverity.WARNING,
            lookback_days=14
        ),
    ]
    
    def __init__(self, db: AsyncSession, rules: List[RuleConfig] = None):
        self.db = db
        self.rules = rules or self.DEFAULT_RULES
        
        # ML components
        self.anomaly_detector = PPCAnomalyDetector()
        self.health_analyzer = KeywordHealthAnalyzer()
        self.forecaster = PerformanceForecaster()
        
        # Cooldown tracking
        self.triggered_alerts: Dict[str, datetime] = {}
    
    async def evaluate_campaign(
        self,
        campaign_id: int,
        target_acos: float = 25.0
    ) -> List[SmartAlert]:
        """
        Evaluate all rules for a campaign using ML capabilities.
        """
        alerts = []
        
        # Get campaign
        query = select(PPCCampaign).where(PPCCampaign.id == campaign_id)
        result = await self.db.execute(query)
        campaign = result.scalars().first()
        
        if not campaign:
            return alerts
        
        # Get performance data
        historical_data = await self._get_historical_data(campaign_id, days=30)
        current_metrics = await self._get_current_metrics(campaign_id, campaign)
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(campaign_id, rule.name):
                continue
            
            alert = await self._evaluate_rule(
                rule, campaign, historical_data, current_metrics, target_acos
            )
            
            if alert:
                self._record_trigger(campaign_id, rule.name)
                alerts.append(alert)
        
        return alerts
    
    async def _get_historical_data(
        self,
        campaign_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical performance data."""
        cutoff = datetime.now() - timedelta(days=days)
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
        
        return [
            {
                'date': r.date,
                'impressions': r.impressions,
                'clicks': r.clicks,
                'spend': float(r.spend),
                'sales': float(r.sales),
                'orders': r.orders,
                'acos': float(r.spend / r.sales * 100) if r.sales > 0 else 0,
                'ctr': r.clicks / r.impressions * 100 if r.impressions > 0 else 0,
                'cpc': float(r.spend / r.clicks) if r.clicks > 0 else 0,
                'roas': float(r.sales / r.spend) if r.spend > 0 else 0
            }
            for r in records
        ]
    
    async def _get_current_metrics(
        self,
        campaign_id: int,
        campaign: PPCCampaign
    ) -> Dict[str, Any]:
        """Get current aggregated metrics."""
        cutoff = datetime.now() - timedelta(days=7)
        query = (
            select(PerformanceRecord)
            .where(
                and_(
                    PerformanceRecord.campaign_id == campaign_id,
                    PerformanceRecord.date >= cutoff
                )
            )
        )
        result = await self.db.execute(query)
        records = result.scalars().all()
        
        if not records:
            return {}
        
        total_spend = sum(float(r.spend) for r in records)
        total_sales = sum(float(r.sales) for r in records)
        total_clicks = sum(r.clicks for r in records)
        total_impressions = sum(r.impressions for r in records)
        
        daily_budget = float(campaign.daily_budget or 0)
        avg_daily_spend = total_spend / 7
        
        return {
            'spend': total_spend,
            'sales': total_sales,
            'clicks': total_clicks,
            'impressions': total_impressions,
            'acos': (total_spend / total_sales * 100) if total_sales > 0 else 0,
            'roas': (total_sales / total_spend) if total_spend > 0 else 0,
            'ctr': (total_clicks / total_impressions * 100) if total_impressions > 0 else 0,
            'cpc': (total_spend / total_clicks) if total_clicks > 0 else 0,
            'budget_utilization': (avg_daily_spend / daily_budget * 100) if daily_budget > 0 else 0
        }
    
    async def _evaluate_rule(
        self,
        rule: RuleConfig,
        campaign: PPCCampaign,
        historical_data: List[Dict[str, Any]],
        current_metrics: Dict[str, Any],
        target_acos: float
    ) -> Optional[SmartAlert]:
        """Evaluate a single rule."""
        
        if rule.trigger_type == TriggerType.THRESHOLD:
            return self._evaluate_threshold_rule(rule, campaign, current_metrics)
        
        elif rule.trigger_type == TriggerType.TREND:
            return self._evaluate_trend_rule(rule, campaign, historical_data)
        
        elif rule.trigger_type == TriggerType.ANOMALY:
            return await self._evaluate_anomaly_rule(
                rule, campaign, historical_data, current_metrics
            )
        
        elif rule.trigger_type == TriggerType.HEALTH_DECLINE:
            return await self._evaluate_health_rule(rule, campaign, target_acos)
        
        elif rule.trigger_type == TriggerType.FORECAST_ALERT:
            return self._evaluate_forecast_rule(rule, campaign, historical_data)
        
        return None
    
    def _evaluate_threshold_rule(
        self,
        rule: RuleConfig,
        campaign: PPCCampaign,
        metrics: Dict[str, Any]
    ) -> Optional[SmartAlert]:
        """Evaluate threshold-based rule."""
        if not rule.metric or rule.metric not in metrics:
            return None
        
        current_value = metrics[rule.metric]
        
        triggered = False
        if rule.threshold_direction == 'above':
            triggered = current_value > rule.threshold_value
        else:
            triggered = current_value < rule.threshold_value
        
        if not triggered:
            return None
        
        # Generate recommended action
        actions = []
        if rule.auto_action:
            actions.append({
                'type': rule.auto_action.value,
                'magnitude': rule.action_magnitude,
                'reasoning': f"{rule.metric} is {current_value:.2f} ({rule.threshold_direction} threshold {rule.threshold_value})"
            })
        
        return SmartAlert(
            trigger_type=rule.trigger_type,
            severity=rule.severity,
            entity_type='campaign',
            entity_id=campaign.id,
            title=rule.name,
            message=f"{rule.metric.upper()} is {current_value:.2f}, {rule.threshold_direction} threshold of {rule.threshold_value}",
            metrics={rule.metric: current_value, 'threshold': rule.threshold_value},
            recommended_actions=actions,
            ml_confidence=1.0,  # Threshold rules are deterministic
            triggered_at=datetime.now()
        )
    
    def _evaluate_trend_rule(
        self,
        rule: RuleConfig,
        campaign: PPCCampaign,
        historical_data: List[Dict[str, Any]]
    ) -> Optional[SmartAlert]:
        """Evaluate trend-based rule."""
        if not rule.metric or len(historical_data) < 14:
            return None
        
        values = [d.get(rule.metric, 0) for d in historical_data]
        
        # Compare recent vs previous period
        mid = len(values) // 2
        recent_avg = np.mean(values[mid:])
        previous_avg = np.mean(values[:mid])
        
        if previous_avg == 0:
            return None
        
        change_pct = (recent_avg - previous_avg) / previous_avg
        
        triggered = False
        if rule.threshold_direction == 'above':
            triggered = change_pct > rule.threshold_value
        else:
            triggered = change_pct < rule.threshold_value
        
        if not triggered:
            return None
        
        direction = "increased" if change_pct > 0 else "decreased"
        
        return SmartAlert(
            trigger_type=rule.trigger_type,
            severity=rule.severity,
            entity_type='campaign',
            entity_id=campaign.id,
            title=rule.name,
            message=f"{rule.metric.upper()} has {direction} by {abs(change_pct)*100:.1f}% over the last {rule.lookback_days} days",
            metrics={
                rule.metric + '_recent': recent_avg,
                rule.metric + '_previous': previous_avg,
                'change_percent': change_pct * 100
            },
            recommended_actions=[],
            ml_confidence=0.85,
            triggered_at=datetime.now()
        )
    
    async def _evaluate_anomaly_rule(
        self,
        rule: RuleConfig,
        campaign: PPCCampaign,
        historical_data: List[Dict[str, Any]],
        current_metrics: Dict[str, Any]
    ) -> Optional[SmartAlert]:
        """Evaluate ML anomaly detection rule."""
        if not historical_data:
            return None
        
        anomalies = self.anomaly_detector.detect_campaign_anomalies(
            campaign.id,
            historical_data,
            current_metrics
        )
        
        if not anomalies:
            return None
        
        # Get most severe anomaly
        severity_order = {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}
        anomalies.sort(key=lambda a: severity_order.get(a.severity.value, 0), reverse=True)
        top_anomaly = anomalies[0]
        
        # Check confidence threshold
        if top_anomaly.deviation < rule.ml_confidence_threshold * 3:  # Z-score threshold
            return None
        
        return SmartAlert(
            trigger_type=rule.trigger_type,
            severity=AlertSeverity(top_anomaly.severity.value) if top_anomaly.severity.value in ['warning', 'critical'] else AlertSeverity.WARNING,
            entity_type='campaign',
            entity_id=campaign.id,
            title=f"Anomaly Detected: {top_anomaly.anomaly_type.value}",
            message=top_anomaly.message,
            metrics={
                'expected': top_anomaly.expected_value,
                'actual': top_anomaly.actual_value,
                'deviation': top_anomaly.deviation
            },
            recommended_actions=[{
                'type': 'investigate',
                'reasoning': top_anomaly.recommended_action
            }],
            ml_confidence=min(1.0, top_anomaly.deviation / 5),
            triggered_at=datetime.now()
        )
    
    async def _evaluate_health_rule(
        self,
        rule: RuleConfig,
        campaign: PPCCampaign,
        target_acos: float
    ) -> Optional[SmartAlert]:
        """Evaluate keyword health decline rule."""
        # Get keywords
        query = select(PPCKeyword).where(PPCKeyword.campaign_id == campaign.id)
        result = await self.db.execute(query)
        keywords = result.scalars().all()
        
        critical_keywords = []
        for kw in keywords:
            kw_data = {
                'keyword_id': kw.id,
                'acos': 50,  # Would need actual calculation
                'ctr': 0.5,
                'impressions': 1000,
                'clicks': 10
            }
            
            report = self.health_analyzer.analyze_keyword_health(kw_data, None, target_acos)
            
            if report.health_status.value in ['critical', 'declining']:
                critical_keywords.append({
                    'keyword_id': kw.id,
                    'status': report.health_status.value,
                    'score': report.health_score
                })
        
        if not critical_keywords:
            return None
        
        return SmartAlert(
            trigger_type=rule.trigger_type,
            severity=AlertSeverity.WARNING,
            entity_type='campaign',
            entity_id=campaign.id,
            title="Keyword Health Decline",
            message=f"{len(critical_keywords)} keywords are in poor health status",
            metrics={
                'critical_count': len(critical_keywords),
                'keywords': critical_keywords[:5]  # Top 5
            },
            recommended_actions=[{
                'type': 'review_keywords',
                'reasoning': 'Review and optimize or pause unhealthy keywords'
            }],
            ml_confidence=0.8,
            triggered_at=datetime.now()
        )
    
    def _evaluate_forecast_rule(
        self,
        rule: RuleConfig,
        campaign: PPCCampaign,
        historical_data: List[Dict[str, Any]]
    ) -> Optional[SmartAlert]:
        """Evaluate forecast-based rule."""
        if len(historical_data) < 14:
            return None
        
        # Forecast sales
        sales_values = [d['sales'] for d in historical_data]
        
        try:
            forecast, lower, upper = self.forecaster.forecast_metric(sales_values, horizon=7)
            
            # Check for declining forecast
            current_avg = np.mean(sales_values[-7:])
            forecast_avg = np.mean(forecast)
            
            if forecast_avg < current_avg * 0.8:  # 20% decline predicted
                return SmartAlert(
                    trigger_type=rule.trigger_type,
                    severity=AlertSeverity.INFO,
                    entity_type='campaign',
                    entity_id=campaign.id,
                    title="Performance Decline Forecast",
                    message=f"Sales predicted to decline by {(1 - forecast_avg/current_avg)*100:.1f}% next week",
                    metrics={
                        'current_avg': current_avg,
                        'forecast_avg': forecast_avg,
                        'forecast': forecast
                    },
                    recommended_actions=[{
                        'type': 'proactive_optimization',
                        'reasoning': 'Consider proactive bid adjustments to maintain performance'
                    }],
                    ml_confidence=rule.ml_confidence_threshold,
                    triggered_at=datetime.now()
                )
        except Exception as e:
            logger.warning(f"Forecast evaluation failed: {e}")
        
        return None
    
    def _is_in_cooldown(self, campaign_id: int, rule_name: str) -> bool:
        """Check if rule is in cooldown period."""
        key = f"{campaign_id}:{rule_name}"
        last_triggered = self.triggered_alerts.get(key)
        
        if last_triggered:
            rule = next((r for r in self.rules if r.name == rule_name), None)
            if rule:
                cooldown_end = last_triggered + timedelta(hours=rule.cooldown_hours)
                return datetime.now() < cooldown_end
        
        return False
    
    def _record_trigger(self, campaign_id: int, rule_name: str):
        """Record when a rule was triggered."""
        key = f"{campaign_id}:{rule_name}"
        self.triggered_alerts[key] = datetime.now()
    
    async def evaluate_all_campaigns(
        self,
        target_acos: float = 25.0
    ) -> Dict[str, List[SmartAlert]]:
        """Evaluate all campaigns and return alerts grouped by campaign."""
        query = select(PPCCampaign.id, PPCCampaign.name).where(PPCCampaign.state == 'enabled')
        result = await self.db.execute(query)
        campaigns = result.all()
        
        all_alerts = {}
        for campaign_id, campaign_name in campaigns:
            alerts = await self.evaluate_campaign(campaign_id, target_acos)
            if alerts:
                all_alerts[campaign_name] = alerts
        
        return all_alerts
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Get all active rules."""
        return [
            {
                'name': r.name,
                'trigger_type': r.trigger_type.value,
                'enabled': r.enabled,
                'severity': r.severity.value,
                'metric': r.metric,
                'threshold': r.threshold_value,
                'uses_ml': r.use_anomaly_detection or r.use_forecasting
            }
            for r in self.rules
        ]
    
    def toggle_rule(self, rule_name: str, enabled: bool):
        """Enable or disable a rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = enabled
                return True
        return False
    
    def add_custom_rule(self, config: RuleConfig):
        """Add a custom rule."""
        self.rules.append(config)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        return len(self.rules) < original_count
