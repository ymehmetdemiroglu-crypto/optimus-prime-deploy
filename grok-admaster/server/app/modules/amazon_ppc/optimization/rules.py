"""
Rule-based triggers for automatic optimization.
Monitors performance and triggers actions based on conditions.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord
from .engine import OptimizationEngine, OptimizationAction, ActionType

logger = logging.getLogger(__name__)


class TriggerCondition(str, Enum):
    """Types of trigger conditions."""
    ACOS_THRESHOLD = "acos_threshold"
    SPEND_SPIKE = "spend_spike"
    CONVERSION_DROP = "conversion_drop"
    CTR_DROP = "ctr_drop"
    NO_SALES = "no_sales"
    BUDGET_DEPLETION = "budget_depletion"
    IMPRESSION_DROP = "impression_drop"


@dataclass
class AlertTrigger:
    """Configuration for an alert trigger."""
    condition: TriggerCondition
    threshold: float
    action: ActionType
    severity: str  # 'warning', 'critical'
    cooldown_hours: int = 24  # Hours before trigger can fire again


@dataclass
class TriggeredAlert:
    """An alert that was triggered."""
    trigger: AlertTrigger
    entity_type: str
    entity_id: int
    current_value: float
    threshold_value: float
    message: str
    triggered_at: datetime
    recommended_action: Optional[OptimizationAction] = None


class RuleEngine:
    """
    Monitors campaigns and triggers alerts based on rules.
    """
    
    # Default ruleset
    DEFAULT_RULES = [
        AlertTrigger(
            condition=TriggerCondition.ACOS_THRESHOLD,
            threshold=50.0,  # 50% ACoS
            action=ActionType.BID_DECREASE,
            severity='critical',
            cooldown_hours=12
        ),
        AlertTrigger(
            condition=TriggerCondition.SPEND_SPIKE,
            threshold=2.0,  # 2x normal spend
            action=ActionType.BUDGET_DECREASE,
            severity='warning',
            cooldown_hours=24
        ),
        AlertTrigger(
            condition=TriggerCondition.NO_SALES,
            threshold=50.0,  # $50 spend without sales
            action=ActionType.PAUSE_KEYWORD,
            severity='critical',
            cooldown_hours=48
        ),
        AlertTrigger(
            condition=TriggerCondition.CTR_DROP,
            threshold=0.5,  # 50% drop
            action=ActionType.BID_INCREASE,
            severity='warning',
            cooldown_hours=24
        ),
        AlertTrigger(
            condition=TriggerCondition.BUDGET_DEPLETION,
            threshold=0.9,  # 90% budget used
            action=ActionType.BUDGET_INCREASE,
            severity='warning',
            cooldown_hours=12
        ),
    ]
    
    def __init__(self, db: AsyncSession, rules: List[AlertTrigger] = None):
        self.db = db
        self.rules = rules or self.DEFAULT_RULES
        self.triggered_alerts: Dict[str, datetime] = {}  # For cooldown tracking
    
    async def evaluate_campaign(self, campaign_id: int) -> List[TriggeredAlert]:
        """
        Evaluate all rules for a campaign.
        Returns list of triggered alerts.
        """
        alerts = []
        
        # Get campaign data
        query = select(PPCCampaign).where(PPCCampaign.id == campaign_id)
        result = await self.db.execute(query)
        campaign = result.scalars().first()
        
        if not campaign:
            return alerts
        
        # Get recent performance
        cutoff = datetime.now() - timedelta(days=7)
        perf_query = (
            select(PerformanceRecord)
            .where(
                and_(
                    PerformanceRecord.campaign_id == campaign_id,
                    PerformanceRecord.date >= cutoff
                )
            )
            .order_by(PerformanceRecord.date.desc())
        )
        perf_result = await self.db.execute(perf_query)
        records = perf_result.scalars().all()
        
        if not records:
            return alerts
        
        # Aggregate metrics
        total_spend = sum(float(r.spend) for r in records)
        total_sales = sum(float(r.sales) for r in records)
        total_clicks = sum(r.clicks for r in records)
        total_impressions = sum(r.impressions for r in records)
        
        current_acos = (total_spend / total_sales * 100) if total_sales > 0 else 999
        current_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        
        for rule in self.rules:
            alert = await self._check_rule(
                rule,
                campaign,
                current_acos=current_acos,
                total_spend=total_spend,
                total_sales=total_sales,
                current_ctr=current_ctr,
                daily_budget=float(campaign.daily_budget or 0)
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    async def _check_rule(
        self,
        rule: AlertTrigger,
        campaign: PPCCampaign,
        **metrics
    ) -> Optional[TriggeredAlert]:
        """Check if a rule is triggered."""
        
        # Check cooldown
        cooldown_key = f"{campaign.id}:{rule.condition.value}"
        last_triggered = self.triggered_alerts.get(cooldown_key)
        if last_triggered:
            cooldown_end = last_triggered + timedelta(hours=rule.cooldown_hours)
            if datetime.now() < cooldown_end:
                return None
        
        triggered = False
        current_value = 0
        message = ""
        
        if rule.condition == TriggerCondition.ACOS_THRESHOLD:
            current_value = metrics.get('current_acos', 0)
            if current_value > rule.threshold:
                triggered = True
                message = f"ACoS {current_value:.1f}% exceeds threshold {rule.threshold}%"
        
        elif rule.condition == TriggerCondition.NO_SALES:
            spend = metrics.get('total_spend', 0)
            sales = metrics.get('total_sales', 0)
            if spend >= rule.threshold and sales == 0:
                triggered = True
                current_value = spend
                message = f"Spent ${spend:.2f} with no sales"
        
        elif rule.condition == TriggerCondition.CTR_DROP:
            current_value = metrics.get('current_ctr', 0)
            # Would need historical comparison for true drop detection
            if current_value < 0.5:  # Simple threshold for now
                triggered = True
                message = f"CTR {current_value:.2f}% is critically low"
        
        elif rule.condition == TriggerCondition.BUDGET_DEPLETION:
            daily_budget = metrics.get('daily_budget', 0)
            daily_spend = metrics.get('total_spend', 0) / 7  # Avg daily
            if daily_budget > 0:
                current_value = daily_spend / daily_budget
                if current_value >= rule.threshold:
                    triggered = True
                    message = f"Spending {current_value*100:.0f}% of daily budget"
        
        if triggered:
            self.triggered_alerts[cooldown_key] = datetime.now()
            
            return TriggeredAlert(
                trigger=rule,
                entity_type='campaign',
                entity_id=campaign.id,
                current_value=current_value,
                threshold_value=rule.threshold,
                message=message,
                triggered_at=datetime.now()
            )
        
        return None
    
    async def evaluate_all_campaigns(self) -> List[TriggeredAlert]:
        """Evaluate rules for all active campaigns."""
        query = select(PPCCampaign.id).where(PPCCampaign.state == 'enabled')
        result = await self.db.execute(query)
        campaign_ids = [row[0] for row in result.all()]
        
        all_alerts = []
        for campaign_id in campaign_ids:
            alerts = await self.evaluate_campaign(campaign_id)
            all_alerts.extend(alerts)
        
        return all_alerts
    
    def add_rule(self, rule: AlertTrigger):
        """Add a custom rule."""
        self.rules.append(rule)
    
    def remove_rule(self, condition: TriggerCondition):
        """Remove rules by condition type."""
        self.rules = [r for r in self.rules if r.condition != condition]
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all active rules."""
        return [
            {
                'condition': r.condition.value,
                'threshold': r.threshold,
                'action': r.action.value,
                'severity': r.severity,
                'cooldown_hours': r.cooldown_hours
            }
            for r in self.rules
        ]
