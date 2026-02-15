"""
Optimization Engine - Core system for applying ML recommendations.
Executes bid changes and budget adjustments in real-time.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, update
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
import asyncio

from ..models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord
from ..features import FeatureEngineer, KeywordFeatureEngineer
from ..ml import BidOptimizer, PPCRLAgent
from ..strategies.config import BidStrategyConfig

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Available optimization strategies."""
    AGGRESSIVE = "aggressive"       # Maximize growth
    BALANCED = "balanced"           # Balance growth and efficiency
    CONSERVATIVE = "conservative"   # Minimize risk
    PROFIT_FOCUSED = "profit"       # Maximize profit margin
    VOLUME_FOCUSED = "volume"       # Maximize impressions/clicks


class ActionType(str, Enum):
    """Types of optimization actions."""
    BID_INCREASE = "bid_increase"
    BID_DECREASE = "bid_decrease"
    PAUSE_KEYWORD = "pause_keyword"
    ENABLE_KEYWORD = "enable_keyword"
    BUDGET_INCREASE = "budget_increase"
    BUDGET_DECREASE = "budget_decrease"
    NO_CHANGE = "no_change"


@dataclass
class OptimizationAction:
    """Represents a single optimization action."""
    action_type: ActionType
    entity_type: str  # 'keyword' or 'campaign'
    entity_id: int
    current_value: float
    recommended_value: float
    change_percent: float
    confidence: float
    reasoning: str
    priority: int = 5  # 1-10, higher = more urgent
    approved: bool = False
    executed: bool = False
    execution_time: Optional[datetime] = None


@dataclass
class OptimizationPlan:
    """Collection of optimization actions for a campaign."""
    campaign_id: int
    campaign_name: str
    strategy: OptimizationStrategy
    target_acos: float
    target_roas: float
    created_at: datetime = field(default_factory=datetime.now)
    actions: List[OptimizationAction] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class OptimizationEngine:
    """
    Core optimization engine that generates and executes optimization plans.
    """
    
    # Strategy-specific thresholds
    STRATEGY_PARAMS = {
        OptimizationStrategy.AGGRESSIVE: {
            'max_bid_increase': 0.30,
            'max_bid_decrease': 0.15,
            'min_data_maturity': 0.3,
            'acos_tolerance': 1.3,  # Allow 30% above target
        },
        OptimizationStrategy.BALANCED: {
            'max_bid_increase': 0.20,
            'max_bid_decrease': 0.20,
            'min_data_maturity': 0.5,
            'acos_tolerance': 1.15,
        },
        OptimizationStrategy.CONSERVATIVE: {
            'max_bid_increase': 0.10,
            'max_bid_decrease': 0.25,
            'min_data_maturity': 0.7,
            'acos_tolerance': 1.05,
        },
        OptimizationStrategy.PROFIT_FOCUSED: {
            'max_bid_increase': 0.15,
            'max_bid_decrease': 0.30,
            'min_data_maturity': 0.6,
            'acos_tolerance': 0.95,
        },
        OptimizationStrategy.VOLUME_FOCUSED: {
            'max_bid_increase': 0.35,
            'max_bid_decrease': 0.10,
            'min_data_maturity': 0.4,
            'acos_tolerance': 1.5,
        },
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.feature_engineer = FeatureEngineer(db)
        self.keyword_engineer = KeywordFeatureEngineer(db)
        self.bid_optimizer = BidOptimizer()
        self.rl_agent = PPCRLAgent()
    
    async def generate_optimization_plan(
        self,
        campaign_id: int,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        target_acos: float = 25.0,
        target_roas: float = 4.0
    ) -> OptimizationPlan:
        """
        Generate a full optimization plan for a campaign.
        """
        # Get campaign info
        query = select(PPCCampaign).where(PPCCampaign.id == campaign_id)
        result = await self.db.execute(query)
        campaign = result.scalars().first()
        
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        plan = OptimizationPlan(
            campaign_id=campaign_id,
            campaign_name=campaign.name,
            strategy=strategy,
            target_acos=target_acos,
            target_roas=target_roas
        )
        
        # Get strategy parameters
        params = self.STRATEGY_PARAMS[strategy]
        
        # Get campaign features
        campaign_features = await self.feature_engineer.compute_full_feature_vector(campaign_id)
        
        # Get keyword features
        keyword_features = await self.keyword_engineer.bulk_compute_features(campaign_id)
        
        # Generate keyword-level actions
        for kw_features in keyword_features:
            action = await self._generate_keyword_action(
                kw_features,
                campaign_features,
                params,
                target_acos,
                target_roas
            )
            if action:
                plan.actions.append(action)
        
        # Generate campaign-level actions
        campaign_action = await self._generate_campaign_action(
            campaign,
            campaign_features,
            params,
            target_acos
        )
        if campaign_action:
            plan.actions.append(campaign_action)
        
        # Sort by priority
        plan.actions.sort(key=lambda a: a.priority, reverse=True)
        
        # Generate summary
        plan.summary = self._generate_plan_summary(plan)
        
        return plan
    
    async def _generate_keyword_action(
        self,
        kw_features: Dict[str, Any],
        campaign_features: Dict[str, Any],
        params: Dict[str, float],
        target_acos: float,
        target_roas: float
    ) -> Optional[OptimizationAction]:
        """Generate optimization action for a single keyword."""
        
        keyword_id = kw_features.get('keyword_id')
        current_bid = kw_features.get('current_bid', 0)
        current_acos = kw_features.get('acos', 0)
        data_maturity = kw_features.get('data_maturity', 0)
        
        # Skip if insufficient data
        if data_maturity < params['min_data_maturity']:
            return None
        
        # Merge features for prediction
        merged_features = {**campaign_features, **kw_features}
        
        # Get ML predictions
        strategy_config = BidStrategyConfig(
            target_acos=target_acos,
            target_roas=target_roas,
            max_bid_increase_factor=1.0 + params['max_bid_increase'],
            max_bid_decrease_factor=1.0 - params['max_bid_decrease']
        )
        gb_prediction = self.bid_optimizer.predict_bid(merged_features, config=strategy_config)
        rl_recommendation = self.rl_agent.get_bid_recommendation(
            merged_features, current_bid, target_acos
        )
        
        # Ensemble the predictions
        predicted_bid = (gb_prediction.predicted_bid + rl_recommendation['recommended_bid']) / 2
        
        # Calculate change
        if current_bid == 0:
            return None
        
        change_ratio = predicted_bid / current_bid
        change_percent = (change_ratio - 1) * 100
        
        # Apply strategy constraints
        max_increase = params['max_bid_increase']
        max_decrease = params['max_bid_decrease']
        
        if change_ratio > 1 + max_increase:
            predicted_bid = current_bid * (1 + max_increase)
            change_percent = max_increase * 100
        elif change_ratio < 1 - max_decrease:
            predicted_bid = current_bid * (1 - max_decrease)
            change_percent = -max_decrease * 100
        
        # Determine action type and priority
        if abs(change_percent) < 2:
            return None  # Change too small
        
        if change_percent > 0:
            action_type = ActionType.BID_INCREASE
            # Higher priority if keyword is profitable
            priority = 7 if current_acos < target_acos * 0.8 else 5
        else:
            action_type = ActionType.BID_DECREASE
            # Higher priority if keyword is losing money
            priority = 9 if current_acos > target_acos * 1.5 else 6
        
        # Consider pausing very poor performers
        if current_acos > target_acos * 2 and kw_features.get('clicks', 0) >= 50:
            action_type = ActionType.PAUSE_KEYWORD
            priority = 10
            predicted_bid = 0
            change_percent = -100
        
        confidence = min(1.0, (gb_prediction.confidence + 0.7) / 2)  # Average with RL confidence
        
        reasoning = f"{gb_prediction.reasoning}. "
        if action_type == ActionType.PAUSE_KEYWORD:
            reasoning = f"ACoS {current_acos:.1f}% is over 2x target. Recommend pausing."
        elif action_type == ActionType.BID_INCREASE:
            reasoning += f"Keyword outperforming at {current_acos:.1f}% ACoS."
        else:
            reasoning += f"Current ACoS {current_acos:.1f}% exceeds target {target_acos}%."
        
        return OptimizationAction(
            action_type=action_type,
            entity_type='keyword',
            entity_id=keyword_id,
            current_value=current_bid,
            recommended_value=round(predicted_bid, 2),
            change_percent=round(change_percent, 1),
            confidence=round(confidence, 2),
            reasoning=reasoning,
            priority=priority
        )
    
    async def _generate_campaign_action(
        self,
        campaign: PPCCampaign,
        features: Dict[str, Any],
        params: Dict[str, float],
        target_acos: float
    ) -> Optional[OptimizationAction]:
        """Generate campaign-level budget action."""
        
        current_budget = float(campaign.daily_budget or 0)
        if current_budget == 0:
            return None
        
        spend_trend = features.get('spend_trend', 1.0)
        sales_trend = features.get('sales_trend', 1.0)
        current_acos = features.get('acos_7d', 50)
        
        # Check if underspending with good performance
        if spend_trend < 0.8 and current_acos < target_acos:
            # Room to increase budget
            recommended_budget = current_budget * 1.15
            return OptimizationAction(
                action_type=ActionType.BUDGET_INCREASE,
                entity_type='campaign',
                entity_id=campaign.id,
                current_value=current_budget,
                recommended_value=round(recommended_budget, 2),
                change_percent=15.0,
                confidence=0.7,
                reasoning=f"Underspending with good performance ({current_acos:.1f}% ACoS). Room to scale.",
                priority=6
            )
        
        # Check if overspending with poor performance
        if spend_trend > 1.2 and current_acos > target_acos * params['acos_tolerance']:
            recommended_budget = current_budget * 0.85
            return OptimizationAction(
                action_type=ActionType.BUDGET_DECREASE,
                entity_type='campaign',
                entity_id=campaign.id,
                current_value=current_budget,
                recommended_value=round(recommended_budget, 2),
                change_percent=-15.0,
                confidence=0.75,
                reasoning=f"Overspending with poor performance ({current_acos:.1f}% ACoS).",
                priority=8
            )
        
        return None
    
    def _generate_plan_summary(self, plan: OptimizationPlan) -> Dict[str, Any]:
        """Generate summary statistics for the plan."""
        
        bid_increases = [a for a in plan.actions if a.action_type == ActionType.BID_INCREASE]
        bid_decreases = [a for a in plan.actions if a.action_type == ActionType.BID_DECREASE]
        pauses = [a for a in plan.actions if a.action_type == ActionType.PAUSE_KEYWORD]
        budget_changes = [a for a in plan.actions if a.action_type in [ActionType.BUDGET_INCREASE, ActionType.BUDGET_DECREASE]]
        
        return {
            'total_actions': len(plan.actions),
            'bid_increases': len(bid_increases),
            'bid_decreases': len(bid_decreases),
            'keywords_to_pause': len(pauses),
            'budget_changes': len(budget_changes),
            'high_priority_actions': len([a for a in plan.actions if a.priority >= 8]),
            'avg_confidence': round(sum(a.confidence for a in plan.actions) / len(plan.actions), 2) if plan.actions else 0,
            'strategy': plan.strategy.value,
            'target_acos': plan.target_acos
        }
    
    async def execute_plan(
        self,
        plan: OptimizationPlan,
        dry_run: bool = True,
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """
        Execute an optimization plan.
        
        Args:
            plan: The optimization plan to execute
            dry_run: If True, only simulate (don't apply changes)
            min_confidence: Minimum confidence to execute action
        """
        executed = []
        skipped = []
        errors = []
        
        for action in plan.actions:
            if not action.approved and not dry_run:
                skipped.append({
                    'entity_id': action.entity_id,
                    'reason': 'Not approved'
                })
                continue
            
            if action.confidence < min_confidence:
                skipped.append({
                    'entity_id': action.entity_id,
                    'reason': f'Confidence {action.confidence} below threshold {min_confidence}'
                })
                continue
            
            if dry_run:
                executed.append({
                    'action_type': action.action_type.value,
                    'entity_type': action.entity_type,
                    'entity_id': action.entity_id,
                    'from': action.current_value,
                    'to': action.recommended_value,
                    'change_percent': action.change_percent,
                    'status': 'simulated'
                })
            else:
                try:
                    await self._apply_action(action)
                    action.executed = True
                    action.execution_time = datetime.now()
                    executed.append({
                        'action_type': action.action_type.value,
                        'entity_type': action.entity_type,
                        'entity_id': action.entity_id,
                        'from': action.current_value,
                        'to': action.recommended_value,
                        'status': 'executed'
                    })
                except Exception as e:
                    errors.append({
                        'entity_id': action.entity_id,
                        'error': str(e)
                    })
        
        return {
            'dry_run': dry_run,
            'executed': executed,
            'skipped': skipped,
            'errors': errors,
            'summary': {
                'total': len(plan.actions),
                'executed': len(executed),
                'skipped': len(skipped),
                'errors': len(errors)
            }
        }
    
    async def _apply_action(self, action: OptimizationAction):
        """Apply a single optimization action to the database."""
        
        if action.entity_type == 'keyword':
            if action.action_type in [ActionType.BID_INCREASE, ActionType.BID_DECREASE]:
                stmt = (
                    update(PPCKeyword)
                    .where(PPCKeyword.id == action.entity_id)
                    .values(bid=Decimal(str(action.recommended_value)))
                )
                await self.db.execute(stmt)
            
            elif action.action_type == ActionType.PAUSE_KEYWORD:
                stmt = (
                    update(PPCKeyword)
                    .where(PPCKeyword.id == action.entity_id)
                    .values(state='paused')
                )
                await self.db.execute(stmt)
        
        elif action.entity_type == 'campaign':
            if action.action_type in [ActionType.BUDGET_INCREASE, ActionType.BUDGET_DECREASE]:
                stmt = (
                    update(PPCCampaign)
                    .where(PPCCampaign.id == action.entity_id)
                    .values(daily_budget=Decimal(str(action.recommended_value)))
                )
                await self.db.execute(stmt)
        
        await self.db.commit()
        logger.info(f"Applied action: {action.action_type.value} on {action.entity_type} {action.entity_id}")
