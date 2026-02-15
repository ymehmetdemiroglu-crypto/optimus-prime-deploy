"""
Advanced Optimization Engine - Integrates all ML capabilities.
Enhanced version with ensemble predictions, anomaly detection, market intelligence.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, update
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
import numpy as np

from ..models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord
from ..features import FeatureEngineer, KeywordFeatureEngineer

# Core ML models
from ..ml import (
    BidOptimizer, PPCRLAgent, PerformanceForecaster,
    DeepBidOptimizer, BidBanditOptimizer, LSTMForecaster,
    BayesianBudgetOptimizer, SpendPacer,
    ModelEnsemble, StackingEnsemble, VotingEnsemble
)

# Specialized capabilities
from ..ml import (
    KeywordSegmenter, PerformanceSegmenter,
    PPCAnomalyDetector,
    SearchTermAnalyzer,
    CompetitorBidEstimator, MarketAnalyzer,
    KeywordHealthAnalyzer, KeywordLifecyclePredictor
)

from .engine import (
    OptimizationStrategy, ActionType, OptimizationAction, OptimizationPlan
)

logger = logging.getLogger(__name__)


class IntelligenceLevel(str, Enum):
    """Level of ML intelligence to use."""
    BASIC = "basic"           # Rule-based + single model
    STANDARD = "standard"      # Ensemble predictions
    ADVANCED = "advanced"      # Full ML suite with anomaly detection
    AUTONOMOUS = "autonomous"  # Self-learning with market intelligence


@dataclass
class EnhancedOptimizationPlan(OptimizationPlan):
    """Extended optimization plan with ML insights."""
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)
    keyword_health: Dict[str, Any] = field(default_factory=dict)
    market_intelligence: Dict[str, Any] = field(default_factory=dict)
    forecast: Dict[str, Any] = field(default_factory=dict)
    segment_analysis: Dict[str, Any] = field(default_factory=dict)
    model_contributions: Dict[str, float] = field(default_factory=dict)
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)


class AdvancedOptimizationEngine:
    """
    Advanced optimization engine integrating all ML capabilities.
    """
    
    STRATEGY_PARAMS = {
        OptimizationStrategy.AGGRESSIVE: {
            'max_bid_increase': 0.35,
            'max_bid_decrease': 0.15,
            'min_data_maturity': 0.3,
            'acos_tolerance': 1.3,
            'confidence_threshold': 0.5,
            'anomaly_sensitivity': 0.7,
        },
        OptimizationStrategy.BALANCED: {
            'max_bid_increase': 0.20,
            'max_bid_decrease': 0.20,
            'min_data_maturity': 0.5,
            'acos_tolerance': 1.15,
            'confidence_threshold': 0.6,
            'anomaly_sensitivity': 0.6,
        },
        OptimizationStrategy.CONSERVATIVE: {
            'max_bid_increase': 0.10,
            'max_bid_decrease': 0.25,
            'min_data_maturity': 0.7,
            'acos_tolerance': 1.05,
            'confidence_threshold': 0.75,
            'anomaly_sensitivity': 0.5,
        },
        OptimizationStrategy.PROFIT_FOCUSED: {
            'max_bid_increase': 0.15,
            'max_bid_decrease': 0.30,
            'min_data_maturity': 0.6,
            'acos_tolerance': 0.95,
            'confidence_threshold': 0.7,
            'anomaly_sensitivity': 0.55,
        },
        OptimizationStrategy.VOLUME_FOCUSED: {
            'max_bid_increase': 0.40,
            'max_bid_decrease': 0.10,
            'min_data_maturity': 0.4,
            'acos_tolerance': 1.5,
            'confidence_threshold': 0.5,
            'anomaly_sensitivity': 0.7,
        },
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer(db)
        self.keyword_engineer = KeywordFeatureEngineer(db)
        
        # Core ML models
        self.bid_optimizer = BidOptimizer()
        self.rl_agent = PPCRLAgent()
        self.deep_optimizer = DeepBidOptimizer()
        self.bandit_optimizer = BidBanditOptimizer()
        
        # Ensembles
        self.ensemble = ModelEnsemble()
        self.stacking_ensemble = StackingEnsemble()
        self.voting_ensemble = VotingEnsemble()
        
        # Forecasting
        self.forecaster = PerformanceForecaster()
        self.lstm_forecaster = LSTMForecaster()
        
        # Budget optimization
        self.budget_optimizer = BayesianBudgetOptimizer()
        self.spend_pacer = SpendPacer()
        
        # Specialized capabilities
        self.anomaly_detector = PPCAnomalyDetector()
        self.keyword_segmenter = KeywordSegmenter()
        self.campaign_segmenter = PerformanceSegmenter()
        self.keyword_health_analyzer = KeywordHealthAnalyzer()
        self.lifecycle_predictor = KeywordLifecyclePredictor()
        self.market_analyzer = MarketAnalyzer()
        self.competitor_estimator = CompetitorBidEstimator()
    
    async def generate_intelligent_plan(
        self,
        campaign_id: int,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        target_acos: float = 25.0,
        target_roas: float = 4.0,
        intelligence_level: IntelligenceLevel = IntelligenceLevel.STANDARD
    ) -> EnhancedOptimizationPlan:
        """
        Generate an intelligent optimization plan using all ML capabilities.
        """
        # Get campaign info
        query = select(PPCCampaign).where(PPCCampaign.id == campaign_id)
        result = await self.db.execute(query)
        campaign = result.scalars().first()
        
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        # Get strategy parameters
        params = self.STRATEGY_PARAMS[strategy]
        
        # Create enhanced plan
        plan = EnhancedOptimizationPlan(
            campaign_id=campaign_id,
            campaign_name=campaign.name,
            strategy=strategy,
            target_acos=target_acos,
            target_roas=target_roas
        )
        
        # Get features
        campaign_features = await self.feature_engineer.compute_full_feature_vector(campaign_id)
        keyword_features = await self.keyword_engineer.bulk_compute_features(campaign_id)
        
        # Get historical data for advanced analysis
        historical_data = await self._get_historical_data(campaign_id)
        
        # === ANOMALY DETECTION ===
        if intelligence_level in [IntelligenceLevel.ADVANCED, IntelligenceLevel.AUTONOMOUS]:
            plan.anomalies_detected = await self._detect_anomalies(
                campaign_id, historical_data, campaign_features, params
            )
        
        # === KEYWORD HEALTH ANALYSIS ===
        if intelligence_level in [IntelligenceLevel.ADVANCED, IntelligenceLevel.AUTONOMOUS]:
            plan.keyword_health = self._analyze_keyword_health(keyword_features, target_acos)
        
        # === KEYWORD SEGMENTATION ===
        plan.segment_analysis = self._segment_keywords(keyword_features, target_acos)
        
        # === FORECASTING ===
        if intelligence_level in [IntelligenceLevel.STANDARD, IntelligenceLevel.ADVANCED, IntelligenceLevel.AUTONOMOUS]:
            plan.forecast = await self._generate_forecast(campaign_id, historical_data)
        
        # === MARKET INTELLIGENCE ===
        if intelligence_level == IntelligenceLevel.AUTONOMOUS:
            plan.market_intelligence = self._analyze_market(keyword_features)
        
        # === GENERATE ACTIONS ===
        for kw_features in keyword_features:
            action = await self._generate_intelligent_action(
                kw_features,
                campaign_features,
                params,
                target_acos,
                target_roas,
                intelligence_level,
                plan
            )
            if action:
                plan.actions.append(action)
        
        # === BUDGET OPTIMIZATION ===
        budget_action = await self._generate_budget_action(
            campaign,
            campaign_features,
            historical_data,
            params,
            target_acos,
            intelligence_level
        )
        if budget_action:
            plan.actions.append(budget_action)
        
        # Sort by priority (anomaly-related actions first)
        plan.actions.sort(key=lambda a: (a.priority, -a.confidence), reverse=True)
        
        # Generate comprehensive summary
        plan.summary = self._generate_enhanced_summary(plan)
        plan.model_contributions = self.ensemble.get_model_status()
        
        return plan
    
    async def _get_historical_data(
        self,
        campaign_id: int,
        days: int = 60
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
                'conversion_rate': r.orders / r.clicks * 100 if r.clicks > 0 else 0
            }
            for r in records
        ]
    
    async def _detect_anomalies(
        self,
        campaign_id: int,
        historical_data: List[Dict[str, Any]],
        current_data: Dict[str, Any],
        params: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using isolation forest and z-score."""
        if not historical_data:
            return []
        
        anomalies = self.anomaly_detector.detect_campaign_anomalies(
            campaign_id,
            historical_data,
            current_data
        )
        
        return [
            {
                'type': a.anomaly_type.value,
                'severity': a.severity.value,
                'metric': a.metric,
                'expected': a.expected_value,
                'actual': a.actual_value,
                'deviation': a.deviation,
                'message': a.message,
                'action': a.recommended_action
            }
            for a in anomalies
        ]
    
    def _analyze_keyword_health(
        self,
        keyword_features: List[Dict[str, Any]],
        target_acos: float
    ) -> Dict[str, Any]:
        """Analyze health of all keywords."""
        health_results = []
        
        for kw in keyword_features:
            report = self.keyword_health_analyzer.analyze_keyword_health(
                kw, None, target_acos
            )
            health_results.append({
                'keyword_id': report.keyword_id,
                'status': report.health_status.value,
                'score': report.health_score,
                'risks': report.risk_factors[:2] if report.risk_factors else [],
                'days_to_decline': report.predicted_days_to_decline
            })
        
        # Summary
        status_counts = {}
        for h in health_results:
            status = h['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        at_risk_keywords = [h for h in health_results if h['status'] in ['at_risk', 'declining', 'critical']]
        
        return {
            'total_analyzed': len(health_results),
            'status_distribution': status_counts,
            'at_risk_count': len(at_risk_keywords),
            'at_risk_keywords': at_risk_keywords[:10],  # Top 10
            'avg_health_score': round(np.mean([h['score'] for h in health_results]), 1) if health_results else 0
        }
    
    def _segment_keywords(
        self,
        keyword_features: List[Dict[str, Any]],
        target_acos: float
    ) -> Dict[str, Any]:
        """Segment keywords for bulk optimization."""
        result = self.keyword_segmenter.segment_keywords(keyword_features, target_acos)
        
        return {
            'segments': [
                {
                    'name': c.name,
                    'count': len(c.keywords),
                    'action': c.recommended_action,
                    'avg_acos': c.avg_performance.get('acos', 0),
                    'avg_roas': c.avg_performance.get('roas', 0)
                }
                for c in result.get('rule_based', [])
            ],
            'summary': result.get('summary', {})
        }
    
    async def _generate_forecast(
        self,
        campaign_id: int,
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate performance forecast."""
        if len(historical_data) < 14:
            return {'error': 'Insufficient data for forecasting'}
        
        try:
            # Use LSTM forecaster
            forecast = self.lstm_forecaster.forecast(historical_data, horizon=7)
            
            return {
                'horizon_days': 7,
                'metrics': forecast.get('metrics', {}),
                'trends': forecast.get('trends', {})
            }
        except Exception as e:
            logger.warning(f"LSTM forecast failed: {e}, falling back to Holt")
            
            # Fallback to simple forecaster
            sales_values = [d['sales'] for d in historical_data]
            forecast, lower, upper = self.forecaster.forecast_metric(sales_values, horizon=7)
            
            return {
                'horizon_days': 7,
                'sales_forecast': forecast,
                'confidence_lower': lower,
                'confidence_upper': upper
            }
    
    def _analyze_market(
        self,
        keyword_features: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze market conditions and opportunities."""
        opportunities = []
        
        for kw in keyword_features[:20]:  # Top 20 keywords
            intel = self.market_analyzer.analyze_keyword_market(kw)
            
            opportunities.append({
                'keyword_id': kw.get('keyword_id'),
                'competition': intel.competition_intensity,
                'cpc_trend': intel.cpc_trend,
                'opportunity_score': intel.opportunity_score,
                'recommended_bid': intel.recommended_bid
            })
        
        # Sort by opportunity
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        return {
            'top_opportunities': opportunities[:10],
            'avg_competition': round(np.mean([o['competition'] for o in opportunities]), 2) if opportunities else 0,
            'market_trend': self._determine_market_trend(opportunities)
        }
    
    def _determine_market_trend(self, opportunities: List[Dict[str, Any]]) -> str:
        """Determine overall market trend."""
        if not opportunities:
            return 'unknown'
        
        rising = sum(1 for o in opportunities if o['cpc_trend'] == 'rising')
        declining = sum(1 for o in opportunities if o['cpc_trend'] == 'declining')
        
        if rising > declining * 1.5:
            return 'CPCs rising - competition increasing'
        elif declining > rising * 1.5:
            return 'CPCs declining - opportunity to scale'
        return 'Stable market conditions'
    
    async def _generate_intelligent_action(
        self,
        kw_features: Dict[str, Any],
        campaign_features: Dict[str, Any],
        params: Dict[str, float],
        target_acos: float,
        target_roas: float,
        intelligence_level: IntelligenceLevel,
        plan: EnhancedOptimizationPlan
    ) -> Optional[OptimizationAction]:
        """Generate action using ensemble predictions and context."""
        
        keyword_id = kw_features.get('keyword_id')
        current_bid = kw_features.get('current_bid', 0)
        current_acos = kw_features.get('acos', 0)
        data_maturity = kw_features.get('data_maturity', 0)
        
        if data_maturity < params['min_data_maturity']:
            return None
        
        merged_features = {**campaign_features, **kw_features}
        
        # === GET PREDICTIONS BASED ON INTELLIGENCE LEVEL ===
        
        if intelligence_level == IntelligenceLevel.BASIC:
            # Single model prediction
            prediction = self.bid_optimizer.predict_bid(merged_features, target_acos, target_roas)
            predicted_bid = prediction.predicted_bid
            confidence = prediction.confidence
            reasoning = prediction.reasoning
            
        elif intelligence_level == IntelligenceLevel.STANDARD:
            # Ensemble prediction
            ensemble_pred = self.ensemble.predict(merged_features, target_acos, target_roas)
            predicted_bid = ensemble_pred.final_bid
            confidence = ensemble_pred.confidence
            reasoning = ensemble_pred.reasoning
            
        else:  # ADVANCED or AUTONOMOUS
            # Full ensemble with voting
            ensemble_pred = self.ensemble.predict(merged_features, target_acos, target_roas)
            voting_pred = self.voting_ensemble.vote(merged_features, target_acos)
            
            # Combine predictions
            predicted_bid = (ensemble_pred.final_bid + voting_pred['recommended_bid']) / 2
            confidence = min(1.0, (ensemble_pred.confidence + voting_pred['consensus_strength']) / 2)
            
            reasoning = f"Ensemble: ${ensemble_pred.final_bid:.2f}, Voting: ${voting_pred['recommended_bid']:.2f} ({voting_pred['decision']})"
            
            # Adjust based on keyword health
            at_risk = any(
                h['keyword_id'] == keyword_id and h['status'] in ['declining', 'critical']
                for h in plan.keyword_health.get('at_risk_keywords', [])
            )
            
            if at_risk:
                # Be more conservative with at-risk keywords
                predicted_bid = min(predicted_bid, current_bid * 0.9)
                reasoning += " [AT RISK - Conservative adjustment]"
            
            # Adjust based on market intelligence (AUTONOMOUS only)
            if intelligence_level == IntelligenceLevel.AUTONOMOUS:
                market_rec = self._get_market_recommendation(keyword_id, plan.market_intelligence)
                if market_rec:
                    predicted_bid = (predicted_bid + market_rec) / 2
                    reasoning += f" [Market suggests: ${market_rec:.2f}]"
        
        # Skip if no current bid
        if current_bid == 0:
            return None
        
        # Calculate change
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
        
        # Skip small changes
        if abs(change_percent) < 2:
            return None
        
        # Determine action type and priority
        action_type, priority = self._determine_action(
            change_percent, current_acos, target_acos, kw_features, plan
        )
        
        # Adjust for pause
        if action_type == ActionType.PAUSE_KEYWORD:
            predicted_bid = 0
            change_percent = -100
        
        # Confidence threshold check
        if confidence < params['confidence_threshold']:
            confidence *= 0.8  # Reduce but don't skip
        
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
    
    def _get_market_recommendation(
        self,
        keyword_id: int,
        market_intelligence: Dict[str, Any]
    ) -> Optional[float]:
        """Get market-based bid recommendation."""
        opportunities = market_intelligence.get('top_opportunities', [])
        for opp in opportunities:
            if opp['keyword_id'] == keyword_id:
                return opp['recommended_bid']
        return None
    
    def _determine_action(
        self,
        change_percent: float,
        current_acos: float,
        target_acos: float,
        kw_features: Dict[str, Any],
        plan: EnhancedOptimizationPlan
    ) -> Tuple[ActionType, int]:
        """Determine action type and priority with context awareness."""
        
        clicks = kw_features.get('clicks', 0)
        
        # Check for anomalies related to this keyword
        has_anomaly = any(
            a['severity'] == 'critical' 
            for a in plan.anomalies_detected 
            if a['metric'] in ['spend', 'acos']
        )
        
        # Pause very poor performers
        if current_acos > target_acos * 2 and clicks >= 50:
            return ActionType.PAUSE_KEYWORD, 10
        
        # Determine direction
        if change_percent > 0:
            action_type = ActionType.BID_INCREASE
            
            # High priority if keyword is very profitable
            if current_acos < target_acos * 0.6:
                priority = 8
            elif current_acos < target_acos * 0.8:
                priority = 7
            else:
                priority = 5
        else:
            action_type = ActionType.BID_DECREASE
            
            # High priority if losing money or anomaly detected
            if has_anomaly:
                priority = 10  # Critical
            elif current_acos > target_acos * 1.5:
                priority = 9
            elif current_acos > target_acos * 1.2:
                priority = 7
            else:
                priority = 5
        
        return action_type, priority
    
    async def _generate_budget_action(
        self,
        campaign: PPCCampaign,
        features: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        params: Dict[str, float],
        target_acos: float,
        intelligence_level: IntelligenceLevel
    ) -> Optional[OptimizationAction]:
        """Generate budget action using Bayesian optimization."""
        
        current_budget = float(campaign.daily_budget or 0)
        if current_budget == 0:
            return None
        
        current_acos = features.get('acos_7d', 50)
        
        if intelligence_level in [IntelligenceLevel.ADVANCED, IntelligenceLevel.AUTONOMOUS]:
            # Use Bayesian budget optimizer
            allocation = self.budget_optimizer.suggest_budget(
                campaign.id,
                current_budget,
                (current_budget * 0.5, current_budget * 2.0)
            )
            
            recommended_budget = allocation.recommended_budget
            confidence = allocation.confidence
            reasoning = allocation.reasoning
            
        else:
            # Simple rule-based
            spend_trend = features.get('spend_trend', 1.0)
            sales_trend = features.get('sales_trend', 1.0)
            
            if spend_trend < 0.8 and current_acos < target_acos:
                recommended_budget = current_budget * 1.15
                confidence = 0.7
                reasoning = f"Underspending with good ACoS ({current_acos:.1f}%)"
            elif spend_trend > 1.2 and current_acos > target_acos * params['acos_tolerance']:
                recommended_budget = current_budget * 0.85
                confidence = 0.75
                reasoning = f"Overspending with poor ACoS ({current_acos:.1f}%)"
            else:
                return None
        
        change_percent = ((recommended_budget / current_budget) - 1) * 100
        
        if abs(change_percent) < 5:
            return None
        
        action_type = ActionType.BUDGET_INCREASE if change_percent > 0 else ActionType.BUDGET_DECREASE
        priority = 6 if action_type == ActionType.BUDGET_INCREASE else 8
        
        return OptimizationAction(
            action_type=action_type,
            entity_type='campaign',
            entity_id=campaign.id,
            current_value=current_budget,
            recommended_value=round(recommended_budget, 2),
            change_percent=round(change_percent, 1),
            confidence=round(confidence, 2),
            reasoning=reasoning,
            priority=priority
        )
    
    def _generate_enhanced_summary(self, plan: EnhancedOptimizationPlan) -> Dict[str, Any]:
        """Generate comprehensive summary with ML insights."""
        
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
            'critical_actions': len([a for a in plan.actions if a.priority == 10]),
            'avg_confidence': round(np.mean([a.confidence for a in plan.actions]), 2) if plan.actions else 0,
            'strategy': plan.strategy.value,
            'target_acos': plan.target_acos,
            
            # ML Insights
            'anomalies_found': len(plan.anomalies_detected),
            'critical_anomalies': len([a for a in plan.anomalies_detected if a.get('severity') == 'critical']),
            'at_risk_keywords': plan.keyword_health.get('at_risk_count', 0),
            'avg_keyword_health': plan.keyword_health.get('avg_health_score', 0),
            'segments': plan.segment_analysis.get('segments', []),
            'forecast_available': bool(plan.forecast and 'error' not in plan.forecast),
            'market_analyzed': bool(plan.market_intelligence),
        }
    
    async def execute_intelligent_plan(
        self,
        plan: EnhancedOptimizationPlan,
        dry_run: bool = True,
        auto_approve_confidence: float = 0.8,
        respect_anomalies: bool = True
    ) -> Dict[str, Any]:
        """
        Execute plan with intelligent decision-making.
        """
        executed = []
        skipped = []
        errors = []
        
        for action in plan.actions:
            # Auto-approve high-confidence actions
            if action.confidence >= auto_approve_confidence:
                action.approved = True
            
            # Skip non-approved in live mode
            if not action.approved and not dry_run:
                skipped.append({
                    'entity_id': action.entity_id,
                    'reason': f'Not approved (confidence: {action.confidence})'
                })
                continue
            
            # Respect anomalies - be extra careful
            if respect_anomalies and plan.anomalies_detected:
                critical = any(a['severity'] == 'critical' for a in plan.anomalies_detected)
                if critical and action.action_type == ActionType.BID_INCREASE:
                    skipped.append({
                        'entity_id': action.entity_id,
                        'reason': 'Critical anomaly detected - skipping bid increases'
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
                    'confidence': action.confidence,
                    'priority': action.priority,
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
            },
            'ml_insights': {
                'anomalies_considered': len(plan.anomalies_detected),
                'keyword_health_considered': plan.keyword_health.get('total_analyzed', 0),
                'forecast_used': 'forecast' in plan.__dict__ and plan.forecast
            }
        }
    
    async def _apply_action(self, action: OptimizationAction):
        """Apply a single optimization action."""
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
