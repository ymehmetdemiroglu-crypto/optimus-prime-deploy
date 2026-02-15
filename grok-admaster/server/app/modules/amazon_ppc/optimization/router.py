"""
API endpoints for optimization operations.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.core.database import get_db
from app.core.config import settings
from ..optimization import (
    OptimizationEngine, 
    OptimizationStrategy,
    RuleEngine,
    OptimizationScheduler,
    OptimizationSchedule,
    ScheduleFrequency,
    # Advanced
    AdvancedOptimizationEngine,
    IntelligenceLevel,
    AdvancedRuleEngine
)

router = APIRouter()

# Global scheduler instance
_scheduler: Optional[OptimizationScheduler] = None


def get_scheduler() -> OptimizationScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = OptimizationScheduler(settings.DATABASE_URL)
    return _scheduler


# ==================== REQUEST MODELS ====================

class OptimizationRequest(BaseModel):
    campaign_id: int
    strategy: str = "balanced"
    target_acos: float = 25.0
    target_roas: float = 4.0


class ExecutePlanRequest(BaseModel):
    action_ids: Optional[List[int]] = None  # If None, execute all
    dry_run: bool = True
    min_confidence: float = 0.6



class KeywordDiagnosticResponse(BaseModel):
    keyword_id: str
    keyword_text: str
    match_type: str
    campaign_name: str
    last_scan: str
    
    # Vitals
    spend: float
    impressions: int
    clicks: int
    current_bid: float
    
    # Diagnosis
    acos: float
    ctr: float
    wasted_spend: float
    
    # Recommendations
    recommendations: List[Dict[str, Any]]
    
    # Harvested
    harvested_keywords: List[Dict[str, Any]]


class ScheduleRequest(BaseModel):
    account_id: int
    campaign_ids: List[int] = []
    strategy: str = "balanced"
    frequency: str = "daily"
    target_acos: float = 25.0
    target_roas: float = 4.0
    auto_execute: bool = False
    min_confidence: float = 0.7


# ==================== OPTIMIZATION ENDPOINTS ====================

@router.post("/generate-plan")
async def generate_optimization_plan(
    request: OptimizationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate an optimization plan for a campaign.
    Returns recommended actions without executing them.
    """
    try:
        strategy = OptimizationStrategy(request.strategy)
    except ValueError:
        raise HTTPException(400, f"Invalid strategy: {request.strategy}")
    
    engine = OptimizationEngine(db)
    
    try:
        plan = await engine.generate_optimization_plan(
            campaign_id=request.campaign_id,
            strategy=strategy,
            target_acos=request.target_acos,
            target_roas=request.target_roas
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    
    return {
        'campaign_id': plan.campaign_id,
        'campaign_name': plan.campaign_name,
        'strategy': plan.strategy.value,
        'target_acos': plan.target_acos,
        'target_roas': plan.target_roas,
        'created_at': plan.created_at.isoformat(),
        'summary': plan.summary,
        'actions': [
            {
                'action_type': a.action_type.value,
                'entity_type': a.entity_type,
                'entity_id': a.entity_id,
                'current_value': a.current_value,
                'recommended_value': a.recommended_value,
                'change_percent': a.change_percent,
                'confidence': a.confidence,
                'reasoning': a.reasoning,
                'priority': a.priority
            }
            for a in plan.actions
        ]
    }


@router.post("/execute")
async def execute_optimization(
    request: OptimizationRequest,
    execute_request: ExecutePlanRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Execute an optimization plan.
    Set dry_run=False to actually apply changes.
    """
    try:
        strategy = OptimizationStrategy(request.strategy)
    except ValueError:
        raise HTTPException(400, f"Invalid strategy: {request.strategy}")
    
    engine = OptimizationEngine(db)
    
    # Generate plan
    plan = await engine.generate_optimization_plan(
        campaign_id=request.campaign_id,
        strategy=strategy,
        target_acos=request.target_acos,
        target_roas=request.target_roas
    )
    
    # Approve specific actions or all
    if execute_request.action_ids:
        for action in plan.actions:
            if action.entity_id in execute_request.action_ids:
                action.approved = True
    else:
        for action in plan.actions:
            action.approved = True
    
    # Execute
    result = await engine.execute_plan(
        plan,
        dry_run=execute_request.dry_run,
        min_confidence=execute_request.min_confidence
    )
    
    return result


@router.get("/strategies")
async def list_strategies():
    """List available optimization strategies."""
    return {
        'strategies': [
            {
                'value': s.value,
                'description': {
                    'aggressive': 'Maximize growth, higher risk tolerance',
                    'balanced': 'Balance growth and efficiency',
                    'conservative': 'Minimize risk, prioritize stability',
                    'profit': 'Maximize profit margin',
                    'volume': 'Maximize impressions and clicks'
                }.get(s.value, '')
            }
            for s in OptimizationStrategy
        ]
    }


# ==================== RULE ENGINE ENDPOINTS ====================

@router.get("/alerts/{campaign_id}")
async def get_campaign_alerts(
    campaign_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Check for triggered alerts on a campaign.
    """
    rule_engine = RuleEngine(db)
    alerts = await rule_engine.evaluate_campaign(campaign_id)
    
    return {
        'campaign_id': campaign_id,
        'alert_count': len(alerts),
        'alerts': [
            {
                'condition': a.trigger.condition.value,
                'severity': a.trigger.severity,
                'current_value': a.current_value,
                'threshold': a.threshold_value,
                'message': a.message,
                'triggered_at': a.triggered_at.isoformat(),
                'recommended_action': a.trigger.action.value
            }
            for a in alerts
        ]
    }


@router.get("/alerts")
async def get_all_alerts(
    db: AsyncSession = Depends(get_db)
):
    """
    Check for alerts across all campaigns.
    """
    rule_engine = RuleEngine(db)
    alerts = await rule_engine.evaluate_all_campaigns()
    
    # Group by severity
    critical = [a for a in alerts if a.trigger.severity == 'critical']
    warnings = [a for a in alerts if a.trigger.severity == 'warning']
    
    return {
        'total_alerts': len(alerts),
        'critical_count': len(critical),
        'warning_count': len(warnings),
        'alerts': [
            {
                'entity_type': a.entity_type,
                'entity_id': a.entity_id,
                'condition': a.trigger.condition.value,
                'severity': a.trigger.severity,
                'message': a.message,
                'recommended_action': a.trigger.action.value
            }
            for a in sorted(alerts, key=lambda x: x.trigger.severity == 'critical', reverse=True)
        ]
    }


@router.get("/rules")
async def list_rules(
    db: AsyncSession = Depends(get_db)
):
    """List all active alert rules."""
    rule_engine = RuleEngine(db)
    return {'rules': rule_engine.get_rules()}


# ==================== SCHEDULER ENDPOINTS ====================

@router.post("/schedule")
async def create_schedule(
    request: ScheduleRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new optimization schedule.
    """
    scheduler = get_scheduler()
    
    try:
        strategy = OptimizationStrategy(request.strategy)
        frequency = ScheduleFrequency(request.frequency)
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    schedule = OptimizationSchedule(
        account_id=request.account_id,
        campaign_ids=request.campaign_ids,
        strategy=strategy,
        frequency=frequency,
        target_acos=request.target_acos,
        target_roas=request.target_roas,
        auto_execute=request.auto_execute,
        min_confidence=request.min_confidence
    )
    
    scheduler.add_schedule(schedule)
    
    return {
        'status': 'scheduled',
        'account_id': request.account_id,
        'next_run': schedule.next_run.isoformat() if schedule.next_run else None
    }


@router.delete("/schedule/{account_id}")
async def delete_schedule(account_id: int):
    """Remove optimization schedule for an account."""
    scheduler = get_scheduler()
    scheduler.remove_schedule(account_id)
    return {'status': 'removed', 'account_id': account_id}


@router.get("/scheduler/status")
async def get_scheduler_status():
    """Get current scheduler status."""
    scheduler = get_scheduler()
    return scheduler.get_status()


@router.post("/scheduler/start")
async def start_scheduler(
    interval_seconds: int = 300,
    background_tasks: BackgroundTasks = None
):
    """Start the optimization scheduler."""
    scheduler = get_scheduler()
    
    if scheduler.state.is_running:
        return {'status': 'already_running'}
    
    # Start in background
    background_tasks.add_task(scheduler.start, interval_seconds)
    
    return {'status': 'started', 'interval_seconds': interval_seconds}


@router.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the optimization scheduler."""
    scheduler = get_scheduler()
    await scheduler.stop()
    return {'status': 'stopped'}


# ==================== QUICK ACTIONS ====================

@router.post("/quick-optimize/{campaign_id}")
async def quick_optimize(
    campaign_id: int,
    strategy: str = "balanced",
    db: AsyncSession = Depends(get_db)
):
    """
    Quick optimization: Generate and simulate execution in one call.
    """
    try:
        strat = OptimizationStrategy(strategy)
    except ValueError:
        raise HTTPException(400, f"Invalid strategy: {strategy}")
    
    engine = OptimizationEngine(db)
    
    plan = await engine.generate_optimization_plan(
        campaign_id=campaign_id,
        strategy=strat
    )
    
    # Simulate execution
    for action in plan.actions:
        action.approved = True
    
    result = await engine.execute_plan(plan, dry_run=True)
    
    return {
        'campaign_id': campaign_id,
        'strategy': strategy,
        'summary': plan.summary,
        'simulation_result': result
    }


# ==================== ADVANCED OPTIMIZATION ENDPOINTS ====================

class AdvancedOptimizationRequest(BaseModel):
    campaign_id: int
    strategy: str = "balanced"
    target_acos: float = 25.0
    target_roas: float = 4.0
    intelligence_level: str = "standard"  # basic, standard, advanced, autonomous


class AdvancedExecuteRequest(BaseModel):
    dry_run: bool = True
    auto_approve_confidence: float = 0.8
    respect_anomalies: bool = True


@router.post("/advanced/generate-plan")
async def generate_intelligent_plan(
    request: AdvancedOptimizationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate an intelligent optimization plan using all ML capabilities.
    
    Intelligence levels:
    - basic: Rule-based + single model
    - standard: Ensemble predictions
    - advanced: Full ML suite with anomaly detection
    - autonomous: Self-learning with market intelligence
    """
    try:
        strategy = OptimizationStrategy(request.strategy)
    except ValueError:
        raise HTTPException(400, f"Invalid strategy: {request.strategy}")
    
    try:
        intelligence = IntelligenceLevel(request.intelligence_level)
    except ValueError:
        raise HTTPException(400, f"Invalid intelligence level: {request.intelligence_level}")
    
    engine = AdvancedOptimizationEngine(db)
    
    try:
        plan = await engine.generate_intelligent_plan(
            campaign_id=request.campaign_id,
            strategy=strategy,
            target_acos=request.target_acos,
            target_roas=request.target_roas,
            intelligence_level=intelligence
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    
    return {
        'campaign_id': plan.campaign_id,
        'campaign_name': plan.campaign_name,
        'strategy': plan.strategy.value,
        'intelligence_level': intelligence.value,
        'target_acos': plan.target_acos,
        'target_roas': plan.target_roas,
        'created_at': plan.created_at.isoformat(),
        'summary': plan.summary,
        
        # ML Insights
        'anomalies_detected': plan.anomalies_detected,
        'keyword_health': plan.keyword_health,
        'segment_analysis': plan.segment_analysis,
        'forecast': plan.forecast,
        'market_intelligence': plan.market_intelligence,
        'model_contributions': plan.model_contributions,
        
        # Actions
        'actions': [
            {
                'action_type': a.action_type.value,
                'entity_type': a.entity_type,
                'entity_id': a.entity_id,
                'current_value': a.current_value,
                'recommended_value': a.recommended_value,
                'change_percent': a.change_percent,
                'confidence': a.confidence,
                'reasoning': a.reasoning,
                'priority': a.priority
            }
            for a in plan.actions
        ]
    }


@router.post("/advanced/execute")
async def execute_intelligent_plan(
    request: AdvancedOptimizationRequest,
    execute_request: AdvancedExecuteRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Execute an intelligent optimization plan with ML-aware decision making.
    """
    try:
        strategy = OptimizationStrategy(request.strategy)
        intelligence = IntelligenceLevel(request.intelligence_level)
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    engine = AdvancedOptimizationEngine(db)
    
    # Generate plan
    plan = await engine.generate_intelligent_plan(
        campaign_id=request.campaign_id,
        strategy=strategy,
        target_acos=request.target_acos,
        target_roas=request.target_roas,
        intelligence_level=intelligence
    )
    
    # Execute with intelligent decision making
    result = await engine.execute_intelligent_plan(
        plan,
        dry_run=execute_request.dry_run,
        auto_approve_confidence=execute_request.auto_approve_confidence,
        respect_anomalies=execute_request.respect_anomalies
    )
    
    return result


@router.post("/advanced/quick-optimize/{campaign_id}")
async def advanced_quick_optimize(
    campaign_id: int,
    strategy: str = "balanced",
    intelligence: str = "standard",
    db: AsyncSession = Depends(get_db)
):
    """
    Advanced quick optimization with ML insights.
    """
    try:
        strat = OptimizationStrategy(strategy)
        intel = IntelligenceLevel(intelligence)
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    engine = AdvancedOptimizationEngine(db)
    
    plan = await engine.generate_intelligent_plan(
        campaign_id=campaign_id,
        strategy=strat,
        intelligence_level=intel
    )
    
    # Simulate execution
    result = await engine.execute_intelligent_plan(plan, dry_run=True)
    
    return {
        'campaign_id': campaign_id,
        'strategy': strategy,
        'intelligence_level': intelligence,
        'summary': plan.summary,
        'ml_insights': {
            'anomalies': len(plan.anomalies_detected),
            'at_risk_keywords': plan.keyword_health.get('at_risk_count', 0),
            'segments': len(plan.segment_analysis.get('segments', [])),
            'forecast_available': bool(plan.forecast)
        },
        'simulation_result': result
    }


# ==================== ADVANCED ALERTS ====================

@router.get("/advanced/alerts/{campaign_id}")
async def get_smart_alerts(
    campaign_id: int,
    target_acos: float = 25.0,
    db: AsyncSession = Depends(get_db)
):
    """
    Get ML-enhanced alerts for a campaign.
    """
    engine = AdvancedRuleEngine(db)
    alerts = await engine.evaluate_campaign(campaign_id, target_acos)
    
    return {
        'campaign_id': campaign_id,
        'alert_count': len(alerts),
        'critical_count': len([a for a in alerts if a.severity.value == 'critical']),
        'alerts': [
            {
                'trigger_type': a.trigger_type.value,
                'severity': a.severity.value,
                'title': a.title,
                'message': a.message,
                'metrics': a.metrics,
                'recommended_actions': a.recommended_actions,
                'ml_confidence': a.ml_confidence,
                'triggered_at': a.triggered_at.isoformat()
            }
            for a in sorted(alerts, key=lambda x: x.severity.value == 'critical', reverse=True)
        ]
    }


@router.get("/advanced/alerts")
async def get_all_smart_alerts(
    target_acos: float = 25.0,
    db: AsyncSession = Depends(get_db)
):
    """
    Get ML-enhanced alerts for all campaigns.
    """
    engine = AdvancedRuleEngine(db)
    all_alerts = await engine.evaluate_all_campaigns(target_acos)
    
    total_count = sum(len(alerts) for alerts in all_alerts.values())
    critical_count = sum(
        len([a for a in alerts if a.severity.value == 'critical'])
        for alerts in all_alerts.values()
    )
    
    return {
        'total_campaigns': len(all_alerts),
        'total_alerts': total_count,
        'critical_count': critical_count,
        'alerts_by_campaign': {
            campaign: [
                {
                    'trigger_type': a.trigger_type.value,
                    'severity': a.severity.value,
                    'title': a.title,
                    'message': a.message,
                    'ml_confidence': a.ml_confidence
                }
                for a in alerts
            ]
            for campaign, alerts in all_alerts.items()
        }
    }


@router.get("/advanced/rules")
async def list_smart_rules(db: AsyncSession = Depends(get_db)):
    """List all ML-enhanced rules."""
    engine = AdvancedRuleEngine(db)
    return {'rules': engine.get_active_rules()}


@router.post("/advanced/rules/toggle")
async def toggle_rule(
    rule_name: str,
    enabled: bool,
    db: AsyncSession = Depends(get_db)
):
    """Enable or disable a smart rule."""
    engine = AdvancedRuleEngine(db)
    success = engine.toggle_rule(rule_name, enabled)
    
    if success:
        return {'status': 'updated', 'rule': rule_name, 'enabled': enabled}
    raise HTTPException(404, f"Rule not found: {rule_name}")


@router.get("/intelligence-levels")
async def list_intelligence_levels():
    """List available intelligence levels."""
    return {
        'levels': [
            {
                'value': 'basic',
                'description': 'Rule-based optimization with single ML model',
                'features': ['Gradient Boosting predictions', 'Rule-based targeting']
            },
            {
                'value': 'standard',
                'description': 'Ensemble predictions for better accuracy',
                'features': ['Model ensemble', 'Multiple algorithm voting', 'Confidence weighting']
            },
            {
                'value': 'advanced',
                'description': 'Full ML suite with anomaly detection',
                'features': ['Anomaly detection', 'Keyword health analysis', 'Performance forecasting', 'Segment optimization']
            },
            {
                'value': 'autonomous',
                'description': 'Self-learning with market intelligence',
                'features': ['Market analysis', 'Competitor bid estimation', 'Opportunity detection', 'Adaptive learning']
            }
        ]
    }



@router.get("/keyword-doctor/{keyword_id}", response_model=KeywordDiagnosticResponse)
async def get_keyword_diagnosis(keyword_id: str):
    """
    Get diagnostic data for a specific keyword (Keyword Doctor).
    """
    return KeywordDiagnosticResponse(
        keyword_id=keyword_id,
        keyword_text="organic face cream",
        match_type="EXACT",
        campaign_name="Skincare_Q3_Core",
        last_scan="Just Now",
        spend=450.00,
        impressions=12402,
        clicks=340,
        current_bid=1.20,
        acos=115.0,
        ctr=0.40,
        wasted_spend=120.50,
        recommendations=[
            {
                "type": "bid_reduction",
                "title": "Bid Reduction Therapy",
                "description": "Current bid is causing inflated ACoS.",
                "current_value": 1.20,
                "proposed_value": 0.85,
                "icon": "trending_down"
            },
            {
                "type": "negative_match",
                "title": "Negative Match Injection",
                "description": "Irrelevant search terms detected.",
                "values": ["cheap", "DIY"],
                "icon": "block"
            }
        ],
        harvested_keywords=[
            {"text": "natural face moisturizer", "vol": "4.2k", "cpc": 0.95, "relevance": 85},
            {"text": "vegan night cream", "vol": "1.8k", "cpc": 1.10, "relevance": 72},
            {"text": "organic skin repair", "vol": "900", "cpc": 0.75, "relevance": 65}
        ]
    )
