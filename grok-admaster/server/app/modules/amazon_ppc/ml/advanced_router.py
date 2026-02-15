"""
API endpoints for specialized ML capabilities.
Includes clustering, anomaly detection, search term analysis, and more.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.core.database import get_db
from ..ml import (
    KeywordSegmenter, PerformanceSegmenter,
    PPCAnomalyDetector,
    SearchTermAnalyzer,
    CompetitorBidEstimator, MarketAnalyzer, AuctionSimulator,
    AttributionEngine, ConversionPathAnalyzer,
    ExperimentManager, StatisticalTester, SampleSizeCalculator,
    KeywordHealthAnalyzer, KeywordLifecyclePredictor
)
from ..ml.attribution import AttributionModel, TouchPoint, Conversion
from ..ml.ab_testing import ExperimentType

router = APIRouter()


# ==================== REQUEST MODELS ====================

class KeywordFeatureList(BaseModel):
    keywords: List[Dict[str, Any]]
    target_acos: float = 25.0


class SearchTermData(BaseModel):
    search_terms: List[Dict[str, Any]]
    target_acos: float = 25.0


class AnomalyCheckRequest(BaseModel):
    campaign_id: int
    historical_data: List[Dict[str, Any]]
    current_data: Dict[str, Any]


class CompetitorAnalysisRequest(BaseModel):
    keyword_data: Dict[str, Any]
    historical_data: Optional[List[Dict[str, Any]]] = None


class AuctionSimulationRequest(BaseModel):
    your_bid: float
    competitor_bids: Optional[List[float]] = None
    n_simulations: int = 1000


class ExperimentRequest(BaseModel):
    name: str
    experiment_type: str  # 'bid_test', 'budget_test', etc.
    control_changes: Dict[str, Any]
    treatment_changes: Dict[str, Any]
    target_metric: str = 'acos'
    mde: float = 0.1
    traffic_split: float = 0.5


class KeywordHealthRequest(BaseModel):
    keyword_data: Dict[str, Any]
    historical_data: Optional[List[Dict[str, Any]]] = None
    target_acos: float = 25.0


# ==================== CLUSTERING ENDPOINTS ====================

@router.post("/segment/keywords")
async def segment_keywords(request: KeywordFeatureList):
    """Segment keywords into performance-based clusters."""
    segmenter = KeywordSegmenter()
    result = segmenter.segment_keywords(request.keywords, request.target_acos)
    
    return {
        'target_acos': request.target_acos,
        'rule_based_segments': [
            {
                'name': c.name,
                'keyword_count': len(c.keywords),
                'keywords': c.keywords[:20],  # First 20
                'avg_performance': c.avg_performance,
                'recommended_action': c.recommended_action,
                'confidence': c.confidence
            }
            for c in result['rule_based']
        ],
        'ml_clusters': [
            {
                'cluster_id': c.cluster_id,
                'keyword_count': len(c.keywords),
                'avg_performance': c.avg_performance,
                'recommended_action': c.recommended_action
            }
            for c in result.get('ml_clusters', [])
        ],
        'summary': result.get('summary', {})
    }


@router.post("/segment/campaigns")
async def segment_campaigns(campaigns: List[Dict[str, Any]]):
    """Segment campaigns into performance tiers."""
    segmenter = PerformanceSegmenter()
    tiers = segmenter.segment_campaigns(campaigns)
    
    return {
        'tiers': tiers,
        'tier_counts': {tier: len(camps) for tier, camps in tiers.items()}
    }


# ==================== ANOMALY DETECTION ====================

@router.post("/anomaly/detect")
async def detect_anomalies(request: AnomalyCheckRequest):
    """Detect anomalies in campaign performance."""
    detector = PPCAnomalyDetector()
    
    anomalies = detector.detect_campaign_anomalies(
        request.campaign_id,
        request.historical_data,
        request.current_data
    )
    
    return {
        'campaign_id': request.campaign_id,
        'anomaly_count': len(anomalies),
        'anomalies': [
            {
                'type': a.anomaly_type.value,
                'severity': a.severity.value,
                'metric': a.metric,
                'expected_value': a.expected_value,
                'actual_value': a.actual_value,
                'deviation': a.deviation,
                'message': a.message,
                'recommended_action': a.recommended_action
            }
            for a in anomalies
        ]
    }


@router.post("/anomaly/trend-changes")
async def detect_trend_changes(values: List[float], metric_name: str = "metric"):
    """Detect significant trend changes in a time series."""
    detector = PPCAnomalyDetector()
    changes = detector.detect_trend_changes(values, metric_name)
    
    return {
        'metric': metric_name,
        'data_points': len(values),
        'change_points': changes
    }


# ==================== SEARCH TERM ANALYSIS ====================

@router.post("/search-terms/analyze")
async def analyze_search_terms(request: SearchTermData):
    """Comprehensive search term analysis."""
    analyzer = SearchTermAnalyzer()
    result = analyzer.analyze_search_terms(request.search_terms, request.target_acos)
    
    return result


@router.post("/search-terms/negatives")
async def find_negative_keywords(
    request: SearchTermData,
    min_spend: float = 10.0,
    max_conversions: int = 0
):
    """Find potential negative keywords."""
    analyzer = SearchTermAnalyzer()
    negatives = analyzer.find_negative_keywords(
        request.search_terms, min_spend, max_conversions
    )
    
    return {
        'negative_candidates': negatives[:50],
        'total_found': len(negatives)
    }


@router.post("/search-terms/exact-match")
async def find_exact_match_candidates(
    request: SearchTermData,
    min_orders: int = 2
):
    """Find search terms that should be exact match keywords."""
    analyzer = SearchTermAnalyzer()
    candidates = analyzer.find_exact_match_candidates(
        request.search_terms, request.target_acos, min_orders
    )
    
    return {
        'exact_match_candidates': candidates[:50],
        'total_found': len(candidates)
    }


# ==================== COMPETITOR ANALYSIS ====================

@router.post("/competitor/estimate-bids")
async def estimate_competitor_bids(request: CompetitorAnalysisRequest):
    """Estimate competitor bids for a keyword."""
    estimator = CompetitorBidEstimator()
    estimate = estimator.estimate_competitor_bids(request.keyword_data)
    
    return {
        'keyword': estimate.keyword,
        'estimated_top_bid': estimate.estimated_top_bid,
        'estimated_avg_bid': estimate.estimated_avg_bid,
        'bid_range': estimate.bid_range,
        'competition_level': estimate.competition_level,
        'impression_share': estimate.impression_share,
        'market_size': estimate.market_size,
        'confidence': estimate.confidence
    }


@router.post("/competitor/market-intelligence")
async def get_market_intelligence(request: CompetitorAnalysisRequest):
    """Get market intelligence for a keyword."""
    analyzer = MarketAnalyzer()
    intel = analyzer.analyze_keyword_market(
        request.keyword_data,
        request.historical_data
    )
    
    return {
        'keyword': intel.keyword,
        'search_volume': intel.search_volume,
        'competition_intensity': intel.competition_intensity,
        'cpc_trend': intel.cpc_trend,
        'opportunity_score': intel.opportunity_score,
        'recommended_bid': intel.recommended_bid
    }


@router.post("/competitor/opportunities")
async def find_keyword_opportunities(
    keywords: List[Dict[str, Any]],
    min_opportunity_score: float = 0.5
):
    """Find high-opportunity keywords."""
    analyzer = MarketAnalyzer()
    opportunities = analyzer.find_keyword_opportunities(keywords, min_opportunity_score)
    
    return {
        'opportunities': [
            {
                'keyword': o.keyword,
                'search_volume': o.search_volume,
                'competition_intensity': o.competition_intensity,
                'opportunity_score': o.opportunity_score,
                'recommended_bid': o.recommended_bid
            }
            for o in opportunities[:20]
        ],
        'total_found': len(opportunities)
    }


@router.post("/competitor/simulate-auction")
async def simulate_auction(request: AuctionSimulationRequest):
    """Simulate auction outcomes."""
    simulator = AuctionSimulator()
    result = simulator.simulate_auction(
        request.your_bid,
        request.competitor_bids,
        request.n_simulations
    )
    
    return result


@router.post("/competitor/optimal-bid")
async def find_optimal_bid(
    competitor_bids: List[float],
    target_position: int = 1,
    max_cpc: float = 5.0
):
    """Find optimal bid for target position."""
    simulator = AuctionSimulator()
    result = simulator.find_optimal_bid(competitor_bids, target_position, max_cpc)
    
    return result


# ==================== A/B TESTING ====================

# Global experiment manager
_experiment_manager = ExperimentManager()


@router.post("/experiments/create")
async def create_experiment(request: ExperimentRequest):
    """Create a new A/B experiment."""
    try:
        exp_type = ExperimentType(request.experiment_type)
    except ValueError:
        return {'error': f'Invalid experiment type: {request.experiment_type}'}
    
    experiment = _experiment_manager.create_experiment(
        name=request.name,
        experiment_type=exp_type,
        control_changes=request.control_changes,
        treatment_changes=request.treatment_changes,
        target_metric=request.target_metric,
        mde=request.mde,
        traffic_split=request.traffic_split
    )
    
    return {
        'experiment_id': experiment.experiment_id,
        'name': experiment.name,
        'status': experiment.status.value,
        'control': experiment.control.__dict__,
        'treatment': experiment.treatment.__dict__
    }


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start an experiment."""
    _experiment_manager.start_experiment(experiment_id)
    
    return {'status': 'started', 'experiment_id': experiment_id}


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop an experiment."""
    _experiment_manager.stop_experiment(experiment_id)
    
    return {'status': 'stopped', 'experiment_id': experiment_id}


@router.post("/experiments/{experiment_id}/results")
async def record_experiment_results(
    experiment_id: str,
    variant_id: str,
    metrics: Dict[str, Any]
):
    """Record results for an experiment variant."""
    _experiment_manager.record_results(experiment_id, variant_id, metrics)
    
    return {'status': 'recorded', 'variant_id': variant_id}


@router.get("/experiments/{experiment_id}/analyze")
async def analyze_experiment(experiment_id: str):
    """Analyze experiment results."""
    return _experiment_manager.analyze_experiment(experiment_id)


@router.get("/experiments/sample-size")
async def calculate_sample_size(
    baseline_metric: float,
    mde: float = 0.1,
    metric_type: str = 'proportion',
    baseline_std: Optional[float] = None
):
    """Calculate required sample size for experiment."""
    return _experiment_manager.estimate_sample_size(
        baseline_metric, mde, metric_type, baseline_std
    )


# ==================== KEYWORD HEALTH ====================

@router.post("/keyword-health/analyze")
async def analyze_keyword_health(request: KeywordHealthRequest):
    """Analyze keyword health status."""
    analyzer = KeywordHealthAnalyzer()
    report = analyzer.analyze_keyword_health(
        request.keyword_data,
        request.historical_data,
        request.target_acos
    )
    
    return {
        'keyword_id': report.keyword_id,
        'keyword_text': report.keyword_text,
        'health_status': report.health_status.value,
        'health_score': report.health_score,
        'risk_factors': report.risk_factors,
        'recommendations': report.recommendations,
        'predicted_days_to_decline': report.predicted_days_to_decline,
        'improvement_potential': report.improvement_potential
    }


@router.post("/keyword-health/bulk")
async def analyze_bulk_keyword_health(
    keywords: List[Dict[str, Any]],
    target_acos: float = 25.0
):
    """Analyze health of multiple keywords."""
    analyzer = KeywordHealthAnalyzer()
    
    results = []
    for kw in keywords:
        report = analyzer.analyze_keyword_health(kw, None, target_acos)
        results.append({
            'keyword_id': report.keyword_id,
            'health_status': report.health_status.value,
            'health_score': report.health_score,
            'risk_factors': report.risk_factors[:2]  # Top 2
        })
    
    # Summary
    status_counts = {}
    for r in results:
        status = r['health_status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return {
        'total_keywords': len(results),
        'status_distribution': status_counts,
        'at_risk_count': sum(1 for r in results if r['health_status'] in ['at_risk', 'declining', 'critical']),
        'results': results
    }


@router.post("/keyword-health/lifecycle")
async def predict_keyword_lifecycle(
    keyword_data: Dict[str, Any],
    days_active: int,
    performance_trend: str  # 'up', 'stable', 'down'
):
    """Predict keyword lifecycle stage."""
    predictor = KeywordLifecyclePredictor()
    result = predictor.predict_lifecycle_stage(keyword_data, days_active, performance_trend)
    
    return result


# ==================== ATTRIBUTION ====================

@router.get("/attribution/models")
async def list_attribution_models():
    """List available attribution models."""
    return {
        'models': [
            {'value': m.value, 'description': _get_model_description(m)}
            for m in AttributionModel
        ]
    }


def _get_model_description(model: AttributionModel) -> str:
    descriptions = {
        AttributionModel.LAST_CLICK: "All credit to last touchpoint before conversion",
        AttributionModel.FIRST_CLICK: "All credit to first touchpoint",
        AttributionModel.LINEAR: "Equal credit to all touchpoints",
        AttributionModel.TIME_DECAY: "More credit to touchpoints closer to conversion",
        AttributionModel.POSITION_BASED: "40% first, 20% middle, 40% last (U-shaped)",
        AttributionModel.DATA_DRIVEN: "Weights based on engagement and position"
    }
    return descriptions.get(model, "")
