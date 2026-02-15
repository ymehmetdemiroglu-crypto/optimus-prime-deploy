"""
A/B Testing Framework for PPC Experiments.
Statistical testing for bid and campaign experiments.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class ExperimentType(str, Enum):
    """Type of experiment."""
    BID_TEST = "bid_test"
    BUDGET_TEST = "budget_test"
    TARGETING_TEST = "targeting_test"
    AD_COPY_TEST = "ad_copy_test"


@dataclass
class ExperimentVariant:
    """A variant in an A/B test."""
    variant_id: str
    name: str
    description: str
    changes: Dict[str, Any]  # e.g., {'bid_multiplier': 1.2}
    traffic_percentage: float  # 0-1


@dataclass
class ExperimentResult:
    """Results of an experiment."""
    variant_id: str
    impressions: int
    clicks: int
    spend: float
    sales: float
    orders: int
    ctr: float
    conversion_rate: float
    acos: float
    roas: float


@dataclass
class Experiment:
    """A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime]
    control: ExperimentVariant
    treatment: ExperimentVariant
    target_metric: str  # 'acos', 'roas', 'ctr', 'conversion_rate'
    minimum_detectable_effect: float  # e.g., 0.1 for 10%
    confidence_level: float  # e.g., 0.95


@dataclass
class StatisticalResult:
    """Statistical test results."""
    is_significant: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    recommendation: str


class StatisticalTester:
    """
    Statistical testing for A/B experiments.
    """
    
    @staticmethod
    def z_test_proportions(
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
        confidence: float = 0.95
    ) -> StatisticalResult:
        """
        Two-proportion z-test for conversion rates.
        """
        # Proportions
        p_a = successes_a / trials_a if trials_a > 0 else 0
        p_b = successes_b / trials_b if trials_b > 0 else 0
        
        # Pooled proportion
        p_pooled = (successes_a + successes_b) / (trials_a + trials_b) if (trials_a + trials_b) > 0 else 0
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/trials_a + 1/trials_b)) if (trials_a > 0 and trials_b > 0) else 1
        
        if se == 0:
            return StatisticalResult(
                is_significant=False,
                p_value=1.0,
                confidence_interval=(0, 0),
                effect_size=0,
                power=0,
                recommendation="Insufficient data"
            )
        
        # Z-statistic
        z = (p_b - p_a) / se
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - (1 - confidence) / 2)
        margin = z_critical * se
        ci = (p_b - p_a - margin, p_b - p_a + margin)
        
        # Effect size (relative change)
        effect_size = (p_b - p_a) / p_a if p_a > 0 else 0
        
        # Power (simplified)
        power = 1 - stats.norm.cdf(z_critical - abs(z))
        
        is_significant = p_value < (1 - confidence)
        
        recommendation = StatisticalTester._generate_recommendation(
            is_significant, effect_size, p_value, power
        )
        
        return StatisticalResult(
            is_significant=is_significant,
            p_value=round(p_value, 4),
            confidence_interval=(round(ci[0], 4), round(ci[1], 4)),
            effect_size=round(effect_size, 4),
            power=round(power, 2),
            recommendation=recommendation
        )
    
    @staticmethod
    def t_test_means(
        values_a: List[float],
        values_b: List[float],
        confidence: float = 0.95
    ) -> StatisticalResult:
        """
        Independent two-sample t-test for means.
        """
        if len(values_a) < 2 or len(values_b) < 2:
            return StatisticalResult(
                is_significant=False,
                p_value=1.0,
                confidence_interval=(0, 0),
                effect_size=0,
                power=0,
                recommendation="Insufficient data"
            )
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        
        # Means and standard deviations
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a, ddof=1)
        std_b = np.std(values_b, ddof=1)
        n_a = len(values_a)
        n_b = len(values_b)
        
        # Pooled standard error
        se = np.sqrt(std_a**2/n_a + std_b**2/n_b)
        
        # Confidence interval
        df = (std_a**2/n_a + std_b**2/n_b)**2 / (
            (std_a**2/n_a)**2/(n_a-1) + (std_b**2/n_b)**2/(n_b-1)
        )
        t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df)
        margin = t_critical * se
        ci = (mean_b - mean_a - margin, mean_b - mean_a + margin)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n_a-1)*std_a**2 + (n_b-1)*std_b**2) / (n_a + n_b - 2))
        effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
        
        is_significant = p_value < (1 - confidence)
        
        recommendation = StatisticalTester._generate_recommendation(
            is_significant, effect_size, p_value, 0.8
        )
        
        return StatisticalResult(
            is_significant=is_significant,
            p_value=round(p_value, 4),
            confidence_interval=(round(ci[0], 4), round(ci[1], 4)),
            effect_size=round(effect_size, 4),
            power=0.8,  # Simplified
            recommendation=recommendation
        )
    
    @staticmethod
    def _generate_recommendation(
        is_significant: bool,
        effect_size: float,
        p_value: float,
        power: float
    ) -> str:
        """Generate recommendation based on results."""
        if not is_significant and power < 0.8:
            return "Inconclusive - Need more data for reliable results"
        
        if is_significant:
            if effect_size > 0.1:
                return "Strong positive effect - Recommend rolling out treatment"
            elif effect_size > 0:
                return "Small positive effect - Consider rolling out with monitoring"
            elif effect_size > -0.1:
                return "Small negative effect - Recommend keeping control"
            else:
                return "Strong negative effect - Do not roll out treatment"
        else:
            return "No significant difference detected - Keep control"


class SampleSizeCalculator:
    """
    Calculate required sample size for experiments.
    """
    
    @staticmethod
    def for_proportion(
        baseline_rate: float,
        minimum_detectable_effect: float,
        confidence: float = 0.95,
        power: float = 0.8
    ) -> int:
        """
        Calculate sample size for proportion test (e.g., CTR, CVR).
        """
        alpha = 1 - confidence
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        # Pooled proportion
        p_bar = (p1 + p2) / 2
        
        numerator = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + 
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2
        
        n = numerator / denominator if denominator > 0 else float('inf')
        
        return int(np.ceil(n))
    
    @staticmethod
    def for_mean(
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        confidence: float = 0.95,
        power: float = 0.8
    ) -> int:
        """
        Calculate sample size for mean test (e.g., ACoS, ROAS).
        """
        alpha = 1 - confidence
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        effect = baseline_mean * minimum_detectable_effect
        
        n = 2 * ((z_alpha + z_beta) * baseline_std / effect) ** 2
        
        return int(np.ceil(n))


class ExperimentManager:
    """
    Manages A/B experiments.
    """
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.results: Dict[str, Dict[str, ExperimentResult]] = {}  # exp_id -> variant_id -> result
    
    def create_experiment(
        self,
        name: str,
        experiment_type: ExperimentType,
        control_changes: Dict[str, Any],
        treatment_changes: Dict[str, Any],
        target_metric: str = 'acos',
        mde: float = 0.1,
        confidence: float = 0.95,
        traffic_split: float = 0.5
    ) -> Experiment:
        """
        Create a new A/B experiment.
        """
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        experiment = Experiment(
            experiment_id=exp_id,
            name=name,
            description=f"{experiment_type.value} experiment",
            experiment_type=experiment_type,
            status=ExperimentStatus.DRAFT,
            start_date=datetime.now(),
            end_date=None,
            control=ExperimentVariant(
                variant_id=f"{exp_id}_control",
                name="Control",
                description="Baseline",
                changes=control_changes,
                traffic_percentage=1 - traffic_split
            ),
            treatment=ExperimentVariant(
                variant_id=f"{exp_id}_treatment",
                name="Treatment",
                description="Test variant",
                changes=treatment_changes,
                traffic_percentage=traffic_split
            ),
            target_metric=target_metric,
            minimum_detectable_effect=mde,
            confidence_level=confidence
        )
        
        self.experiments[exp_id] = experiment
        self.results[exp_id] = {}
        
        return experiment
    
    def start_experiment(self, experiment_id: str):
        """Start an experiment."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].status = ExperimentStatus.RUNNING
            self.experiments[experiment_id].start_date = datetime.now()
    
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].status = ExperimentStatus.STOPPED
            self.experiments[experiment_id].end_date = datetime.now()
    
    def record_results(
        self,
        experiment_id: str,
        variant_id: str,
        metrics: Dict[str, Any]
    ):
        """Record results for a variant."""
        impressions = metrics.get('impressions', 0)
        clicks = metrics.get('clicks', 0)
        spend = metrics.get('spend', 0)
        sales = metrics.get('sales', 0)
        orders = metrics.get('orders', 0)
        
        result = ExperimentResult(
            variant_id=variant_id,
            impressions=impressions,
            clicks=clicks,
            spend=spend,
            sales=sales,
            orders=orders,
            ctr=round(clicks / impressions * 100, 2) if impressions > 0 else 0,
            conversion_rate=round(orders / clicks * 100, 2) if clicks > 0 else 0,
            acos=round(spend / sales * 100, 2) if sales > 0 else 0,
            roas=round(sales / spend, 2) if spend > 0 else 0
        )
        
        self.results[experiment_id][variant_id] = result
    
    def analyze_experiment(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Analyze experiment results.
        """
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        exp = self.experiments[experiment_id]
        results = self.results.get(experiment_id, {})
        
        control_result = results.get(exp.control.variant_id)
        treatment_result = results.get(exp.treatment.variant_id)
        
        if not control_result or not treatment_result:
            return {'error': 'Results not available for both variants'}
        
        # Run statistical test based on target metric
        if exp.target_metric in ['ctr', 'conversion_rate']:
            # Proportion test
            if exp.target_metric == 'ctr':
                stat_result = StatisticalTester.z_test_proportions(
                    control_result.clicks,
                    control_result.impressions,
                    treatment_result.clicks,
                    treatment_result.impressions,
                    exp.confidence_level
                )
            else:  # conversion_rate
                stat_result = StatisticalTester.z_test_proportions(
                    control_result.orders,
                    control_result.clicks,
                    treatment_result.orders,
                    treatment_result.clicks,
                    exp.confidence_level
                )
        else:
            # For ACoS/ROAS, we'd need daily values - simplified here
            stat_result = StatisticalResult(
                is_significant=False,
                p_value=0.5,
                confidence_interval=(0, 0),
                effect_size=0,
                power=0,
                recommendation="Use daily values for proper analysis"
            )
        
        return {
            'experiment': {
                'id': exp.experiment_id,
                'name': exp.name,
                'status': exp.status.value,
                'target_metric': exp.target_metric,
                'confidence_level': exp.confidence_level
            },
            'control': {
                'variant_id': control_result.variant_id,
                'impressions': control_result.impressions,
                'clicks': control_result.clicks,
                exp.target_metric: getattr(control_result, exp.target_metric)
            },
            'treatment': {
                'variant_id': treatment_result.variant_id,
                'impressions': treatment_result.impressions,
                'clicks': treatment_result.clicks,
                exp.target_metric: getattr(treatment_result, exp.target_metric)
            },
            'statistical_analysis': {
                'is_significant': stat_result.is_significant,
                'p_value': stat_result.p_value,
                'confidence_interval': stat_result.confidence_interval,
                'effect_size': stat_result.effect_size,
                'power': stat_result.power,
                'recommendation': stat_result.recommendation
            }
        }
    
    def estimate_sample_size(
        self,
        baseline_metric: float,
        mde: float,
        metric_type: str = 'proportion',
        baseline_std: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Estimate required sample size for various power levels.
        """
        results = {}
        
        for power in [0.8, 0.9, 0.95]:
            if metric_type == 'proportion':
                n = SampleSizeCalculator.for_proportion(
                    baseline_metric, mde, power=power
                )
            else:
                std = baseline_std or baseline_metric * 0.3
                n = SampleSizeCalculator.for_mean(
                    baseline_metric, std, mde, power=power
                )
            
            results[f'power_{int(power*100)}'] = n
        
        return results
