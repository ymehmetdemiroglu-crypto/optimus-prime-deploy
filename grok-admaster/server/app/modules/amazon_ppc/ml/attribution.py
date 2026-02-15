"""
Attribution Modeling for Multi-Touch Conversions.
Assigns credit to different touchpoints in the customer journey.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AttributionModel(str, Enum):
    """Attribution model types."""
    LAST_CLICK = "last_click"
    FIRST_CLICK = "first_click"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"
    DATA_DRIVEN = "data_driven"


@dataclass
class TouchPoint:
    """A single touchpoint in the customer journey."""
    campaign_id: int
    keyword_id: Optional[int]
    ad_group_id: Optional[int]
    match_type: str
    timestamp: datetime
    click: bool
    impression: bool
    cost: float


@dataclass
class Conversion:
    """A conversion event."""
    order_id: str
    revenue: float
    timestamp: datetime
    touchpoints: List[TouchPoint]


@dataclass
class AttributedValue:
    """Value attributed to an entity."""
    entity_type: str  # 'campaign', 'keyword', 'ad_group'
    entity_id: int
    attributed_revenue: float
    attributed_conversions: float
    attributed_cost: float
    roas: float


class AttributionEngine:
    """
    Multi-touch attribution engine.
    """
    
    def __init__(self, model: AttributionModel = AttributionModel.POSITION_BASED):
        self.model = model
        self.decay_rate = 0.5  # For time decay model
        self.position_weights = (0.4, 0.2, 0.4)  # First, middle, last
    
    def attribute_conversion(
        self,
        conversion: Conversion
    ) -> List[AttributedValue]:
        """
        Attribute a conversion to touchpoints.
        """
        touchpoints = conversion.touchpoints
        
        if not touchpoints:
            return []
        
        # Calculate weights based on model
        if self.model == AttributionModel.LAST_CLICK:
            weights = self._last_click_weights(touchpoints)
        elif self.model == AttributionModel.FIRST_CLICK:
            weights = self._first_click_weights(touchpoints)
        elif self.model == AttributionModel.LINEAR:
            weights = self._linear_weights(touchpoints)
        elif self.model == AttributionModel.TIME_DECAY:
            weights = self._time_decay_weights(touchpoints, conversion.timestamp)
        elif self.model == AttributionModel.POSITION_BASED:
            weights = self._position_based_weights(touchpoints)
        elif self.model == AttributionModel.DATA_DRIVEN:
            weights = self._data_driven_weights(touchpoints)
        else:
            weights = self._last_click_weights(touchpoints)
        
        # Apply weights to attribution
        attributed = []
        for tp, weight in zip(touchpoints, weights):
            attributed.append(AttributedValue(
                entity_type='keyword' if tp.keyword_id else 'campaign',
                entity_id=tp.keyword_id or tp.campaign_id,
                attributed_revenue=conversion.revenue * weight,
                attributed_conversions=weight,
                attributed_cost=tp.cost,
                roas=round((conversion.revenue * weight) / tp.cost, 2) if tp.cost > 0 else 0
            ))
        
        return attributed
    
    def _last_click_weights(self, touchpoints: List[TouchPoint]) -> List[float]:
        """All credit to last touchpoint."""
        weights = [0.0] * len(touchpoints)
        if touchpoints:
            weights[-1] = 1.0
        return weights
    
    def _first_click_weights(self, touchpoints: List[TouchPoint]) -> List[float]:
        """All credit to first touchpoint."""
        weights = [0.0] * len(touchpoints)
        if touchpoints:
            weights[0] = 1.0
        return weights
    
    def _linear_weights(self, touchpoints: List[TouchPoint]) -> List[float]:
        """Equal credit to all touchpoints."""
        n = len(touchpoints)
        if n == 0:
            return []
        return [1.0 / n] * n
    
    def _time_decay_weights(
        self,
        touchpoints: List[TouchPoint],
        conversion_time: datetime
    ) -> List[float]:
        """More credit to touchpoints closer to conversion."""
        if not touchpoints:
            return []
        
        # Calculate decay based on time difference
        weights = []
        for tp in touchpoints:
            time_diff = (conversion_time - tp.timestamp).total_seconds() / 86400  # Days
            weight = np.exp(-self.decay_rate * time_diff)
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights
    
    def _position_based_weights(self, touchpoints: List[TouchPoint]) -> List[float]:
        """40% first, 20% middle, 40% last (U-shaped)."""
        n = len(touchpoints)
        
        if n == 0:
            return []
        if n == 1:
            return [1.0]
        if n == 2:
            return [0.5, 0.5]
        
        first_weight, middle_weight, last_weight = self.position_weights
        
        weights = [0.0] * n
        weights[0] = first_weight
        weights[-1] = last_weight
        
        # Distribute middle weight
        middle_count = n - 2
        if middle_count > 0:
            per_middle = middle_weight / middle_count
            for i in range(1, n - 1):
                weights[i] = per_middle
        
        return weights
    
    def _data_driven_weights(self, touchpoints: List[TouchPoint]) -> List[float]:
        """
        Data-driven attribution using Shapley values approximation.
        Simplified version - combines position, recency, and engagement.
        """
        if not touchpoints:
            return []
        
        n = len(touchpoints)
        weights = []
        
        for i, tp in enumerate(touchpoints):
            # Position factor
            if i == 0:
                position_score = 0.3
            elif i == n - 1:
                position_score = 0.4
            else:
                position_score = 0.2
            
            # Engagement factor (clicked vs impression)
            engagement_score = 1.0 if tp.click else 0.3
            
            # Cost factor (higher spend = more influence)
            cost_score = min(1.0, tp.cost / 2.0)
            
            weight = (position_score * 0.4 + 
                     engagement_score * 0.4 + 
                     cost_score * 0.2)
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights


class ConversionPathAnalyzer:
    """
    Analyzes conversion paths to understand customer journeys.
    """
    
    def __init__(self):
        self.paths: List[Conversion] = []
    
    def analyze_paths(
        self,
        conversions: List[Conversion]
    ) -> Dict[str, Any]:
        """
        Analyze all conversion paths.
        """
        self.paths = conversions
        
        # Path length analysis
        path_lengths = [len(c.touchpoints) for c in conversions]
        
        # Time to conversion
        time_to_convert = []
        for c in conversions:
            if c.touchpoints:
                first_touch = min(tp.timestamp for tp in c.touchpoints)
                time_diff = (c.timestamp - first_touch).total_seconds() / 3600  # Hours
                time_to_convert.append(time_diff)
        
        # Common paths
        path_patterns = self._extract_path_patterns(conversions)
        
        # Channel sequences
        channel_sequences = self._analyze_channel_sequences(conversions)
        
        return {
            'total_conversions': len(conversions),
            'total_revenue': sum(c.revenue for c in conversions),
            'path_length': {
                'avg': round(np.mean(path_lengths), 1) if path_lengths else 0,
                'min': min(path_lengths) if path_lengths else 0,
                'max': max(path_lengths) if path_lengths else 0,
                'distribution': self._bucket_distribution(path_lengths)
            },
            'time_to_conversion': {
                'avg_hours': round(np.mean(time_to_convert), 1) if time_to_convert else 0,
                'median_hours': round(np.median(time_to_convert), 1) if time_to_convert else 0
            },
            'common_paths': path_patterns[:10],
            'channel_sequences': channel_sequences
        }
    
    def _extract_path_patterns(
        self,
        conversions: List[Conversion]
    ) -> List[Dict[str, Any]]:
        """Extract common path patterns."""
        pattern_counts = defaultdict(lambda: {'count': 0, 'revenue': 0})
        
        for c in conversions:
            # Create pattern string
            pattern = ' > '.join([
                str(tp.campaign_id) for tp in c.touchpoints
            ])
            
            pattern_counts[pattern]['count'] += 1
            pattern_counts[pattern]['revenue'] += c.revenue
        
        # Sort by count
        patterns = [
            {'pattern': p, **stats}
            for p, stats in pattern_counts.items()
        ]
        
        return sorted(patterns, key=lambda x: x['count'], reverse=True)
    
    def _analyze_channel_sequences(
        self,
        conversions: List[Conversion]
    ) -> Dict[str, Any]:
        """Analyze channel (match type) sequences."""
        first_touch = defaultdict(int)
        last_touch = defaultdict(int)
        transitions = defaultdict(int)
        
        for c in conversions:
            if not c.touchpoints:
                continue
            
            first_touch[c.touchpoints[0].match_type] += 1
            last_touch[c.touchpoints[-1].match_type] += 1
            
            for i in range(len(c.touchpoints) - 1):
                transition = f"{c.touchpoints[i].match_type} > {c.touchpoints[i+1].match_type}"
                transitions[transition] += 1
        
        return {
            'first_touch_by_match': dict(first_touch),
            'last_touch_by_match': dict(last_touch),
            'common_transitions': dict(sorted(
                transitions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }
    
    def _bucket_distribution(self, values: List[int]) -> Dict[str, int]:
        """Create bucketed distribution."""
        buckets = {'1': 0, '2-3': 0, '4-5': 0, '6-10': 0, '10+': 0}
        
        for v in values:
            if v == 1:
                buckets['1'] += 1
            elif v <= 3:
                buckets['2-3'] += 1
            elif v <= 5:
                buckets['4-5'] += 1
            elif v <= 10:
                buckets['6-10'] += 1
            else:
                buckets['10+'] += 1
        
        return buckets


class MarkovAttribution:
    """
    Markov Chain-based attribution model.
    Calculates removal effect for each channel.
    """
    
    def __init__(self):
        self.transition_matrix: Dict[str, Dict[str, float]] = {}
        self.channels: List[str] = []
    
    def fit(self, conversions: List[Conversion]):
        """
        Fit Markov chain on conversion paths.
        """
        # Extract all channels
        channel_set = set()
        for c in conversions:
            for tp in c.touchpoints:
                channel_set.add(str(tp.campaign_id))
        
        self.channels = list(channel_set)
        
        # Add start and end states
        all_states = ['START'] + self.channels + ['CONVERT', 'NULL']
        
        # Build transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        
        for c in conversions:
            if not c.touchpoints:
                continue
            
            # Start -> First
            first_channel = str(c.touchpoints[0].campaign_id)
            transitions['START'][first_channel] += 1
            
            # Intermediate
            for i in range(len(c.touchpoints) - 1):
                from_ch = str(c.touchpoints[i].campaign_id)
                to_ch = str(c.touchpoints[i+1].campaign_id)
                transitions[from_ch][to_ch] += 1
            
            # Last -> Convert
            last_channel = str(c.touchpoints[-1].campaign_id)
            transitions[last_channel]['CONVERT'] += 1
        
        # Normalize to probabilities
        for from_state, to_states in transitions.items():
            total = sum(to_states.values())
            self.transition_matrix[from_state] = {
                to: count / total for to, count in to_states.items()
            }
    
    def calculate_removal_effect(self, channel: str) -> float:
        """
        Calculate effect of removing a channel.
        Returns the reduction in conversion probability.
        """
        # Calculate baseline conversion probability
        baseline = self._calculate_conversion_prob()
        
        # Calculate with channel removed
        removed = self._calculate_conversion_prob(remove_channel=channel)
        
        return baseline - removed
    
    def _calculate_conversion_prob(
        self,
        remove_channel: Optional[str] = None,
        max_steps: int = 100
    ) -> float:
        """Calculate probability of reaching conversion."""
        # Simulation approach
        n_simulations = 1000
        conversions = 0
        
        for _ in range(n_simulations):
            state = 'START'
            
            for _ in range(max_steps):
                if state == 'CONVERT':
                    conversions += 1
                    break
                if state == 'NULL':
                    break
                if state == remove_channel:
                    state = 'NULL'
                    break
                
                # Transition
                if state not in self.transition_matrix:
                    state = 'NULL'
                    break
                
                probs = self.transition_matrix[state]
                states = list(probs.keys())
                weights = list(probs.values())
                
                if not states:
                    state = 'NULL'
                    break
                
                state = np.random.choice(states, p=weights)
        
        return conversions / n_simulations
    
    def get_channel_attributions(self) -> Dict[str, float]:
        """Get attributed value for each channel based on removal effect."""
        removal_effects = {}
        
        for channel in self.channels:
            effect = self.calculate_removal_effect(channel)
            removal_effects[channel] = max(0, effect)
        
        # Normalize
        total = sum(removal_effects.values())
        if total > 0:
            return {ch: effect / total for ch, effect in removal_effects.items()}
        
        return removal_effects
