"""
Keyword Clustering and Segmentation.
Groups keywords by performance patterns for bulk optimization.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class KeywordCluster:
    """A cluster of similar keywords."""
    cluster_id: int
    name: str
    keywords: List[int]  # keyword IDs
    centroid: Dict[str, float]
    avg_performance: Dict[str, float]
    recommended_action: str
    confidence: float


class KMeansClusterer:
    """K-Means clustering implementation."""
    
    def __init__(self, n_clusters: int = 5, max_iterations: int = 100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit K-Means and return cluster assignments."""
        n_samples = len(X)
        
        if n_samples < self.n_clusters:
            return np.zeros(n_samples, dtype=int)
        
        # Initialize centroids with K-Means++
        self.centroids = self._kmeans_plus_plus_init(X)
        
        for _ in range(self.max_iterations):
            # Assign points to nearest centroid
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        return self._assign_clusters(X)
    
    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using K-Means++ algorithm."""
        n_samples = len(X)
        centroids = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            distances = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            probs = distances / distances.sum()
            next_idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[next_idx])
        
        return np.array(centroids)
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid."""
        labels = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            distances = [np.sum((x - c) ** 2) for c in self.centroids]
            labels[i] = np.argmin(distances)
        return labels


class KeywordSegmenter:
    """
    Segments keywords into actionable groups based on performance.
    """
    
    # Feature columns for clustering
    CLUSTER_FEATURES = [
        'ctr', 'conversion_rate', 'acos', 'roas', 
        'cpc', 'impressions_log', 'clicks_log', 'spend_log'
    ]
    
    # Predefined segment definitions
    SEGMENTS = {
        'stars': {
            'description': 'High volume, high performance',
            'action': 'Increase bids to maximize volume',
            'acos_range': (0, 20),
            'volume': 'high'
        },
        'potential': {
            'description': 'Good performance, low volume',
            'action': 'Increase bids to scale',
            'acos_range': (0, 25),
            'volume': 'low'
        },
        'workhorses': {
            'description': 'High volume, average performance',
            'action': 'Optimize bids carefully',
            'acos_range': (20, 35),
            'volume': 'high'
        },
        'underperformers': {
            'description': 'Poor performance, still active',
            'action': 'Reduce bids or pause',
            'acos_range': (35, 100),
            'volume': 'medium'
        },
        'zombies': {
            'description': 'Very low activity',
            'action': 'Consider pausing or bid increase test',
            'acos_range': (0, 100),
            'volume': 'none'
        }
    }
    
    def __init__(self, n_clusters: int = 5):
        self.clusterer = KMeansClusterer(n_clusters=n_clusters)
        self.feature_means = None
        self.feature_stds = None
    
    def segment_keywords(
        self,
        keyword_features: List[Dict[str, Any]],
        target_acos: float = 25.0
    ) -> Dict[str, List[KeywordCluster]]:
        """
        Segment keywords into performance-based clusters.
        """
        if not keyword_features:
            return {'segments': [], 'rule_based': []}
        
        # Rule-based segmentation first
        rule_segments = self._rule_based_segmentation(keyword_features, target_acos)
        
        # ML-based clustering
        ml_clusters = self._ml_clustering(keyword_features)
        
        return {
            'rule_based': rule_segments,
            'ml_clusters': ml_clusters,
            'summary': self._generate_summary(rule_segments)
        }
    
    def _rule_based_segmentation(
        self,
        features: List[Dict[str, Any]],
        target_acos: float
    ) -> List[KeywordCluster]:
        """Segment using rule-based approach."""
        
        segments = defaultdict(list)
        
        for kw in features:
            kw_id = kw.get('keyword_id')
            acos = kw.get('acos', 0)
            clicks = kw.get('clicks', 0)
            impressions = kw.get('impressions', 0)
            
            # Classify by volume
            if impressions < 100:
                volume = 'none'
            elif impressions < 1000:
                volume = 'low'
            elif impressions < 10000:
                volume = 'medium'
            else:
                volume = 'high'
            
            # Classify by performance
            if volume == 'none':
                segment = 'zombies'
            elif acos == 0 or clicks == 0:
                segment = 'zombies'
            elif acos < target_acos * 0.8:
                segment = 'stars' if volume in ['high', 'medium'] else 'potential'
            elif acos < target_acos * 1.2:
                segment = 'workhorses'
            else:
                segment = 'underperformers'
            
            segments[segment].append(kw_id)
        
        # Build cluster objects
        clusters = []
        for segment_name, keyword_ids in segments.items():
            segment_info = self.SEGMENTS[segment_name]
            
            # Calculate average performance for segment
            segment_features = [f for f in features if f.get('keyword_id') in keyword_ids]
            avg_perf = self._calculate_avg_performance(segment_features)
            
            clusters.append(KeywordCluster(
                cluster_id=list(self.SEGMENTS.keys()).index(segment_name),
                name=segment_name,
                keywords=keyword_ids,
                centroid={},
                avg_performance=avg_perf,
                recommended_action=segment_info['action'],
                confidence=0.85
            ))
        
        return clusters
    
    def _ml_clustering(self, features: List[Dict[str, Any]]) -> List[KeywordCluster]:
        """Segment using K-Means clustering."""
        
        if len(features) < 5:
            return []
        
        # Prepare feature matrix
        X = self._prepare_features(features)
        
        # Normalize
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        X_normalized = (X - self.feature_means) / self.feature_stds
        
        # Cluster
        labels = self.clusterer.fit(X_normalized)
        
        # Build cluster objects
        clusters = []
        for k in range(self.clusterer.n_clusters):
            mask = labels == k
            cluster_features = [f for i, f in enumerate(features) if mask[i]]
            
            if not cluster_features:
                continue
            
            avg_perf = self._calculate_avg_performance(cluster_features)
            action = self._determine_cluster_action(avg_perf)
            
            clusters.append(KeywordCluster(
                cluster_id=k,
                name=f"cluster_{k}",
                keywords=[f.get('keyword_id') for f in cluster_features],
                centroid=dict(zip(self.CLUSTER_FEATURES, self.clusterer.centroids[k])) if self.clusterer.centroids is not None else {},
                avg_performance=avg_perf,
                recommended_action=action,
                confidence=0.7
            ))
        
        return clusters
    
    def _prepare_features(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare feature matrix for clustering."""
        X = []
        for f in features:
            row = [
                f.get('ctr', 0),
                f.get('conversion_rate', 0),
                f.get('acos', 0),
                f.get('roas', 0),
                f.get('cpc', 0),
                np.log1p(f.get('impressions', 0)),
                np.log1p(f.get('clicks', 0)),
                np.log1p(f.get('spend', 0))
            ]
            X.append(row)
        return np.array(X)
    
    def _calculate_avg_performance(self, features: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average performance metrics."""
        if not features:
            return {}
        
        metrics = ['acos', 'roas', 'ctr', 'conversion_rate', 'cpc']
        avg = {}
        for m in metrics:
            values = [f.get(m, 0) for f in features if f.get(m) is not None]
            avg[m] = round(np.mean(values), 2) if values else 0
        
        avg['keyword_count'] = len(features)
        avg['total_spend'] = round(sum(f.get('spend', 0) for f in features), 2)
        avg['total_sales'] = round(sum(f.get('sales', 0) for f in features), 2)
        
        return avg
    
    def _determine_cluster_action(self, avg_perf: Dict[str, float]) -> str:
        """Determine recommended action for cluster."""
        acos = avg_perf.get('acos', 50)
        roas = avg_perf.get('roas', 0)
        
        if acos < 20:
            return "Increase bids - High performers"
        elif acos < 30:
            return "Maintain bids - Good performance"
        elif acos < 45:
            return "Decrease bids - Below target"
        else:
            return "Significantly reduce or pause - Poor performance"
    
    def _generate_summary(self, clusters: List[KeywordCluster]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_keywords = sum(len(c.keywords) for c in clusters)
        
        return {
            'total_keywords': total_keywords,
            'segment_distribution': {
                c.name: {
                    'count': len(c.keywords),
                    'percentage': round(len(c.keywords) / total_keywords * 100, 1) if total_keywords > 0 else 0,
                    'action': c.recommended_action
                }
                for c in clusters
            }
        }


class PerformanceSegmenter:
    """
    Segments campaigns by performance tier.
    """
    
    TIERS = {
        'platinum': {'min_roas': 5.0, 'max_acos': 20, 'description': 'Elite performers'},
        'gold': {'min_roas': 3.5, 'max_acos': 28, 'description': 'Strong performers'},
        'silver': {'min_roas': 2.5, 'max_acos': 40, 'description': 'Average performers'},
        'bronze': {'min_roas': 1.5, 'max_acos': 60, 'description': 'Below average'},
        'needs_attention': {'min_roas': 0, 'max_acos': 100, 'description': 'Requires immediate action'}
    }
    
    def segment_campaigns(
        self,
        campaign_metrics: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Segment campaigns into performance tiers."""
        
        tiers = defaultdict(list)
        
        for campaign in campaign_metrics:
            roas = campaign.get('roas', 0)
            acos = campaign.get('acos', 100)
            
            tier = self._determine_tier(roas, acos)
            tiers[tier].append({
                'campaign_id': campaign.get('campaign_id'),
                'name': campaign.get('name'),
                'roas': roas,
                'acos': acos,
                'tier': tier
            })
        
        return dict(tiers)
    
    def _determine_tier(self, roas: float, acos: float) -> str:
        """Determine performance tier."""
        for tier, criteria in self.TIERS.items():
            if roas >= criteria['min_roas'] and acos <= criteria['max_acos']:
                return tier
        return 'needs_attention'
