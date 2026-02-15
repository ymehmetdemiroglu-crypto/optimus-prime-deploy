"""
Competitor Bid Estimation and Market Intelligence.
Infers competitor bids and market conditions from impression/position data.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompetitorEstimate:
    """Estimated competitor bid information."""
    keyword: str
    estimated_top_bid: float
    estimated_avg_bid: float
    bid_range: Tuple[float, float]
    competition_level: str  # 'low', 'medium', 'high', 'very_high'
    impression_share: float
    market_size: int  # estimated total market impressions
    confidence: float


@dataclass
class MarketIntelligence:
    """Market intelligence summary."""
    keyword: str
    search_volume: int
    competition_intensity: float
    cpc_trend: str  # 'rising', 'stable', 'declining'
    opportunity_score: float
    recommended_bid: float


class CompetitorBidEstimator:
    """
    Estimates competitor bids using auction theory and historical data.
    Uses first-price auction model common in Amazon advertising.
    """
    
    def __init__(self):
        # Conversion factors based on Amazon's auction dynamics
        self.position_to_bid_multiplier = {
            1: 1.0,    # Top position = reference bid
            2: 0.85,
            3: 0.75,
            4: 0.65,
            5: 0.55
        }
    
    def estimate_competitor_bids(
        self,
        keyword_data: Dict[str, Any]
    ) -> CompetitorEstimate:
        """
        Estimate competitor bids for a keyword.
        
        Uses the relationship between:
        - Your bid
        - Your average position
        - Your win rate (impressions / estimated total impressions)
        """
        keyword = keyword_data.get('keyword', '')
        your_bid = keyword_data.get('bid', 1.0)
        avg_position = keyword_data.get('avg_position', 3.0)
        impressions = keyword_data.get('impressions', 0)
        impression_share = keyword_data.get('impression_share', 0.1)  # If available
        clicks = keyword_data.get('clicks', 0)
        cpc = keyword_data.get('cpc', your_bid * 0.8)
        
        # Estimate position multiplier
        position_mult = self._interpolate_position_multiplier(avg_position)
        
        # Estimate top bid (reverse engineer from your position)
        estimated_top_bid = your_bid / position_mult if position_mult > 0 else your_bid * 1.5
        
        # Estimate average market bid
        estimated_avg_bid = estimated_top_bid * 0.7
        
        # Calculate bid range
        bid_range = (
            round(estimated_avg_bid * 0.5, 2),
            round(estimated_top_bid * 1.2, 2)
        )
        
        # Determine competition level
        competition_level = self._assess_competition(
            cpc, your_bid, avg_position, impression_share
        )
        
        # Estimate market size
        if impression_share > 0:
            market_size = int(impressions / impression_share)
        else:
            market_size = impressions * 10  # Rough estimate
        
        # Calculate confidence
        confidence = self._calculate_confidence(impressions, clicks)
        
        return CompetitorEstimate(
            keyword=keyword,
            estimated_top_bid=round(estimated_top_bid, 2),
            estimated_avg_bid=round(estimated_avg_bid, 2),
            bid_range=bid_range,
            competition_level=competition_level,
            impression_share=round(impression_share, 2),
            market_size=market_size,
            confidence=round(confidence, 2)
        )
    
    def _interpolate_position_multiplier(self, position: float) -> float:
        """Interpolate bid multiplier for a given position."""
        if position <= 1:
            return 1.0
        if position >= 5:
            return 0.5
        
        lower_pos = int(position)
        upper_pos = lower_pos + 1
        fraction = position - lower_pos
        
        lower_mult = self.position_to_bid_multiplier.get(lower_pos, 0.6)
        upper_mult = self.position_to_bid_multiplier.get(upper_pos, 0.5)
        
        return lower_mult + fraction * (upper_mult - lower_mult)
    
    def _assess_competition(
        self,
        cpc: float,
        bid: float,
        position: float,
        impression_share: float
    ) -> str:
        """Assess competition level."""
        # CPC close to bid suggests high competition
        bid_fill_rate = cpc / bid if bid > 0 else 0.5
        
        # High position despite low impression share suggests many competitors
        competition_score = 0
        
        if bid_fill_rate > 0.9:
            competition_score += 3
        elif bid_fill_rate > 0.7:
            competition_score += 2
        else:
            competition_score += 1
        
        if position > 3:
            competition_score += 2
        elif position > 2:
            competition_score += 1
        
        if impression_share < 0.1:
            competition_score += 2
        elif impression_share < 0.3:
            competition_score += 1
        
        if competition_score >= 6:
            return 'very_high'
        elif competition_score >= 4:
            return 'high'
        elif competition_score >= 2:
            return 'medium'
        return 'low'
    
    def _calculate_confidence(self, impressions: int, clicks: int) -> float:
        """Calculate confidence based on data volume."""
        # More data = higher confidence
        if impressions >= 10000 and clicks >= 100:
            return 0.9
        elif impressions >= 1000 and clicks >= 20:
            return 0.75
        elif impressions >= 100:
            return 0.6
        return 0.4


class MarketAnalyzer:
    """
    Analyzes market conditions and opportunities.
    """
    
    def __init__(self):
        self.bid_estimator = CompetitorBidEstimator()
    
    def analyze_keyword_market(
        self,
        keyword_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> MarketIntelligence:
        """
        Generate market intelligence for a keyword.
        """
        keyword = keyword_data.get('keyword', '')
        impressions = keyword_data.get('impressions', 0)
        clicks = keyword_data.get('clicks', 0)
        spend = keyword_data.get('spend', 0)
        sales = keyword_data.get('sales', 0)
        current_bid = keyword_data.get('bid', 1.0)
        cpc = keyword_data.get('cpc', current_bid * 0.8)
        
        # Estimate search volume (impressions adjusted for share)
        impression_share = keyword_data.get('impression_share', 0.1)
        search_volume = int(impressions / impression_share) if impression_share > 0 else impressions * 10
        
        # Calculate competition intensity (0-1 scale)
        competition_intensity = self._calculate_competition_intensity(keyword_data)
        
        # Determine CPC trend
        cpc_trend = self._analyze_cpc_trend(historical_data) if historical_data else 'stable'
        
        # Calculate opportunity score
        opportunity_score = self._calculate_opportunity_score(
            keyword_data, competition_intensity, cpc_trend
        )
        
        # Recommend optimal bid
        recommended_bid = self._recommend_bid(
            keyword_data, competition_intensity, opportunity_score
        )
        
        return MarketIntelligence(
            keyword=keyword,
            search_volume=search_volume,
            competition_intensity=round(competition_intensity, 2),
            cpc_trend=cpc_trend,
            opportunity_score=round(opportunity_score, 2),
            recommended_bid=round(recommended_bid, 2)
        )
    
    def _calculate_competition_intensity(self, data: Dict[str, Any]) -> float:
        """Calculate competition intensity score."""
        bid = data.get('bid', 1.0)
        cpc = data.get('cpc', bid * 0.8)
        position = data.get('avg_position', 3.0)
        impression_share = data.get('impression_share', 0.1)
        
        # Higher CPC/bid ratio = more competition
        cpc_ratio = min(1.0, cpc / bid) if bid > 0 else 0.5
        
        # Lower position = more competition
        position_factor = min(1.0, position / 5)
        
        # Lower impression share = more competition
        share_factor = 1 - min(1.0, impression_share)
        
        return (cpc_ratio * 0.4 + position_factor * 0.3 + share_factor * 0.3)
    
    def _analyze_cpc_trend(self, historical_data: List[Dict[str, Any]]) -> str:
        """Analyze CPC trend over time."""
        if not historical_data or len(historical_data) < 7:
            return 'stable'
        
        cpcs = [d.get('cpc', 0) for d in historical_data]
        
        if len(cpcs) < 2:
            return 'stable'
        
        # Simple linear regression for trend
        x = np.arange(len(cpcs))
        slope = np.polyfit(x, cpcs, 1)[0]
        
        avg_cpc = np.mean(cpcs)
        relative_slope = slope / avg_cpc if avg_cpc > 0 else 0
        
        if relative_slope > 0.02:
            return 'rising'
        elif relative_slope < -0.02:
            return 'declining'
        return 'stable'
    
    def _calculate_opportunity_score(
        self,
        data: Dict[str, Any],
        competition: float,
        cpc_trend: str
    ) -> float:
        """
        Calculate opportunity score (0-1).
        Higher = better opportunity.
        """
        spend = data.get('spend', 0)
        sales = data.get('sales', 0)
        clicks = data.get('clicks', 0)
        impressions = data.get('impressions', 0)
        
        # ROAS factor
        roas = sales / spend if spend > 0 else 0
        roas_factor = min(1.0, roas / 5)  # Normalize to 0-1
        
        # Conversion factor
        cvr = (data.get('orders', 0) / clicks * 100) if clicks > 0 else 0
        cvr_factor = min(1.0, cvr / 20)  # 20% CVR = max score
        
        # Volume factor
        volume_factor = min(1.0, np.log1p(impressions) / 10)
        
        # Competition factor (less competition = better)
        competition_factor = 1 - competition
        
        # Trend factor
        trend_factor = {
            'declining': 0.7,  # CPC declining = good
            'stable': 0.5,
            'rising': 0.3     # CPC rising = bad
        }.get(cpc_trend, 0.5)
        
        return (
            roas_factor * 0.3 +
            cvr_factor * 0.25 +
            volume_factor * 0.15 +
            competition_factor * 0.2 +
            trend_factor * 0.1
        )
    
    def _recommend_bid(
        self,
        data: Dict[str, Any],
        competition: float,
        opportunity: float
    ) -> float:
        """Recommend optimal bid based on market conditions."""
        current_bid = data.get('bid', 1.0)
        cpc = data.get('cpc', current_bid * 0.8)
        
        # Base recommendation on current CPC
        base_bid = cpc * 1.1
        
        # Adjust for opportunity
        if opportunity > 0.7:
            # High opportunity - bid more aggressively
            adjustment = 1.2
        elif opportunity > 0.4:
            adjustment = 1.0
        else:
            # Low opportunity - bid conservatively
            adjustment = 0.8
        
        # Adjust for competition
        if competition > 0.7:
            # High competition - need higher bid
            adjustment *= 1.1
        elif competition < 0.3:
            # Low competition - can bid lower
            adjustment *= 0.9
        
        recommended = base_bid * adjustment
        
        # Don't recommend more than 50% increase or decrease
        min_bid = current_bid * 0.5
        max_bid = current_bid * 1.5
        
        return max(min_bid, min(recommended, max_bid))
    
    def find_keyword_opportunities(
        self,
        keywords_data: List[Dict[str, Any]],
        min_opportunity_score: float = 0.5
    ) -> List[MarketIntelligence]:
        """Find high-opportunity keywords."""
        opportunities = []
        
        for kw_data in keywords_data:
            intel = self.analyze_keyword_market(kw_data)
            
            if intel.opportunity_score >= min_opportunity_score:
                opportunities.append(intel)
        
        return sorted(opportunities, key=lambda x: x.opportunity_score, reverse=True)


class AuctionSimulator:
    """
    Simulates auction outcomes to optimize bidding strategy.
    """
    
    def __init__(self, competitors: int = 5):
        self.competitors = competitors
    
    def simulate_auction(
        self,
        your_bid: float,
        competitor_bids: Optional[List[float]] = None,
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Simulate auction outcomes.
        """
        if competitor_bids is None:
            # Generate random competitor bids around your bid
            competitor_bids = [
                np.random.lognormal(np.log(your_bid), 0.3)
                for _ in range(self.competitors)
            ]
        
        wins = 0
        positions = []
        cpcs = []
        
        for _ in range(n_simulations):
            # Add noise to competitor bids
            noisy_bids = [
                max(0.01, b * np.random.normal(1, 0.1))
                for b in competitor_bids
            ]
            
            all_bids = [your_bid] + noisy_bids
            sorted_bids = sorted(all_bids, reverse=True)
            
            your_position = sorted_bids.index(your_bid) + 1
            positions.append(your_position)
            
            if your_position == 1:
                wins += 1
                # CPC = second highest bid (second-price-ish)
                cpc = sorted_bids[1] if len(sorted_bids) > 1 else your_bid
            else:
                # CPC is your bid when you don't win top
                cpc = your_bid * 0.9
            
            cpcs.append(cpc)
        
        return {
            'your_bid': your_bid,
            'win_rate': round(wins / n_simulations, 3),
            'avg_position': round(np.mean(positions), 2),
            'avg_cpc': round(np.mean(cpcs), 2),
            'estimated_reach': round(wins / n_simulations * 100, 1)
        }
    
    def find_optimal_bid(
        self,
        competitor_bids: List[float],
        target_position: int = 1,
        max_cpc: float = 5.0
    ) -> Dict[str, Any]:
        """
        Find optimal bid for target position.
        """
        bids_to_test = np.linspace(0.1, max_cpc, 50)
        results = []
        
        for bid in bids_to_test:
            sim = self.simulate_auction(bid, competitor_bids)
            sim['efficiency'] = sim['win_rate'] / bid if bid > 0 else 0
            results.append(sim)
        
        # Find bid that achieves target with best efficiency
        target_met = [r for r in results if r['avg_position'] <= target_position]
        
        if target_met:
            best = max(target_met, key=lambda x: x['efficiency'])
        else:
            best = min(results, key=lambda x: x['avg_position'])
        
        return {
            'optimal_bid': best['your_bid'],
            'expected_position': best['avg_position'],
            'expected_win_rate': best['win_rate'],
            'expected_cpc': best['avg_cpc']
        }
