"""
Share of Voice Calculator for Competitive Intelligence
Calculates organic and paid visibility metrics.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import random  # For simulation

@dataclass
class SearchResult:
    position: int
    asin: str
    brand: str
    is_sponsored: bool
    ad_type: Optional[str] = None  # "SP", "SB", "SB_Video"

class ShareOfVoiceCalculator:
    def __init__(self):
        self.weights = {
            "position_1": 30,
            "position_2": 20,
            "position_3": 15,
            "position_4_10": 5,
            "position_11_20": 2,
            "position_21_plus": 1
        }
    
    def calculate_sov(self, keyword: str, search_results: List[SearchResult], 
                      your_brand: str) -> Dict[str, Any]:
        """
        Calculate Share of Voice for a keyword.
        """
        total_weight = 0
        brand_weights = {}
        your_organic_weight = 0
        your_paid_weight = 0
        
        for result in search_results:
            weight = self._get_position_weight(result.position)
            total_weight += weight
            
            if result.brand not in brand_weights:
                brand_weights[result.brand] = {"organic": 0, "paid": 0, "total": 0}
            
            if result.is_sponsored:
                brand_weights[result.brand]["paid"] += weight
                if result.brand.lower() == your_brand.lower():
                    your_paid_weight += weight
            else:
                brand_weights[result.brand]["organic"] += weight
                if result.brand.lower() == your_brand.lower():
                    your_organic_weight += weight
            
            brand_weights[result.brand]["total"] += weight
        
        # Calculate percentages
        sov_breakdown = {}
        for brand, weights in brand_weights.items():
            sov_breakdown[brand] = {
                "organic_sov": round((weights["organic"] / total_weight) * 100, 2) if total_weight > 0 else 0,
                "paid_sov": round((weights["paid"] / total_weight) * 100, 2) if total_weight > 0 else 0,
                "total_sov": round((weights["total"] / total_weight) * 100, 2) if total_weight > 0 else 0
            }
        
        your_total_sov = your_organic_weight + your_paid_weight
        
        return {
            "keyword": keyword,
            "your_brand": your_brand,
            "your_sov": {
                "organic": round((your_organic_weight / total_weight) * 100, 2) if total_weight > 0 else 0,
                "paid": round((your_paid_weight / total_weight) * 100, 2) if total_weight > 0 else 0,
                "total": round((your_total_sov / total_weight) * 100, 2) if total_weight > 0 else 0
            },
            "competitor_breakdown": sov_breakdown,
            "total_positions_analyzed": len(search_results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _get_position_weight(self, position: int) -> float:
        """Get weight based on search result position."""
        if position == 1:
            return self.weights["position_1"]
        elif position == 2:
            return self.weights["position_2"]
        elif position == 3:
            return self.weights["position_3"]
        elif position <= 10:
            return self.weights["position_4_10"]
        elif position <= 20:
            return self.weights["position_11_20"]
        else:
            return self.weights["position_21_plus"]
    
    def calculate_multi_keyword_sov(self, keyword_results: Dict[str, List[SearchResult]], 
                                    your_brand: str) -> Dict[str, Any]:
        """
        Calculate aggregate SOV across multiple keywords.
        """
        all_sov = []
        keyword_breakdown = []
        
        for keyword, results in keyword_results.items():
            sov = self.calculate_sov(keyword, results, your_brand)
            all_sov.append(sov)
            keyword_breakdown.append({
                "keyword": keyword,
                "your_total_sov": sov["your_sov"]["total"],
                "your_organic_sov": sov["your_sov"]["organic"],
                "your_paid_sov": sov["your_sov"]["paid"]
            })
        
        # Aggregate
        avg_total = sum(s["your_sov"]["total"] for s in all_sov) / max(len(all_sov), 1)
        avg_organic = sum(s["your_sov"]["organic"] for s in all_sov) / max(len(all_sov), 1)
        avg_paid = sum(s["your_sov"]["paid"] for s in all_sov) / max(len(all_sov), 1)
        
        # Find top competitors across all keywords
        competitor_totals = {}
        for sov in all_sov:
            for brand, data in sov["competitor_breakdown"].items():
                if brand.lower() != your_brand.lower():
                    if brand not in competitor_totals:
                        competitor_totals[brand] = 0
                    competitor_totals[brand] += data["total_sov"]
        
        top_competitors = sorted(competitor_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "aggregate_sov": {
                "total": round(avg_total, 2),
                "organic": round(avg_organic, 2),
                "paid": round(avg_paid, 2)
            },
            "keywords_analyzed": len(keyword_results),
            "keyword_breakdown": sorted(keyword_breakdown, key=lambda x: x["your_total_sov"], reverse=True),
            "top_competitors": [{"brand": b, "avg_sov": round(s / len(all_sov), 2)} for b, s in top_competitors],
            "opportunities": self._identify_opportunities(keyword_breakdown),
            "threats": self._identify_threats(keyword_breakdown, all_sov)
        }
    
    def _identify_opportunities(self, keyword_breakdown: List[Dict]) -> List[Dict]:
        """Identify keywords where we can gain SOV."""
        opportunities = []
        for kw in keyword_breakdown:
            if kw["your_total_sov"] < 10 and kw["your_paid_sov"] < 5:
                opportunities.append({
                    "keyword": kw["keyword"],
                    "current_sov": kw["your_total_sov"],
                    "gap": "Low paid presence",
                    "recommendation": "Increase bids on this keyword"
                })
        return opportunities[:5]
    
    def _identify_threats(self, keyword_breakdown: List[Dict], all_sov: List[Dict]) -> List[Dict]:
        """Identify competitive threats."""
        threats = []
        for i, kw in enumerate(keyword_breakdown):
            # Find dominant competitor
            if i < len(all_sov):
                competitors = all_sov[i]["competitor_breakdown"]
                for brand, data in competitors.items():
                    if data["total_sov"] > 30:  # Dominant position
                        threats.append({
                            "keyword": kw["keyword"],
                            "threat_brand": brand,
                            "their_sov": data["total_sov"],
                            "your_sov": kw["your_total_sov"],
                            "gap": round(data["total_sov"] - kw["your_total_sov"], 2)
                        })
                        break
        return sorted(threats, key=lambda x: x["gap"], reverse=True)[:5]

    def track_sov_trend(self, historical_sov: List[Dict]) -> Dict[str, Any]:
        """
        Analyze SOV trends over time.
        """
        if len(historical_sov) < 2:
            return {"trend": "insufficient_data"}
        
        recent = historical_sov[-1]["your_sov"]["total"]
        previous = historical_sov[-2]["your_sov"]["total"]
        oldest = historical_sov[0]["your_sov"]["total"]
        
        short_term_change = recent - previous
        long_term_change = recent - oldest
        
        return {
            "current_sov": recent,
            "short_term_trend": {
                "change": round(short_term_change, 2),
                "direction": "up" if short_term_change > 0 else "down" if short_term_change < 0 else "stable"
            },
            "long_term_trend": {
                "change": round(long_term_change, 2),
                "direction": "up" if long_term_change > 0 else "down" if long_term_change < 0 else "stable",
                "periods": len(historical_sov)
            },
            "assessment": self._assess_trend(short_term_change, long_term_change)
        }
    
    def _assess_trend(self, short: float, long: float) -> str:
        if short > 2 and long > 5:
            return "STRONG GROWTH: Consistently gaining market share"
        elif short > 0 and long > 0:
            return "HEALTHY: Positive trajectory"
        elif short < -2 and long < -5:
            return "ALERT: Losing market share - action required"
        elif short < 0:
            return "WATCH: Recent decline - monitor closely"
        else:
            return "STABLE: Maintaining position"


# Simulated SERP fetcher
class SerpSimulator:
    """Simulates search result data. Replace with real SERP API."""
    
    def get_search_results(self, keyword: str, marketplace: str = "US") -> List[SearchResult]:
        """Simulate fetching search results."""
        brands = ["YourBrand", "RivalBrand", "CompetitorX", "BrandY", "GenericCo"]
        results = []
        
        for pos in range(1, 21):
            is_sponsored = pos <= 4 or random.random() < 0.2
            brand = random.choice(brands)
            results.append(SearchResult(
                position=pos,
                asin=f"B0{brand[:3].upper()}{pos:03d}",
                brand=brand,
                is_sponsored=is_sponsored,
                ad_type="SP" if is_sponsored else None
            ))
        
        return results


if __name__ == "__main__":
    calc = ShareOfVoiceCalculator()
    serp = SerpSimulator()
    
    # Single keyword analysis
    results = serp.get_search_results("wireless earbuds")
    sov = calc.calculate_sov("wireless earbuds", results, "YourBrand")
    
    print("Share of Voice Analysis:")
    print(f"  Your Total SOV: {sov['your_sov']['total']}%")
    print(f"  Your Organic SOV: {sov['your_sov']['organic']}%")
    print(f"  Your Paid SOV: {sov['your_sov']['paid']}%")
    
    # Multi-keyword analysis
    keywords = ["wireless earbuds", "bluetooth headphones", "noise cancelling"]
    multi_results = {kw: serp.get_search_results(kw) for kw in keywords}
    multi_sov = calc.calculate_multi_keyword_sov(multi_results, "YourBrand")
    
    print(f"\nAggregate SOV across {len(keywords)} keywords: {multi_sov['aggregate_sov']['total']}%")
    print(f"Top Competitors: {[c['brand'] for c in multi_sov['top_competitors'][:3]]}")
