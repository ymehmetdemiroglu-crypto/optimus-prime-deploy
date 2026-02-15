from fastmcp import FastMCP
from typing import List, Dict

mcp = FastMCP("DSP-War-Room")

@mcp.tool()
def get_dsp_audiences() -> List[Dict]:
    """Fetch active DSP audiences and their current metrics."""
    # This would normally query the DB or Amazon API
    return [
        {"name": "Abandonment Recovery", "type": "Retargeting", "reach": "12.4k", "relevance": 98, "status": "high"},
        {"name": "Competitor: Anker Rivals", "type": "Conquest", "reach": "85k", "relevance": 82, "status": "active"},
        {"name": "Eco-Friendly Lifestyle", "type": "Lookalike", "reach": "1.2M", "relevance": 65, "status": "expanding"},
        {"name": "High-LTV Custom Seed", "type": "1P Data", "reach": "5.2k", "relevance": 94, "status": "steady"},
    ]

@mcp.tool()
def simulate_attack(budget: float, target_type: str) -> Dict:
    """
    Simulate the impact of a DSP attack strategy.
    Args:
        budget: The budget in USD for the attack.
        target_type: The type of target (e.g., 'conquest', 'retargeting', 'awareness').
    """
    if target_type == "conquest":
        return {
            "predicted_share_theft": "18%",
            "organic_lift": "1.4x",
            "estimated_reach": f"{int(budget * 5.6)} users",
            "confidence_score": 0.88
        }
    return {
        "predicted_share_theft": "5%",
        "organic_lift": "1.1x",
        "estimated_reach": "Unknown",
        "confidence_score": 0.5
    }

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.researcher import perform_research, searchapi_amazon_search

@mcp.tool()
async def analyze_market_position(keyword: str, asin: str) -> Dict:
    """
    Analyze the market position of a specific ASIN for a target keyword using live SearchApi data.
    Determine if we are a 'Leader', 'Challenger', or 'Invisible' relative to the top 3 results.
    """
    results = searchapi_amazon_search(query=keyword)
    if isinstance(results, str):
        return {"error": results}
    
    if not isinstance(results, list):
         return {"error": "Unexpected response format from SearchApi"}
    
    # 1. Locate our ASIN
    our_rank = -1
    our_item = None
    for i, item in enumerate(results):
        if item.get("asin") == asin:
            our_rank = i + 1
            our_item = item
            break
            
    # 2. Analyze Top 3 Competitors
    top_competitors = results[:3]
    
    # 3. Calculate Deltas if we found our item
    status = "Invisible"
    opportunity = "Growth" # Default
    
    if our_rank > 0:
        if our_rank <= 3:
            status = "Leader"
            opportunity = "Defend"
        else:
            status = "Challenger"
            opportunity = "Attack"
            
        # Refine opportunity based on price
        avg_competitor_price = 0
        valid_prices = 0
        for comp in top_competitors:
             # handle price strings like "$19.99"
             try:
                 p = float(comp.get("price", "0").replace("$", "").replace(",", ""))
                 if p > 0:
                     avg_competitor_price += p
                     valid_prices += 1
             except:
                 pass
        
        if valid_prices > 0:
            avg_competitor_price /= valid_prices
            try:
                our_price = float(our_item.get("price", "0").replace("$", "").replace(",", ""))
                if our_price < avg_competitor_price:
                    opportunity += " (Price Advantage)"
            except:
                pass

    return {
        "keyword": keyword,
        "asin": asin,
        "rank": our_rank if our_rank > 0 else "Not Ranked in Top 10",
        "status": status,
        "opportunity": opportunity,
        "top_competitors": [c.get("title") for c in top_competitors]
    }

@mcp.tool()
async def get_ppc_recommendations(asin: str = None, title: str = None, strategic_context: Dict = None) -> Dict:
    """
    Generate AI-powered PPC recommendations (keywords and bids) for a specific ASIN or Product Title.
    Accepts an optional 'strategic_context' from analyze_market_position to refine the strategy.
    """
    target = title if title else asin
    
    # Base Recommendations
    strategy = "auto_pilot"
    base_keywords = []
    
    # Manual Override Logic for Project and Stanley (kept for demo consistency)
    if title and "Stanley" in title:
         base_keywords = [
            {"text": "stanley aerolight transit mug", "match_type": "exact", "suggested_bid": 1.95},
            {"text": "stanley leak proof travel mug", "match_type": "phrase", "suggested_bid": 1.65},
        ]
    else:
        # Default Project Keywords
        base_keywords = [
            {"text": "gan travel charger 145w", "match_type": "exact", "suggested_bid": 2.45},
            {"text": "multi port usb c charger", "match_type": "phrase", "suggested_bid": 1.85},
        ]
        
    # Strategic Refinement logic
    if strategic_context:
        opportunity = strategic_context.get("opportunity", "")
        
        if "Attack" in opportunity:
            strategy = "aggressive_growth"
            # Add Conquesting Keywords
            competitors = strategic_context.get("top_competitors", [])
            for i, comp in enumerate(competitors):
                # Simple extraction of first 3 words for brand keyword
                brand_key = " ".join(comp.split()[:2])
                base_keywords.append({
                    "text": brand_key.lower(),
                    "match_type": "exact", 
                    "suggested_bid": 3.50, # Higher bid for conquest
                    "note": "Conquesting Target" 
                })
        
        elif "Defend" in opportunity:
            strategy = "profit_guard"
            # Ensure we own our brand
            base_keywords.append({
                "text": f"{target} official", 
                "match_type": "exact", 
                "suggested_bid": 5.00, # Moat Bid
                "note": "Brand Defense"
            })
            
    return {
        "target": target,
        "recommended_keywords": base_keywords,
        "suggested_daily_budget": 50.0 if strategy == "aggressive_growth" else 35.0,
        "strategy": strategy,
        "context_used": strategic_context is not None
    }

@mcp.tool()
def create_ppc_campaign(name: str, daily_budget: float, strategy: str, keywords: List[Dict]) -> Dict:
    """
    Provision a new PPC campaign in the Optimus AdMaster system.
    Args:
        name: Name of the campaign.
        daily_budget: Daily budget in USD.
        strategy: AI strategy (manual, auto_pilot, aggressive_growth, profit_guard).
        keywords: List of dictionaries with 'text', 'match_type', and 'bid'.
    """
    # Simulate DB insertion
    campaign_id = f"amzn-ppc-{os.urandom(4).hex()}"
    return {
        "status": "success",
        "message": f"Campaign '{name}' has been provisioned and is pending Amazon sync.",
        "campaign_id": campaign_id,
        "config": {
            "name": name,
            "budget": daily_budget,
            "strategy": strategy,
            "keyword_count": len(keywords)
        }
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
