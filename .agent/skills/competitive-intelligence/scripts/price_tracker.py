"""
Price Tracker for Competitive Intelligence
Monitors competitor pricing changes and detects price wars.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import random  # For simulation; replace with real API calls

@dataclass
class PricePoint:
    asin: str
    price: float
    timestamp: datetime
    source: str = "amazon"
    is_deal: bool = False
    deal_type: Optional[str] = None

@dataclass
class PriceChange:
    asin: str
    competitor: str
    old_price: float
    new_price: float
    change_pct: float
    change_type: str  # "drop", "increase"
    detected_at: datetime
    likely_reason: str

class PriceTracker:
    def __init__(self, db_client=None):
        self.db = db_client
        self.price_history: Dict[str, List[PricePoint]] = {}
        self.alert_thresholds = {
            "significant_drop": 10,  # %
            "price_war_indicator": 15,  # %
            "deal_detection": 5  # %
        }
    
    def track_price(self, asin: str, current_price: float, 
                    competitor: str = "unknown", is_deal: bool = False) -> Optional[PriceChange]:
        """
        Record a price point and detect changes.
        """
        now = datetime.now()
        
        price_point = PricePoint(
            asin=asin,
            price=current_price,
            timestamp=now,
            is_deal=is_deal
        )
        
        if asin not in self.price_history:
            self.price_history[asin] = []
        
        # Check for price change
        change = None
        if self.price_history[asin]:
            last_price = self.price_history[asin][-1].price
            if last_price != current_price:
                change_pct = ((current_price - last_price) / last_price) * 100
                
                change = PriceChange(
                    asin=asin,
                    competitor=competitor,
                    old_price=last_price,
                    new_price=current_price,
                    change_pct=round(change_pct, 2),
                    change_type="drop" if change_pct < 0 else "increase",
                    detected_at=now,
                    likely_reason=self._infer_reason(change_pct, is_deal)
                )
        
        self.price_history[asin].append(price_point)
        return change
    
    def _infer_reason(self, change_pct: float, is_deal: bool) -> str:
        """Infer likely reason for price change."""
        if is_deal:
            return "Lightning Deal or Promotion active"
        elif change_pct < -20:
            return "Possible inventory clearance or aggressive market grab"
        elif change_pct < -10:
            return "Competitive repositioning or price matching"
        elif change_pct < 0:
            return "Minor price adjustment"
        elif change_pct > 20:
            return "Supply shortage or demand-based pricing"
        elif change_pct > 10:
            return "Cost increase pass-through or premium positioning"
        else:
            return "Normal price fluctuation"
    
    def get_price_history(self, asin: str, days: int = 30) -> List[Dict]:
        """Get price history for an ASIN."""
        if asin not in self.price_history:
            return []
        
        cutoff = datetime.now() - timedelta(days=days)
        history = [
            asdict(p) for p in self.price_history[asin]
            if p.timestamp >= cutoff
        ]
        return history
    
    def detect_price_war(self, category_asins: List[str]) -> Dict[str, Any]:
        """
        Detect if a price war is occurring in a category.
        """
        recent_drops = 0
        total_drop_pct = 0
        affected_products = []
        
        for asin in category_asins:
            if asin in self.price_history and len(self.price_history[asin]) >= 2:
                recent = self.price_history[asin][-1].price
                week_ago_prices = [
                    p.price for p in self.price_history[asin]
                    if p.timestamp >= datetime.now() - timedelta(days=7)
                ]
                if week_ago_prices:
                    earliest = week_ago_prices[0]
                    change = ((recent - earliest) / earliest) * 100
                    if change < -5:
                        recent_drops += 1
                        total_drop_pct += abs(change)
                        affected_products.append({"asin": asin, "drop_pct": round(change, 2)})
        
        price_war_score = min(100, (recent_drops / max(len(category_asins), 1)) * 100 + (total_drop_pct / 10))
        
        return {
            "price_war_detected": price_war_score > 50,
            "severity_score": round(price_war_score, 1),
            "products_with_drops": recent_drops,
            "average_drop_pct": round(total_drop_pct / max(recent_drops, 1), 2),
            "affected_products": affected_products[:10],
            "recommendation": self._get_price_war_recommendation(price_war_score)
        }
    
    def _get_price_war_recommendation(self, score: float) -> str:
        if score > 75:
            return "HIGH ALERT: Aggressive price war in progress. Consider defensive pricing or differentiation strategy."
        elif score > 50:
            return "MODERATE: Price competition intensifying. Monitor closely and prepare response."
        elif score > 25:
            return "LOW: Some competitive pricing activity. Normal market dynamics."
        else:
            return "STABLE: No significant price war indicators."
    
    def compare_to_competitors(self, your_asin: str, competitor_asins: List[str]) -> Dict[str, Any]:
        """
        Compare your price to competitors.
        """
        if your_asin not in self.price_history or not self.price_history[your_asin]:
            return {"error": "No price data for your product"}
        
        your_price = self.price_history[your_asin][-1].price
        
        comparisons = []
        for comp_asin in competitor_asins:
            if comp_asin in self.price_history and self.price_history[comp_asin]:
                comp_price = self.price_history[comp_asin][-1].price
                diff = your_price - comp_price
                diff_pct = (diff / comp_price) * 100
                
                comparisons.append({
                    "competitor_asin": comp_asin,
                    "competitor_price": comp_price,
                    "price_difference": round(diff, 2),
                    "difference_pct": round(diff_pct, 2),
                    "position": "higher" if diff > 0 else "lower" if diff < 0 else "equal"
                })
        
        avg_competitor_price = sum(c["competitor_price"] for c in comparisons) / max(len(comparisons), 1)
        
        return {
            "your_price": your_price,
            "avg_competitor_price": round(avg_competitor_price, 2),
            "your_position": "premium" if your_price > avg_competitor_price * 1.1 else 
                           "budget" if your_price < avg_competitor_price * 0.9 else "mid-market",
            "comparisons": comparisons
        }


# Simulated data fetcher (replace with real API integration)
import sys
import os
import asyncio

# Attempt to find the server module to import DataForSEOClient
# Strategy: Look for 'grok-admaster/server' relative to this script or workspace root
current_dir = os.path.dirname(os.path.abspath(__file__))
# .agent/skills/competitive-intelligence/scripts -> optimus pryme
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
server_path = os.path.join(project_root, "grok-admaster", "server")

if os.path.exists(server_path):
    sys.path.append(server_path)
    try:
        from app.services.dataforseo_client import dfs_client
        from app.core.config import settings
        # Force load credentials if not present, because script might run outside uvicorn context
        from dotenv import load_dotenv
        load_dotenv(os.path.join(server_path, '.env'))
        # Re-init client to pick up env vars
        dfs_client.__init__()
        
        print("Successfully imported DataForSEO Client")
    except ImportError as e:
        print(f"Could not import DataForSEO Client: {e}")
        dfs_client = None
else:
    print(f"Could not locate server path: {server_path}")
    dfs_client = None

class AmazonPriceFetcher:
    """
    Fetches real prices from Amazon using DataForSEO.
    """
    
    async def fetch_price(self, asin: str) -> Optional[Dict]:
        """Fetch real price from Amazon via DataForSEO and persist to DB."""
        if not dfs_client:
             print("Error: DataForSEO client not available.")
             return None

        # DataForSEO search by ASIN (using keyword search which works for ASINs)
        try:
            # We use the raw client, but we will ingest via the ingester service
            products = await dfs_client.get_amazon_products(asin)
            
            if products:
                # Assuming first result is the product if searching by ASIN
                product = products[0]
                
                # PERSISTENCE DATA FLOW:
                # Save this fetch to the database immediately
                try:
                    # Import here to avoid circular imports if any, or just ensuring context
                    from app.services.market_intelligence_ingester import market_ingester
                    
                    # We treat this ASIN lookup as a "competitor" check usually
                    await market_ingester.ingest_amazon_products(
                        keyword=f"ASIN:{asin}", 
                        products=[product], 
                        mark_as_competitors=True
                    )
                    print(f"  [DB] Persisted price point for {asin}")
                except Exception as db_e:
                    print(f"  [DB Warning] Failed to persist price: {db_e}")

                price = product.get("price")
                
                # Check if price is available (DataForSEO returns float or None)
                if price is not None:
                    return {
                        "asin": asin,
                        "price": float(price),
                        "currency": "USD", # Defaulting to USD/location 2840
                        "in_stock": True, # basic assumption if price exists
                        "is_deal": False, # TODO: parsing deal info if available
                        "fetched_at": datetime.now().isoformat(),
                        "title": product.get("title")
                    }
            return None
        except Exception as e:
            print(f"Error fetching price for {asin}: {e}")
            return None
    
    async def fetch_bulk_prices(self, asins: List[str]) -> List[Dict]:
        """Fetch prices for multiple ASINs."""
        # Parallel fetch for speed
        tasks = [self.fetch_price(asin) for asin in asins]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]


if __name__ == "__main__":
    async def main():
        # Demo
        tracker = PriceTracker()
        fetcher = AmazonPriceFetcher()
        
        # Test with real ASINs if possible, or expect failure/empty with placeholders
        # B07W7K98V6: Specific keyboard/mouse example (Simulated or Real)
        asins = ["B07W7K98V6", "B07K1M3TH9"] 
        
        print("Tracking competitor prices...")
        # Since we are fetching live data, we just do one pass usually, 
        # or we can simulate a loop with delay, but APIs cost money.
        # Let's do just ONE pass for the demo.
        
        for asin in asins:
            print(f"Fetching data for {asin}...")
            data = await fetcher.fetch_price(asin)
            if data:
                print(f"  Found: {data['title']} - ${data['price']}")
                change = tracker.track_price(asin, data["price"], "RivalBrand", data["is_deal"])
                if change:
                    print(f"  Price Change: {asin} ${change.old_price} -> ${change.new_price} ({change.change_pct}%)")
            else:
                print(f"  No data found for {asin}")
        
        print("\nPrice War Analysis (Snapshot):")
        war_analysis = tracker.detect_price_war(asins)
        print(f"  Score: {war_analysis['severity_score']}")
        print(f"  Recommendation: {war_analysis['recommendation']}")

    asyncio.run(main())
