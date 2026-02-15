"""
Profitability Calculator for Optimus Pryme
Calculates true product and account profitability including COGS and fees.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

@dataclass
class CostStructure:
    product_cost_per_unit: float
    amazon_referral_fee: float
    fba_fulfillment_fee: float
    fba_storage_fee_per_unit: float = 0.0
    miscellaneous_cost: float = 0.0
    
    @property
    def total_unit_cost(self) -> float:
        return (self.product_cost_per_unit + 
                self.amazon_referral_fee + 
                self.fba_fulfillment_fee + 
                self.fba_storage_fee_per_unit + 
                self.miscellaneous_cost)

@dataclass
class ProfitabilityMetrics:
    revenue: float
    units_sold: int
    total_costs: float
    gross_profit: float
    gross_margin_pct: float
    net_profit_per_unit: float
    ad_spend: float
    contribution_margin: float
    contribution_margin_pct: float
    tacos: float  # Total ACoS
    roas: float

class ProfitabilityCalculator:
    def __init__(self):
        # Simulated fee structure (In prod: fetch from Amazon Fees API)
        self.category_referral_fees = {
            "Electronics": 0.15,
            "Clothing": 0.17,
            "Home": 0.15,
            "Beauty": 0.08  # for under $10, etc.
        }
    
    def calculate_product_profitability(self, 
                                      asin: str,
                                      price: float,
                                      units_sold: int,
                                      ad_spend: float,
                                      cogs: float,
                                      category: str = "Electronics",
                                      ad_sales: float = 0.0) -> Dict:
        """
        Calculate deep profitability metrics for a single product.
        """
        # 1. Calculate Revenue
        gross_revenue = price * units_sold
        
        # 2. Calculate Costs
        referral_rate = self.category_referral_fees.get(category, 0.15)
        referral_fee = price * referral_rate
        
        # Simplified FBA fee lookup (In prod: use FBA Fee API based on dimensions)
        fba_fee = self._estimate_fba_fee(price)
        
        storage_fee = 0.50 # Estimate
        
        costs = CostStructure(
            product_cost_per_unit=cogs,
            amazon_referral_fee=referral_fee,
            fba_fulfillment_fee=fba_fee,
            fba_storage_fee_per_unit=storage_fee
        )
        
        total_cogs_amount = cogs * units_sold
        total_amazon_fees = (referral_fee + fba_fee + storage_fee) * units_sold
        total_variable_costs = total_cogs_amount + total_amazon_fees
        
        # 3. Calculate Profit
        gross_profit_before_ads = gross_revenue - total_variable_costs
        net_profit = gross_profit_before_ads - ad_spend
        
        # 4. Calculate Margins
        gross_margin_pct = (gross_profit_before_ads / gross_revenue) * 100 if gross_revenue > 0 else 0
        net_margin_pct = (net_profit / gross_revenue) * 100 if gross_revenue > 0 else 0
        
        unit_net_profit = net_profit / units_sold if units_sold > 0 else 0
        
        # 5. Ad Metrics
        tacos = (ad_spend / gross_revenue) * 100 if gross_revenue > 0 else 0
        roas = ad_sales / ad_spend if ad_spend > 0 else 0
        
        # 6. Break-even Analysis
        # Break-even ACoS = Net Margin before Ads
        break_even_acos = gross_margin_pct
        
        return {
            "asin": asin,
            "financials": {
                "gross_revenue": round(gross_revenue, 2),
                "ad_spend": round(ad_spend, 2),
                "total_cogs": round(total_cogs_amount, 2),
                "total_amazon_fees": round(total_amazon_fees, 2),
                "net_profit": round(net_profit, 2)
            },
            "unit_metrics": {
                "selling_price": price,
                "total_unit_cost": round(costs.total_unit_cost, 2),
                "net_profit_per_unit": round(unit_net_profit, 2),
                "break_even_acos": round(break_even_acos, 2)
            },
            "margins": {
                "gross_margin_pct": round(gross_margin_pct, 1),
                "net_margin_pct": round(net_margin_pct, 1),
                "tacos": round(tacos, 1),
                "roas": round(roas, 2)
            },
            "insights": self._generate_insights(net_margin_pct, tacos, break_even_acos)
        }

    def _estimate_fba_fee(self, price: float) -> float:
        """Simple heuristic for FBA fees based on price proxy for size."""
        if price < 15: return 3.50  # Small/Light
        if price < 30: return 5.50  # Standard
        if price < 100: return 8.50 # Large Standard
        return 12.00 # Oversize

    def _generate_insights(self, net_margin: float, tacos: float, break_even_acos: float) -> List[str]:
        insights = []
        
        if net_margin < 0:
            insights.append("Product is generating a net LOSS. Immediate action required.")
            if tacos > break_even_acos:
                insights.append(f"Ad spend (TACoS {tacos}%) exceeds profit margin ({break_even_acos}%). Reduce bids immediately.")
        elif net_margin < 10:
            insights.append("Low profit margin (<10%). Sensitive to bid changes.")
        else:
            insights.append("Healthy profit margin. Potential to scale ad spend.")
            
        if tacos < 10 and net_margin > 20:
            insights.append("High efficiency (Low TACoS). Considerable room to scale ads aggressively.")
            
        return insights

if __name__ == "__main__":
    # Test
    calc = ProfitabilityCalculator()
    report = calc.calculate_product_profitability(
        asin="B0TEST001",
        price=49.99,
        units_sold=100,
        ad_spend=1200.0,
        cogs=12.50,
        category="Electronics",
        ad_sales=5000.0
    )
    print(json.dumps(report, indent=2))
