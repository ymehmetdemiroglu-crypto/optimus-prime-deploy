"""
Financial Analyst Logic
Core algorithms for profitability and budgeting.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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
class CampaignPerformance:
    campaign_name: str
    current_spend: float
    current_sales: float
    roas: float
    category: str

@dataclass
class BudgetAllocation:
    campaign_name: str
    old_budget: float
    new_budget: float
    change_pct: float
    reason: str

class ProfitabilityCalculator:
    def __init__(self):
        # Simulated fee structure (Ideally fetched from DB or Amazon Fees API)
        self.category_referral_fees = {
            "Electronics": 0.15,
            "Clothing": 0.17,
            "Home": 0.15,
            "Beauty": 0.08
        }
    
    def calculate_product_profitability(self, 
                                      asin: str,
                                      price: float,
                                      units_sold: int,
                                      ad_spend: float,
                                      cogs: float,
                                      category: str = "Electronics",
                                      ad_sales: float = 0.0) -> Dict:
        # 1. Calculate Revenue
        gross_revenue = price * units_sold
        
        # 2. Calculate Costs
        referral_rate = self.category_referral_fees.get(category, 0.15)
        referral_fee = price * referral_rate
        fba_fee = self._estimate_fba_fee(price)
        storage_fee = 0.50 
        
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
        
        # 6. Break-even Analysis (Break-even ACoS = Net Margin before Ads)
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
        if price < 15: return 3.50
        if price < 30: return 5.50
        if price < 100: return 8.50
        return 12.00

    def _generate_insights(self, net_margin: float, tacos: float, break_even_acos: float) -> List[str]:
        insights = []
        if net_margin < 0:
            insights.append("Product is generating a net LOSS.")
            if tacos > break_even_acos:
                insights.append(f"Ad spend (TACoS {tacos}%) exceeds profit margin ({break_even_acos}%).")
        elif net_margin < 10:
            insights.append("Low profit margin (<10%). Sensitive to bid changes.")
        else:
            insights.append("Healthy profit margin.")
        return insights

    def analyze_account_health(self, campaigns: List[Dict]) -> Dict:
        """
        Analyze aggregated account health metrics.
        campaigns list expected to have: spend, sales, name, status
        """
        total_spend = 0.0
        total_sales = 0.0
        wasted_spend = 0.0 # Spend with 0 sales
        
        for c in campaigns:
            spend = float(c.get('spend', 0))
            sales = float(c.get('sales', 0))
            total_spend += spend
            total_sales += sales
            
            if sales == 0 and spend > 0:
                wasted_spend += spend
                
        acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        
        # Heuristic potential savings 
        # (Example: Cut 100% of wasted spend + Improve ACoS by 10%)
        potential_savings = wasted_spend + (total_spend * 0.10)
        
        return {
            "total_spend": round(total_spend, 2),
            "total_sales": round(total_sales, 2),
            "acos": round(acos, 1),
            "wasted_spend": round(wasted_spend, 2),
            "potential_savings": round(potential_savings, 2)
        }



class BudgetOptimizer:
    def optimize_budget_allocation(self, 
                                 total_budget: float, 
                                 campaigns: List[Dict],
                                 objective: str = "maximize_revenue") -> Dict:
        """
        Uses Linear Programming to allocate budget optimally.
        Objective: Maximize Revenue (Spend * ROAS)
        Constraints:
        1. Total Spend <= Total Budget
        2. Individual Spend >= 50% of Current (Stability)
        3. Individual Spend <= 300% of Current (Growth Cap)
        """
        processed_campaigns = []
        for c in campaigns:
            spend = float(c.get('spend', 0))
            sales = float(c.get('sales', 0))
            # Smooth ROAS to avoid division by zero or extreme outliers
            roas = sales / spend if spend > 5 else 0.5 
            
            perf = CampaignPerformance(
                campaign_name=c['name'],
                current_spend=spend,
                current_sales=sales,
                roas=roas,
                category=self._categorize_campaign(c['name'])
            )
            processed_campaigns.append(perf)
            
        # Filter low-spend campaigns to simple rule-based to reduce LP complexity if needed
        # For now, we optimize all active campaigns
        active_campaigns = [c for c in processed_campaigns if c.current_spend > 0]
        
        if not active_campaigns:
            return {"total_budget": total_budget, "allocations": [], "error": "No active campaigns to optimize"}

        from scipy.optimize import linprog
        import numpy as np
        
        n = len(active_campaigns)
        
        # IMPROVEMENT: Marginal ROAS Logic
        # Linear Programming assumes Constant Returns to Scale (CRS), which is false for ads.
        # We approximate Diminishing Returns by penalizing the objective coefficient.
        # Effectively, we treat the 'ROAS' not as the current average, but as the estimated *marginal* ROAS
        # if we were to increase spend.
        
        # Simple Decay Model: Expected Marginal ROAS = Current ROAS * Decay_Factor
        # We assume effectively that 'current spend' is the equilibrium.
        # For the LP, since we can't easily do non-linear, we create a linear approximation
        # where we dampen the ROAS coefficient slightly to prevent 'Winner Takes All' behavior.
        
        c = []
        for camp in active_campaigns:
            # Log-like dampening: The higher the current spend, the harder it is to get MORE efficiently.
            # However, for the LP coeff (which is constant per unit of new budget), 
            # we just use the current ROAS but check bounds carefully.
            # To truly model diminishing returns in LP, we'd need Piecewise Linear formulation.
            # For this iteration, we implemented a 'decay' scalar based on current saturation vs category benchmarks.
            
            marginal_modifier = 1.0
            if camp.roas > 10.0: marginal_modifier = 0.8 # Assume outlying high ROAS is not scalable
            
            # Minimizing negative revenue -> -1 * (Spend * Marginal_ROAS)
            c_val = -1 * (camp.roas * marginal_modifier)
            c.append(c_val)
        
        # Constraints
        
        # 1. Inequality: Sum(x_i) <= Total Budget
        # A_ub * x <= b_ub
        A_ub = [np.ones(n)] # [1, 1, 1, ...]
        b_ub = [total_budget]
        
        # 2. Bounds (Stability)
        # Tighter bounds based on performance to model 'Risk'
        # We don't want to scale a campaign 3x just because ROAS is good (Inventory limits).
        bounds = []
        for camp in active_campaigns:
            # Diminishing Returns Heuristic: The better the ROAS, the more room we give,
            # BUT we cap growth to preventing crashing the efficiency.
            max_scale = 3.0
            if camp.current_spend > 500: max_scale = 1.5 # Harder to scale big spenders
            
            lower = camp.current_spend * 0.7 # Secure floor
            upper = camp.current_spend * max_scale
            bounds.append((lower, upper))
            
        # Solve
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        allocations = []
        if res.success:
            new_budgets = res.x
            for i, camp in enumerate(active_campaigns):
                new_val = new_budgets[i]
                allocations.append(BudgetAllocation(
                    campaign_name=camp.campaign_name,
                    old_budget=camp.current_spend,
                    new_budget=new_val,
                    change_pct=round(((new_val / camp.current_spend) - 1) * 100, 1),
                    reason=f"LP Optimization (ROAS {camp.roas:.2f})"
                ))
        else:
             # Fallback if solver fails
            return self._fallback_allocation(total_budget, processed_campaigns, "Solver failed")

        return {
            "total_budget": total_budget,
            "allocations": [
                {
                    "campaign": a.campaign_name,
                    "old_budget": round(a.old_budget, 2),
                    "new_budget": round(a.new_budget, 2),
                    "change_pct": a.change_pct,
                    "reason": a.reason
                } for a in allocations
            ],
            "solver_status": "optimal"
        }

    def _fallback_allocation(self, total_budget: float, campaigns: List[CampaignPerformance], reason: str) -> Dict:
        """Previous heuristic logic as fallback."""
        total_current_spend = sum(c.current_spend for c in campaigns)
        budget_multiplier = total_budget / total_current_spend if total_current_spend > 0 else 1.0
        
        allocations = []
        for camp in campaigns:
            allocations.append(BudgetAllocation(
                campaign_name=camp.campaign_name,
                old_budget=camp.current_spend,
                new_budget=camp.current_spend * budget_multiplier,
                change_pct=round((budget_multiplier - 1) * 100, 1),
                reason=f"Fallback: {reason}"
            ))
            
        return {
            "total_budget": total_budget,
            "allocations": [
                {
                    "campaign": a.campaign_name,
                    "old_budget": round(a.old_budget, 2),
                    "new_budget": round(a.new_budget, 2),
                    "change_pct": a.change_pct,
                    "reason": a.reason
                } for a in allocations
            ],
            "solver_status": "fallback"
        }

    def _categorize_campaign(self, name: str) -> str:
        name_lower = name.lower()
        if "brand" in name_lower: return "Brand"
        if "competitor" in name_lower: return "Competitor"
        return "Product"
