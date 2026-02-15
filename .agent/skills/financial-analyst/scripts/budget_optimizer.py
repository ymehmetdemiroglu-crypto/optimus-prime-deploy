"""
Budget Optimizer for Optimus Pryme
Allocates budgets across campaigns to maximize ROI using marginal analysis.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import json

@dataclass
class CampaignPerformance:
    campaign_name: str
    current_spend: float
    current_sales: float
    roas: float
    category: str  # Brand, Product, Generic

@dataclass
class BudgetAllocation:
    campaign_name: str
    old_budget: float
    new_budget: float
    change_pct: float
    reason: str

class BudgetOptimizer:
    def __init__(self):
        pass

    def optimize_budget_allocation(self, 
                                 total_budget: float, 
                                 campaigns: List[Dict],
                                 objective: str = "maximize_profit") -> Dict:
        """
        Allocate total budget across campaigns based on performance.
        Objective: 'maximize_profit', 'maximize_revenue', 'efficiency'
        """
        
        # 1. Parse and categorize campaigns
        processed_campaigns = []
        for c in campaigns:
            perf = CampaignPerformance(
                campaign_name=c['name'],
                current_spend=c['spend'],
                current_sales=c['sales'],
                roas=c['sales'] / c['spend'] if c['spend'] > 0 else 0,
                category=self._categorize_campaign(c['name'])
            )
            processed_campaigns.append(perf)
            
        # 2. Sort by efficiency (ROAS)
        # In a real marginal analysis, we'd look at marginal ROAS, not average.
        # Here we use a heuristic: higher ROAS gets more budget.
        processed_campaigns.sort(key=lambda x: x.roas, reverse=True)
        
        # 3. Allocation Algorithm
        allocations = []
        remaining_budget = total_budget
        
        total_current_spend = sum(c.current_spend for c in processed_campaigns)
        budget_multiplier = total_budget / total_current_spend if total_current_spend > 0 else 1.0
        
        for camp in processed_campaigns:
            # Base logic:
            # High ROAS (>4): Increase +20-50%
            # Mid ROAS (2-4): Maintain or slight increase
            # Low ROAS (<2): Decrease -20-50%
            
            # Use current spend as baseline
            baseline = camp.current_spend
            
            if camp.roas > 6.0:
                factor = 1.4  # +40%
                reason = "Elite performance (ROAS > 6). Scale aggressively."
            elif camp.roas > 4.0:
                factor = 1.2  # +20%
                reason = "Strong performance. Scale budget."
            elif camp.roas > 2.5:
                factor = 1.05 # +5%
                reason = "Stable performance. Maintain/Slight increase."
            elif camp.roas > 1.5:
                factor = 0.9  # -10%
                reason = "Mediocre performance. Slight reduction."
            else:
                factor = 0.6  # -40%
                reason = "Poor performance. Significant cut."
            
            # Apply global multiplier constraint (scaling up or down heavily)
            if budget_multiplier > 1.2:
                factor *= 1.1 # Bonus for extra available budget
            elif budget_multiplier < 0.8:
                factor *= 0.9 # Penalty for tight budget
                
            new_budget = baseline * factor
            
            allocations.append(BudgetAllocation(
                campaign_name=camp.campaign_name,
                old_budget=baseline,
                new_budget=new_budget,
                change_pct=round((factor - 1) * 100, 1),
                reason=reason
            ))
            
        # 4. Normalize to fit exact Total Budget
        prelim_total = sum(a.new_budget for a in allocations)
        if prelim_total > 0:
            normalization = total_budget / prelim_total
            for a in allocations:
                a.new_budget *= normalization
        
        return {
            "total_budget_available": total_budget,
            "objective": objective,
            "allocations": [
                {
                    "campaign": a.campaign_name,
                    "old_budget": round(a.old_budget, 2),
                    "new_budget": round(a.new_budget, 2),
                    "change": f"{a.change_pct}%",
                    "reason": a.reason
                }
                for a in allocations
            ],
            "projected_outcome": self._project_outcome(allocations, processed_campaigns)
        }

    def _categorize_campaign(self, name: str) -> str:
        name_lower = name.lower()
        if "brand" in name_lower or "defen" in name_lower: return "Brand"
        if "comp" in name_lower or "conq" in name_lower: return "Competitor"
        return "Generic/Product"

    def _project_outcome(self, allocations: List[BudgetAllocation], history: List[CampaignPerformance]) -> Dict:
        """Roughly project revenue based on new budgets and historical ROAS."""
        projected_rev = 0
        current_rev = 0
        
        hist_map = {c.campaign_name: c for c in history}
        
        for alloc in allocations:
            camp = hist_map.get(alloc.campaign_name)
            if camp:
                current_rev += camp.current_sales
                # Assume diminishing returns: ROAS decays by 5% for every 20% spend increase
                spend_increase_ratio = alloc.new_budget / alloc.old_budget if alloc.old_budget > 0 else 1
                roas_decay = 1.0
                if spend_increase_ratio > 1.2:
                    roas_decay = 0.95
                
                proj_roas = camp.roas * roas_decay
                projected_rev += alloc.new_budget * proj_roas
                
        return {
            "current_revenue": round(current_rev, 2),
            "projected_revenue": round(projected_rev, 2),
            "estimated_growth": round(((projected_rev - current_rev) / current_rev) * 100, 1) if current_rev > 0 else 0
        }

if __name__ == "__main__":
    optimizer = BudgetOptimizer()
    campaigns = [
        {"name": "SP - Brand - Exact", "spend": 1000, "sales": 8500},  # 8.5 ROAS
        {"name": "SP - Product - Generic", "spend": 2500, "sales": 7500}, # 3.0 ROAS
        {"name": "SP - Competitor - Auto", "spend": 1500, "sales": 1800}  # 1.2 ROAS
    ]
    result = optimizer.optimize_budget_allocation(6000, campaigns)
    print(json.dumps(result, indent=2))
