"""
Launch Planner for Optimus Pryme
Generates strategic product launch plans with phased budgets and tactics.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

@dataclass
class LaunchPhase:
    name: str
    duration_days: int
    budget_allocation_pct: float
    objectives: List[str]
    tactics: List[str]
    success_metrics: Dict[str, Any]

class LaunchPlanner:
    def __init__(self):
        pass

    def create_launch_plan(self, 
                         product_name: str, 
                         asin: str, 
                         launch_date: str, 
                         total_budget: float,
                         aggressiveness: str = "balanced") -> Dict:
        """
        Create a comprehensive 60-day product launch strategy.
        Aggressiveness: 'conservative', 'balanced', 'blitz'
        """
        
        # 1. Define Strategy Parameters based on Aggressiveness
        if aggressiveness == "blitz":
            budget_split = [0.40, 0.35, 0.25] # Front-loaded
            bid_strategy = "Aggressive (+50% over suggested)"
            acos_target = 80.0 # Willing to lose money for rank
            primary_tactic = "Dominance"
        elif aggressiveness == "conservative":
            budget_split = [0.20, 0.30, 0.50] # Back-loaded (test first)
            bid_strategy = "Conservative (Suggested bid)"
            acos_target = 40.0
            primary_tactic = "Efficiency"
        else: # Balanced
            budget_split = [0.30, 0.35, 0.35]
            bid_strategy = "Moderate (+20% over suggested)"
            acos_target = 60.0 # Break-even focus
            primary_tactic = "Growth"

        # 2. Build Phases
        phases = []
        
        # Phase 1: Traction (Days 0-14)
        phases.append(LaunchPhase(
            name="Phase 1: Traction & Data",
            duration_days=14,
            budget_allocation_pct=budget_split[0],
            objectives=[
                "Generate first 10-20 reviews",
                "Indentify converting keywords via Auto campaigns",
                "Establish initial conversion rate history"
            ],
            tactics=[
                "SP Auto Campaign (High Bid, Down-only)",
                "SP Manual Exact (Top 5 highly relevant keywords)",
                "Product Targeting (Competitors with lower ratings)",
                "Early Reviewer Program enrollment"
            ],
            success_metrics={
                "daily_sales": 5 if aggressiveness == 'conservative' else 15,
                "ctr_target": "0.4%",
                "acos_target": acos_target
            }
        ))
        
        # Phase 2: Refinement (Days 15-30)
        phases.append(LaunchPhase(
            name="Phase 2: Refinement & Ranking",
            duration_days=16,
            budget_allocation_pct=budget_split[1],
            objectives=[
                "Improve organic rank for core keywords",
                "Cut wasted spend (negative keywords)",
                "Improve ACoS while maintaining velocity"
            ],
            tactics=[
                "Harvest performing search terms to Manual Exact",
                "Aggressive negative keyword blocking",
                "Launch Sponsored Brands Video",
                "Category targeting defense"
            ],
            success_metrics={
                "daily_sales": 10 if aggressiveness == 'conservative' else 30,
                "organic_rank": "Top 30",
                "acos_target": acos_target * 0.8
            }
        ))
        
        # Phase 3: Profitability (Days 31-60)
        phases.append(LaunchPhase(
            name="Phase 3: Scale & Profit",
            duration_days=30,
            budget_allocation_pct=budget_split[2],
            objectives=[
                "Reach break-even or profitable ACoS",
                "Maximize impression share on winning keywords",
                "Expand to broad match types"
            ],
            tactics=[
                "Bid optimization for ROAS",
                "Launch Sponsored Display retargeting",
                "Expand keywrod coverage",
                "Dayparting implementation"
            ],
            success_metrics={
                "daily_sales": 20 if aggressiveness == 'conservative' else 50,
                "organic_rank": "Page 1",
                "acos_target": 30.0
            }
        ))

        # 3. Calculate Budgets per Phase
        detailed_phases = []
        start_dt = datetime.strptime(launch_date, "%Y-%m-%d")
        current_dt = start_dt
        
        for phase in phases:
            phase_budget = total_budget * phase.budget_allocation_pct
            daily_budget = phase_budget / phase.duration_days
            
            phase_output = {
                "name": phase.name,
                "dates": f"{current_dt.strftime('%Y-%m-%d')} to {(current_dt + timedelta(days=phase.duration_days)).strftime('%Y-%m-%d')}",
                "total_budget": round(phase_budget, 2),
                "daily_budget": round(daily_budget, 2),
                "objectives": phase.objectives,
                "tactics": phase.tactics,
                "metrics": phase.success_metrics
            }
            detailed_phases.append(phase_output)
            current_dt += timedelta(days=phase.duration_days)

        return {
            "launch_summary": {
                "product": product_name,
                "asin": asin,
                "launch_date": launch_date,
                "strategy_type": aggressiveness.upper(),
                "core_objective": primary_tactic,
                "total_budget": total_budget,
                "bid_strategy": bid_strategy
            },
            "checklist": self._get_pre_launch_checklist(),
            "phases": detailed_phases
        }

    def _get_pre_launch_checklist(self) -> List[Dict]:
        return [
            {"task": "Listing Images Optimized (7 slots filled)", "status": "pending"},
            {"task": "A+ Content Published", "status": "pending"},
            {"task": "Backend Search Terms Filled (249 bytes)", "status": "pending"},
            {"task": "Coupons Created (5-10% off badge)", "status": "pending"},
            {"task": "Inventory Received in FBA", "status": "pending"}
        ]

if __name__ == "__main__":
    planner = LaunchPlanner()
    plan = planner.create_launch_plan(
        product_name="Wireless Noise Cancelling Earbuds",
        asin="B0NEWBUD",
        launch_date="2026-03-01",
        total_budget=5000,
        aggressiveness="blitz"
    )
    print(json.dumps(plan, indent=2))
