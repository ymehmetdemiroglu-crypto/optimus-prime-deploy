"""
Campaign Strategist Logic
Algorithms for campaign structure and launch planning.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class LaunchPhase:
    name: str
    duration_days: int
    budget_allocation_pct: float
    objectives: List[str]
    tactics: List[str]
    success_metrics: Dict[str, Any]

class LaunchPlanner:
    def create_launch_plan(self, 
                         product_name: str, 
                         asin: str, 
                         launch_date: str, 
                         total_budget: float,
                         aggressiveness: str = "balanced") -> Dict:
        
        if aggressiveness == "blitz":
            budget_split = [0.40, 0.35, 0.25]
            bid_strategy = "Aggressive (+50% over suggested)"
            acos_target = 80.0
            primary_tactic = "Dominance"
        elif aggressiveness == "conservative":
            budget_split = [0.20, 0.30, 0.50]
            bid_strategy = "Conservative (Suggested bid)"
            acos_target = 40.0
            primary_tactic = "Efficiency"
        else:
            budget_split = [0.30, 0.35, 0.35]
            bid_strategy = "Moderate (+20% over suggested)"
            acos_target = 60.0
            primary_tactic = "Growth"

        phases = []
        phases.append(LaunchPhase(
            name="Phase 1: Traction & Data",
            duration_days=14,
            budget_allocation_pct=budget_split[0],
            objectives=["Generate reviews", "Identify keywords"],
            tactics=["SP Auto (High Bid)", "SP Manual Exact", "Product Targeting"],
            success_metrics={"daily_sales": 15, "acos_target": acos_target}
        ))
        
        phases.append(LaunchPhase(
            name="Phase 2: Refinement",
            duration_days=16,
            budget_allocation_pct=budget_split[1],
            objectives=["Improve rank", "Cut wasted spend"],
            tactics=["Harvest search terms", "Negative keywords", "SB Video"],
            success_metrics={"daily_sales": 30, "acos_target": acos_target * 0.8}
        ))
        
        phases.append(LaunchPhase(
            name="Phase 3: Scale",
            duration_days=30,
            budget_allocation_pct=budget_split[2],
            objectives=["Profitability", "Maximize IS"],
            tactics=["Bid opt", "SD Retargeting", "Broad match"],
            success_metrics={"daily_sales": 50, "acos_target": 30.0}
        ))

        detailed_phases = []
        start_dt = datetime.strptime(launch_date, "%Y-%m-%d")
        current_dt = start_dt
        
        for phase in phases:
            phase_budget = total_budget * phase.budget_allocation_pct
            phase_output = {
                "name": phase.name,
                "dates": f"{current_dt.strftime('%Y-%m-%d')} to {(current_dt + timedelta(days=phase.duration_days)).strftime('%Y-%m-%d')}",
                "total_budget": round(phase_budget, 2),
                "tactics": phase.tactics,
                "metrics": phase.success_metrics
            }
            detailed_phases.append(phase_output)
            current_dt += timedelta(days=phase.duration_days)

        return {
            "summary": {
                "product": product_name,
                "strategy": aggressiveness.upper(),
                "total_budget": total_budget
            },
            "phases": detailed_phases
        }

class ArchitectureDesigner:
    def audit_structure(self, campaigns: List[Dict]) -> Dict:
        score = 100
        issues = []
        
        # 1. Naming
        bad_names = [c['name'] for c in campaigns if not self._is_naming_valid(c['name'])]
        if bad_names:
            score -= 15
            issues.append({"type": "Naming", "msg": f"{len(bad_names)} campaigns have unclear names."})
            
        # 2. Types
        types = set(c.get('targetingType', c.get('type', '')) for c in campaigns)
        # Note: Amazon API returns 'targetingType': 'manual'/'auto', needed for heuristic
        
        return {
            "score": score,
            "issues": issues,
            "campaign_count": len(campaigns)
        }

    def design_structure(self, product_name: str, asin: str) -> Dict:
        # (Simplified for brevity as exact logic is in original script)
        return {
            "product": product_name,
            "campaigns": [
                {"name": f"{product_name} - SP - Auto", "type": "SP Auto"},
                {"name": f"{product_name} - SP - Manual Exact", "type": "SP Manual"},
                {"name": f"{product_name} - SB - Video", "type": "SB Video"}
            ]
        }

    def _is_naming_valid(self, name: str) -> bool:
        return "-" in name or "|" in name or "_" in name
