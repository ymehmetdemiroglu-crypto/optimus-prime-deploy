"""
Financial Analyst Skill Package
Profitability, budgeting, and unit economics analysis.
"""

from .scripts.profitability_calculator import ProfitabilityCalculator, CostStructure, ProfitabilityMetrics
from .scripts.budget_optimizer import BudgetOptimizer, BudgetAllocation, CampaignPerformance

__all__ = [
    "ProfitabilityCalculator",
    "CostStructure",
    "ProfitabilityMetrics",
    "BudgetOptimizer",
    "BudgetAllocation",
    "CampaignPerformance"
]

__version__ = "1.0.0"
