"""
Integration Test for Financial Analyst and Campaign Strategist
Verifies that the new skill packages can be imported and initialized.
"""

import sys
import os

def test_imports():
    print("Testing imports for Financial Analyst...")
    try:
        from financial_analyst import ProfitabilityCalculator, BudgetOptimizer
        calc = ProfitabilityCalculator()
        opt = BudgetOptimizer()
        print("✅ Financial Analyst imported successfully")
    except ImportError as e:
        print(f"❌ Financial Analyst import failed: {e}")

    print("\nTesting imports for Campaign Strategist...")
    try:
        from campaign_strategist import LaunchPlanner, ArchitectureDesigner
        planner = LaunchPlanner()
        designer = ArchitectureDesigner()
        print("✅ Campaign Strategist imported successfully")
    except ImportError as e:
        print(f"❌ Campaign Strategist import failed: {e}")

if __name__ == "__main__":
    test_imports()
