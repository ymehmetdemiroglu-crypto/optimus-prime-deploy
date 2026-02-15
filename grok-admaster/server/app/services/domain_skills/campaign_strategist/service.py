"""
Campaign Strategist Service
Connects Amazon Ads API data to strategy logic.
"""

from typing import Dict, Any, List
from app.modules.amazon_ppc.ingestion.client import AmazonAdsAPIClient
from .logic import LaunchPlanner, ArchitectureDesigner

class CampaignStrategistService:
    def __init__(self, api_client: AmazonAdsAPIClient):
        self.client = api_client
        self.planner = LaunchPlanner()
        self.designer = ArchitectureDesigner()

    async def generate_launch_plan(self, product_name: str, asin: str, launch_date: str, budget: float, style: str) -> Dict:
        """
        Generate a launch plan. 
        (Future integration: Check inventory levels or current category CPCs via API)
        """
        return self.planner.create_launch_plan(product_name, asin, launch_date, budget, style)

    async def audit_account_structure(self, profile_id: str) -> Dict[str, Any]:
        """
        Audit the current campaign structure using real account data.
        """
        # 1. Fetch all campaigns
        campaigns = await self.client.get_campaigns(profile_id)
        
        # 2. Run Audit Logic
        return self.designer.audit_structure(campaigns)

    async def design_campaigns_for_product(self, product_name: str, asin: str) -> Dict[str, Any]:
        """
        Design ideal campaign structure.
        """
        return self.designer.design_structure(product_name, asin)
