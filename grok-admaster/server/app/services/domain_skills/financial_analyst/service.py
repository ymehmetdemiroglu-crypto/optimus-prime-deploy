"""
Financial Analyst Service
Connects Amazon Ads API data to financial analysis logic.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import logging
from app.modules.amazon_ppc.ingestion.client import AmazonAdsAPIClient
from .logic import ProfitabilityCalculator, BudgetOptimizer

logger = logging.getLogger(__name__)

class FinancialAnalystService:
    def __init__(self, api_client: AmazonAdsAPIClient):
        self.client = api_client
        self.profit_calc = ProfitabilityCalculator()
        self.budget_opt = BudgetOptimizer()

    async def _fetch_report_data(self, profile_id: str, report_type: str, metrics: List[str]) -> List[Dict[str, Any]]:
        """
        Helper to request, wait for, and download a report.
        Real data fetching from Amazon Ads API.
        """
        # 1. Request Report (Yesterday's data for stability)
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        report_id = await self.client.create_report_request(
            profile_id=profile_id,
            report_type=report_type,
            report_date=yesterday,
            metrics=metrics
        )
        logger.info(f"Requested {report_type} report: {report_id}")
        
        # 2. Poll for completion
        status = "PENDING"
        download_url = None
        retries = 0
        while status != "SUCCESS" and retries < 20:
            await asyncio.sleep(2) # Wait 2s between checks
            report_meta = await self.client.get_report(profile_id, report_id)
            status = report_meta.get("status")
            if status == "SUCCESS":
                download_url = report_meta.get("location")
                break
            elif status == "FAILURE":
                raise Exception(f"Report generation failed: {report_meta}")
            retries += 1
            
        if not download_url:
            raise Exception("Report timed out")
            
        # 3. Download Data
        return await self.client.download_report(download_url)

    async def get_product_profitability(self, profile_id: str, asin: str, price: float, cogs: float) -> Dict[str, Any]:
        """
        Calculate profitability for a specific ASIN using real ad data.
        """
        # Fetch Advertised Product Report
        metrics = ["asin", "cost", "attributedSales1d", "unitsSold1d"]
        report_data = await self._fetch_report_data(profile_id, "product", metrics)
        
        # Filter for our ASIN
        asin_data = next((r for r in report_data if r.get("asin") == asin), None)
        
        if not asin_data:
            # No ad spend found, return organic-only view
            return self.profit_calc.calculate_product_profitability(
                asin=asin,
                price=price,
                units_sold=0, # Unknown without Order API
                ad_spend=0.0,
                cogs=cogs
            )
            
        # Extract metrics
        ad_spend = float(asin_data.get("cost", 0))
        ad_sales = float(asin_data.get("attributedSales1d", 0))
        ad_units = int(asin_data.get("unitsSold1d", 0))
        
        # Note: We need Total Units for accurate profitability. 
        # Campaign API only gives Ad Units.
        # Ideally, we would fetch Total Units from SP-API (Selling Partner API).
        # For now, we will create a comprehensive view assuming Ad Units + Estimated Organic (or User Supplied)
        # Here we default to Ad Units only for strict ad-profitability
        
        return self.profit_calc.calculate_product_profitability(
            asin=asin,
            price=price,
            units_sold=ad_units, # Conservative (Ad units only)
            ad_spend=ad_spend,
            cogs=cogs,
            ad_sales=ad_sales
        )

    async def optimize_budget(self, profile_id: str, total_budget: float) -> Dict[str, Any]:
        """
        Optimize budget allocation based on real campaign performance.
        """
        # 1. Fetch Campaign List
        campaigns = await self.client.get_campaigns(profile_id)
        
        # 2. Fetch Performance Data (Campaign Report)
        metrics = ["campaignId", "cost", "attributedSales1d"]
        report_data = await self._fetch_report_data(profile_id, "campaigns", metrics)
        
        # 3. Merge Data
        perf_map = {str(r["campaignId"]): r for r in report_data}
        
        campaign_inputs = []
        for camp in campaigns:
            c_id = str(camp["campaignId"])
            perf = perf_map.get(c_id, {})
            
            campaign_inputs.append({
                "name": camp["name"],
                "spend": float(perf.get("cost", 0)),
                "sales": float(perf.get("attributedSales1d", 0))
            })
            
        return self.budget_opt.optimize_budget_allocation(total_budget, campaign_inputs)

    async def analyze_account(self, profile_id: str) -> Dict[str, Any]:
        """
        Analyze overall account health (Spend, Wasted Spend, ACoS).
        """
        # 1. Fetch Campaign List
        campaigns = await self.client.get_campaigns(profile_id)
        
        # 2. Fetch Performance Data
        metrics = ["campaignId", "cost", "attributedSales1d"]
        report_data = await self._fetch_report_data(profile_id, "campaigns", metrics)
        
        # 3. Merge Data
        perf_map = {str(r["campaignId"]): r for r in report_data}
        
        campaign_inputs = []
        for camp in campaigns:
            c_id = str(camp["campaignId"])
            perf = perf_map.get(c_id, {})
            
            campaign_inputs.append({
                "name": camp["name"],
                "spend": float(perf.get("cost", 0)),
                "sales": float(perf.get("attributedSales1d", 0))
            })
            
        return self.profit_calc.analyze_account_health(campaign_inputs)
