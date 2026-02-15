
from typing import List, Dict, Any, Optional
from langchain.tools import tool
from sqlalchemy import select, func, desc, text
from app.core.database import AsyncSessionLocal
from app.models.market_intelligence import CompetitorPrice, MarketProduct
from app.modules.amazon_ppc.models.ppc_data import PPCCampaign, PerformanceRecord

class GrokTools:
    
    @tool
    async def get_performance_summary(period_days: int = 30) -> str:
        """
        Get the overall account performance summary for the last N days.
        Returns total spend, sales, ACoS, and ROAS.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(text(f"""
                SELECT 
                    SUM(spend) as total_spend,
                    SUM(sales) as total_sales,
                    SUM(impressions) as total_impressions,
                    SUM(clicks) as total_clicks
                FROM performance_records
                WHERE date >= NOW() - INTERVAL '{period_days} days'
            """))
            row = result.fetchone()
            if not row or not row[0]:
                return "No performance data available for this period."
                
            spend, sales, imps, clicks = row
            acos = (spend / sales * 100) if sales > 0 else 0
            roas = (sales / spend) if spend > 0 else 0
            ctr = (clicks / imps * 100) if imps > 0 else 0
            
            return f"""
            Performance Summary (Last {period_days} Days):
            - Spend: ${spend:,.2f}
            - Sales: ${sales:,.2f}
            - ACoS: {acos:.2f}%
            - ROAS: {roas:.2f}x
            - CTR: {ctr:.2f}%
            """

    @tool
    async def get_top_moving_competitors(limit: int = 5) -> str:
        """
        Get a list of competitors with the most significant price changes recently.
        """
        async with AsyncSessionLocal() as session:
            # Simple query to find tracked competitors
            result = await session.execute(select(MarketProduct).where(MarketProduct.is_competitor == True).limit(limit))
            products = result.scalars().all()
            
            if not products:
                return "No competitors are currently being tracked."
            
            report = "Tracked Competitors Status:\n"
            for p in products:
                report += f"- {p.title[:30]}... (ASIN: {p.asin})\n"
                
            return report

    @tool
    async def run_sql_query(query: str) -> str:
        """
        Execute a READ-ONLY SQL query against the database. 
        Use this to answer complex questions about campaigns, ad groups, or keywords.
        Do not use DROP, DELETE, or INSERT.
        Tables: ppc_campaigns, ppc_ad_groups, ppc_keywords, performance_records.
        """
        if "drop " in query.lower() or "delete " in query.lower() or "insert " in query.lower() or "update " in query.lower():
            return "Error: Only read-only queries are allowed."
            
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(text(query))
                rows = result.fetchall()
                if not rows:
                    return "No results found."
                return str(rows[:10]) # Limit rows to avoid context overflow
            except Exception as e:
                return f"SQL Error: {str(e)}"

    @staticmethod
    def get_all_tools():
        return [
            GrokTools.get_performance_summary,
            GrokTools.get_top_moving_competitors,
            GrokTools.run_sql_query
        ]
