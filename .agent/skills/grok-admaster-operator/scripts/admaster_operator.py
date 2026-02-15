#!/usr/bin/env python3
"""
Grok AdMaster Operator - Helper Script
=======================================
Provides command-line interface for common Grok AdMaster operations.

Usage:
    python operator.py <command> [options]

Commands:
    dashboard              Get dashboard summary
    campaigns              List all campaigns
    campaign <id>          Get specific campaign details
    optimize <id>          Generate optimization plan for campaign
    execute <id>           Execute optimization (dry-run by default)
    anomalies              Get all anomalies
    explain-anomaly        Explain anomaly with GPT-4
    generate-headlines     Generate ad headlines with Claude
    features <id>          Get campaign features
    schedule               Create optimization schedule
    sync                   Sync all accounts
    health                 Check system health
"""

import asyncio
import httpx
import json
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
TIMEOUT = 30.0


class GrokAdMasterClient:
    """Client for interacting with Grok AdMaster API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def get(self, endpoint: str) -> Dict[str, Any]:
        """GET request"""
        response = await self.client.get(f"{self.base_url}{endpoint}")
        response.raise_for_status()
        return response.json()
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST request"""
        response = await self.client.post(
            f"{self.base_url}{endpoint}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def patch(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """PATCH request"""
        response = await self.client.patch(
            f"{self.base_url}{endpoint}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """DELETE request"""
        response = await self.client.delete(f"{self.base_url}{endpoint}")
        response.raise_for_status()
        return response.json()


class GrokAdMasterOperator:
    """High-level operations for Grok AdMaster"""
    
    def __init__(self):
        self.client = GrokAdMasterClient()
    
    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard KPI summary"""
        async with self.client:
            return await self.client.get("/dashboard/summary")
    
    async def get_chart_data(self, time_range: str = "7d") -> List[Dict[str, Any]]:
        """Get time-series chart data"""
        async with self.client:
            return await self.client.get(f"/dashboard/chart-data?range={time_range}")
    
    async def get_ai_actions(self) -> List[Dict[str, Any]]:
        """Get AI-driven action recommendations"""
        async with self.client:
            return await self.client.get("/dashboard/ai-actions")
    
    async def list_campaigns(self) -> List[Dict[str, Any]]:
        """List all campaigns"""
        async with self.client:
            return await self.client.get("/campaigns")
    
    async def get_campaign(self, campaign_id: int) -> Dict[str, Any]:
        """Get specific campaign details"""
        async with self.client:
            return await self.client.get(f"/campaigns/{campaign_id}")
    
    async def update_campaign_strategy(
        self, 
        campaign_id: int, 
        ai_mode: str
    ) -> Dict[str, Any]:
        """Update campaign AI strategy"""
        async with self.client:
            return await self.client.patch(
                f"/campaigns/{campaign_id}/strategy",
                {"ai_mode": ai_mode}
            )
    
    async def generate_optimization_plan(
        self,
        campaign_id: int,
        strategy: str = "balanced",
        target_acos: float = 25.0
    ) -> Dict[str, Any]:
        """Generate optimization plan"""
        async with self.client:
            return await self.client.post(
                "/optimization/generate-plan",
                {
                    "campaign_id": campaign_id,
                    "strategy": strategy,
                    "target_acos": target_acos
                }
            )
    
    async def execute_optimization(
        self,
        campaign_id: int,
        strategy: str = "balanced",
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Execute optimization plan"""
        async with self.client:
            return await self.client.post(
                "/optimization/execute",
                {
                    "campaign_id": campaign_id,
                    "strategy": strategy,
                    "dry_run": dry_run
                }
            )
    
    async def quick_optimize(
        self,
        campaign_id: int,
        strategy: str = "balanced"
    ) -> Dict[str, Any]:
        """Quick optimize (generate + simulate)"""
        async with self.client:
            return await self.client.post(
                f"/optimization/quick-optimize/{campaign_id}?strategy={strategy}",
                {}
            )
    
    async def get_alerts(self, campaign_id: Optional[int] = None) -> Dict[str, Any]:
        """Get alerts (all or for specific campaign)"""
        async with self.client:
            if campaign_id:
                return await self.client.get(f"/optimization/alerts/{campaign_id}")
            return await self.client.get("/optimization/alerts")
    
    async def create_schedule(
        self,
        account_id: int,
        strategy: str = "balanced",
        frequency: str = "daily",
        auto_execute: bool = False,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """Create optimization schedule"""
        async with self.client:
            return await self.client.post(
                "/optimization/schedule",
                {
                    "account_id": account_id,
                    "strategy": strategy,
                    "frequency": frequency,
                    "auto_execute": auto_execute,
                    "min_confidence": min_confidence
                }
            )
    
    async def explain_anomaly(
        self,
        anomaly: Dict[str, Any],
        keyword_context: Optional[Dict[str, Any]] = None,
        campaign_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Explain anomaly using GPT-4"""
        async with self.client:
            return await self.client.post(
                "/anomalies/explain",
                {
                    "anomaly": anomaly,
                    "keyword_context": keyword_context,
                    "campaign_context": campaign_context
                }
            )
    
    async def generate_headlines(
        self,
        product_name: str,
        keywords: List[str],
        unique_selling_points: List[str],
        target_audience: str = "general",
        tone: str = "persuasive"
    ) -> Dict[str, Any]:
        """Generate ad headlines using Claude"""
        async with self.client:
            return await self.client.post(
                "/creative/headlines",
                {
                    "product_name": product_name,
                    "keywords": keywords,
                    "unique_selling_points": unique_selling_points,
                    "target_audience": target_audience,
                    "tone": tone
                }
            )
    
    async def enhance_description(
        self,
        current_description: str,
        focus_keywords: List[str]
    ) -> Dict[str, Any]:
        """Enhance product description using Claude"""
        async with self.client:
            return await self.client.post(
                "/creative/description",
                {
                    "current_description": current_description,
                    "focus_keywords": focus_keywords
                }
            )
    
    async def get_campaign_features(
        self,
        campaign_id: int,
        refresh: bool = False
    ) -> Dict[str, Any]:
        """Get campaign features"""
        async with self.client:
            return await self.client.get(
                f"/features/campaign/{campaign_id}?refresh={refresh}"
            )
    
    async def get_rolling_metrics(
        self,
        campaign_id: int,
        windows: str = "7,14,30"
    ) -> Dict[str, Any]:
        """Get rolling metrics"""
        async with self.client:
            return await self.client.get(
                f"/features/campaign/{campaign_id}/rolling?windows={windows}"
            )
    
    async def get_keyword_features(self, keyword_id: int) -> Dict[str, Any]:
        """Get keyword features"""
        async with self.client:
            return await self.client.get(f"/features/keyword/{keyword_id}")
    
    async def get_bid_recommendations(
        self,
        keyword_id: int,
        target_acos: float = 25.0,
        target_roas: float = 4.0
    ) -> Dict[str, Any]:
        """Get bid recommendations for keyword"""
        async with self.client:
            return await self.client.get(
                f"/features/keyword/{keyword_id}/bid-recommendations"
                f"?target_acos={target_acos}&target_roas={target_roas}"
            )
    
    async def sync_all_accounts(self) -> Dict[str, Any]:
        """Sync all accounts from Amazon Ads API"""
        async with self.client:
            return await self.client.post("/ingestion/sync-all", {})
    
    async def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        async with self.client:
            response = await self.client.client.get("http://localhost:8000/health")
            response.raise_for_status()
            return response.json()


# CLI Commands

def print_json(data: Any):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2, default=str))


async def cmd_dashboard():
    """Dashboard summary command"""
    operator = GrokAdMasterOperator()
    summary = await operator.get_dashboard_summary()
    print("\n📊 Dashboard Summary")
    print("=" * 50)
    print(f"Total Sales:  ${summary.get('total_sales', 0):,.2f}")
    print(f"Ad Spend:     ${summary.get('ad_spend', 0):,.2f}")
    print(f"ACoS:         {summary.get('acos', 0):.1f}%")
    print(f"ROAS:         {summary.get('roas', 0):.2f}x")
    print(f"Trend:        {summary.get('velocity_trend', 'N/A')}")
    print()


async def cmd_campaigns():
    """List campaigns command"""
    operator = GrokAdMasterOperator()
    campaigns = await operator.list_campaigns()
    print(f"\n📋 Campaigns ({len(campaigns)} total)")
    print("=" * 80)
    for c in campaigns:
        print(f"ID: {c['id']:3d} | {c['name']:30s} | "
              f"Status: {c['status']:10s} | "
              f"ACoS: {c.get('acos', 0):5.1f}% | "
              f"Spend: ${c.get('spend', 0):8,.2f}")
    print()


async def cmd_campaign(campaign_id: int):
    """Get campaign details command"""
    operator = GrokAdMasterOperator()
    campaign = await operator.get_campaign(campaign_id)
    print(f"\n🎯 Campaign Details: {campaign['name']}")
    print("=" * 50)
    print_json(campaign)
    print()


async def cmd_optimize(campaign_id: int, strategy: str = "balanced"):
    """Generate optimization plan command"""
    operator = GrokAdMasterOperator()
    plan = await operator.generate_optimization_plan(campaign_id, strategy)
    print(f"\n⚡ Optimization Plan for Campaign {campaign_id}")
    print("=" * 50)
    summary = plan.get('summary', {})
    print(f"Strategy:           {plan.get('strategy')}")
    print(f"Target ACoS:        {plan.get('target_acos')}%")
    print(f"Total Actions:      {summary.get('total_actions')}")
    print(f"Bid Increases:      {summary.get('bid_increases')}")
    print(f"Bid Decreases:      {summary.get('bid_decreases')}")
    print(f"Keywords to Pause:  {summary.get('keywords_to_pause')}")
    print(f"Avg Confidence:     {summary.get('avg_confidence', 0):.2f}")
    print(f"\nTop Actions:")
    for i, action in enumerate(plan.get('actions', [])[:5], 1):
        print(f"\n{i}. {action['action_type'].upper()}")
        print(f"   Priority: {action['priority']}/10")
        print(f"   Confidence: {action['confidence']:.2f}")
        print(f"   Reasoning: {action['reasoning']}")
    print()


async def cmd_execute(campaign_id: int, strategy: str = "balanced", live: bool = False):
    """Execute optimization command"""
    operator = GrokAdMasterOperator()
    dry_run = not live
    result = await operator.execute_optimization(campaign_id, strategy, dry_run)
    mode = "LIVE" if live else "DRY-RUN"
    print(f"\n🚀 Optimization Execution ({mode})")
    print("=" * 50)
    print_json(result)
    print()


async def cmd_anomalies():
    """Get anomalies command"""
    operator = GrokAdMasterOperator()
    alerts = await operator.get_alerts()
    print(f"\n⚠️  Anomalies & Alerts")
    print("=" * 50)
    print(f"Total Alerts:    {alerts.get('total_alerts', 0)}")
    print(f"Critical:        {alerts.get('critical_count', 0)}")
    print(f"Warnings:        {alerts.get('warning_count', 0)}")
    print("\nRecent Alerts:")
    for alert in alerts.get('alerts', [])[:10]:
        print(f"\n[{alert['severity'].upper()}] {alert['message']}")
        print(f"  Entity: {alert['entity_type']} #{alert['entity_id']}")
        print(f"  Action: {alert['recommended_action']}")
    print()


async def cmd_features(campaign_id: int):
    """Get campaign features command"""
    operator = GrokAdMasterOperator()
    features = await operator.get_campaign_features(campaign_id)
    print(f"\n🔬 Campaign Features: {campaign_id}")
    print("=" * 50)
    print_json(features)
    print()


async def cmd_headlines(
    product_name: str,
    keywords: str,
    usps: str
):
    """Generate headlines command"""
    operator = GrokAdMasterOperator()
    keywords_list = [k.strip() for k in keywords.split(',')]
    usps_list = [u.strip() for u in usps.split(',')]
    
    result = await operator.generate_headlines(
        product_name=product_name,
        keywords=keywords_list,
        unique_selling_points=usps_list
    )
    
    print(f"\n✨ Generated Headlines for: {product_name}")
    print("=" * 50)
    print_json(result)
    print()


async def cmd_schedule(
    account_id: int,
    strategy: str = "balanced",
    frequency: str = "daily"
):
    """Create schedule command"""
    operator = GrokAdMasterOperator()
    result = await operator.create_schedule(account_id, strategy, frequency)
    print(f"\n⏰ Optimization Schedule Created")
    print("=" * 50)
    print_json(result)
    print()


async def cmd_sync():
    """Sync accounts command"""
    operator = GrokAdMasterOperator()
    print("\n🔄 Syncing all accounts...")
    result = await operator.sync_all_accounts()
    print("=" * 50)
    print_json(result)
    print()


async def cmd_health():
    """Health check command"""
    operator = GrokAdMasterOperator()
    health = await operator.health_check()
    print("\n💚 System Health")
    print("=" * 50)
    print_json(health)
    print()


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "dashboard":
            asyncio.run(cmd_dashboard())
        
        elif command == "campaigns":
            asyncio.run(cmd_campaigns())
        
        elif command == "campaign":
            if len(sys.argv) < 3:
                print("Usage: operator.py campaign <id>")
                sys.exit(1)
            campaign_id = int(sys.argv[2])
            asyncio.run(cmd_campaign(campaign_id))
        
        elif command == "optimize":
            if len(sys.argv) < 3:
                print("Usage: operator.py optimize <id> [strategy]")
                sys.exit(1)
            campaign_id = int(sys.argv[2])
            strategy = sys.argv[3] if len(sys.argv) > 3 else "balanced"
            asyncio.run(cmd_optimize(campaign_id, strategy))
        
        elif command == "execute":
            if len(sys.argv) < 3:
                print("Usage: operator.py execute <id> [strategy] [--live]")
                sys.exit(1)
            campaign_id = int(sys.argv[2])
            strategy = "balanced"
            live = False
            if len(sys.argv) > 3:
                if sys.argv[3] == "--live":
                    live = True
                else:
                    strategy = sys.argv[3]
            if len(sys.argv) > 4 and sys.argv[4] == "--live":
                live = True
            asyncio.run(cmd_execute(campaign_id, strategy, live))
        
        elif command == "anomalies":
            asyncio.run(cmd_anomalies())
        
        elif command == "features":
            if len(sys.argv) < 3:
                print("Usage: operator.py features <campaign_id>")
                sys.exit(1)
            campaign_id = int(sys.argv[2])
            asyncio.run(cmd_features(campaign_id))
        
        elif command == "headlines":
            if len(sys.argv) < 5:
                print('Usage: operator.py headlines "<product>" "<keywords>" "<usps>"')
                print('Example: operator.py headlines "Wireless Headphones" "bluetooth,wireless" "40hr battery,premium sound"')
                sys.exit(1)
            asyncio.run(cmd_headlines(sys.argv[2], sys.argv[3], sys.argv[4]))
        
        elif command == "schedule":
            if len(sys.argv) < 3:
                print("Usage: operator.py schedule <account_id> [strategy] [frequency]")
                sys.exit(1)
            account_id = int(sys.argv[2])
            strategy = sys.argv[3] if len(sys.argv) > 3 else "balanced"
            frequency = sys.argv[4] if len(sys.argv) > 4 else "daily"
            asyncio.run(cmd_schedule(account_id, strategy, frequency))
        
        elif command == "sync":
            asyncio.run(cmd_sync())
        
        elif command == "health":
            asyncio.run(cmd_health())
        
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
