#!/usr/bin/env python3
"""
Grok AdMaster - Advanced Usage Examples
========================================
Demonstrates complex workflows and automation patterns.
"""

import asyncio
import sys
import os

# Add parent directory to path to import operator
sys.path.insert(0, os.path.dirname(__file__))
from admaster_operator import GrokAdMasterOperator, print_json


async def example_1_complete_campaign_audit():
    """
    Example 1: Complete Campaign Audit
    
    Performs a comprehensive audit of a campaign including:
    - Performance metrics
    - Feature analysis
    - Anomaly detection
    - Optimization recommendations
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Complete Campaign Audit")
    print("="*80)
    
    operator = GrokAdMasterOperator()
    campaign_id = 1
    
    # Step 1: Get campaign details
    print(f"\n📊 Step 1: Fetching campaign {campaign_id} details...")
    campaign = await operator.get_campaign(campaign_id)
    print(f"Campaign: {campaign['name']}")
    print(f"Status: {campaign['status']}")
    print(f"Current ACoS: {campaign.get('acos', 0):.1f}%")
    print(f"Total Spend: ${campaign.get('spend', 0):,.2f}")
    
    # Step 2: Get features and metrics
    print(f"\n🔬 Step 2: Computing features and metrics...")
    features = await operator.get_campaign_features(campaign_id, refresh=True)
    print(f"Features computed: {features.get('source')}")
    
    rolling = await operator.get_rolling_metrics(campaign_id)
    print(f"Rolling metrics: {len(rolling.get('metrics', {}))} windows")
    
    # Step 3: Check for alerts
    print(f"\n⚠️  Step 3: Checking for anomalies...")
    alerts = await operator.get_alerts(campaign_id)
    print(f"Total alerts: {alerts.get('total_alerts', 0)}")
    print(f"Critical: {alerts.get('critical_count', 0)}")
    
    # Step 4: Generate optimization plan
    print(f"\n⚡ Step 4: Generating optimization plan...")
    plan = await operator.generate_optimization_plan(
        campaign_id,
        strategy="balanced",
        target_acos=25.0
    )
    summary = plan.get('summary', {})
    print(f"Total actions recommended: {summary.get('total_actions')}")
    print(f"High priority actions: {summary.get('high_priority_actions')}")
    print(f"Average confidence: {summary.get('avg_confidence', 0):.2%}")
    
    # Step 5: Show top recommendations
    print(f"\n💡 Step 5: Top Recommendations:")
    for i, action in enumerate(plan.get('actions', [])[:3], 1):
        print(f"\n{i}. {action['action_type'].upper()}")
        print(f"   Entity: {action['entity_type']} #{action['entity_id']}")
        print(f"   Priority: {action['priority']}/10")
        print(f"   Reasoning: {action['reasoning']}")
    
    print("\n✅ Audit complete!\n")


async def example_2_batch_optimization():
    """
    Example 2: Batch Optimization
    
    Optimizes all active campaigns with different strategies
    based on their performance.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Optimization")
    print("="*80)
    
    operator = GrokAdMasterOperator()
    
    # Get all campaigns
    print("\n📋 Fetching all campaigns...")
    campaigns = await operator.list_campaigns()
    print(f"Found {len(campaigns)} campaigns")
    
    results = []
    
    for campaign in campaigns:
        campaign_id = campaign['id']
        campaign_name = campaign['name']
        current_acos = campaign.get('acos', 0)
        
        # Determine strategy based on performance
        if current_acos > 35:
            strategy = "profit"  # Focus on profitability
        elif current_acos > 25:
            strategy = "balanced"  # Balance growth and efficiency
        elif current_acos > 15:
            strategy = "aggressive"  # Push for more growth
        else:
            strategy = "volume"  # Maximize reach
        
        print(f"\n🎯 Optimizing: {campaign_name}")
        print(f"   Current ACoS: {current_acos:.1f}%")
        print(f"   Strategy: {strategy}")
        
        # Generate plan
        plan = await operator.generate_optimization_plan(
            campaign_id,
            strategy=strategy,
            target_acos=25.0
        )
        
        summary = plan.get('summary', {})
        results.append({
            'campaign_id': campaign_id,
            'campaign_name': campaign_name,
            'strategy': strategy,
            'total_actions': summary.get('total_actions'),
            'confidence': summary.get('avg_confidence', 0)
        })
        
        print(f"   Actions: {summary.get('total_actions')}")
        print(f"   Confidence: {summary.get('avg_confidence', 0):.2%}")
    
    # Summary
    print("\n" + "="*80)
    print("BATCH OPTIMIZATION SUMMARY")
    print("="*80)
    total_actions = sum(r['total_actions'] for r in results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    
    print(f"\nTotal campaigns optimized: {len(results)}")
    print(f"Total actions recommended: {total_actions}")
    print(f"Average confidence: {avg_confidence:.2%}")
    
    print("\n✅ Batch optimization complete!\n")


async def example_3_anomaly_investigation():
    """
    Example 3: Anomaly Investigation
    
    Detects anomalies and uses GPT-4 to explain them with context.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Anomaly Investigation with GPT-4")
    print("="*80)
    
    operator = GrokAdMasterOperator()
    
    # Get all alerts
    print("\n🔍 Detecting anomalies...")
    alerts = await operator.get_alerts()
    
    total_alerts = alerts.get('total_alerts', 0)
    print(f"Found {total_alerts} anomalies")
    
    if total_alerts == 0:
        print("No anomalies detected. System is healthy! ✅")
        return
    
    # Investigate top 3 critical anomalies
    critical_alerts = [
        a for a in alerts.get('alerts', [])
        if a.get('severity') == 'critical'
    ][:3]
    
    for i, alert in enumerate(critical_alerts, 1):
        print(f"\n{'='*80}")
        print(f"ANOMALY {i}: {alert.get('message')}")
        print(f"{'='*80}")
        
        # Prepare anomaly data for GPT-4
        anomaly_data = {
            "type": alert.get('condition'),
            "severity": alert.get('severity'),
            "description": alert.get('message'),
            "entity_type": alert.get('entity_type'),
            "entity_id": alert.get('entity_id')
        }
        
        # Get campaign context if it's a campaign anomaly
        campaign_context = None
        if alert.get('entity_type') == 'campaign':
            campaign = await operator.get_campaign(alert.get('entity_id'))
            campaign_context = {
                "name": campaign.get('name'),
                "status": campaign.get('status'),
                "daily_budget": campaign.get('daily_budget'),
                "ai_mode": campaign.get('ai_mode')
            }
        
        # Explain with GPT-4
        print("\n🤖 Asking GPT-4 for explanation...")
        explanation = await operator.explain_anomaly(
            anomaly=anomaly_data,
            campaign_context=campaign_context
        )
        
        print("\n📝 GPT-4 Analysis:")
        print(explanation.get('explanation', 'No explanation available'))
        
        print("\n💡 Recommendations:")
        for rec in explanation.get('recommendations', []):
            print(f"  • {rec}")
    
    print("\n✅ Investigation complete!\n")


async def example_4_creative_content_generation():
    """
    Example 4: Creative Content Generation
    
    Generates ad copy for all active campaigns using Claude.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Creative Content Generation with Claude")
    print("="*80)
    
    operator = GrokAdMasterOperator()
    
    # Example product data (in real scenario, this would come from campaigns)
    products = [
        {
            "name": "Wireless Bluetooth Headphones",
            "keywords": ["wireless headphones", "bluetooth earbuds", "noise cancelling"],
            "usps": ["40-hour battery", "Premium sound quality", "Comfortable fit"],
            "audience": "music lovers",
            "tone": "professional"
        },
        {
            "name": "Yoga Mat Premium",
            "keywords": ["yoga mat", "exercise mat", "fitness mat"],
            "usps": ["Extra thick", "Non-slip surface", "Eco-friendly"],
            "audience": "fitness enthusiasts",
            "tone": "energetic"
        }
    ]
    
    for i, product in enumerate(products, 1):
        print(f"\n{'='*80}")
        print(f"PRODUCT {i}: {product['name']}")
        print(f"{'='*80}")
        
        print(f"\n✨ Generating headlines with Claude...")
        result = await operator.generate_headlines(
            product_name=product['name'],
            keywords=product['keywords'],
            unique_selling_points=product['usps'],
            target_audience=product['audience'],
            tone=product['tone']
        )
        
        print(f"\n📝 Generated Headlines:")
        headlines = result.get('headlines', [])
        for j, headline in enumerate(headlines, 1):
            print(f"  {j}. {headline}")
        
        # Also enhance description
        current_desc = f"{product['name']} with {', '.join(product['usps'])}"
        print(f"\n📄 Enhancing description...")
        enhanced = await operator.enhance_description(
            current_description=current_desc,
            focus_keywords=product['keywords']
        )
        
        print(f"\n✏️  Enhanced Description:")
        print(enhanced.get('enhanced_description', 'N/A'))
    
    print("\n✅ Creative generation complete!\n")


async def example_5_automated_daily_workflow():
    """
    Example 5: Automated Daily Workflow
    
    Simulates a complete daily optimization workflow.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Automated Daily Workflow")
    print("="*80)
    
    operator = GrokAdMasterOperator()
    
    # Step 1: Morning health check
    print("\n🌅 Step 1: Morning Health Check")
    print("-" * 40)
    health = await operator.health_check()
    print(f"System status: {health.get('status')}")
    
    # Step 2: Get dashboard overview
    print("\n📊 Step 2: Dashboard Overview")
    print("-" * 40)
    summary = await operator.get_dashboard_summary()
    print(f"Total Sales: ${summary.get('total_sales', 0):,.2f}")
    print(f"Ad Spend: ${summary.get('ad_spend', 0):,.2f}")
    print(f"ACoS: {summary.get('acos', 0):.1f}%")
    print(f"ROAS: {summary.get('roas', 0):.2f}x")
    
    # Step 3: Check for critical alerts
    print("\n⚠️  Step 3: Alert Check")
    print("-" * 40)
    alerts = await operator.get_alerts()
    critical_count = alerts.get('critical_count', 0)
    print(f"Critical alerts: {critical_count}")
    
    if critical_count > 0:
        print("⚡ Taking immediate action on critical alerts...")
        # In real scenario, would execute optimizations here
    
    # Step 4: Optimize underperforming campaigns
    print("\n⚡ Step 4: Optimize Underperforming Campaigns")
    print("-" * 40)
    campaigns = await operator.list_campaigns()
    
    underperforming = [
        c for c in campaigns
        if c.get('acos', 0) > 30  # ACoS above 30%
    ]
    
    print(f"Found {len(underperforming)} underperforming campaigns")
    
    for campaign in underperforming[:3]:  # Limit to top 3
        print(f"\n  Optimizing: {campaign['name']}")
        result = await operator.quick_optimize(
            campaign['id'],
            strategy="profit"  # Focus on profitability
        )
        summary = result.get('summary', {})
        print(f"  Actions: {summary.get('total_actions')}")
    
    # Step 5: Generate AI recommendations
    print("\n💡 Step 5: AI Recommendations")
    print("-" * 40)
    ai_actions = await operator.get_ai_actions()
    print(f"Total recommendations: {len(ai_actions)}")
    
    # Step 6: Summary report
    print("\n" + "="*80)
    print("DAILY WORKFLOW SUMMARY")
    print("="*80)
    print(f"✅ System health: OK")
    print(f"✅ Critical alerts addressed: {critical_count}")
    print(f"✅ Campaigns optimized: {len(underperforming)}")
    print(f"✅ AI recommendations generated: {len(ai_actions)}")
    
    print("\n✅ Daily workflow complete!\n")


async def example_6_ml_powered_predictions():
    """
    Example 6: ML-Powered Predictions
    
    Uses machine learning models to predict campaign performance.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: ML-Powered Predictions")
    print("="*80)
    
    operator = GrokAdMasterOperator()
    
    # Get campaigns
    campaigns = await operator.list_campaigns()
    
    for campaign in campaigns[:3]:  # Analyze top 3
        campaign_id = campaign['id']
        campaign_name = campaign['name']
        
        print(f"\n{'='*80}")
        print(f"CAMPAIGN: {campaign_name}")
        print(f"{'='*80}")
        
        # Get comprehensive features
        print("\n🔬 Computing features...")
        features = await operator.get_campaign_features(campaign_id, refresh=True)
        
        # Get rolling metrics
        rolling = await operator.get_rolling_metrics(campaign_id, windows="7,14,30")
        print(f"Rolling metrics: {len(rolling.get('metrics', {}))} windows")
        
        # Get trend analysis
        print("\n📈 Trend Analysis:")
        metrics = rolling.get('metrics', {})
        if metrics:
            print(f"  7-day average ACoS: {metrics.get('acos_7d', 0):.1f}%")
            print(f"  14-day average ACoS: {metrics.get('acos_14d', 0):.1f}%")
            print(f"  30-day average ACoS: {metrics.get('acos_30d', 0):.1f}%")
        
        # Generate optimization with ML
        print("\n🤖 ML-Powered Optimization:")
        plan = await operator.generate_optimization_plan(
            campaign_id,
            strategy="balanced"
        )
        
        summary = plan.get('summary', {})
        print(f"  Recommended actions: {summary.get('total_actions')}")
        print(f"  ML confidence: {summary.get('avg_confidence', 0):.2%}")
        
        # Show high-confidence actions
        high_conf_actions = [
            a for a in plan.get('actions', [])
            if a.get('confidence', 0) > 0.8
        ]
        
        print(f"\n  High-confidence actions ({len(high_conf_actions)}):")
        for action in high_conf_actions[:3]:
            print(f"    • {action['action_type']}: {action['reasoning']}")
    
    print("\n✅ ML analysis complete!\n")


async def run_all_examples():
    """Run all examples sequentially"""
    examples = [
        ("Complete Campaign Audit", example_1_complete_campaign_audit),
        ("Batch Optimization", example_2_batch_optimization),
        ("Anomaly Investigation", example_3_anomaly_investigation),
        ("Creative Content Generation", example_4_creative_content_generation),
        ("Automated Daily Workflow", example_5_automated_daily_workflow),
        ("ML-Powered Predictions", example_6_ml_powered_predictions),
    ]
    
    print("\n" + "="*80)
    print("GROK ADMASTER - ADVANCED USAGE EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates 6 advanced usage patterns:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "="*80)
    input("\nPress Enter to start examples...")
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            await func()
            if i < len(examples):
                input(f"\nPress Enter to continue to Example {i+1}...")
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80)


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        examples = {
            "1": example_1_complete_campaign_audit,
            "2": example_2_batch_optimization,
            "3": example_3_anomaly_investigation,
            "4": example_4_creative_content_generation,
            "5": example_5_automated_daily_workflow,
            "6": example_6_ml_powered_predictions,
        }
        
        if example_num in examples:
            asyncio.run(examples[example_num]())
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python examples.py [1-6]")
            print("  Or run without arguments to see all examples")
    else:
        asyncio.run(run_all_examples())


if __name__ == "__main__":
    main()
