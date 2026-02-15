"""
GPT-4 Anomaly Explanation Service
----------------------------------
Uses GPT-4 via OpenRouter to explain the 'why' behind detected anomalies
and provide actionable recommendations.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.services.openrouter_client import call_ai_model, ModelRole

logger = logging.getLogger(__name__)


async def explain_anomaly(
    anomaly: Dict[str, Any],
    keyword_context: Optional[Dict[str, Any]] = None,
    campaign_context: Optional[Dict[str, Any]] = None,
    historical_data: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Use GPT-4 to explain why an anomaly occurred and what to do about it.
    
    Args:
        anomaly: The detected anomaly from the anomaly detection system
            Expected keys: type, keyword, severity, description, metric, expected_value, actual_value
        keyword_context: Additional keyword data (bid history, match type, etc.)
        campaign_context: Campaign-level data (budget, target_acos, strategy)
        historical_data: Recent performance history for trend analysis
    
    Returns:
        Dict with 'explanation', 'root_causes', 'recommendations', and 'risk_level'
    """
    
    # Build the context section
    context_parts = []
    
    if keyword_context:
        context_parts.append(f"""
Keyword Information:
- Keyword: "{keyword_context.get('keyword_text', anomaly.get('keyword', 'Unknown'))}"
- Match Type: {keyword_context.get('match_type', 'Unknown')}
- Current Bid: ${keyword_context.get('bid', 'Unknown')}
- Impressions (7d): {keyword_context.get('impressions', 'Unknown')}
- Clicks (7d): {keyword_context.get('clicks', 'Unknown')}
- CTR: {keyword_context.get('ctr', 'Unknown')}%
- Conversions: {keyword_context.get('orders', 'Unknown')}
- ACoS: {keyword_context.get('acos', 'Unknown')}%""")
    
    if campaign_context:
        context_parts.append(f"""
Campaign Context:
- Campaign: "{campaign_context.get('name', 'Unknown')}"
- Type: {campaign_context.get('campaign_type', 'Unknown')}
- Daily Budget: ${campaign_context.get('daily_budget', 'Unknown')}
- Target ACoS: {campaign_context.get('target_acos', 'Unknown')}%
- AI Strategy: {campaign_context.get('ai_mode', 'manual')}
- Overall Campaign ACoS: {campaign_context.get('current_acos', 'Unknown')}%""")
    
    if historical_data and len(historical_data) > 0:
        # Summarize the trend
        recent = historical_data[-7:] if len(historical_data) >= 7 else historical_data
        context_parts.append(f"""
Recent Trend (Last {len(recent)} days):
- Spend Trend: {_calculate_trend([d.get('spend', 0) for d in recent])}
- Sales Trend: {_calculate_trend([d.get('sales', 0) for d in recent])}
- Impression Trend: {_calculate_trend([d.get('impressions', 0) for d in recent])}""")
    
    context_str = "\n".join(context_parts) if context_parts else "Limited context available."
    
    # Build the main prompt
    prompt = f"""An anomaly has been detected in an Amazon PPC campaign. Analyze and explain:

## Anomaly Details
- Type: {anomaly.get('type', 'Unknown')}
- Severity: {anomaly.get('severity', 'Unknown')}
- Description: {anomaly.get('description', 'Anomaly detected')}
- Metric Affected: {anomaly.get('metric', 'Unknown')}
- Expected Value: {anomaly.get('expected_value', 'N/A')}
- Actual Value: {anomaly.get('actual_value', 'N/A')}
- Deviation: {anomaly.get('deviation_percent', 'N/A')}%

## Context
{context_str}

## Your Analysis Required

Provide a structured analysis with:

1. **ROOT CAUSE ANALYSIS** (2-3 most likely causes)
   Consider: seasonality, competition changes, Amazon algorithm updates, bid changes, budget constraints, market trends, product listing issues

2. **IMMEDIATE ACTIONS** (prioritized list)
   Specific, actionable steps to address the anomaly right now

3. **STRATEGIC RECOMMENDATIONS** (longer-term)
   How to prevent this in the future or leverage it if positive

4. **RISK ASSESSMENT**
   - Urgency level (Critical/High/Medium/Low)
   - Potential revenue impact if not addressed
   - Confidence in your analysis (High/Medium/Low)

Be specific and actionable. Reference the actual numbers when making recommendations."""

    # Call AI Model (Strategist Role)
    result = await call_ai_model(
        prompt=prompt,
        role=ModelRole.STRATEGIST,
        system_prompt="""You are a senior Amazon PPC specialist with 10+ years of experience managing $10M+ in ad spend. 
You excel at diagnosing performance issues and providing clear, actionable recommendations.
Always be specific with numbers and percentages. Never give generic advice.""",
        max_tokens=1500,
        temperature=0.4  # Lower temperature for more focused analysis
    )
    
    if result.get("error"):
        logger.error(f"AI anomaly explanation failed: {result['error']}")
        return {
            "explanation": "Unable to generate AI explanation at this time.",
            "error": result["error"],
            "fallback_recommendation": _get_fallback_recommendation(anomaly)
        }
    
    # Parse the response into structured format
    explanation_text = result["content"]
    
    return {
        "anomaly_id": anomaly.get("id"),
        "anomaly_type": anomaly.get("type"),
        "explanation": explanation_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": result.get("model", "gpt-4-turbo"),
        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
        "keyword": anomaly.get("keyword") or keyword_context.get("keyword_text") if keyword_context else None
    }


async def explain_multiple_anomalies(
    anomalies: List[Dict[str, Any]],
    campaign_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze multiple anomalies together to find patterns and systemic issues.
    More efficient than explaining each one separately.
    """
    if not anomalies:
        return {"summary": "No anomalies to analyze", "explanations": []}
    
    # Group anomalies by type
    anomaly_summary = {}
    for a in anomalies:
        atype = a.get("type", "unknown")
        if atype not in anomaly_summary:
            anomaly_summary[atype] = []
        anomaly_summary[atype].append(a)
    
    # Build summary for GPT-4
    anomaly_list = "\n".join([
        f"- {a.get('keyword', 'Campaign-level')}: {a.get('type')} ({a.get('severity')}) - {a.get('description')}"
        for a in anomalies[:20]  # Limit to 20 for prompt size
    ])
    
    prompt = f"""Multiple anomalies detected in an Amazon PPC campaign. Analyze the patterns:

## Anomalies Detected ({len(anomalies)} total)
{anomaly_list}

## Campaign Context
- Campaign: {campaign_context.get('name', 'Unknown') if campaign_context else 'Unknown'}
- Target ACoS: {campaign_context.get('target_acos', 'N/A') if campaign_context else 'N/A'}%
- Daily Budget: ${campaign_context.get('daily_budget', 'N/A') if campaign_context else 'N/A'}

## Analysis Required

1. **PATTERN IDENTIFICATION**
   What patterns do you see across these anomalies? Are they related?

2. **SYSTEMIC ROOT CAUSE**
   Is there a single underlying issue causing multiple anomalies?

3. **PRIORITY TRIAGE**
   Which anomalies should be addressed first and why?

4. **UNIFIED ACTION PLAN**
   A single cohesive strategy to address multiple issues efficiently

Be concise but specific. Focus on actionable insights."""

    result = await call_ai_model(
        prompt=prompt,
        role=ModelRole.STRATEGIST,
        max_tokens=1200,
        temperature=0.3
    )
    
    if result.get("error"):
        return {
            "summary": "Unable to generate pattern analysis",
            "error": result["error"],
            "individual_anomalies": anomalies
        }
    
    return {
        "summary": result["content"],
        "anomaly_count": len(anomalies),
        "anomaly_types": list(anomaly_summary.keys()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "individual_anomalies": anomalies
    }


def _calculate_trend(values: List[float]) -> str:
    """Calculate simple trend description from a list of values."""
    if len(values) < 2:
        return "Insufficient data"
    
    first_half = sum(values[:len(values)//2]) / max(len(values)//2, 1)
    second_half = sum(values[len(values)//2:]) / max(len(values) - len(values)//2, 1)
    
    if first_half == 0:
        return "New data (no baseline)"
    
    change = ((second_half - first_half) / first_half) * 100
    
    if change > 20:
        return f"Strong upward (+{change:.0f}%)"
    elif change > 5:
        return f"Slight upward (+{change:.0f}%)"
    elif change < -20:
        return f"Strong downward ({change:.0f}%)"
    elif change < -5:
        return f"Slight downward ({change:.0f}%)"
    else:
        return "Stable"


def _get_fallback_recommendation(anomaly: Dict[str, Any]) -> str:
    """Provide a basic fallback recommendation if GPT-4 fails."""
    anomaly_type = anomaly.get("type", "").lower()
    severity = anomaly.get("severity", "medium").lower()
    
    fallbacks = {
        "spend_spike": "Monitor bid levels and set budget caps. Check for new competitor activity.",
        "acos_increase": "Review keyword bids - consider reducing bids on underperforming keywords.",
        "impression_drop": "Check if bids are competitive. Review keyword relevance and quality score.",
        "conversion_drop": "Review product listing, pricing, and reviews. Check stock availability.",
        "ctr_drop": "Update ad copy and headlines. Review search term relevance.",
    }
    
    for key, recommendation in fallbacks.items():
        if key in anomaly_type:
            return recommendation
    
    return "Review campaign settings and recent changes. Consider pausing and investigating if severity is high."
