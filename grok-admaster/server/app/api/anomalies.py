"""
Anomaly API Endpoints
---------------------
Exposes GPT-4 powered anomaly explanation to the frontend.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal
import logging

from app.core.config import settings
from app.services.gpt4_anomaly_explainer import explain_anomaly, explain_multiple_anomalies

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/anomalies", tags=["Anomalies"])


class AnomalyFeedItem(BaseModel):
    id: str
    priority: Literal["high", "medium", "low"]
    title: str
    subtitle: str
    metric: str
    metric_value: str
    metric_label: str
    timestamp: str
    ai_recommendation: str
    status: Literal["active", "resolved", "ignored"] = "active"


class AnomalyExplainRequest(BaseModel):
    """Request body for single anomaly explanation."""
    anomaly: Dict[str, Any]  # type, keyword, severity, description, metric, expected_value, actual_value
    keyword_context: Optional[Dict[str, Any]] = None
    campaign_context: Optional[Dict[str, Any]] = None
    historical_data: Optional[List[Dict[str, Any]]] = None


class MultiAnomalyRequest(BaseModel):
    """Request body for multiple anomaly analysis."""
    anomalies: List[Dict[str, Any]]
    campaign_context: Optional[Dict[str, Any]] = None


@router.post("/explain")
async def explain_single_anomaly(request: AnomalyExplainRequest):
    """
    Use GPT-4 to explain a single anomaly and provide actionable recommendations.
    
    Example anomaly:
    ```json
    {
        "anomaly": {
            "type": "acos_spike",
            "keyword": "wireless headphones",
            "severity": "high",
            "description": "ACoS increased by 45% compared to 7-day average",
            "metric": "acos",
            "expected_value": 22.5,
            "actual_value": 32.6,
            "deviation_percent": 45
        },
        "keyword_context": {
            "keyword_text": "wireless headphones",
            "match_type": "exact",
            "bid": 1.50,
            "impressions": 12500,
            "clicks": 450,
            "ctr": 3.6,
            "orders": 18,
            "acos": 32.6
        },
        "campaign_context": {
            "name": "Holiday Electronics",
            "campaign_type": "sponsoredProducts",
            "daily_budget": 150,
            "target_acos": 25,
            "ai_mode": "auto_pilot"
        }
    }
    ```
    """
    try:
        result = await explain_anomaly(
            anomaly=request.anomaly,
            keyword_context=request.keyword_context,
            campaign_context=request.campaign_context,
            historical_data=request.historical_data
        )
        if result.get("error"):
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly explanation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain-batch")
async def explain_batch_anomalies(request: MultiAnomalyRequest):
    """
    Analyze multiple anomalies together to identify patterns and systemic issues.
    More efficient than explaining each one separately.
    """
    try:
        result = await explain_multiple_anomalies(
            anomalies=request.anomalies,
            campaign_context=request.campaign_context
        )
        if result.get("error"):
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch anomaly explanation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-gpt4")
async def test_gpt4_connection():
    """
    Test endpoint to verify GPT-4 via OpenRouter is working.
    Only available in non-production environments.
    """
    if settings.ENV.lower() == "production":
        raise HTTPException(status_code=403, detail="Diagnostic endpoint disabled in production")

    from app.services.openrouter_client import call_gpt4
    
    result = await call_gpt4(
        prompt="Say 'GPT-4 connection successful!' in exactly those words.",
        max_tokens=50,
        temperature=0.0
    )
    
    if result.get("error"):
        return {
            "status": "error",
            "message": result["error"],
            "connected": False
        }
    
    return {
        "status": "success",
        "response": result["content"],
        "model": result.get("model"),
        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
        "connected": True
    }


@router.get("/feed", response_model=List[AnomalyFeedItem])
async def get_anomaly_feed():
    """
    Get the feed of detected anomalies.
    """
    # Mock data based on the template
    return [
        AnomalyFeedItem(
            id="anom_001",
            priority="high",
            title="Sudden Ad Spend Spike",
            subtitle="Campaign 'Summer_Promo_24'",
            metric="+420%",
            metric_value="$1,240.50",
            metric_label="Impact Cost",
            timestamp="14:02 PM",
            ai_recommendation="Pause Campaign immediately"
        ),
        AnomalyFeedItem(
            id="anom_002",
            priority="medium",
            title="Inventory Stockout Risk",
            subtitle="SKU: B08XG9 - Gaming Mouse Pad",
            metric="15%",
            metric_value="4 days",
            metric_label="Days until stockout",
            timestamp="13:45 PM",
            ai_recommendation="Create Shipment Plan"
        ),
        AnomalyFeedItem(
            id="anom_003",
            priority="low",
            title="Listing Content Change",
            subtitle="Main image resolution flagged as low quality",
            metric="Quality",
            metric_value="Low",
            metric_label="Resolution",
            timestamp="11:20 AM",
            ai_recommendation="Update Image"
        ),
        AnomalyFeedItem(
            id="anom_004",
            priority="high",
            title="Buy Box Loss Alert",
            subtitle="ASIN: B09Y22 - Wireless Earbuds",
            metric="-$0.50",
            metric_value="CompetitorXYZ",
            metric_label="Lost To",
            timestamp="10:15 AM",
            ai_recommendation="Match Price"
        )
    ]
