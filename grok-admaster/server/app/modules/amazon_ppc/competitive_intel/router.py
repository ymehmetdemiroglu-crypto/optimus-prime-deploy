from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from sqlalchemy.orm import Session
from typing import List, Dict
import json
import logging

from app.core.database import get_db
from app.modules.amazon_ppc.competitive_intel.service import CompetitiveIntelligenceService
from app.modules.amazon_ppc.competitive_intel.models import (
    PriceChangeEvent, CompetitorForecast, UndercutProbability, 
    StrategicSimulation, KeywordCannibalization
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/competitive", tags=["Competitive Intelligence"])

def get_service(db: Session = Depends(get_db)):
    return CompetitiveIntelligenceService(db)

# --- Price Monitoring ---
@router.post("/price-monitor/{asin}/scan", response_model=List[dict])
def scan_price_changes(asin: str, service: CompetitiveIntelligenceService = Depends(get_service)):
    """
    Trigger binary segmentation change-point detection on price history.
    """
    try:
        events = service.detect_price_changes_for_asin(asin)
        # Convert ORM to dict manually if needed or use Pydantic response_model properly
        return [{"date": e.change_date, "old": e.old_price, "new": e.new_price, "type": e.change_type} for e in events]
    except Exception as e:
        logger.error(f"Error scanning asin {asin}: {str(e)}")
        raise HTTPException(status_code=500, detail="Scan failed")

@router.get("/price-monitor/{asin}/history")
def get_price_history(asin: str, db: Session = Depends(get_db)):
    """Fetch raw history."""
    pass # Implementation details

# --- Forecasting ---
@router.post("/forecast/{asin}", response_model=dict)
def generate_forecast(asin: str, service: CompetitiveIntelligenceService = Depends(get_service)):
    """
    Generate LSTM forecast for next 7 days.
    """
    forecast = service.generate_price_forecast(asin)
    if not forecast:
        raise HTTPException(status_code=404, detail="Insufficient data for forecast")
    
    return {
        "asin": forecast.asin,
        "predicted_price": forecast.predicted_price,
        "range": [forecast.confidence_interval_low, forecast.confidence_interval_high],
        "date": forecast.forecast_date
    }

# --- Undercut Prediction (XGBoost) ---
@router.post("/undercut-prediction", response_model=dict)
def predict_undercut(
    asin: str = Body(...),
    price_gap: float = Body(...),
    demand_index: int = Body(...),
    service: CompetitiveIntelligenceService = Depends(get_service)
):
    """
    Predict probability of competitor undercutting based on current market state.
    """
    prediction = service.analyze_undercut_probability(asin, price_gap, demand_index)
    return {
        "probability": prediction.probability,
        "action": prediction.recommended_action,
        "drivers": prediction.drivers
    }

# --- Game Theory Simulation ---
@router.post("/simulate-strategy", response_model=dict)
def simulate_strategy(
    my_cost: float = Body(...),
    their_cost: float = Body(...),
    current_price: float = Body(...),
    service: CompetitiveIntelligenceService = Depends(get_service)
):
    """
    Run Nash Equilibrium simulation for pricing strategy.
    """
    result = service.run_strategic_simulation(
        "Standard Simulation", my_cost, their_cost, current_price
    )
    return {
        "matrix": result.payoff_matrix,
        "equilibrium": result.nash_equilibrium,
        "recommendation": result.recommended_strategy,
        "expected_value": result.expected_value
    }

# --- Cannibalization Detection ---
@router.post("/detect-cannibalization", response_model=List[dict])
def upload_gsc_data(
    file: UploadFile = File(...),
    service: CompetitiveIntelligenceService = Depends(get_service)
):
    """
    Upload Google Search Console JSON/CSV to detect cannibalization.
    """
    try:
        content = file.file.read()
        # Assume JSON for now for simplicity
        if file.content_type == "application/json":
            data = json.loads(content)
        else:
            raise HTTPException(status_code=400, detail="Only JSON supported for MVP")
            
        conflicts = service.detect_cannibalization(data)
        
        return [
            {
                "keyword": c.keyword_text,
                "urls": c.cannibalizing_urls,
                "loss": c.ctr_loss_estimate
            }
            for c in conflicts
        ]
    except Exception as e:
        logger.error(f"Cannibalization detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
