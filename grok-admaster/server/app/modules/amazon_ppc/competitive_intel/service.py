from sqlalchemy.orm import Session
from sqlalchemy import select, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from app.modules.amazon_ppc.competitive_intel.models import (
    CompetitorPriceHistory, PriceChangeEvent, CompetitorForecast, 
    UndercutProbability, StrategicSimulation, KeywordCannibalization
)
from app.modules.amazon_ppc.competitive_intel.detectors import PriceChangeDetector, CannibalizationDetector
from app.modules.amazon_ppc.competitive_intel.forecasting import Forecaster
from app.modules.amazon_ppc.competitive_intel.strategy import UndercutPredictor, GameTheorySimulator

logger = logging.getLogger(__name__)

class CompetitiveIntelligenceService:
    def __init__(self, db: Session):
        self.db = db
        self.change_detector = PriceChangeDetector()
        self.price_forecaster = Forecaster()
        self.undercut_predictor = UndercutPredictor()
        self.strategy_simulator = GameTheorySimulator()
        self.cannibalization_detector = CannibalizationDetector()

    # --- Price Monitoring ---
    def detect_price_changes_for_asin(self, asin: str) -> List[PriceChangeEvent]:
        """
        Fetch history, run binary segmentation, save events.
        """
        # 1. Fetch history
        history = self.db.query(CompetitorPriceHistory).filter(
            CompetitorPriceHistory.asin == asin
        ).order_by(CompetitorPriceHistory.captured_at).all()
        
        if not history:
            return []

        prices = [h.price for h in history]
        dates = [h.captured_at for h in history]
        
        # 2. Run detection
        detected_events = self.change_detector.detect_changes(prices, dates)
        
        saved_events = []
        for event_data in detected_events:
            # Check if duplicate
            existing = self.db.query(PriceChangeEvent).filter(
                and_(
                    PriceChangeEvent.asin == asin,
                    PriceChangeEvent.change_date == event_data['change_date']
                )
            ).first()
            
            if not existing:
                new_event = PriceChangeEvent(
                    asin=asin,
                    change_date=event_data['change_date'],
                    old_price=event_data['old_price'],
                    new_price=event_data['new_price'],
                    change_percent=event_data['change_percent'],
                    change_type=event_data['change_type'],
                    confidence_score=event_data['confidence']
                )
                self.db.add(new_event)
                saved_events.append(new_event)
        
        self.db.commit()
        return saved_events

    # --- Forecasting ---
    def generate_price_forecast(self, asin: str) -> Optional[CompetitorForecast]:
        """
        Train LSTM on history -> Predict next 7 days.
        """
        history = self.db.query(CompetitorPriceHistory).filter(
            CompetitorPriceHistory.asin == asin
        ).order_by(CompetitorPriceHistory.captured_at).all()
        
        if len(history) < 30:
            logger.warning(f"Not enough data to forecast for {asin}")
            return None
            
        prices = [h.price for h in history]
        
        # Train on fly (MVP approach - ideally pre-trained models)
        # In production, load model from disk
        self.price_forecaster.train(prices, epochs=50) 
        
        forecast_prices = self.price_forecaster.predict(prices)
        avg_next_price = sum(forecast_prices) / len(forecast_prices)
        
        forecast = CompetitorForecast(
            asin=asin,
            forecast_date=datetime.utcnow().date() + timedelta(days=7),
            predicted_price=avg_next_price,
            confidence_interval_low=min(forecast_prices),
            confidence_interval_high=max(forecast_prices),
            model_version="lstm_v1"
        )
        
        self.db.add(forecast)
        self.db.commit()
        return forecast

    # --- Strategy ---
    def analyze_undercut_probability(self, asin: str, 
                                   price_gap: float, 
                                   demand_index: int) -> UndercutProbability:
        """
        Predict probability of undercut using XGBoost logic.
        """
        # Feature vector: [price_gap, days_since_change, is_weekend, inventory, demand]
        # Mocking some features for now
        features = [price_gap, 5, 0, 1, demand_index] # Example
        
        prob = self.undercut_predictor.predict_probability(features)
        
        action = "Monitor"
        if prob > 0.8: action = "Preemptive Strike"
        elif prob > 0.5: action = "Prepare Defense"
        
        prediction = UndercutProbability(
            asin=asin,
            prediction_date=datetime.utcnow().date(),
            probability=prob,
            drivers={"price_gap": price_gap, "demand": demand_index},
            recommended_action=action
        )
        
        self.db.add(prediction)
        self.db.commit()
        return prediction

    def run_strategic_simulation(self, 
                               simulation_name: str,
                               my_cost: float, 
                               their_cost: float, 
                               current_price: float) -> StrategicSimulation:
        """
        Run Game Theory Nash Equilibrium solver.
        """
        result = self.strategy_simulator.solve_nash_equilibrium(
            my_cost, their_cost, current_price
        )
        
        simulation = StrategicSimulation(
            simulation_name=simulation_name,
            scenario_data={"my_cost": my_cost, "their_cost": their_cost},
            payoff_matrix=result["matrix"],
            nash_equilibrium=result["equilibria"],
            recommended_strategy=result["recommendation"],
            expected_value=result["expected_payoff"]
        )
        
        self.db.add(simulation)
        self.db.commit()
        return simulation

    # --- SEO Cannibalization ---
    def detect_cannibalization(self, gsc_data: List[Dict]) -> List[KeywordCannibalization]:
        """
        Process Google Search Console data.
        """
        conflicts = self.cannibalization_detector.detect_conflicts(gsc_data)
        
        saved_conflicts = []
        for c in conflicts:
            # Check existing
            existing = self.db.query(KeywordCannibalization).filter(
                and_(
                    KeywordCannibalization.keyword_text == c['keyword'],
                    KeywordCannibalization.status == 'detected'
                )
            ).first()
            
            if not existing:
                new_c = KeywordCannibalization(
                    keyword_text=c['keyword'],
                    cannibalizing_urls=c['conflicting_urls'],
                    search_volume=c['total_volume'],
                    ctr_loss_estimate=c['ctr_loss_estimate'],
                    status="detected"
                )
                self.db.add(new_c)
                saved_conflicts.append(new_c)
                
        self.db.commit()
        return saved_conflicts
