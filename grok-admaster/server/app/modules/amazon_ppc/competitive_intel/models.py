from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, JSON
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from app.core.database import Base

class CompetitorPriceHistory(Base):
    __tablename__ = "competitor_price_history"

    id = Column(Integer, primary_key=True, index=True)
    asin = Column(String(50), nullable=False, index=True)
    competitor_name = Column(String(255))
    price = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    is_promotion = Column(Boolean, default=False)
    captured_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String(50), default="manual")

class PriceChangeEvent(Base):
    __tablename__ = "price_change_events"

    id = Column(Integer, primary_key=True, index=True)
    asin = Column(String(50), nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    change_date = Column(Date, nullable=False)
    old_price = Column(Float)
    new_price = Column(Float)
    change_percent = Column(Float)
    change_type = Column(String(20)) # 'drop', 'hike'
    confidence_score = Column(Float)
    is_acknowledged = Column(Boolean, default=False)

class CompetitorForecast(Base):
    __tablename__ = "competitor_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    asin = Column(String(50), nullable=False)
    forecast_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    predicted_price = Column(Float)
    confidence_interval_low = Column(Float)
    confidence_interval_high = Column(Float)
    model_version = Column(String(50), default="v1.0")

class UndercutProbability(Base):
    __tablename__ = "undercut_probability"

    id = Column(Integer, primary_key=True, index=True)
    asin = Column(String(50), nullable=False)
    prediction_date = Column(Date, nullable=False)
    probability = Column(Float)
    drivers = Column(JSON) # e.g. {"price_gap": 0.4}
    recommended_action = Column(String(50))
    executed_action = Column(String(50))
    outcome_verified = Column(Boolean, default=False)

class StrategicSimulation(Base):
    __tablename__ = "strategic_simulations"

    id = Column(Integer, primary_key=True, index=True)
    simulation_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    scenario_data = Column(JSON)
    payoff_matrix = Column(JSON)
    nash_equilibrium = Column(JSON)
    recommended_strategy = Column(String(100))
    expected_value = Column(Float)

class KeywordCannibalization(Base):
    __tablename__ = "keyword_cannibalization"

    id = Column(Integer, primary_key=True, index=True)
    keyword_text = Column(String(255), nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    cannibalizing_urls = Column(JSON)
    search_volume = Column(Integer)
    ctr_loss_estimate = Column(Float)
    status = Column(String(20), default="detected")
    resolution_action = Column(String(255))
