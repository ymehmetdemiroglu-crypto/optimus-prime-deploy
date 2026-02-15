-- Phase 7: Competitive Intelligence Schema
-- Supports Price Monitoring, LSTM Forecasting, Strategic Simulation, and SEO Cannibalization

-- 1. COMPETITOR PRICE HISTORY (Foundation for LSTM/Change Detection)
CREATE TABLE IF NOT EXISTS competitor_price_history (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(50) NOT NULL,
    competitor_name VARCHAR(255),
    price DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    is_promotion BOOLEAN DEFAULT FALSE,
    captured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source VARCHAR(50) DEFAULT 'manual' -- 'api', 'scraper', 'manual'
);

CREATE INDEX idx_comp_price_asin_date ON competitor_price_history(asin, captured_at DESC);

-- 2. PRICE CHANGE EVENTS (Change-Point Detection Results)
CREATE TABLE IF NOT EXISTS price_change_events (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(50) NOT NULL,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    change_date DATE NOT NULL, -- The retrospectively identified date of change
    old_price DECIMAL(10, 2),
    new_price DECIMAL(10, 2),
    change_percent DECIMAL(5, 2),
    change_type VARCHAR(20), -- 'drop', 'hike', 'volatile'
    confidence_score DECIMAL(3, 2), -- 0.0 to 1.0 (RUPTURES confidence)
    is_acknowledged BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_price_change_asin ON price_change_events(asin);

-- 3. PRICE FORECASTS (LSTM Outputs)
CREATE TABLE IF NOT EXISTS competitor_forecasts (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(50) NOT NULL,
    forecast_date DATE NOT NULL, -- Target date for prediction
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    predicted_price DECIMAL(10, 2),
    confidence_interval_low DECIMAL(10, 2),
    confidence_interval_high DECIMAL(10, 2),
    model_version VARCHAR(50) DEFAULT 'v1.0'
);

CREATE INDEX idx_forecast_asin_date ON competitor_forecasts(asin, forecast_date);

-- 4. UNDERCUT PROBABILITY (XGBoost Outputs)
CREATE TABLE IF NOT EXISTS undercut_probability (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(50) NOT NULL,
    prediction_date DATE NOT NULL,
    probability DECIMAL(3, 2), -- 0.00 to 1.00
    drivers JSONB, -- Feature importance: {"price_gap": 0.4, "demand": 0.2}
    recommended_action VARCHAR(50), -- 'hold', 'match', 'differentiate'
    executed_action VARCHAR(50), -- To track what we actually did
    outcome_verified BOOLEAN DEFAULT FALSE -- Later used for retraining
);

-- 5. STRATEGIC SIMULATIONS (Game Theory Results)
CREATE TABLE IF NOT EXISTS strategic_simulations (
    id SERIAL PRIMARY KEY,
    simulation_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    scenario_data JSONB, -- Inputs: { "my_cost": 50, "their_cost_est": 55 }
    payoff_matrix JSONB, -- The generated 2x2 or 3x3 matrix
    nash_equilibrium JSONB, -- The solved equilibrium state
    recommended_strategy VARCHAR(100),
    expected_value DECIMAL(12, 2)
);

-- 6. KEYWORD CANNIBALIZATION (SEO/PPC Conflict)
CREATE TABLE IF NOT EXISTS keyword_cannibalization (
    id SERIAL PRIMARY KEY,
    keyword_text VARCHAR(255) NOT NULL,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    cannibalizing_urls JSONB, -- ["/product-a", "/blog-post-b"]
    search_volume INTEGER,
    ctr_loss_estimate DECIMAL(5, 2), -- Estimated loss due to split ranking
    status VARCHAR(20) DEFAULT 'detected', -- 'detected', 'resolved', 'ignored'
    resolution_action VARCHAR(255) -- 'redirect', 'merge', 'de-optimize'
);

CREATE INDEX idx_cannibalization_status ON keyword_cannibalization(status);
