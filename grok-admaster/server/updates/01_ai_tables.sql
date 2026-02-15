-- =============================================
-- AI MODEL PERSISTENCE TABLES
-- Migration: 01_ai_tables.sql
-- Purpose: Support for Advanced ML Models (Ensemble, Bandits, LSTM, Bayesian)
-- =============================================

-- Model Registry: Track active model versions and their metadata
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL UNIQUE, -- 'ensemble', 'bandit', 'lstm_forecaster', 'bayesian_budget'
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'bidding', 'forecasting', 'budget', 'exploration'
    is_active BOOLEAN DEFAULT TRUE,
    performance_metric NUMERIC(10, 4), -- e.g., average confidence or accuracy
    last_trained_at TIMESTAMP WITH TIME ZONE,
    config JSONB, -- Store hyperparameters, weights, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bandit Arms: Store Thompson Sampling & UCB state for Multi-Armed Bandits
CREATE TABLE IF NOT EXISTS bandit_arms (
    id SERIAL PRIMARY KEY,
    keyword_id INTEGER REFERENCES keywords(id) ON DELETE CASCADE, -- Nullable for global bandits
    arm_id INTEGER NOT NULL, -- Index of the arm (e.g., 0-10 for different bid multipliers)
    multiplier NUMERIC(5, 3) NOT NULL, -- e.g., 0.500, 1.200, 1.500
    
    -- Thompson Sampling (Beta Distribution)
    alpha NUMERIC(10, 4) DEFAULT 1.0,
    beta NUMERIC(10, 4) DEFAULT 1.0,
    
    -- UCB Statistics
    pulls INTEGER DEFAULT 0,
    total_reward NUMERIC(15, 4) DEFAULT 0,
    sum_squared_reward NUMERIC(20, 4) DEFAULT 0,
    
    -- Context (for Contextual Bandits)
    context_vector JSONB, -- Store feature values if using LinUCB
    
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(keyword_id, arm_id) -- Prevent duplicate arms per keyword
);

-- Prediction Logs: Record every AI decision for debugging and future training
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    keyword_id INTEGER REFERENCES keywords(id) ON DELETE CASCADE,
    campaign_id INTEGER REFERENCES campaigns(id) ON DELETE CASCADE,
    
    -- Input Features (snapshot of state at prediction time)
    input_features JSONB NOT NULL,
    
    -- Prediction Output
    predicted_bid NUMERIC(10, 2),
    confidence_score NUMERIC(4, 3), -- 0.000 - 1.000
    model_predictions JSONB, -- Individual predictions from ensemble members
    
    -- Reasoning
    reasoning_text TEXT,
    model_version VARCHAR(50),
    
    -- Execution Context
    was_executed BOOLEAN DEFAULT FALSE,
    actual_outcome JSONB, -- Store {acos, roas, sales} after execution
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Training Jobs: Track background model training status
CREATE TABLE IF NOT EXISTS training_jobs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    job_type VARCHAR(50) NOT NULL, -- 'initial', 'incremental', 'retrain'
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    
    -- Data Range
    training_start_date DATE,
    training_end_date DATE,
    samples_count INTEGER,
    
    -- Results
    final_loss NUMERIC(15, 6),
    epochs_completed INTEGER,
    performance_metrics JSONB, -- {accuracy, precision, etc.}
    
    -- Execution
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add index for fast lookups
CREATE INDEX idx_prediction_logs_keyword ON prediction_logs(keyword_id, created_at DESC);
CREATE INDEX idx_prediction_logs_campaign ON prediction_logs(campaign_id, created_at DESC);
CREATE INDEX idx_bandit_arms_keyword ON bandit_arms(keyword_id);
CREATE INDEX idx_training_jobs_status ON training_jobs(status, created_at DESC);

-- Insert default model registry entries
INSERT INTO model_registry (model_name, model_version, model_type, is_active, config)
VALUES 
    ('ensemble', 'v1.0', 'bidding', TRUE, '{"weights": {"gradient_boost": 0.30, "deep_nn": 0.25, "rl_agent": 0.25, "bandit": 0.20}}'),
    ('bandit_optimizer', 'v1.0', 'exploration', TRUE, '{"arms": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}'),
    ('lstm_forecaster', 'v1.0', 'forecasting', TRUE, '{"sequence_length": 14, "hidden_size": 32}'),
    ('bayesian_budget', 'v1.0', 'budget', TRUE, '{"kernel": "rbf", "acquisition": "expected_improvement"}')
ON CONFLICT (model_name) DO NOTHING;

-- Verification: Show created tables
SELECT 'AI Tables Created Successfully!' AS status;
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('model_registry', 'bandit_arms', 'prediction_logs', 'training_jobs');
