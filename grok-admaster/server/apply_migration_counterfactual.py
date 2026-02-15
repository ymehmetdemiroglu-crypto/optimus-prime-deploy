"""
Database Migration: Add Off-Policy Evaluation Tables

Creates tables required for counterfactual learning and off-policy evaluation:
1. prediction_logs - Logged decisions with propensities
2. backtest_results - Policy evaluation results
3. policy_deployments - Deployment tracking and rollback

Run with:
    python apply_migration_counterfactual.py
"""
from __future__ import annotations

import asyncio
import asyncpg
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database URL from environment
import os
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("SUPABASE_DB_URL")

MIGRATION_SQL = """
-- ═══════════════════════════════════════════════════════════════════════
-- 1. Prediction Logs Table (Logged Decisions)
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS prediction_logs (
    log_id BIGSERIAL PRIMARY KEY,
    keyword_id INTEGER NOT NULL REFERENCES ppc_keywords(id) ON DELETE CASCADE,
    campaign_id INTEGER NOT NULL REFERENCES ppc_campaigns(id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL,
    
    -- Input features at decision time
    input_features JSONB,
    
    -- Model prediction
    predicted_bid DECIMAL(10, 4) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    
    -- Decision metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    was_executed BOOLEAN DEFAULT FALSE,
    execution_timestamp TIMESTAMPTZ,
    
    -- Outcome (populated after attribution window, e.g., 7 days)
    actual_spend DECIMAL(10, 2),
    actual_sales DECIMAL(10, 2),
    actual_clicks INTEGER,
    actual_impressions INTEGER,
    actual_orders INTEGER,
    outcome_recorded_at TIMESTAMPTZ,
    attribution_window_days INTEGER DEFAULT 7
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_prediction_logs_keyword_created 
    ON prediction_logs(keyword_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_campaign_created 
    ON prediction_logs(campaign_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_profile_created 
    ON prediction_logs(profile_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_executed 
    ON prediction_logs(was_executed, created_at DESC) 
    WHERE was_executed = TRUE;

CREATE INDEX IF NOT EXISTS idx_prediction_logs_outcome 
    ON prediction_logs(outcome_recorded_at) 
    WHERE outcome_recorded_at IS NOT NULL;

-- Partial index for unprocessed outcomes (needs attribution)
CREATE INDEX IF NOT EXISTS idx_prediction_logs_pending_outcome 
    ON prediction_logs(created_at, was_executed) 
    WHERE was_executed = TRUE AND outcome_recorded_at IS NULL;

-- ═══════════════════════════════════════════════════════════════════════
-- 2. Backtest Results Table (Policy Evaluations)
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS backtest_results (
    id BIGSERIAL PRIMARY KEY,
    profile_id TEXT NOT NULL,
    
    -- Policy identification
    strategy_name VARCHAR(200) NOT NULL,
    policy_description TEXT,
    
    -- Evaluation period
    test_period_start DATE NOT NULL,
    test_period_end DATE NOT NULL,
    
    -- Results (JSONB for flexibility)
    hypothetical_performance JSONB NOT NULL,  -- IPS/DR estimates, CIs
    actual_performance JSONB NOT NULL,        -- Baseline/logging policy
    improvement_metrics JSONB,                -- Lift, p-values, safety checks
    
    -- Evaluation metadata
    evaluation_method VARCHAR(20) NOT NULL,   -- 'ips', 'snips', 'dr'
    estimated_value DECIMAL(10, 4),
    variance DECIMAL(10, 6),
    ci_lower DECIMAL(10, 4),
    ci_upper DECIMAL(10, 4),
    n_samples INTEGER,
    effective_sample_size DECIMAL(10, 2),
    
    -- Confidence and safety
    confidence DECIMAL(5, 4) CHECK (confidence >= 0 AND confidence <= 1),
    is_safe_to_deploy BOOLEAN DEFAULT FALSE,
    safety_warnings TEXT[],
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    evaluated_by VARCHAR(100)  -- Model or user who ran evaluation
);

CREATE INDEX IF NOT EXISTS idx_backtest_results_profile_created 
    ON backtest_results(profile_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy 
    ON backtest_results(strategy_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_backtest_results_period 
    ON backtest_results(test_period_start, test_period_end);

CREATE INDEX IF NOT EXISTS idx_backtest_results_safe 
    ON backtest_results(is_safe_to_deploy, estimated_value DESC) 
    WHERE is_safe_to_deploy = TRUE;

-- ═══════════════════════════════════════════════════════════════════════
-- 3. Policy Deployments Table (Rollout Tracking)
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS policy_deployments (
    id BIGSERIAL PRIMARY KEY,
    profile_id TEXT NOT NULL,
    
    -- Policy identification
    policy_name VARCHAR(200) NOT NULL,
    policy_version VARCHAR(50),
    backtest_result_id INTEGER REFERENCES backtest_results(id),
    
    -- Deployment configuration
    traffic_percentage INTEGER NOT NULL CHECK (traffic_percentage >= 0 AND traffic_percentage <= 100),
    target_campaigns INTEGER[],  -- NULL = all campaigns
    target_keywords INTEGER[],   -- NULL = all keywords
    
    -- Rollout schedule
    started_at TIMESTAMPTZ DEFAULT NOW(),
    scheduled_ramp_to INTEGER,  -- Next traffic % to ramp to
    scheduled_ramp_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- 'active', 'paused', 'completed', 'rolled_back'
    rollback_reason TEXT,
    rollback_at TIMESTAMPTZ,
    
    -- Performance monitoring
    live_performance_metrics JSONB,  -- Real-time A/B test results
    alert_triggers TEXT[],
    
    -- Metadata
    deployed_by VARCHAR(100),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_policy_deployments_profile_status 
    ON policy_deployments(profile_id, status, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_policy_deployments_active 
    ON policy_deployments(status, traffic_percentage) 
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_policy_deployments_policy 
    ON policy_deployments(policy_name, started_at DESC);

-- ═══════════════════════════════════════════════════════════════════════
-- 4. Row Level Security (RLS)
-- ═══════════════════════════════════════════════════════════════════════

-- Enable RLS on all tables
ALTER TABLE prediction_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE policy_deployments ENABLE ROW LEVEL SECURITY;

-- Policies: Users can only access their own profile's data
CREATE POLICY prediction_logs_profile_isolation ON prediction_logs
    FOR ALL
    USING (profile_id = current_setting('app.current_profile_id', TRUE));

CREATE POLICY backtest_results_profile_isolation ON backtest_results
    FOR ALL
    USING (profile_id = current_setting('app.current_profile_id', TRUE));

CREATE POLICY policy_deployments_profile_isolation ON policy_deployments
    FOR ALL
    USING (profile_id = current_setting('app.current_profile_id', TRUE));

-- ═══════════════════════════════════════════════════════════════════════
-- 5. Helper Functions
-- ═══════════════════════════════════════════════════════════════════════

-- Function to compute ROAS from prediction log
CREATE OR REPLACE FUNCTION compute_roas(log_row prediction_logs)
RETURNS DECIMAL AS $$
BEGIN
    IF log_row.actual_spend IS NULL OR log_row.actual_spend <= 0 THEN
        RETURN NULL;
    END IF;
    RETURN COALESCE(log_row.actual_sales, 0) / log_row.actual_spend;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to check if attribution window has passed
CREATE OR REPLACE FUNCTION attribution_window_complete(log_row prediction_logs)
RETURNS BOOLEAN AS $$
BEGIN
    IF NOT log_row.was_executed THEN
        RETURN FALSE;
    END IF;
    RETURN NOW() > (log_row.execution_timestamp + (log_row.attribution_window_days || ' days')::INTERVAL);
END;
$$ LANGUAGE plpgsql STABLE;

"""

async def apply_migration():
    """Apply the counterfactual learning migration."""
    if not DB_URL:
        logger.error("SUPABASE_DB_URL not found in environment")
        return False
    
    try:
        logger.info("Connecting to database...")
        conn = await asyncpg.connect(DB_URL)
        
        logger.info("Applying counterfactual learning migration...")
        await conn.execute(MIGRATION_SQL)
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('prediction_logs', 'backtest_results', 'policy_deployments')
            ORDER BY table_name
        """)
        
        logger.info("✅ Migration complete. Created tables:")
        for table in tables:
            logger.info(f"   - {table['table_name']}")
        
        # Check table sizes
        for table in tables:
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {table['table_name']}"
            )
            logger.info(f"   {table['table_name']}: {count} rows")
        
        await conn.close()
        logger.info("Migration successful!")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(apply_migration())
    exit(0 if success else 1)
