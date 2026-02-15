-- Domain Skills & Automation Database Tables
-- [RESET & RECREATE] 
-- This script DROPS existing domain skill tables and recreates them with 'profile_id' metadata.
-- Use this to fix the "column profile_id does not exist" error.

-- ============================================
-- 1. DROP EXISTING TABLES (Cleanup)
-- ============================================

DROP TABLE IF EXISTS competitor_prices CASCADE;
DROP TABLE IF EXISTS price_changes CASCADE;
DROP TABLE IF EXISTS share_of_voice CASCADE;
DROP TABLE IF EXISTS product_reviews_analysis CASCADE;
DROP TABLE IF EXISTS ml_models CASCADE;
DROP TABLE IF EXISTS detected_anomalies CASCADE;
DROP TABLE IF EXISTS customer_segments CASCADE;
DROP TABLE IF EXISTS alert_configs CASCADE;
DROP TABLE IF EXISTS alert_history CASCADE;
DROP TABLE IF EXISTS scheduled_reports CASCADE;
DROP TABLE IF EXISTS scheduled_tasks CASCADE;
DROP TABLE IF EXISTS task_executions CASCADE;

-- ============================================
-- 2. CREATE TABLES (With profile_id)
-- ============================================

-- Competitive Intelligence
-- ------------------------

CREATE TABLE competitor_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL, -- Amazon Profile ID
    asin TEXT NOT NULL,
    competitor_brand TEXT,
    price NUMERIC(10,2) NOT NULL,
    currency TEXT DEFAULT 'USD',
    is_deal BOOLEAN DEFAULT FALSE,
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE price_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    asin TEXT NOT NULL,
    competitor_brand TEXT,
    old_price NUMERIC(10,2),
    new_price NUMERIC(10,2),
    change_pct NUMERIC(5,2),
    change_type TEXT CHECK (change_type IN ('drop', 'increase')),
    likely_reason TEXT,
    detected_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE share_of_voice (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    brand TEXT NOT NULL,
    organic_sov NUMERIC(5,2),
    paid_sov NUMERIC(5,2),
    total_sov NUMERIC(5,2),
    position_breakdown JSONB,
    analyzed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE product_reviews_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    asin TEXT NOT NULL,
    review_id TEXT,
    rating INT,
    sentiment TEXT CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    sentiment_score NUMERIC(3,2),
    features_mentioned JSONB,
    is_actionable BOOLEAN DEFAULT FALSE,
    analyzed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Data Scientist
-- --------------

CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    training_metrics JSONB,
    feature_importance JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    trained_at TIMESTAMPTZ DEFAULT NOW(),
    next_retrain_at TIMESTAMPTZ
);

CREATE TABLE detected_anomalies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    detected_value NUMERIC,
    expected_value NUMERIC,
    severity TEXT CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    probable_cause TEXT,
    status TEXT CHECK (status IN ('new', 'investigating', 'resolved', 'ignored')) DEFAULT 'new',
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE TABLE customer_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    segment_name TEXT NOT NULL,
    customer_count INT,
    avg_ltv NUMERIC(10,2),
    segment_criteria JSONB,
    recommended_strategy TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Automation
-- ----------

CREATE TABLE alert_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    channels JSONB,
    priority TEXT CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    conditions JSONB,
    recipients JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(profile_id, alert_type)
);

CREATE TABLE alert_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    config_id UUID REFERENCES alert_configs(id),
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    status TEXT CHECK (status IN ('sent', 'failed', 'acknowledged', 'resolved')) DEFAULT 'sent',
    sent_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ
);

CREATE TABLE scheduled_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    name TEXT NOT NULL,
    report_type TEXT NOT NULL,
    frequency TEXT CHECK (frequency IN ('hourly', 'daily', 'weekly', 'monthly', 'custom')),
    schedule_config JSONB,
    recipients JSONB,
    formats JSONB,
    enabled BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE scheduled_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    name TEXT NOT NULL,
    handler TEXT NOT NULL,
    cron_schedule TEXT NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    priority INT DEFAULT 1,
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE task_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    task_id UUID REFERENCES scheduled_tasks(id),
    status TEXT CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms INT,
    result_summary JSONB,
    error_message TEXT
);

-- ============================================
-- 3. CREATE INDEXES
-- ============================================

CREATE INDEX idx_competitor_prices_profile ON competitor_prices(profile_id);
CREATE INDEX idx_competitor_prices_asin ON competitor_prices(asin);
CREATE INDEX idx_share_of_voice_profile ON share_of_voice(profile_id);
CREATE INDEX idx_product_reviews_profile ON product_reviews_analysis(profile_id);
CREATE INDEX idx_detected_anomalies_profile ON detected_anomalies(profile_id);
CREATE INDEX idx_alert_history_profile ON alert_history(profile_id);
CREATE INDEX idx_scheduled_tasks_profile ON scheduled_tasks(profile_id);

-- ============================================
-- 4. ENABLE RLS
-- ============================================

ALTER TABLE competitor_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE price_changes ENABLE ROW LEVEL SECURITY;
ALTER TABLE share_of_voice ENABLE ROW LEVEL SECURITY;
ALTER TABLE product_reviews_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE detected_anomalies ENABLE ROW LEVEL SECURITY;
ALTER TABLE customer_segments ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE scheduled_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE scheduled_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE task_executions ENABLE ROW LEVEL SECURITY;

-- Simple RLS Policies
CREATE POLICY "Allow authenticated full access competitor_prices" ON competitor_prices FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access price_changes" ON price_changes FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access share_of_voice" ON share_of_voice FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access product_reviews_analysis" ON product_reviews_analysis FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access ml_models" ON ml_models FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access detected_anomalies" ON detected_anomalies FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access customer_segments" ON customer_segments FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access alert_configs" ON alert_configs FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access alert_history" ON alert_history FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access scheduled_reports" ON scheduled_reports FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access scheduled_tasks" ON scheduled_tasks FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access task_executions" ON task_executions FOR ALL TO public USING (auth.role() = 'authenticated');
