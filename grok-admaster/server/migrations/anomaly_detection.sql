-- Anomaly Detection Database Migration
-- Run this SQL in your Supabase SQL editor or psql

-- ═══════════════════════════════════════════════════════════════════════
--  1. Create anomaly_alerts table
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id SERIAL PRIMARY KEY,
    
    -- Entity identification
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    entity_name VARCHAR(500),
    
    -- Profile/account association
    profile_id INTEGER NOT NULL,
    
    -- Anomaly details
    anomaly_score FLOAT NOT NULL,
    threshold FLOAT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    
    -- Values
    actual_value FLOAT,
    expected_value FLOAT,
    reconstruction_error FLOAT,
    
    -- Detection metadata
    detector_type VARCHAR(50) NOT NULL,
    detection_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Explanations and root cause
    explanation JSONB,
    root_causes JSONB,
    
    -- Alert management
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(200),
    
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);

-- Indexes for anomaly_alerts
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_entity_type ON anomaly_alerts(entity_type);
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_entity_id ON anomaly_alerts(entity_id);
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_profile_id ON anomaly_alerts(profile_id);
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_severity ON anomaly_alerts(severity);
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_detection_timestamp ON anomaly_alerts(detection_timestamp);
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_is_acknowledged ON anomaly_alerts(is_acknowledged);
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_is_resolved ON anomaly_alerts(is_resolved);
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_profile_severity ON anomaly_alerts(profile_id, severity);
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_entity_type_id ON anomaly_alerts(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS ix_anomaly_alerts_unresolved ON anomaly_alerts(is_resolved, detection_timestamp);

-- ═══════════════════════════════════════════════════════════════════════
--  2. Create anomaly_history table
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS anomaly_history (
    id SERIAL PRIMARY KEY,
    
    -- Entity identification
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    entity_name VARCHAR(500),
    
    -- Profile/account association
    profile_id INTEGER NOT NULL,
    
    -- Anomaly details
    anomaly_score FLOAT NOT NULL,
    threshold FLOAT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    
    -- Values at time of detection
    actual_value FLOAT,
    expected_value FLOAT,
    reconstruction_error FLOAT,
    
    -- Detection metadata
    detector_type VARCHAR(50) NOT NULL,
    detection_timestamp TIMESTAMP NOT NULL,
    
    -- Context at time of detection
    explanation JSONB,
    root_causes JSONB,
    market_conditions JSONB,
    campaign_settings JSONB,
    
    -- Resolution tracking
    was_resolved BOOLEAN DEFAULT FALSE,
    resolution_time_minutes INTEGER,
    resolution_action VARCHAR(100),
    
    -- Impact measurement
    revenue_impact FLOAT,
    performance_degradation FLOAT
);

-- Indexes for anomaly_history
CREATE INDEX IF NOT EXISTS ix_anomaly_history_entity_type ON anomaly_history(entity_type);
CREATE INDEX IF NOT EXISTS ix_anomaly_history_entity_id ON anomaly_history(entity_id);
CREATE INDEX IF NOT EXISTS ix_anomaly_history_profile_id ON anomaly_history(profile_id);
CREATE INDEX IF NOT EXISTS ix_anomaly_history_severity ON anomaly_history(severity);
CREATE INDEX IF NOT EXISTS ix_anomaly_history_detection_timestamp ON anomaly_history(detection_timestamp);
CREATE INDEX IF NOT EXISTS ix_anomaly_history_profile_date ON anomaly_history(profile_id, detection_timestamp);
CREATE INDEX IF NOT EXISTS ix_anomaly_history_entity_type_id ON anomaly_history(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS ix_anomaly_history_severity_date ON anomaly_history(severity, detection_timestamp);

-- ═══════════════════════════════════════════════════════════════════════
--  3. Create anomaly_training_data table
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS anomaly_training_data (
    id SERIAL PRIMARY KEY,
    
    -- Reference
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    profile_id INTEGER NOT NULL,
    
    -- Time series data
    sequence_data JSONB NOT NULL,
    feature_snapshot JSONB NOT NULL,
    
    -- Labels
    is_true_anomaly BOOLEAN NOT NULL,
    labeled_by VARCHAR(200),
    labeled_at TIMESTAMP DEFAULT NOW(),
    
    -- Model performance tracking
    predicted_score FLOAT,
    was_correctly_classified BOOLEAN
);

-- Indexes for anomaly_training_data
CREATE INDEX IF NOT EXISTS ix_training_profile_id ON anomaly_training_data(profile_id);
CREATE INDEX IF NOT EXISTS ix_training_labeled_at ON anomaly_training_data(labeled_at);
CREATE INDEX IF NOT EXISTS ix_training_profile_labeled ON anomaly_training_data(profile_id, labeled_at);

-- ═══════════════════════════════════════════════════════════════════════
--  4. Helper Functions
-- ═══════════════════════════════════════════════════════════════════════

-- Function to archive old alerts (>90 days)
CREATE OR REPLACE FUNCTION archive_old_anomaly_alerts()
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- Alerts older than 90 days that are resolved
    WITH archived AS (
        DELETE FROM anomaly_alerts
        WHERE detection_timestamp < NOW() - INTERVAL '90 days'
            AND is_resolved = TRUE
        RETURNING *
    )
    SELECT COUNT(*) INTO archived_count FROM archived;
    
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get anomaly statistics
CREATE OR REPLACE FUNCTION get_anomaly_stats(p_profile_id INTEGER)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_alerts', COUNT(*),
        'unresolved', COUNT(*) FILTER (WHERE is_resolved = FALSE),
        'critical', COUNT(*) FILTER (WHERE severity = 'critical' AND is_resolved = FALSE),
        'high', COUNT(*) FILTER (WHERE severity = 'high' AND is_resolved = FALSE),
        'medium', COUNT(*) FILTER (WHERE severity = 'medium' AND is_resolved = FALSE),
        'low', COUNT(*) FILTER (WHERE severity = 'low' AND is_resolved = FALSE),
        'last_24h', COUNT(*) FILTER (WHERE detection_timestamp >= NOW() - INTERVAL '24 hours')
    )
    INTO result
   FROM anomaly_alerts
    WHERE profile_id = p_profile_id;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ═══════════════════════════════════════════════════════════════════════
--  5. Comments
-- ═══════════════════════════════════════════════════════════════════════

COMMENT ON TABLE anomaly_alerts IS 'Real-time anomaly alerts with 90-day retention';
COMMENT ON TABLE anomaly_history IS 'Historical anomaly tracking for ML training';
COMMENT ON TABLE anomaly_training_data IS 'Labeled data for model retraining';

COMMENT ON FUNCTION archive_old_anomaly_alerts() IS 'Archive resolved alerts older than 90 days';
COMMENT ON FUNCTION get_anomaly_stats(INTEGER) IS 'Get aggregated anomaly statistics for a profile';
