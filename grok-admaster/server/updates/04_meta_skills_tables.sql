-- Meta-Skills System Database Tables
-- [UPDATED] Added profile_id to all tables for strict multi-tenant isolation
-- This ensures Memory, Decisions, and Workflows are customized for EACH account.

-- ============================================
-- 1. DROP EXISTING TABLES (Cleanup)
-- ============================================

DROP TABLE IF EXISTS skill_executions CASCADE;
DROP TABLE IF EXISTS workflow_templates CASCADE;
DROP TABLE IF EXISTS memory_patterns CASCADE;
DROP TABLE IF EXISTS case_library CASCADE;
DROP TABLE IF EXISTS decision_audit CASCADE;
DROP TABLE IF EXISTS model_performance_tracking CASCADE;
DROP TABLE IF EXISTS generated_skills CASCADE;
DROP TABLE IF EXISTS skill_versions CASCADE;

-- ============================================
-- 2. CREATE TABLES (With profile_id)
-- ============================================

-- Orchestrator Maestro
-- --------------------

CREATE TABLE skill_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL, -- Isolated execution context
    workflow_id UUID,
    skill_name TEXT NOT NULL,
    input_data JSONB,
    output_data JSONB,
    execution_order INT,
    status TEXT CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE workflow_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL, -- Custom workflows per account
    name TEXT NOT NULL,
    description TEXT,
    skill_sequence JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(profile_id, name)
);

-- Memory Palace
-- -------------

CREATE TABLE memory_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL, -- Unique memory per account
    pattern_type TEXT NOT NULL CHECK (pattern_type IN ('seasonal', 'situational', 'user_preference', 'day_of_week')),
    pattern_signature JSONB NOT NULL,
    occurrences INT DEFAULT 1,
    success_rate NUMERIC(5,4) CHECK (success_rate >= 0 AND success_rate <= 1),
    context JSONB,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE case_library (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    scenario_description TEXT NOT NULL,
    actions_taken JSONB NOT NULL,
    outcome JSONB,
    lessons_learned TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Consciousness Engine
-- --------------------

CREATE TABLE decision_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    options_considered JSONB,
    chosen_option TEXT NOT NULL,
    reasoning TEXT,
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    actual_outcome JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_performance_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prediction_accuracy NUMERIC(5,4) CHECK (prediction_accuracy >= 0 AND prediction_accuracy <= 1),
    usage_count INT DEFAULT 0,
    avg_confidence NUMERIC(5,4) CHECK (avg_confidence >= 0 AND avg_confidence <= 1),
    last_calibrated TIMESTAMPTZ,
    performance_trend TEXT CHECK (performance_trend IN ('improving', 'stable', 'degrading')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(profile_id, model_name)
);

-- Skill Creator
-- -------------

CREATE TABLE generated_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    description TEXT,
    capabilities JSONB,
    template_used TEXT,
    validation_status TEXT CHECK (validation_status IN ('passed', 'failed', 'pending')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    approved_by TEXT,
    UNIQUE(profile_id, skill_name)
);

CREATE TABLE skill_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    version TEXT NOT NULL,
    changes TEXT,
    performance_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (profile_id, skill_name, version)
);

-- ============================================
-- 3. CREATE INDEXES
-- ============================================

CREATE INDEX idx_skill_executions_profile ON skill_executions(profile_id);
CREATE INDEX idx_workflow_templates_profile ON workflow_templates(profile_id);
CREATE INDEX idx_memory_patterns_profile ON memory_patterns(profile_id);
CREATE INDEX idx_memory_patterns_type ON memory_patterns(pattern_type);
CREATE INDEX idx_decision_audit_profile ON decision_audit(profile_id);
CREATE INDEX idx_model_performance_profile ON model_performance_tracking(profile_id);
CREATE INDEX idx_generated_skills_profile ON generated_skills(profile_id);

-- ============================================
-- 4. ENABLE RLS
-- ============================================

ALTER TABLE skill_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE case_library ENABLE ROW LEVEL SECURITY;
ALTER TABLE decision_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_performance_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE generated_skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE skill_versions ENABLE ROW LEVEL SECURITY;

-- Simple RLS Policies (Allow authenticated access)
CREATE POLICY "Allow authenticated full access skill_executions" ON skill_executions FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access workflow_templates" ON workflow_templates FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access memory_patterns" ON memory_patterns FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access case_library" ON case_library FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access decision_audit" ON decision_audit FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access model_performance_tracking" ON model_performance_tracking FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access generated_skills" ON generated_skills FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access skill_versions" ON skill_versions FOR ALL TO public USING (auth.role() = 'authenticated');
