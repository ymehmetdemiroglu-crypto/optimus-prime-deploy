-- Tier 2 Meta-Skills Database Tables
-- [UPDATED] Added profile_id to all tables for strict multi-tenant isolation
-- This ensures Evolution, Simulations, and Insights are customized for EACH account.

-- ============================================
-- 1. DROP EXISTING TABLES (Cleanup)
-- ============================================

DROP TABLE IF EXISTS strategy_lineage CASCADE;
DROP TABLE IF EXISTS evolution_cycles CASCADE;
DROP TABLE IF EXISTS simulation_runs CASCADE;
DROP TABLE IF EXISTS backtest_results CASCADE;
DROP TABLE IF EXISTS synthesized_insights CASCADE;
DROP TABLE IF EXISTS external_knowledge CASCADE;

-- ============================================
-- 2. CREATE TABLES (With profile_id)
-- ============================================

-- Evolution Engine
-- ----------------

CREATE TABLE strategy_lineage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL, -- Custom strategy evolution per account
    strategy_name TEXT NOT NULL,
    parent_id UUID REFERENCES strategy_lineage(id),
    generation INT NOT NULL,
    dna JSONB NOT NULL,
    fitness_score NUMERIC(5,4) CHECK (fitness_score >= 0 AND fitness_score <= 1),
    mutation_type TEXT CHECK (mutation_type IN ('parameter_tweak', 'threshold_shift', 'feature_toggle', 'structural_change', 'crossover')),
    status TEXT CHECK (status IN ('active', 'extinct', 'archived')) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE evolution_cycles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    generation_number INT NOT NULL,
    population_size INT NOT NULL,
    best_fitness NUMERIC(5,4),
    avg_fitness NUMERIC(5,4),
    diversity_score NUMERIC(5,4),
    completed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Simulation Lab
-- --------------

CREATE TABLE simulation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL, -- Simulations run on specific account data
    simulation_type TEXT NOT NULL CHECK (simulation_type IN ('monte_carlo', 'backtest', 'scenario', 'risk_analysis')),
    input_parameters JSONB NOT NULL,
    iterations INT,
    results JSONB,
    confidence_intervals JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE backtest_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    test_period_start DATE NOT NULL,
    test_period_end DATE NOT NULL,
    hypothetical_performance JSONB,
    actual_performance JSONB,
    improvement_metrics JSONB,
    confidence NUMERIC(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge Synthesizer
-- ---------------------

CREATE TABLE synthesized_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL, -- Insights specific to this account
    insight_type TEXT NOT NULL CHECK (insight_type IN ('cross_product', 'trend', 'competitive', 'correlation', 'pattern')),
    source_data JSONB NOT NULL,
    insight_description TEXT NOT NULL,
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    actionable_recommendations JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE external_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id TEXT NOT NULL, -- Track what knowledge is relevant to this account
    source TEXT NOT NULL,
    source_type TEXT CHECK (source_type IN ('blog', 'forum', 'research', 'trend_report', 'news', 'social_media')),
    content TEXT,
    relevance_score NUMERIC(5,4) CHECK (relevance_score >= 0 AND relevance_score <= 1),
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- 3. CREATE INDEXES
-- ============================================

CREATE INDEX idx_strategy_lineage_profile ON strategy_lineage(profile_id);
CREATE INDEX idx_evolution_cycles_profile ON evolution_cycles(profile_id);
CREATE INDEX idx_simulation_runs_profile ON simulation_runs(profile_id);
CREATE INDEX idx_backtest_results_profile ON backtest_results(profile_id);
CREATE INDEX idx_synthesized_insights_profile ON synthesized_insights(profile_id);
CREATE INDEX idx_external_knowledge_profile ON external_knowledge(profile_id);

-- ============================================
-- 4. ENABLE RLS
-- ============================================

ALTER TABLE strategy_lineage ENABLE ROW LEVEL SECURITY;
ALTER TABLE evolution_cycles ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulation_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE synthesized_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE external_knowledge ENABLE ROW LEVEL SECURITY;

-- Simple RLS Policies
CREATE POLICY "Allow authenticated full access strategy_lineage" ON strategy_lineage FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access evolution_cycles" ON evolution_cycles FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access simulation_runs" ON simulation_runs FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access backtest_results" ON backtest_results FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access synthesized_insights" ON synthesized_insights FOR ALL TO public USING (auth.role() = 'authenticated');
CREATE POLICY "Allow authenticated full access external_knowledge" ON external_knowledge FOR ALL TO public USING (auth.role() = 'authenticated');
