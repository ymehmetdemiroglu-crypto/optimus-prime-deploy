-- =============================================
-- SECURITY MIGRATION: ENABLE RLS
-- Migration: 03_enable_rls.sql
-- Purpose: Enable Row Level Security on AI tables to comply with security best practices
-- =============================================

-- 1. Enable RLS on all AI tables
ALTER TABLE prediction_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE creative_suggestions ENABLE ROW LEVEL SECURITY;
ALTER TABLE bandit_arms ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_registry ENABLE ROW LEVEL SECURITY;

-- 2. Create Policies
-- Note: In this single-tenant application, we allow 'authenticated' users (the main user) 
-- full access. Service role (backend) always has full access by default.

-- Policy for prediction_logs
DROP POLICY IF EXISTS "Enable all for authenticated users" ON prediction_logs;
CREATE POLICY "Enable all for authenticated users" ON prediction_logs
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy for training_jobs
DROP POLICY IF EXISTS "Enable all for authenticated users" ON training_jobs;
CREATE POLICY "Enable all for authenticated users" ON training_jobs
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy for creative_suggestions
DROP POLICY IF EXISTS "Enable all for authenticated users" ON creative_suggestions;
CREATE POLICY "Enable all for authenticated users" ON creative_suggestions
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy for bandit_arms
DROP POLICY IF EXISTS "Enable all for authenticated users" ON bandit_arms;
CREATE POLICY "Enable all for authenticated users" ON bandit_arms
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy for model_registry
DROP POLICY IF EXISTS "Enable all for authenticated users" ON model_registry;
CREATE POLICY "Enable all for authenticated users" ON model_registry
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Verification
SELECT 'RLS Enabled and Policies Applied' as status;
