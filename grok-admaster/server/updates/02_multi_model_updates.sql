-- =============================================
-- MULTI-MODEL SUPPORT MIGRATION
-- Migration: 02_multi_model_updates.sql
-- Purpose: Support for Creative AI and LLM Usage Tracking
-- =============================================

-- 1. Create table for storing AI-generated creative suggestions
CREATE TABLE IF NOT EXISTS creative_suggestions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id BIGINT REFERENCES campaigns(id),
    suggestion_type TEXT NOT NULL, -- 'headline', 'description', 'email'
    content TEXT NOT NULL,
    ai_model TEXT NOT NULL, -- e.g. 'claude-3.5-sonnet'
    status TEXT DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    meta_data JSONB -- specific context like 'target_audience', 'tone'
);

-- 2. Register the external LLM models
INSERT INTO model_registry (model_name, model_version, model_type, is_active)
VALUES 
    ('openai/gpt-4-turbo', '2024-04-09', 'strategist', true),
    ('anthropic/claude-3.5-sonnet', '3.5', 'creative', true),
    ('anthropic/claude-3-haiku', '3.0', 'analyst', true),
    ('meta-llama/llama-3-8b', '8b-instruct', 'fast_chat', true)
ON CONFLICT (model_name) DO UPDATE 
SET model_version = EXCLUDED.model_version,
    model_type = EXCLUDED.model_type,
    is_active = EXCLUDED.is_active;
