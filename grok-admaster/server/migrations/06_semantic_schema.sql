-- =============================================
-- MIGRATION 06: Semantic Intelligence Layer
-- =============================================

-- Ensure extensions schema is in path for everyone running this script
SET search_path = public, extensions;

-- 1. Enable vector extension (Safe if already enabled)
CREATE EXTENSION IF NOT EXISTS vector SCHEMA extensions;
-- Note: 'SCHEMA extensions' ensures it goes there. Most setups default to extensions or public.
-- If user's vector is in public, removing 'SCHEMA extensions' is safer or just rely on search_path.
-- Reverting to simple 'CREATE EXTENSION' because user said it's enabled.

-- 2. Create Search Query Report Table
-- We create the table WITHOUT foreign keys first to avoid permission issues
-- on existing tables if you lack ownership.
CREATE TABLE IF NOT EXISTS search_term_reports (
    id SERIAL PRIMARY KEY,
    -- Storing IDs as integers/strings without hard enforcement
    campaign_id INTEGER, 
    ad_group_id VARCHAR(50),
    keyword_id INTEGER,
    date DATE NOT NULL,
    search_term TEXT NOT NULL,
    impressions INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    cost NUMERIC(15, 2) DEFAULT 0,
    sales NUMERIC(15, 2) DEFAULT 0,
    orders INTEGER DEFAULT 0,
    converted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Search Term Embeddings
CREATE TABLE IF NOT EXISTS search_term_embeddings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    term TEXT NOT NULL,
    embedding VECTOR(384),
    account_id INTEGER, -- FK removed
    campaign_id INTEGER, -- FK removed
    source VARCHAR(50) DEFAULT 'search_query_report',
    impressions INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    spend NUMERIC(15, 2) DEFAULT 0,
    sales NUMERIC(15, 2) DEFAULT 0,
    orders INTEGER DEFAULT 0,
    acos NUMERIC(8, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Product Embeddings
CREATE TABLE IF NOT EXISTS product_embeddings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    asin VARCHAR(20) NOT NULL,
    title TEXT,
    source_text TEXT,
    embedding VECTOR(384),
    account_id INTEGER, -- FK removed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    -- UNIQUE constraint internal to table
    , UNIQUE(asin, account_id)
);

-- 5. Semantic Bleed Log
CREATE TABLE IF NOT EXISTS semantic_bleed_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    search_term_embedding_id UUID REFERENCES search_term_embeddings(id),
    product_embedding_id UUID REFERENCES product_embeddings(id),
    semantic_distance NUMERIC(6, 4) NOT NULL,
    spend_at_detection NUMERIC(15, 2) DEFAULT 0,
    action_taken VARCHAR(50) DEFAULT 'flagged',
    operator VARCHAR(50) DEFAULT 'autonomous',
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. Semantic Opportunity Log
CREATE TABLE IF NOT EXISTS semantic_opportunity_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    term TEXT NOT NULL,
    closest_product_asin VARCHAR(20),
    semantic_similarity NUMERIC(6, 4) NOT NULL,
    estimated_monthly_volume INTEGER,
    suggested_match_type VARCHAR(20) DEFAULT 'exact',
    suggested_bid NUMERIC(10, 2),
    status VARCHAR(50) DEFAULT 'discovered',
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. Autonomous Patrol Log
CREATE TABLE IF NOT EXISTS autonomous_patrol_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    patrol_cycle INTEGER NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    target_entity VARCHAR(255),
    details JSONB,
    status VARCHAR(50) DEFAULT 'success',
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 8. Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ste_term ON search_term_embeddings(term);
CREATE INDEX IF NOT EXISTS idx_ste_account ON search_term_embeddings(account_id);
CREATE INDEX IF NOT EXISTS idx_ste_campaign ON search_term_embeddings(campaign_id);
CREATE INDEX IF NOT EXISTS idx_pe_asin ON product_embeddings(asin);
CREATE INDEX IF NOT EXISTS idx_bleed_detected ON semantic_bleed_log(detected_at);
CREATE INDEX IF NOT EXISTS idx_opportunity_status ON semantic_opportunity_log(status);
CREATE INDEX IF NOT EXISTS idx_patrol_cycle ON autonomous_patrol_log(patrol_cycle);

-- 9. Helper function: Find semantically similar terms
CREATE OR REPLACE FUNCTION find_semantic_bleed(
    p_product_asin VARCHAR,
    p_account_id INTEGER,
    p_threshold NUMERIC DEFAULT 0.40,
    p_min_spend NUMERIC DEFAULT 1.00
)
RETURNS TABLE (
    search_term TEXT,
    semantic_similarity NUMERIC,
    spend NUMERIC,
    clicks INTEGER,
    acos NUMERIC
)
LANGUAGE sql
-- CRITICAL FIX: Ensure 'extensions' schema is in search path for vector operators
SET search_path = public, extensions
AS $$
    SELECT 
        ste.term AS search_term,
        ROUND((1 - (ste.embedding <=> pe.embedding))::NUMERIC, 4) AS semantic_similarity,
        ste.spend,
        ste.clicks,
        ste.acos
    FROM search_term_embeddings ste
    CROSS JOIN product_embeddings pe
    WHERE pe.asin = p_product_asin
      AND pe.account_id = p_account_id
      AND ste.account_id = p_account_id
      AND ste.spend >= p_min_spend
      AND (1 - (ste.embedding <=> pe.embedding)) < p_threshold
    ORDER BY ste.spend DESC;
$$;

-- 10. Helper function: Find untapped opportunities
CREATE OR REPLACE FUNCTION find_semantic_opportunities(
    p_product_asin VARCHAR,
    p_account_id INTEGER,
    p_similarity_floor NUMERIC DEFAULT 0.70,
    p_min_impressions INTEGER DEFAULT 50
)
RETURNS TABLE (
    search_term TEXT,
    semantic_similarity NUMERIC,
    impressions INTEGER,
    clicks INTEGER,
    sales NUMERIC
)
LANGUAGE sql
-- CRITICAL FIX: Ensure 'extensions' schema is in search path for vector operators
SET search_path = public, extensions
AS $$
    SELECT 
        ste.term AS search_term,
        ROUND((1 - (ste.embedding <=> pe.embedding))::NUMERIC, 4) AS semantic_similarity,
        ste.impressions,
        ste.clicks,
        ste.sales
    FROM search_term_embeddings ste
    CROSS JOIN product_embeddings pe
    WHERE pe.asin = p_product_asin
      AND pe.account_id = p_account_id
      AND ste.account_id = p_account_id
      AND (1 - (ste.embedding <=> pe.embedding)) >= p_similarity_floor
      AND ste.impressions >= p_min_impressions
      AND ste.orders > 0
    ORDER BY semantic_similarity DESC, ste.sales DESC;
$$;
