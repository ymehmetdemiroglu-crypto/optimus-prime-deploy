-- =============================================
-- MIGRATION 08: HNSW Vector Indexes for Scale
-- =============================================
-- Adds Hierarchical Navigable Small World (HNSW) indexes to both
-- embedding columns so pgvector can serve approximate nearest-neighbor
-- queries in O(log N) instead of the current O(N) sequential scan.
--
-- Before this migration every bleed/opportunity query performs a full
-- CROSS JOIN computing cosine distance for every (search_term, product)
-- pair.  With HNSW the planner can use ORDER BY … <=> … LIMIT k
-- queries that are accelerated by the spatial index.

SET search_path = public, extensions;

-- Increase maintenance_work_mem for faster index builds.
-- This is session-scoped and resets after the migration completes.
SET maintenance_work_mem = '512MB';

-- ────────────────────────────────────────────────────────────────────
-- 1. HNSW index on search_term_embeddings (the large, growing table)
-- ────────────────────────────────────────────────────────────────────
-- m = 16:  standard for 384-dim vectors; balances build time vs recall
-- ef_construction = 128: higher quality graph for better query recall
-- vector_cosine_ops: matches the <=> operator used in all queries
CREATE INDEX IF NOT EXISTS idx_ste_embedding_hnsw
    ON search_term_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- ────────────────────────────────────────────────────────────────────
-- 2. HNSW index on product_embeddings (smaller but still benefits)
-- ────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_pe_embedding_hnsw
    ON product_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- ────────────────────────────────────────────────────────────────────
-- 3. Rewrite helper functions to use ORDER BY <=> LIMIT pattern
--    that pgvector's HNSW index can accelerate.
-- ────────────────────────────────────────────────────────────────────

-- 3a. Bleed detection: find search terms FARTHEST from the product
--     (low similarity = high distance = wasted spend)
CREATE OR REPLACE FUNCTION find_semantic_bleed(
    p_product_asin VARCHAR,
    p_account_id INTEGER,
    p_threshold NUMERIC DEFAULT 0.40,
    p_min_spend NUMERIC DEFAULT 1.00,
    p_candidate_limit INTEGER DEFAULT 200
)
RETURNS TABLE (
    search_term TEXT,
    semantic_similarity NUMERIC,
    spend NUMERIC,
    clicks INTEGER,
    acos NUMERIC
)
LANGUAGE plpgsql
SET search_path = public, extensions
AS $$
DECLARE
    product_vec vector(384);
BEGIN
    -- Step 1: fetch the product embedding vector
    SELECT pe.embedding INTO product_vec
    FROM product_embeddings pe
    WHERE pe.asin = p_product_asin
      AND pe.account_id = p_account_id
    LIMIT 1;

    IF product_vec IS NULL THEN
        RETURN;  -- no product embedding found
    END IF;

    -- Step 2: KNN scan ordered by distance (HNSW-accelerated)
    --         then post-filter by similarity threshold
    RETURN QUERY
    SELECT
        sub.term AS search_term,
        sub.similarity AS semantic_similarity,
        sub.spend,
        sub.clicks,
        sub.acos
    FROM (
        SELECT
            ste.term,
            ROUND((1 - (ste.embedding <=> product_vec))::NUMERIC, 4) AS similarity,
            ste.spend,
            ste.clicks,
            ste.acos
        FROM search_term_embeddings ste
        WHERE ste.account_id = p_account_id
          AND ste.spend >= p_min_spend
        ORDER BY ste.embedding <=> product_vec DESC  -- farthest first
        LIMIT p_candidate_limit
    ) sub
    WHERE sub.similarity < p_threshold
    ORDER BY sub.spend DESC;
END;
$$;

-- 3b. Opportunity discovery: find search terms NEAREST to the product
--     (high similarity + conversions = untapped targets)
CREATE OR REPLACE FUNCTION find_semantic_opportunities(
    p_product_asin VARCHAR,
    p_account_id INTEGER,
    p_similarity_floor NUMERIC DEFAULT 0.70,
    p_min_impressions INTEGER DEFAULT 50,
    p_candidate_limit INTEGER DEFAULT 200
)
RETURNS TABLE (
    search_term TEXT,
    semantic_similarity NUMERIC,
    impressions INTEGER,
    clicks INTEGER,
    sales NUMERIC
)
LANGUAGE plpgsql
SET search_path = public, extensions
AS $$
DECLARE
    product_vec vector(384);
BEGIN
    -- Step 1: fetch the product embedding vector
    SELECT pe.embedding INTO product_vec
    FROM product_embeddings pe
    WHERE pe.asin = p_product_asin
      AND pe.account_id = p_account_id
    LIMIT 1;

    IF product_vec IS NULL THEN
        RETURN;
    END IF;

    -- Step 2: KNN scan ordered by distance ASC (HNSW-accelerated)
    --         then post-filter by similarity floor
    RETURN QUERY
    SELECT
        sub.term AS search_term,
        sub.similarity AS semantic_similarity,
        sub.impressions,
        sub.clicks,
        sub.sales
    FROM (
        SELECT
            ste.term,
            ROUND((1 - (ste.embedding <=> product_vec))::NUMERIC, 4) AS similarity,
            ste.impressions,
            ste.clicks,
            ste.sales
        FROM search_term_embeddings ste
        WHERE ste.account_id = p_account_id
          AND ste.impressions >= p_min_impressions
          AND ste.orders > 0
        ORDER BY ste.embedding <=> product_vec ASC  -- nearest first
        LIMIT p_candidate_limit
    ) sub
    WHERE sub.similarity >= p_similarity_floor
    ORDER BY sub.similarity DESC, sub.sales DESC;
END;
$$;

-- Reset maintenance_work_mem to default
RESET maintenance_work_mem;
