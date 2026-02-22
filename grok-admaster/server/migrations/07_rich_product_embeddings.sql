-- =============================================
-- MIGRATION 07: Rich Product Embeddings (Cosmo Alignment)
-- =============================================
-- Adds structured product metadata columns to product_embeddings so
-- the embedding vector incorporates brand, category, attributes,
-- price tier, and review signals — matching the feature set that
-- Amazon's Cosmo ranking system uses for query–product relevance.

SET search_path = public, extensions;

-- 1. Add rich metadata columns
ALTER TABLE product_embeddings
  ADD COLUMN IF NOT EXISTS brand            VARCHAR(255),
  ADD COLUMN IF NOT EXISTS category_path    TEXT,
  ADD COLUMN IF NOT EXISTS product_type     VARCHAR(100),
  ADD COLUMN IF NOT EXISTS bullet_points    JSONB,
  ADD COLUMN IF NOT EXISTS attributes       JSONB,
  ADD COLUMN IF NOT EXISTS price            NUMERIC(10, 2),
  ADD COLUMN IF NOT EXISTS review_score     NUMERIC(3, 2),
  ADD COLUMN IF NOT EXISTS review_count     INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS parent_asin      VARCHAR(20),
  ADD COLUMN IF NOT EXISTS embedding_version VARCHAR(30) DEFAULT 'v1_title_only',
  ADD COLUMN IF NOT EXISTS cosmo_alignment_score NUMERIC(6, 4);

-- 2. Index for brand + category lookups (common filtering axes)
CREATE INDEX IF NOT EXISTS idx_pe_brand ON product_embeddings(brand);
CREATE INDEX IF NOT EXISTS idx_pe_category ON product_embeddings(product_type);
CREATE INDEX IF NOT EXISTS idx_pe_parent ON product_embeddings(parent_asin);

-- 3. Add intent_type + query_source columns added by migration 06b (idempotent)
ALTER TABLE search_term_embeddings
  ADD COLUMN IF NOT EXISTS intent_type       VARCHAR(30) DEFAULT 'unclassified',
  ADD COLUMN IF NOT EXISTS query_source      VARCHAR(20) DEFAULT 'organic',
  ADD COLUMN IF NOT EXISTS intent_confidence NUMERIC(6, 4);
