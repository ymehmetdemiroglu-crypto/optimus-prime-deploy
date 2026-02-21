-- Migration: 08_rufus_cosmo_optimizations.sql
-- Description: Adds query_source to search_term_embeddings, cosmo_embedding to product_embeddings, and HNSW indexes for both.

-- Ensure pgvector is enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Add query_source to search_term_embeddings with NOT NULL and default
ALTER TABLE search_term_embeddings
ADD COLUMN IF NOT EXISTS query_source VARCHAR(50) NOT NULL DEFAULT 'organic';

-- 2. Add cosmo_embedding to product_embeddings (1024-dim for e5-large-v2)
ALTER TABLE product_embeddings
ADD COLUMN IF NOT EXISTS cosmo_embedding vector(1024);

-- 3. Add HNSW indexes for scale
-- Using CONCURRENTLY to avoid locking writes on live tables.
-- Note: CONCURRENTLY cannot be run inside a transaction block.
-- Using ef_construction = 128 for better recall stability as the table grows.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_search_term_embedding_cosine 
ON search_term_embeddings USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 128);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_product_cosmo_embedding_cosine 
ON product_embeddings USING hnsw (cosmo_embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 128);
