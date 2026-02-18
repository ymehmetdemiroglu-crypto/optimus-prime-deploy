-- =============================================
-- MIGRATION 07: Vector Indexes + Action Approval Queue
-- =============================================
-- Fix 1: Add HNSW approximate-nearest-neighbour indexes to embedding columns.
--        Without these, every bleed/opportunity query performs an O(n*m) exact
--        scan (CROSS JOIN + full table distance computation). HNSW reduces this
--        to O(log n) at query time with a small build-time cost.
--
-- Fix 2: Introduce action_review_queue — a staging table that holds every
--        autonomous recommendation before it is applied. Allows human operators
--        to approve or reject individual actions, breaking the direct path from
--        detection to execution.
-- =============================================

SET search_path = public, extensions;

-- -----------------------------------------------------------------
-- 1. HNSW vector indexes
--    vector_cosine_ops matches the <=> cosine-distance operator used
--    in all bleed/opportunity SQL queries.
--
--    m=16, ef_construction=64 are the pgvector defaults and a good
--    starting point for 384-dim MiniLM vectors at typical PPC scale.
--    Tune ef_search at query time if recall needs adjustment.
-- -----------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_ste_embedding_hnsw
    ON search_term_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_pe_embedding_hnsw
    ON product_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- -----------------------------------------------------------------
-- 2. Action review queue
--    The autonomous operator writes every recommendation here with
--    status='pending_review' instead of executing it immediately.
--    An admin approves or rejects via the /operator-actions API.
--    Approved actions are passed to the execution layer; rejected
--    ones are archived for audit.
-- -----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS action_review_queue (
    id               UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    patrol_cycle     INTEGER NOT NULL,
    account_id       INTEGER,
    asin             VARCHAR(20),
    action_type      VARCHAR(50) NOT NULL,   -- 'add_negative' | 'add_target'
    term             TEXT NOT NULL,
    semantic_similarity  NUMERIC(6, 4),
    spend_at_detection   NUMERIC(15, 2),
    suggested_bid        NUMERIC(10, 2),
    suggested_match_type VARCHAR(20),
    urgency          VARCHAR(20) DEFAULT 'MEDIUM',  -- 'HIGH' | 'MEDIUM' | 'LOW'
    status           VARCHAR(30) DEFAULT 'pending_review',
    --   pending_review  → awaiting human decision
    --   approved        → cleared for execution
    --   rejected        → discarded, will not execute
    --   executed        → action has been applied downstream
    reviewed_by      VARCHAR(255),
    reviewed_at      TIMESTAMP WITH TIME ZONE,
    review_note      TEXT,
    details          JSONB,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_arq_status     ON action_review_queue(status);
CREATE INDEX IF NOT EXISTS idx_arq_account    ON action_review_queue(account_id);
CREATE INDEX IF NOT EXISTS idx_arq_asin       ON action_review_queue(asin);
CREATE INDEX IF NOT EXISTS idx_arq_created    ON action_review_queue(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_arq_cycle      ON action_review_queue(patrol_cycle);
