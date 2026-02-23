-- =============================================
-- Migration 09: Rufus Attribution Tracking
-- Data foundation to measure Rufus AI assistant
-- influence on conversions and PPC performance
-- =============================================

-- Rufus Attribution Events
-- Captures each interaction where Amazon's Rufus AI
-- surfaces our ASIN/keyword to a shopper
CREATE TABLE IF NOT EXISTS rufus_attribution_events (
    id                      SERIAL PRIMARY KEY,
    profile_id              VARCHAR(255) REFERENCES profiles(profile_id) ON DELETE CASCADE,
    asin                    VARCHAR(50)  NOT NULL,
    keyword_id              INTEGER      REFERENCES ppc_keywords(id) ON DELETE SET NULL,
    campaign_id             INTEGER      REFERENCES ppc_campaigns(id) ON DELETE SET NULL,

    -- What the shopper asked Rufus
    rufus_query             TEXT         NOT NULL,
    query_intent            VARCHAR(50)  NOT NULL DEFAULT 'informational',
    -- 'informational', 'transactional', 'navigational', 'comparison'

    -- Rufus scoring for this ASIN
    rufus_rank              INTEGER,           -- position in Rufus response (1 = first)
    rufus_confidence        NUMERIC(5,4),      -- model confidence 0-1
    rufus_intent_probability NUMERIC(5,4),     -- probability this query leads to purchase

    -- Contextual features at time of event (mirrors contextual_features.py)
    context_snapshot        JSONB,             -- full context vector at event time

    -- Outcome tracking
    attributed_order_id     VARCHAR(255),      -- Amazon order ID if converted
    attributed_revenue      NUMERIC(12,2),     -- revenue credited to Rufus touch
    converted               BOOLEAN      DEFAULT FALSE,
    conversion_delay_hours  NUMERIC(8,2),      -- hours from Rufus touch to purchase

    -- Channel credit split (sums to 1.0)
    rufus_credit            NUMERIC(5,4) DEFAULT 1.0,   -- portion attributed to Rufus
    ppc_credit              NUMERIC(5,4) DEFAULT 0.0,   -- portion attributed to PPC click
    organic_credit          NUMERIC(5,4) DEFAULT 0.0,   -- portion attributed to organic

    event_at                TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_rufus_events_profile
    ON rufus_attribution_events(profile_id);
CREATE INDEX IF NOT EXISTS ix_rufus_events_asin
    ON rufus_attribution_events(asin);
CREATE INDEX IF NOT EXISTS ix_rufus_events_keyword
    ON rufus_attribution_events(keyword_id);
CREATE INDEX IF NOT EXISTS ix_rufus_events_event_at
    ON rufus_attribution_events(event_at DESC);
CREATE INDEX IF NOT EXISTS ix_rufus_events_converted
    ON rufus_attribution_events(converted);

-- Rufus Channel Comparison
-- Aggregated daily snapshot for Rufus vs. PPC vs. Organic attribution
CREATE TABLE IF NOT EXISTS rufus_channel_comparison (
    id                      SERIAL PRIMARY KEY,
    profile_id              VARCHAR(255) REFERENCES profiles(profile_id) ON DELETE CASCADE,
    asin                    VARCHAR(50)  NOT NULL,
    date                    DATE         NOT NULL,

    -- Rufus channel totals
    rufus_impressions       INTEGER      DEFAULT 0,   -- times ASIN surfaced in Rufus
    rufus_conversions       INTEGER      DEFAULT 0,
    rufus_revenue           NUMERIC(12,2) DEFAULT 0,
    rufus_conversion_rate   NUMERIC(6,4) DEFAULT 0,
    avg_rufus_rank          NUMERIC(5,2),             -- average position in Rufus results
    avg_conversion_delay_h  NUMERIC(8,2),             -- avg hours Rufus→purchase

    -- PPC channel totals (for comparison)
    ppc_clicks              INTEGER      DEFAULT 0,
    ppc_conversions         INTEGER      DEFAULT 0,
    ppc_revenue             NUMERIC(12,2) DEFAULT 0,
    ppc_spend               NUMERIC(12,2) DEFAULT 0,

    -- Organic channel totals
    organic_sessions        INTEGER      DEFAULT 0,
    organic_conversions     INTEGER      DEFAULT 0,
    organic_revenue         NUMERIC(12,2) DEFAULT 0,

    -- Composite attribution
    total_attributed_revenue NUMERIC(12,2) DEFAULT 0,
    rufus_revenue_share     NUMERIC(5,4) DEFAULT 0,   -- rufus_revenue / total_attributed_revenue
    rufus_roas              NUMERIC(8,4),              -- rufus_revenue / ppc_spend (halo effect)
    rufus_incrementality    NUMERIC(5,4),              -- incremental lift over PPC baseline

    computed_at             TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(profile_id, asin, date)
);

CREATE INDEX IF NOT EXISTS ix_rufus_channel_profile_date
    ON rufus_channel_comparison(profile_id, date DESC);
CREATE INDEX IF NOT EXISTS ix_rufus_channel_asin
    ON rufus_channel_comparison(asin, date DESC);

-- Enable RLS
ALTER TABLE rufus_attribution_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE rufus_channel_comparison  ENABLE ROW LEVEL SECURITY;

-- Service-role bypass policies (match pattern of existing tables)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'rufus_attribution_events'
          AND policyname = 'rufus_events_service_all'
    ) THEN
        CREATE POLICY rufus_events_service_all ON rufus_attribution_events
            USING (TRUE) WITH CHECK (TRUE);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'rufus_channel_comparison'
          AND policyname = 'rufus_channel_service_all'
    ) THEN
        CREATE POLICY rufus_channel_service_all ON rufus_channel_comparison
            USING (TRUE) WITH CHECK (TRUE);
    END IF;
END $$;
