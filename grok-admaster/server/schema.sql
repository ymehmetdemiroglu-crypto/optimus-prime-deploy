-- Optymus Pryme Database Schema
-- Optimized for Amazon PPC Management with ML Capabilities

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- CREATE EXTENSION IF NOT EXISTS "timescaledb"; -- If available on your tier, otherwise standard tables work

-- =============================================
-- 1. CAMPAIGN STRUCTURE
-- =============================================

-- Accounts (Amazon Seller/Vendor Accounts)
CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    amazon_account_id VARCHAR(255) UNIQUE NOT NULL,
    profile_id VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    region VARCHAR(50) NOT NULL,
    currency_code VARCHAR(10) DEFAULT 'USD',
    timezone VARCHAR(50) DEFAULT 'UTC',
    status VARCHAR(50) DEFAULT 'onboarding', -- 'onboarding', 'active', 'restricted', 'error'
    last_sync_at TIMESTAMP WITH TIME ZONE,
    health_score INTEGER DEFAULT 100,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolios
CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    amazon_portfolio_id VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    budget NUMERIC(15, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(account_id, amazon_portfolio_id)
);

-- Campaigns
CREATE TABLE IF NOT EXISTS campaigns (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    portfolio_id INTEGER REFERENCES portfolios(id),
    amazon_campaign_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    campaign_type VARCHAR(50) NOT NULL, -- 'sponsoredProducts', 'sponsoredBrands', etc.
    targeting_type VARCHAR(50), -- 'manual', 'auto'
    status VARCHAR(50) NOT NULL, -- 'enabled', 'paused', 'archived'
    daily_budget NUMERIC(15, 2),
    start_date DATE,
    end_date DATE,
    bidding_strategy VARCHAR(50), -- 'legacyForSales', 'autoForSales', 'manual'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(account_id, amazon_campaign_id)
);

-- Ad Groups
CREATE TABLE IF NOT EXISTS ad_groups (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id),
    amazon_ad_group_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    default_bid NUMERIC(10, 2),
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(campaign_id, amazon_ad_group_id)
);

-- Keywords
CREATE TABLE IF NOT EXISTS keywords (
    id SERIAL PRIMARY KEY,
    ad_group_id INTEGER REFERENCES ad_groups(id),
    campaign_id INTEGER REFERENCES campaigns(id),
    amazon_keyword_id VARCHAR(255) NOT NULL,
    keyword_text TEXT NOT NULL,
    match_type VARCHAR(20) NOT NULL, -- 'exact', 'phrase', 'broad'
    status VARCHAR(50) NOT NULL,
    bid NUMERIC(10, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(ad_group_id, amazon_keyword_id)
);

-- Targets (Product Targeting / Auto Targeting)
CREATE TABLE IF NOT EXISTS targets (
    id SERIAL PRIMARY KEY,
    ad_group_id INTEGER REFERENCES ad_groups(id),
    campaign_id INTEGER REFERENCES campaigns(id),
    amazon_target_id VARCHAR(255) NOT NULL,
    expression_type VARCHAR(50) NOT NULL, -- 'auto', 'manual'
    expression TEXT NOT NULL, -- JSON or string representation of targeting
    bid NUMERIC(10, 2),
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(ad_group_id, amazon_target_id)
);

-- Products (Advertised ASINs)
CREATE TABLE IF NOT EXISTS product_ads (
    id SERIAL PRIMARY KEY,
    ad_group_id INTEGER REFERENCES ad_groups(id),
    campaign_id INTEGER REFERENCES campaigns(id),
    amazon_ad_id VARCHAR(255) NOT NULL,
    asin VARCHAR(20) NOT NULL,
    sku VARCHAR(255),
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(ad_group_id, amazon_ad_id)
);

-- =============================================
-- 2. METRICS (Time Series Data)
-- =============================================

CREATE TABLE IF NOT EXISTS daily_campaign_metrics (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id),
    date DATE NOT NULL,
    impressions INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    cost NUMERIC(15, 2) DEFAULT 0,
    sales NUMERIC(15, 2) DEFAULT 0,
    orders INTEGER DEFAULT 0,
    units_sold INTEGER DEFAULT 0,
    UNIQUE(campaign_id, date)
);

CREATE TABLE IF NOT EXISTS daily_keyword_metrics (
    id SERIAL PRIMARY KEY,
    keyword_id INTEGER REFERENCES keywords(id),
    date DATE NOT NULL,
    impressions INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    cost NUMERIC(15, 2) DEFAULT 0,
    sales NUMERIC(15, 2) DEFAULT 0,
    orders INTEGER DEFAULT 0,
    UNIQUE(keyword_id, date)
);

CREATE TABLE IF NOT EXISTS search_term_reports (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id),
    ad_group_id INTEGER REFERENCES ad_groups(id),
    keyword_id INTEGER REFERENCES keywords(id), -- Nullable for auto campaigns
    date DATE NOT NULL,
    search_term TEXT NOT NULL,
    impressions INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    cost NUMERIC(15, 2) DEFAULT 0,
    sales NUMERIC(15, 2) DEFAULT 0,
    orders INTEGER DEFAULT 0,
    converted BOOLEAN DEFAULT FALSE
);

-- =============================================
-- 3. OPTIMIZATION & ML
-- =============================================

-- API Credentials (Encrypted)
CREATE TABLE IF NOT EXISTS credentials (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    api_type VARCHAR(50) NOT NULL, -- 'amazon_ads', 'seller_central'
    client_id TEXT NOT NULL,
    client_secret TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    access_token TEXT, -- Optional, cached
    token_expires_at TIMESTAMP WITH TIME ZONE,
    is_valid BOOLEAN DEFAULT TRUE,
    last_validated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optimization Profiles / Strategies
CREATE TABLE IF NOT EXISTS optimization_policies (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    name VARCHAR(255) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL, -- 'target_acos', 'max_sales', etc.
    target_acos NUMERIC(5, 2),
    target_roas NUMERIC(5, 2),
    min_bid NUMERIC(10, 2) DEFAULT 0.05,
    max_bid NUMERIC(10, 2) DEFAULT 5.00,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optimization History / Plans
CREATE TABLE IF NOT EXISTS optimization_plans (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id),
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) NOT NULL, -- 'pending', 'executed', 'failed'
    intelligence_level VARCHAR(50) DEFAULT 'standard',
    summary JSONB -- Stores summary stats
);

-- Actions within a plan
CREATE TABLE IF NOT EXISTS optimization_actions (
    id SERIAL PRIMARY KEY,
    plan_id INTEGER REFERENCES optimization_plans(id),
    action_type VARCHAR(50) NOT NULL, -- 'bid_change', 'keyword_pause'
    entity_type VARCHAR(50) NOT NULL, -- 'keyword', 'campaign', 'ad_group'
    entity_id INTEGER NOT NULL,
    old_value TEXT,
    new_value TEXT,
    confidence_score NUMERIC(4, 3),
    reasoning TEXT,
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'approved', 'rejected', 'executed'
    executed_at TIMESTAMP WITH TIME ZONE
);

-- =============================================
-- 4. ADVANCED ML INSIGHTS
-- =============================================

-- Anomaly Logs
CREATE TABLE IF NOT EXISTS anomalies (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    campaign_id INTEGER REFERENCES campaigns(id),
    metric_name VARCHAR(50) NOT NULL, -- 'spend', 'cpc', 'impressions'
    severity VARCHAR(20) NOT NULL, -- 'info', 'warning', 'critical'
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expected_value NUMERIC(15, 2),
    actual_value NUMERIC(15, 2),
    deviation_score NUMERIC(10, 2),
    status VARCHAR(50) DEFAULT 'new' -- 'new', 'acknowledged', 'resolved'
);

-- Keyword Health Scores
CREATE TABLE IF NOT EXISTS keyword_health (
    id SERIAL PRIMARY KEY,
    keyword_id INTEGER REFERENCES keywords(id),
    health_score INTEGER NOT NULL, -- 0-100
    status VARCHAR(50) NOT NULL, -- 'excellent', 'good', 'at_risk', 'critical'
    last_analyzed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    risk_factors JSONB -- Array of strings e.g. ["high_acos", "declining_ctr"]
);

-- Sales Forecasts
CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id),
    forecast_date DATE NOT NULL,
    predicted_sales NUMERIC(15, 2),
    predicted_spend NUMERIC(15, 2),
    confidence_interval_lower NUMERIC(15, 2),
    confidence_interval_upper NUMERIC(15, 2),
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_campaign_metrics_date ON daily_campaign_metrics(date);
CREATE INDEX idx_keyword_metrics_date ON daily_keyword_metrics(date);
CREATE INDEX idx_search_terms_text ON search_term_reports(search_term);
CREATE INDEX idx_actions_plan_id ON optimization_actions(plan_id);
CREATE INDEX idx_anomalies_created ON anomalies(detected_at);

-- =============================================
-- 5. MARKET INTELLIGENCE (DataForSEO Integration)
-- =============================================

-- Tracked Market Products (Competitors & Own)
CREATE TABLE IF NOT EXISTS market_products (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) UNIQUE NOT NULL,
    title TEXT,
    brand VARCHAR(255),
    category VARCHAR(255),
    image_url TEXT,
    product_url TEXT,
    is_competitor BOOLEAN DEFAULT FALSE,
    is_our_product BOOLEAN DEFAULT FALSE,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Competitor Price History (Time Series)
CREATE TABLE IF NOT EXISTS competitor_prices (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES market_products(id),
    price NUMERIC(15, 2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    is_deal BOOLEAN DEFAULT FALSE,
    deal_type VARCHAR(50),
    discount_percent NUMERIC(5, 2),
    in_stock BOOLEAN DEFAULT TRUE,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Keyword Rankings (Where products rank over time)
CREATE TABLE IF NOT EXISTS keyword_rankings (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES market_products(id),
    keyword VARCHAR(500) NOT NULL,
    rank_position INTEGER,
    rank_page INTEGER,
    rating NUMERIC(3, 1),
    reviews_count INTEGER,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Keyword Volume Data (Market Sizing)
CREATE TABLE IF NOT EXISTS market_keyword_volumes (
    id SERIAL PRIMARY KEY,
    keyword VARCHAR(500) NOT NULL,
    search_volume INTEGER,
    cpc NUMERIC(10, 2),
    competition NUMERIC(4, 3),
    location_code INTEGER DEFAULT 2840,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for Market Intelligence
CREATE INDEX idx_market_products_asin ON market_products(asin);
CREATE INDEX idx_competitor_prices_product_date ON competitor_prices(product_id, recorded_at);
CREATE INDEX idx_keyword_rankings_keyword_date ON keyword_rankings(keyword, recorded_at);
CREATE INDEX idx_market_keyword_volumes_keyword ON market_keyword_volumes(keyword);

