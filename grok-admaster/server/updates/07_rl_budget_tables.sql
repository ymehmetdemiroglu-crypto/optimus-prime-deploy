
-- Optimus Pryme RL Budget Allocator Tables

CREATE TABLE IF NOT EXISTS rl_portfolio_state (
    id SERIAL PRIMARY KEY,
    profile_id VARCHAR(255) NOT NULL,
    state_vector JSONB NOT NULL,
    total_budget NUMERIC(15, 2) NOT NULL,
    budget_remaining NUMERIC(15, 2) NOT NULL,
    allocation_strategy VARCHAR(50) DEFAULT 'policy_gradient',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_state_profile ON rl_portfolio_state(profile_id);
CREATE INDEX IF NOT EXISTS idx_rl_state_time ON rl_portfolio_state(timestamp);

CREATE TABLE IF NOT EXISTS rl_budget_actions (
    id SERIAL PRIMARY KEY,
    portfolio_state_id INTEGER REFERENCES rl_portfolio_state(id) ON DELETE CASCADE,
    campaign_id INTEGER REFERENCES ppc_campaigns(id) ON DELETE CASCADE,
    allocated_budget NUMERIC(15, 2) NOT NULL,
    confidence_score NUMERIC(5, 4),
    reasoning TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_actions_state ON rl_budget_actions(portfolio_state_id);
CREATE INDEX IF NOT EXISTS idx_rl_actions_campaign ON rl_budget_actions(campaign_id);
