CREATE TABLE IF NOT EXISTS bandit_arms (
    keyword_id INTEGER NOT NULL REFERENCES ppc_keywords(id) ON DELETE CASCADE,
    arm_id INTEGER NOT NULL,
    multiplier NUMERIC(4, 2) NOT NULL,
    alpha NUMERIC(10, 4) DEFAULT 1.0,
    beta NUMERIC(10, 4) DEFAULT 1.0,
    pulls INTEGER DEFAULT 0,
    total_reward NUMERIC(15, 4) DEFAULT 0,
    sum_squared_reward NUMERIC(15, 4) DEFAULT 0, -- For UCB calculation if needed
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (keyword_id, arm_id)
);

CREATE INDEX IF NOT EXISTS idx_bandit_arms_keyword_id ON bandit_arms(keyword_id);
