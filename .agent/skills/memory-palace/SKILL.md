---
name: memory-palace
description: Long-term memory and pattern recognition across time. Stores institutional knowledge, learns from historical successes/failures, detects seasonal patterns, and adapts to user preferences for intelligent decision-making.
---

# Memory Palace Skill

The **Memory Palace** is Optimus Pryme's long-term memory system. It stores, indexes,

 and retrieves patterns, lessons, and knowledge accumulated over time, enabling the system to learn from experience and make increasingly intelligent decisions.

## Core Capabilities

### 1. **Historical Pattern Library**
- Detect recurring patterns across campaigns and time
- Seasonal trend identification (Q4 surge, Prime Day, etc.)
- Day-of-week and time-of-day performance patterns
- Product lifecycle patterns
- Category-specific behaviors

### 2. **Case-Based Reasoning**
- Store successful strategies and their outcomes
- Learn from failures and near-misses
- Retrieve similar past scenarios for decision support
- "This worked last time" recommendations
- Context-aware pattern matching

### 3. **Situational Memory**
- Remember market conditions and responses
- Crisis response playbooks from experience
- Competitive action-reaction pairs
- Recovery strategies that worked

### 4. **User Preference Learning**
- Track which recommendations user accepts/rejects
- Learn risk tolerance and management style
- Adapt confidence thresholds per user
- Remember custom rules and constraints

### 5. **Institutional Knowledge**
- Vendor/supplier reliability tracking
- Seasonality calendars
- Product bundling success rates
- Keyword performance history

## Pattern Types

### Seasonal Patterns
```json
{
  "pattern_type": "seasonal",
  "pattern_signature": {
    "month": "December",
    "category": "electronics",
    "metric": "sales_velocity"
  },
  "observed_effect": "3.2x increase in sales",
  "occurrences": 3,
  "success_rate": 1.0,
  "recommendation": "Increase bids 30-50% in November, scale budget by 3x"
}
```

### Situational Patterns
```json
{
  "pattern_type": "situational",
  "scenario": "high_acos_sudden_spike",
  "past_actions": [
    "decrease_bid_20_percent",
    "pause_underperforming_keywords"
  ],
  "outcome": {
    "acos_recovered": true,
    "recovery_time_days": 3,
    "revenue_impact": -5
  },
  "lessons": "Quick bid reduction more effective than keyword pausing"
}
```

### User Preference Patterns
```json
{
  "pattern_type": "user_preference",
  "preference_category": "risk_tolerance",
  "observed_behavior": {
    "auto_approves_below_budget": 100,
    "manual_review_above": 100,
    "rejects_aggressive_strategies": 0.8
  },
  "inferred_preference": "conservative",
  "confidence": 0.92
}
```

## API Operations

### Store Pattern

```json
{
  "action": "store_pattern",
  "pattern": {
    "type": "seasonal",
    "signature": {...},
    "context": {...},
    "outcome": {...}
  }
}
```

### Retrieve Similar Patterns

```json
{
  "action": "find_similar",
  "current_situation": {
    "campaign_id": 123,
    "metrics": {...},
    "context": "high_acos"
  },
  "limit": 5
}
```

**Response**:
```json
{
  "similar_cases": [
    {
      "similarity_score": 0.89,
      "past_scenario": {...},
      "actions_taken": [...],
      "outcome": {...},
      "recommendation": "Based on 3 similar cases, decrease bid by 15-20%"
    }
  ]
}
```

### Learn User Preference

```json
{
  "action": "update_preference",
  "user_action": "rejected",
  "recommendation_context": {
    "strategy": "aggressive",
    "budget_increase": 200
  }
}
```

## Usage Patterns

### Pattern 1: "What Worked Last Time?"

**Scenario**: Facing a performance issue

```
CURRENT: High ACoS on campaign X
MEMORY PALACE:
1. Search for similar past scenarios
2. Find 2 cases with high ACoS
3. Review actions taken and outcomes
4. Recommend best-performing approach
```

### Pattern 2: Seasonal Prediction

**Scenario**: Approaching known seasonal event

```
TRIGGER: 6 weeks before Prime Day
MEMORY PALACE:
1. Retrieve Prime Day patterns from last 2 years
2. Identify: 3x demand spike, 40% higher competition
3. Recommend: Scale budget early, increase bids 2 weeks before
4. Alert user proactively
```

### Pattern 3: User Preference Adaptation

**Scenario**: Generating recommendations

```
BEFORE RECOMMENDATION:
1. Check user's historical acceptance rate for similar recommendations
2. If user typically rejects "aggressive" → soften recommendation
3. If user auto-approves below $X → don't require manual approval
4. Adapt confidence thresholds to user's style
```

## Database Schema

```sql
-- From server/updates/04_meta_skills_tables.sql

memory_patterns (
  pattern_type,        -- 'seasonal', 'situational', 'user_preference'
  pattern_signature,   -- JSON description of pattern
  occurrences,         -- How many times observed
  success_rate,        -- Success rate (0.0 - 1.0)
  context,             -- Additional context
  first_seen,
  last_seen
)

case_library (
  scenario_description,
  actions_taken,      -- What was done
  outcome,            -- What happened
  lessons_learned,    -- Key insights
  created_at
)
```

## Integration with Other Skills

**Works with**:
- **orchestrator-maestro**: Provide historical workflow success rates
- **evolution-engine**: Supply failure patterns to avoid
- **consciousness-engine**: Share prediction accuracy history
- **grok-admaster-operator**: Inform optimization decisions

## Files

```
.agent/skills/memory-palace/
├── SKILL.md
├── scripts/
│   ├── pattern_miner.py          # Detect patterns in historical data
│   ├── memory_indexer.py         # Efficient storage and retrieval
│   └── similarity_matcher.py     # Find similar past scenarios
├── resources/
│   └── seasonal_patterns.json    # Known seasonal trends
└── tests/
    └── test_pattern_miner.py
```

## Example Invocation

```
USER: "Should I increase my bid on this keyword?"

MEMORY PALACE ACTION:
1. Check: Has this keyword been optimized before?
2. Find: Yes, 3 months ago, bid increased 20% → ACoS worsened
3. Find: Similar keywords in this category respond better to 10% increases
4. Retrieve: User typically prefers conservative changes
5. Recommend: "Increase by 10% (historical data shows this range works better for this category)"
```

## Notes

- Patterns strengthen with repeated observations
- Old patterns can "decay" if not recently observed
- User preferences are continuously refined
- All pattern storage respects privacy and data retention policies
- Patterns from different accounts are kept separate

---

**This skill transforms Optimus Pryme from reactive to predictive, learning from every action and getting smarter over time.**
