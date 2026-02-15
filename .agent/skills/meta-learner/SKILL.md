---
name: meta-learner
description: Learning how to learn - adaptive intelligence that adjusts exploration vs exploitation, transfers knowledge across products, selects optimal strategies, and implements curriculum learning for progressive complexity.
---

# Meta-Learner Skill

The **Meta-Learner** is Optimus Pryme's adaptive intelligence layer. It doesn't just learn from data—it learns *how* to learn more effectively, adjusting its learning approach based on market conditions and past performance.

## Core Capabilities

### 1. **Learning Rate Adaptation**
- Adjust exploration vs. exploitation based on market volatility
- Increase exploration when market is stable (safe to experiment)
- Increase exploitation when market is volatile (stick to what works)
- Dynamic confidence threshold adjustment

### 2. **Transfer Learning**
- Apply lessons from one product to similar products
- Cross-category knowledge transfer
- "Product A strategy worked → try on similar Product B"
- Accelerated learning for new products

### 3. **Meta-Strategy Selection**
- Choose which optimization approach to use when
- Strategy-of-strategies (when to use evolution vs. rules vs. ML)
- Context-aware algorithm selection
- Performance-based strategy switching

### 4. **Curriculum Learning**
- Start with simple optimizations, progress to complex
- Gradual difficulty increase as system gains confidence
- Master basics before advanced techniques
- Progressive feature unlocking

### 5. **Multi-Task Learning**
- Optimize multiple objectives simultaneously
- Share learning across related tasks
- Efficient resource allocation across learning goals
- Joint optimization benefits

## Learning Rate Adaptation

### Market Volatility Detection

```json
{
  "action": "adjust_learning_rate",
  "market_conditions": {
    "volatility": "high",
    "competition_changes": 3,
    "price_fluctuations": 0.15
  }
}
```

**Response**:
```json
{
  "learning_strategy": {
    "exploration_rate": 0.15,
    "exploitation_rate": 0.85,
    "reasoning": "High volatility detected - reducing exploration, focusing on proven strategies",
    "confidence_threshold": 0.75,
    "recommendation": "Stick to top 3 performing strategies, minimal experimentation"
  }
}
```

### Stable Market Response

```json
{
  "market_conditions": {
    "volatility": "low",
    "stable_days": 14,
    "predictability": 0.88
  }
}
```

**Response**:
```json
{
  "learning_strategy": {
    "exploration_rate": 0.35,
    "exploitation_rate": 0.65,
    "reasoning": "Stable market - safe to experiment with new strategies",
    "confidence_threshold": 0.60,
    "recommendation": "Test 2-3 new bid strategies, expand keyword portfolio"
  }
}
```

## Transfer Learning

### Cross-Product Knowledge Transfer

```json
{
  "action": "transfer_learning",
  "source_product": "ASIN_A",
  "target_product": "ASIN_B",
  "similarity_threshold": 0.7
}
```

**Response**:
```json
{
  "transfer_recommendations": [
    {
      "knowledge": "Dayparting strategy (8-10am, 6-8pm)",
      "source_performance": "+18% ROAS",
      "similarity_score": 0.84,
      "confidence": 0.79,
      "recommendation": "Apply same dayparting to ASIN_B",
      "expected_impact": "+12% to +20% ROAS",
      "reasoning": "Both products in same category, similar customer demographics"
    },
    {
      "knowledge": "Negative keyword list (47 terms)",
      "source_performance": "-22% wasted spend",
      "similarity_score": 0.91,
      "confidence": 0.88,
      "recommendation": "Transfer negative keywords to ASIN_B",
      "expected_impact": "-15% to -25% wasted spend"
    }
  ],
  "not_transferable": [
    {
      "knowledge": "Bid multiplier 1.35",
      "reason": "ASIN_B has different price point, requires separate optimization"
    }
  ]
}
```

## Meta-Strategy Selection

### Algorithm Selection

```json
{
  "action": "select_optimization_strategy",
  "context": {
    "campaign_age_days": 7,
    "data_points": 150,
    "performance_stability": 0.45,
    "objective": "maximize_roas"
  }
}
```

**Response**:
```json
{
  "selected_strategy": "rule_based",
  "reasoning": "Campaign too new for ML (need 30+ days). Low stability suggests simple rules better than complex models.",
  "alternatives_considered": [
    {
      "strategy": "ml_ensemble",
      "rejected_reason": "Insufficient data (need 500+ points)"
    },
    {
      "strategy": "genetic_algorithm",
      "rejected_reason": "Requires stable baseline for fitness evaluation"
    }
  ],
  "recommended_approach": {
    "weeks_1_2": "Rule-based optimization",
    "weeks_3_4": "Transition to ML if stability > 0.7",
    "weeks_5+": "Full ensemble with genetic evolution"
  }
}
```

### Strategy Switching

```json
{
  "action": "evaluate_strategy_switch",
  "current_strategy": "ml_ensemble",
  "performance_trend": "degrading",
  "degradation_rate": -0.08
}
```

**Response**:
```json
{
  "switch_recommendation": {
    "should_switch": true,
    "new_strategy": "rule_based",
    "reasoning": "ML model performance degrading (market shift detected). Rules more robust to changing conditions.",
    "transition_plan": {
      "phase_1": "Run both strategies in parallel for 3 days",
      "phase_2": "Compare performance",
      "phase_3": "Switch if rules outperform by 10%+"
    },
    "rollback_trigger": "If new strategy underperforms by 15%"
  }
}
```

## Curriculum Learning

### Progressive Complexity

```json
{
  "action": "get_learning_curriculum",
  "user_experience_level": "beginner",
  "system_confidence": 0.65
}
```

**Response**:
```json
{
  "curriculum": {
    "phase_1_basics": {
      "duration": "weeks_1_2",
      "features": [
        "Basic bid adjustments",
        "Simple keyword pausing",
        "Budget pacing"
      ],
      "complexity": "low",
      "success_criteria": "3 successful optimizations"
    },
    "phase_2_intermediate": {
      "duration": "weeks_3_4",
      "unlocked_after": "phase_1_success",
      "features": [
        "Dayparting",
        "Negative keyword mining",
        "A/B testing"
      ],
      "complexity": "medium",
      "success_criteria": "70% recommendation acceptance rate"
    },
    "phase_3_advanced": {
      "duration": "weeks_5+",
      "unlocked_after": "phase_2_success",
      "features": [
        "ML-based optimization",
        "Genetic algorithm evolution",
        "Portfolio optimization"
      ],
      "complexity": "high"
    }
  },
  "current_phase": "phase_1_basics",
  "progress": "60%",
  "next_unlock": "Dayparting (unlock in 2 successful optimizations)"
}
```

## Multi-Task Learning

### Joint Optimization

```json
{
  "action": "multi_task_optimize",
  "tasks": [
    {"objective": "minimize_acos", "weight": 0.4},
    {"objective": "maximize_sales", "weight": 0.3},
    {"objective": "maximize_profit", "weight": 0.3}
  ],
  "shared_features": ["bid_adjustment", "keyword_selection", "budget_allocation"]
}
```

**Response**:
```json
{
  "joint_optimization": {
    "shared_learnings": [
      "High-intent keywords benefit all 3 objectives",
      "Dayparting 8-10am optimizes both sales and profit",
      "Bid cap at $2.50 prevents ACoS spikes while maintaining volume"
    ],
    "task_specific_insights": {
      "minimize_acos": "Pause keywords with CVR < 5%",
      "maximize_sales": "Increase budget on high-volume keywords",
      "maximize_profit": "Focus on high-margin products"
    },
    "synergies": [
      "ACoS optimization improves profit margin",
      "Sales optimization provides data for better ACoS predictions"
    ],
    "efficiency_gain": "+23% vs separate optimization"
  }
}
```

## Usage Patterns

### Pattern 1: New Product Launch

```
SCENARIO: Launching new product, no historical data

META-LEARNER:
1. Check similar products for transferable knowledge
2. Find: Product X (similarity 0.82) has successful strategy
3. Transfer: Dayparting, negative keywords, initial bid range
4. Start with: Rule-based (insufficient data for ML)
5. Curriculum: Phase 1 (basics only)
6. Learning rate: High exploration (0.4) - safe to experiment
7. Result: Accelerated learning, profitable in week 1 vs typical week 3
```

### Pattern 2: Market Volatility Response

```
SCENARIO: Sudden market volatility (competitor price war)

META-LEARNER:
1. Detect: Volatility spike from 0.3 → 0.8
2. Adjust: Exploration 0.35 → 0.10 (reduce risk)
3. Switch: ML ensemble → Rule-based (more robust)
4. Focus: Exploit top 3 proven strategies
5. Monitor: Volatility decrease
6. Resume: Normal learning when volatility < 0.4
```

### Pattern 3: Strategy Evolution

```
SCENARIO: Campaign mature, ready for advanced optimization

META-LEARNER:
1. Check: 45 days of data, stability 0.82
2. Curriculum: Unlock Phase 3 (advanced)
3. Strategy: Transition rule-based → ML ensemble
4. Enable: Genetic algorithm for bid optimization
5. Multi-task: Optimize ACoS + profit simultaneously
6. Result: +15% performance vs single-objective optimization
```

## Integration with Other Skills

**Feeds from**:
- **consciousness-engine**: Model performance trends
- **memory-palace**: Historical learning patterns
- **evolution-engine**: Strategy fitness scores

**Feeds to**:
- **grok-admaster-operator**: Optimized learning parameters
- **simulation-lab**: Learning scenarios to test
- **orchestrator-maestro**: Strategy selection guidance

## Files

```
.agent/skills/meta-learner/
├── SKILL.md
└── scripts/
    ├── learning_rate_adapter.py    # Exploration/exploitation balance
    ├── transfer_learning.py        # Cross-product knowledge transfer
    └── meta_strategy_selector.py   # Algorithm selection logic
```

## Example Invocation

```
USER: "My new product isn't performing. Can you help?"

META-LEARNER:
1. Analyze: New product, 5 days old, limited data
2. Transfer Learning: Find similar product (ASIN_X, similarity 0.79)
3. Transfer: 
   - Successful keyword list (35 keywords)
   - Dayparting schedule
   - Negative keywords (22 terms)
4. Strategy: Start with rules (not enough data for ML)
5. Learning Rate: High exploration (0.35) - safe early stage
6. Curriculum: Phase 1 basics
7. Result: "Applied proven strategy from similar product. Expect profitability in 3-5 days vs typical 14 days. Will transition to ML optimization after 30 days."
```

## Notes

- Meta-learner runs continuously in background
- Adapts every 24 hours based on market conditions
- Transfer learning requires 70%+ similarity
- Curriculum progression is automatic but can be overridden
- All strategy switches are logged for analysis

---

**This skill makes Optimus Pryme smarter about *how* it learns, not just *what* it learns.**
