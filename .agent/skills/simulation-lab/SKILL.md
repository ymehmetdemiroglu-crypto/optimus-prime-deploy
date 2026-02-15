---
name: simulation-lab
description: What-if scenario testing and strategy validation. Runs Monte Carlo simulations, backtests strategies on historical data, performs risk analysis, and provides sandbox environments for safe experimentation.
---

# Simulation Lab Skill

The **Simulation Lab** is Optimus Pryme's testing ground. It allows you to safely test strategies, forecast outcomes, and understand risks before committing real budget.

## Core Capabilities

### 1. **Monte Carlo Simulation**
- Forecast campaign outcomes with uncertainty
- Run thousands of scenarios to understand probability distributions
- Confidence intervals for key metrics
- Risk quantification (Value at Risk, worst-case scenarios)

### 2. **Strategy Backtesting**
- Test strategies on historical data
- "What if I had used this strategy last quarter?"
- Compare hypothetical vs. actual performance
- Identify which strategies would have worked

### 3. **Scenario Analysis**
- Test market condition changes (competition +20%, CPC +15%)
- Budget constraint scenarios
- Seasonal impact modeling
- Crisis simulation (sudden ACoS spike, budget cut)

### 4. **Risk Analysis**
- Value at Risk (VaR) calculations
- Downside risk quantification
- Stress testing under extreme conditions
- Portfolio risk across campaigns

### 5. **Sandbox Environment**
- Safe testing without affecting live campaigns
- Dry-run mode for all operations
- Rollback capability
- Isolated experimentation

## Monte Carlo Simulation

### Campaign Forecast Example

```json
{
  "action": "monte_carlo_forecast",
  "campaign_id": 123,
  "forecast_days": 30,
  "iterations": 10000,
  "variables": {
    "daily_budget": {
      "distribution": "normal",
      "mean": 100,
      "std": 10
    },
    "cpc": {
      "distribution": "lognormal",
      "mean": 1.50,
      "std": 0.30
    },
    "conversion_rate": {
      "distribution": "beta",
      "alpha": 20,
      "beta": 180
    }
  }
}
```

**Output**:
```json
{
  "forecast": {
    "total_sales": {
      "mean": 4500,
      "median": 4350,
      "p5": 3200,
      "p95": 5900,
      "confidence_90": [3200, 5900]
    },
    "acos": {
      "mean": 0.24,
      "median": 0.23,
      "p5": 0.18,
      "p95": 0.32
    },
    "probability_profitable": 0.87,
    "value_at_risk_5pct": -450
  },
  "iterations": 10000
}
```

## Backtesting

### Historical Strategy Test

```json
{
  "action": "backtest_strategy",
  "strategy": {
    "name": "aggressive_growth",
    "rules": {
      "bid_increase_on_high_roas": true,
      "auto_pause_high_acos": true,
      "dayparting_enabled": true
    }
  },
  "historical_period": {
    "start": "2025-10-01",
    "end": "2025-12-31"
  },
  "baseline": "actual_performance"
}
```

**Output**:
```json
{
  "backtest_results": {
    "hypothetical_performance": {
      "total_sales": 125000,
      "total_spend": 28000,
      "acos": 0.224,
      "roas": 4.46
    },
    "actual_performance": {
      "total_sales": 110000,
      "total_spend": 30000,
      "acos": 0.273,
      "roas": 3.67
    },
    "improvement": {
      "sales": "+13.6%",
      "spend": "-6.7%",
      "acos": "-17.9%",
      "roas": "+21.5%"
    },
    "confidence": 0.85,
    "notes": "Strategy would have significantly outperformed during Q4"
  }
}
```

## Scenario Analysis

### Market Condition Scenarios

```json
{
  "action": "scenario_analysis",
  "base_case": {
    "daily_budget": 100,
    "avg_cpc": 1.50,
    "conversion_rate": 0.10
  },
  "scenarios": [
    {
      "name": "high_competition",
      "changes": {"avg_cpc": "+20%", "conversion_rate": "-5%"}
    },
    {
      "name": "budget_cut",
      "changes": {"daily_budget": "-30%"}
    },
    {
      "name": "seasonal_surge",
      "changes": {"conversion_rate": "+50%", "avg_cpc": "+10%"}
    }
  ],
  "metrics": ["sales", "acos", "profit"]
}
```

**Output**:
```json
{
  "scenario_results": [
    {
      "scenario": "base_case",
      "sales": 3000,
      "acos": 0.25,
      "profit": 750
    },
    {
      "scenario": "high_competition",
      "sales": 2550,
      "acos": 0.31,
      "profit": 450,
      "impact_vs_base": {"sales": "-15%", "profit": "-40%"}
    },
    {
      "scenario": "budget_cut",
      "sales": 2100,
      "acos": 0.24,
      "profit": 600,
      "impact_vs_base": {"sales": "-30%", "profit": "-20%"}
    },
    {
      "scenario": "seasonal_surge",
      "sales": 4200,
      "acos": 0.27,
      "profit": 1050,
      "impact_vs_base": {"sales": "+40%", "profit": "+40%"}
    }
  ],
  "recommendation": "Prepare for high_competition scenario by increasing budget flexibility"
}
```

## Risk Analysis

### Value at Risk (VaR)

```json
{
  "action": "calculate_var",
  "portfolio": ["campaign_123", "campaign_456", "campaign_789"],
  "confidence_level": 0.95,
  "time_horizon_days": 30
}
```

**Output**:
```json
{
  "value_at_risk": {
    "var_95": -2500,
    "interpretation": "95% confident losses won't exceed $2,500 in next 30 days",
    "expected_shortfall": -3200,
    "worst_case_1pct": -4800
  },
  "risk_contributors": [
    {"campaign": 123, "contribution": 0.45},
    {"campaign": 456, "contribution": 0.35},
    {"campaign": 789, "contribution": 0.20}
  ]
}
```

## Sandbox Environment

### Safe Testing Mode

```json
{
  "action": "create_sandbox",
  "clone_from": "campaign_123",
  "test_duration_days": 7,
  "test_strategy": {
    "bid_multiplier": 1.25,
    "budget_increase": 50
  }
}
```

**Features**:
- No real budget spent
- Uses historical data patterns
- Simulates market responses
- Full rollback capability
- Side-by-side comparison with live campaign

## Usage Patterns

### Pattern 1: Pre-Launch Forecast

```
USER: "Should I launch this new campaign with $100/day budget?"

SIMULATION LAB:
1. Run Monte Carlo with 10,000 iterations
2. Forecast 30-day outcomes
3. Results:
   - Expected sales: $4,500 ± $1,200
   - Expected ACoS: 24% ± 6%
   - Probability of profitability: 87%
   - Worst case (5%): Loss of $450
4. Recommendation: "Launch recommended. 87% chance of profit, limited downside risk."
```

### Pattern 2: Strategy Comparison

```
USER: "Which strategy is better: aggressive or conservative?"

SIMULATION LAB:
1. Backtest both on last 90 days
2. Aggressive: +18% sales, +12% spend, 22% ACoS
3. Conservative: +5% sales, -8% spend, 19% ACoS
4. Run forward simulation for next 30 days
5. Recommendation: "Aggressive for growth, conservative for efficiency"
```

### Pattern 3: Risk Assessment

```
USER: "What's my downside risk if competition increases?"

SIMULATION LAB:
1. Scenario: CPC +25%, CVR -10%
2. Monte Carlo: 10,000 iterations
3. Results:
   - VaR (95%): -$2,800
   - Expected loss: -$1,200
   - Recovery time: 14 days
4. Recommendation: "Maintain $3,000 reserve budget for this scenario"
```

## Database Schema

```sql
-- From server/updates/05_tier2_meta_skills_tables.sql

simulation_runs (
  simulation_type,    -- 'monte_carlo', 'backtest', 'scenario'
  input_parameters,
  iterations,
  results,
  confidence_intervals,
  created_at
)

backtest_results (
  strategy_name,
  test_period_start,
  test_period_end,
  hypothetical_performance,
  actual_performance,
  created_at
)
```

## Integration with Other Skills

**Feeds from**:
- **memory-palace**: Historical data for backtesting
- **evolution-engine**: Test evolved strategies before deployment
- **grok-admaster-operator**: Live campaign data

**Feeds to**:
- **consciousness-engine**: Log simulation insights
- **narrative-architect**: Generate scenario reports

## Files

```
.agent/skills/simulation-lab/
├── SKILL.md
├── scripts/
│   ├── monte_carlo_simulator.py  # Probabilistic forecasting
│   ├── backtester.py              # Historical testing
│   └── risk_analyzer.py           # VaR and risk metrics
└── resources/
    └── market_scenarios.json      # Pre-defined scenarios
```

## Example Invocation

```
USER: "I'm thinking of doubling my budget. What could happen?"

SIMULATION LAB:
1. Current: $100/day, 3000 sales/month, 25% ACoS
2. Simulate: $200/day budget
3. Monte Carlo: 10,000 iterations
4. Results:
   - Sales: 5,400 ± 800 (80% likely between 4,600-6,200)
   - ACoS: 27% ± 4% (increased competition effect)
   - Profit: +$1,200/month (median)
   - Risk: 15% chance of worse performance than current
5. Recommendation: "Budget increase likely profitable, but expect 2% ACoS increase due to diminishing returns. Consider phased rollout: +50% for 2 weeks, then evaluate."
```

## Safety Features

- All simulations clearly marked as "hypothetical"
- Confidence intervals always provided
- Worst-case scenarios highlighted
- No automatic deployment from simulations
- User approval required for strategy changes

## Notes

- Simulations use historical patterns but can't predict black swan events
- Higher iteration counts = more accurate but slower
- Backtests assume market conditions were similar
- Always combine simulation insights with human judgment

---

**This skill lets you test before you invest, turning uncertainty into calculated risk.**
