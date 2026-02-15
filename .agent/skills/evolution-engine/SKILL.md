---
name: evolution-engine
description: Continuous self-improvement through genetic algorithms. Evolves strategies, hyperparameters, and rules using mutation, crossover, and selection based on multi-objective fitness functions.
---

# Evolution Engine Skill

The **Evolution Engine** enables Optimus Pryme to evolve and improve its own strategies over time using genetic algorithms. It treats optimization strategies as "DNA" that can mutate, combine, and compete for survival based on performance.

## Core Capabilities

### 1. **Strategy Evolution**
- Treat bid strategies as genetic code
- Mutate parameters (bid adjustments, thresholds, timing)
- Crossover successful strategies to create hybrids
- Natural selection based on performance metrics
- Generational improvement tracking

### 2. **Model Hyperparameter Evolution**
- Evolve ML model configurations
- Optimize learning rates, regularization, architecture
- Multi-objective optimization (accuracy + speed + cost)
- Automatic hyperparameter tuning

### 3. **Rule Evolution**
- Dynamic threshold adjustment
- Evolve decision rules and conditions
- Adapt alert triggers based on outcomes
- Self-tuning automation rules

### 4. **Fitness Functions**
- Multi-objective optimization (ACoS, ROAS, profit, volume)
- Pareto frontier discovery
- Risk-adjusted fitness scoring
- User preference weighting

### 5. **Lineage Tracking**
- Track strategy genealogy (parent → child)
- Identify successful mutations
- Preserve high-performing genes
- Convergence detection

## Genetic Algorithm Workflow

```
GENERATION 0: Initial population of strategies
↓
EVALUATE: Run strategies, measure fitness
↓
SELECT: Keep top performers (elitism)
↓
CROSSOVER: Combine successful strategies
↓
MUTATE: Random variations
↓
GENERATION 1: New population
↓
REPEAT for N generations or until convergence
```

## Strategy DNA Example

```json
{
  "strategy_id": "strat_gen5_003",
  "generation": 5,
  "parent_ids": ["strat_gen4_001", "strat_gen4_007"],
  "dna": {
    "bid_adjustment_factor": 1.15,
    "acos_threshold": 0.25,
    "pause_threshold_days": 7,
    "min_conversions_required": 3,
    "dayparting_enabled": true,
    "dayparting_hours": [8, 9, 10, 18, 19, 20]
  },
  "fitness_score": 0.87,
  "metrics": {
    "acos": 0.22,
    "roas": 4.5,
    "total_sales": 15420,
    "profit": 3200
  },
  "mutation_type": "parameter_tweak",
  "status": "active"
}
```

## Mutation Types

### 1. Parameter Tweak
```python
# Small random adjustments
bid_adjustment *= random.uniform(0.95, 1.05)
```

### 2. Threshold Shift
```python
# Adjust decision boundaries
acos_threshold += random.uniform(-0.02, 0.02)
```

### 3. Feature Toggle
```python
# Enable/disable features
dayparting_enabled = not dayparting_enabled
```

### 4. Structural Change
```python
# Add/remove strategy components
if random.random() < 0.1:
    add_new_rule()
```

## Crossover Strategies

### Single-Point Crossover
```python
# Take first half from parent A, second half from parent B
child_dna = {
    **parent_a_dna[:split_point],
    **parent_b_dna[split_point:]
}
```

### Uniform Crossover
```python
# Randomly pick each gene from either parent
for gene in genes:
    child_dna[gene] = random.choice([parent_a[gene], parent_b[gene]])
```

### Weighted Blend
```python
# Blend based on fitness scores
weight_a = fitness_a / (fitness_a + fitness_b)
child_value = parent_a_value * weight_a + parent_b_value * (1 - weight_a)
```

## Fitness Function

```python
def calculate_fitness(strategy_results):
    """
    Multi-objective fitness with user preference weighting
    """
    # Normalize metrics to 0-1 scale
    acos_score = 1.0 - (acos / target_acos)
    roas_score = roas / target_roas
    profit_score = profit / max_profit_seen
    volume_score = sales / max_sales_seen
    
    # User preference weights (from memory-palace)
    weights = {
        "acos": 0.3,
        "roas": 0.3,
        "profit": 0.25,
        "volume": 0.15
    }
    
    # Weighted sum
    fitness = (
        weights["acos"] * acos_score +
        weights["roas"] * roas_score +
        weights["profit"] * profit_score +
        weights["volume"] * volume_score
    )
    
    # Penalty for risk
    if volatility > threshold:
        fitness *= 0.8
    
    return fitness
```

## API Operations

### Start Evolution Cycle

```json
{
  "action": "start_evolution",
  "population_size": 20,
  "generations": 10,
  "mutation_rate": 0.15,
  "crossover_rate": 0.7,
  "elitism_count": 2,
  "fitness_objectives": ["acos", "roas", "profit"]
}
```

### Get Best Strategy

```json
{
  "action": "get_best_strategy",
  "generation": "latest",
  "objective": "multi_objective"
}
```

**Response**:
```json
{
  "strategy_id": "strat_gen10_001",
  "generation": 10,
  "fitness_score": 0.92,
  "dna": {...},
  "lineage": ["gen0_005", "gen3_012", "gen7_003", "gen10_001"],
  "improvement_over_baseline": "+23%"
}
```

### Track Lineage

```json
{
  "action": "get_lineage",
  "strategy_id": "strat_gen10_001"
}
```

## Usage Patterns

### Pattern 1: Evolve Bid Strategy

```
INITIAL: 5 baseline bid strategies
GENERATION 1: Run all 5, measure performance
SELECT: Keep top 2 (fitness 0.75, 0.72)
CROSSOVER: Create 2 hybrids
MUTATE: Create 1 random variation
GENERATION 2: Evaluate new population of 5
REPEAT: 10 generations
RESULT: Best strategy has fitness 0.89 (+18% vs baseline)
```

### Pattern 2: Hyperparameter Tuning

```
OBJECTIVE: Optimize anomaly detection model
PARAMETERS: threshold, lookback_window, sensitivity
POPULATION: 15 configurations
EVOLVE: 20 generations
FITNESS: Detection accuracy + false positive rate
RESULT: Optimal config found at generation 12
```

### Pattern 3: Adaptive Rule Evolution

```
SCENARIO: Alert rules triggering too often
EVOLVE: Alert thresholds and conditions
FITNESS: True positive rate - false positive penalty
GENERATIONS: 5
RESULT: Reduced false alerts by 60%, maintained detection
```

## Database Schema

```sql
-- From server/updates/05_tier2_meta_skills_tables.sql

strategy_lineage (
  strategy_name,
  parent_id,          -- Reference to parent strategy
  generation,
  dna,                -- Strategy parameters as JSON
  fitness_score,
  mutation_type,
  status,             -- 'active', 'extinct', 'archived'
  created_at
)

evolution_cycles (
  generation_number,
  population_size,
  best_fitness,
  avg_fitness,
  diversity_score,    -- Genetic diversity measure
  completed_at
)
```

## Integration with Other Skills

**Feeds from**:
- **consciousness-engine**: Get model performance data for fitness
- **memory-palace**: Retrieve historical strategy outcomes
- **simulation-lab**: Test evolved strategies in sandbox

**Feeds to**:
- **grok-admaster-operator**: Deploy evolved strategies
- **memory-palace**: Store successful mutations as patterns

## Files

```
.agent/skills/evolution-engine/
├── SKILL.md
├── scripts/
│   ├── genetic_optimizer.py      # Core GA implementation
│   ├── fitness_evaluator.py      # Multi-objective fitness
│   └── lineage_tracker.py        # Genealogy tracking
└── resources/
    └── baseline_strategies.json  # Initial population
```

## Example Invocation

```
USER: "My current bid strategy isn't performing well. Can you evolve a better one?"

EVOLUTION ENGINE:
1. Load current strategy as baseline
2. Create population of 15 variations
3. Run evolution for 10 generations
4. Fitness = 0.6 ACoS + 0.4 ROAS
5. Generation 1: Best fitness 0.68
6. Generation 5: Best fitness 0.79
7. Generation 10: Best fitness 0.88
8. RESULT: "Evolved strategy improves fitness by 30%. Key mutations: 
   - Bid adjustment factor: 1.12 → 1.18
   - ACoS threshold: 0.25 → 0.22
   - Added dayparting (8am-10am, 6pm-8pm)
   Would you like to deploy this strategy?"
```

## Safety Mechanisms

- **Elitism**: Always preserve top 2 strategies
- **Diversity maintenance**: Prevent premature convergence
- **Bounds checking**: Parameters stay within safe ranges
- **Rollback capability**: Can revert to any previous generation
- **Dry-run testing**: Test evolved strategies before deployment

## Notes

- Evolution runs in background (doesn't block operations)
- Convergence typically occurs in 10-20 generations
- Diversity score prevents local optima trapping
- User can define custom fitness functions
- Extinct strategies are archived for future reference

---

**This skill enables Optimus Pryme to continuously improve itself, discovering strategies that humans might never think of.**
