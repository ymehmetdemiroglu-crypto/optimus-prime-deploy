---
name: orchestrator-maestro
description: Multi-skill coordination and workflow intelligence. Chains skills together, resolves dependencies, maintains context across operations, and executes complex workflows through intelligent orchestration.
---

# Orchestrator Maestro Skill

The **Orchestrator Maestro** is a meta-skill that coordinates multiple skills into cohesive workflows. It acts as the conductor of the Optimus Pryme skill orchestra, determining execution order, managing dependencies, and preserving context.

## Core Capabilities

### 1. **Workflow Choreography**
- Determine optimal execution sequence for multi-skill operations
- Parallel execution when dependencies allow
- Sequential execution when required
- Context handoff between skills

### 2. **Dependency Resolution**
- Analyze skill requirements and outputs
- Build dependency graphs
- Detect circular dependencies
- Auto-resolve execution order

### 3. **Context Preservation**
- Maintain conversation state across skill invocations
- Pass relevant context between skills
- Aggregate results from multiple skills
- Track workflow state

### 4. **Goal Decomposition**
- Break complex user requests into skill sequences
- Map intents to appropriate skills
- Generate efficient execution plans
- Adaptive replanning on failures

### 5. **Workflow Templates**
- Pre-built workflow patterns for common operations
- Template library management
- Custom workflow creation
- Template versioning

## Workflow Templates

### Template: New Product Launch
```yaml
name: new_product_launch
description: Complete product launch workflow
skills:
  - skill: market-researcher
    input: { asin: "{{product_asin}}", depth: "comprehensive" }
    output_key: market_data
    
  - skill: amazon-listing-optimizer
    input: { asin: "{{product_asin}}", market_insights: "{{market_data}}" }
    output_key: listing_optimizations
    parallel: false
    
  - skill: grok-admaster-operator
    action: create_launch_campaign
    input: { 
      product: "{{product_asin}}",
      strategy: "aggressive",
      keywords: "{{market_data.keywords}}"
    }
    parallel: false
```

### Template: Performance Crisis Response
```yaml
name: crisis_response
description: Rapid response to performance issues
skills:
  - skill: grok-admaster-operator
    action: get_dashboard_summary
    output_key: current_metrics
    
  - skill: consciousness-engine
    action: diagnose_issues
    input: { metrics: "{{current_metrics}}" }
    output_key: diagnosis
    parallel: false
    
  - skill: simulation-lab
    action: test_recovery_strategies
    input: { diagnosis: "{{diagnosis}}" }
    output_key: recovery_plans
    parallel: false
    
  - skill: grok-admaster-operator
    action: execute_plan
    input: { plan: "{{recovery_plans.best_plan}}" }
    parallel: false
```

### Template: Quarterly Strategy Refresh
```yaml
name: quarterly_strategy_refresh
description: Comprehensive quarterly review and optimization
skills:
  - skill: memory-palace
    action: retrieve_quarterly_patterns
    output_key: historical_insights
    
  - skill: knowledge-synthesizer
    action: analyze_market_trends
    output_key: market_trends
    parallel: true
    
  - skill: competitive-intelligence
    action: competitor_analysis
    output_key: competitive_landscape
    parallel: true
    
  - skill: evolution-engine
    action: evolve_strategies
    input: {
      history: "{{historical_insights}}",
      trends: "{{market_trends}}",
      competition: "{{competitive_landscape}}"
    }
    output_key: new_strategies
    parallel: false
```

## API Operations

### Execute Workflow

**Endpoint**: Internal skill invocation (not HTTP API)

**Input**:
```json
{
  "workflow_template": "new_product_launch",
  "parameters": {
    "product_asin": "B0DWK3C1R7"
  },
  "execution_mode": "sequential|parallel_where_possible",
  "dry_run": false
}
```

**Output**:
```json
{
  "workflow_id": "wf_abc123",
  "status": "completed",
  "execution_time_seconds": 45.2,
  "skills_executed": 3,
  "results": {
    "market_data": {...},
    "listing_optimizations": {...},
    "campaign_created": {...}
  },
  "execution_log": [
    {
      "skill": "market-researcher",
      "status": "completed",
      "duration_seconds": 12.3
    }
  ]
}
```

### Create Custom Workflow

```json
{
  "action": "create_workflow_template",
  "name": "custom_optimization_flow",
  "description": "My custom workflow",
  "skills_sequence": [
    {
      "skill": "grok-admaster-operator",
      "action": "analyze_campaigns"
    },
    {
      "skill": "simulation-lab",
      "action": "test_strategies"
    }
  ]
}
```

## Usage Patterns

### Pattern 1: Chain Multiple Skills

**Scenario**: Research market → Optimize listing → Launch campaign

```
USER: "Launch my new charger product (ASIN B0DWK3C1R7) with optimized listings and PPC"

ORCHESTRATOR:
1. Invoke market-researcher(asin=B0DWK3C1R7)
2. Pass results to amazon-listing-optimizer
3. Create campaign via grok-admaster-operator
4. Return consolidated results
```

### Pattern 2: Parallel Execution

**Scenario**: Run independent analyses simultaneously

```
USER: "Give me a complete competitive overview"

ORCHESTRATOR:
1. Parallel execution:
   - market-researcher: trend analysis
   - competitive-intelligence: competitor data
   - knowledge-synthesizer: industry insights
2. Aggregate all results
3. Generate unified report
```

### Pattern 3: Adaptive Replanning

**Scenario**: Skill fails, orchestrator adapts

```
WORKFLOW:
1. Try primary strategy
2. IF FAILS → Fall back to alternative
3. Continue workflow with adjusted plan
```

## Integration with Other Skills

**All skills** can be orchestrated. The maestro:
- Reads skill SKILL.md files to understand capabilities
- Parses input/output specifications
- Builds dependency chains
- Executes in optimal order

## Database Schema

Workflows and executions are tracked in the database:

```sql
-- Defined in server/updates/04_meta_skills_tables.sql
skill_executions (
  workflow_id,
  skill_name,
  input_data,
  output_data,
  execution_order,
  status,
  started_at,
  completed_at
)

workflow_templates (
  name,
  description,
  skill_sequence,
  created_at
)
```

## Files

```
.agent/skills/orchestrator-maestro/
├── SKILL.md                          # This file
├── scripts/
│   ├── workflow_engine.py            # Core execution engine
│   ├── dependency_resolver.py        # Dependency graph builder
│   └── context_manager.py            # Context preservation
├── resources/
│   ├── workflow_templates.json       # Template library
│   └── skill_registry.json           # Known skills catalog
└── tests/
    └── test_workflow_engine.py       # Unit tests
```

## Example Invocation

```
USER: "I want to launch a new product. ASIN is B0ABC123. Do the full research, optimize the listing, and create an aggressive PPC campaign."

ORCHESTRATOR ACTION:
1. Detect intent: product launch
2. Load template: new_product_launch
3. Inject parameters: {asin: "B0ABC123"}
4. Execute workflow:
   - market-researcher → 12s
   - amazon-listing-optimizer → 8s
   - grok-admaster-operator (create campaign) → 5s
5. Return: "Product launch workflow completed. Market research shows high demand for your category. Listing optimized with 15 keyword improvements. Campaign 'Aggressive Launch B0ABC123' created with $50/day budget."
```

## Notes

- The orchestrator runs **within the agent context**, not as a separate service
- It coordinates existing skills but doesn't replace them
- Workflows are **reusable** and **version-controlled**
- Failed skills can trigger **fallback strategies**
- All executions are **logged** for analysis by consciousness-engine

---

**This skill enables seamless multi-skill operations, turning Optimus Pryme into a true autonomous system.**
