---
name: consciousness-engine
description: Self-awareness and introspection layer. Monitors system performance, tracks decision accuracy, generates audit trails, calibrates confidence, and provides proactive self-diagnostics.
---

# Consciousness Engine Skill

The **Consciousness Engine** provides self-awareness to Optimus Pryme. It tracks what the system knows, how well it's performing, and proactively identifies when something isn't working as expected.

## Core Capabilities

### 1. **Performance Introspection**
- Track which ML models are actually being used vs. sitting idle
- Measure feature utilization rates
- Identify bottlenecks in optimization pipelines
- Monitor API response times and failure rates

### 2. **Confidence Calibration**
- Compare predicted outcomes vs. actual results
- Adjust confidence scores based on historical accuracy
- Flag when system is overconfident or underconfident
- Per-model accuracy tracking

### 3. **Decision Audit Trails**
- Log every autonomous decision with full context
- Natural language explanations: "Why did I choose X over Y?"
- Traceable decision paths for compliance
- Timeline reconstruction of actions

### 4. **Self-Diagnostics**
- "My bid predictions have been off by 15% this week"
- "Users are rejecting 80% of my keyword recommendations"
- Proactive problem identification before user notices
- Performance degradation alerts

### 5. **Feature Attribution**
- Which features provide the most value?
- What insights are users actually using?
- Dead code and unused feature detection

## Decision Audit Example

```json
{
  "decision_id": "dec_20260205_001",
  "decision_type": "bid_optimization",
  "timestamp": "2026-02-05T11:00:00Z",
  "options_considered": [
    {"action": "increase_bid", "value": 1.80, "confidence": 0.72},
    {"action": "decrease_bid", "value": 1.20, "confidence": 0.65},
    {"action": "no_change", "value": 1.50, "confidence": 0.58}
  ],
  "chosen_option": "increase_bid",
  "reasoning": "Ensemble model predicted 0.72 confidence for bid increase. Historical data shows this keyword responds well to bid increases (+12% sales on average). Current ACoS of 18% is below target of 25%, allowing room for investment.",
  "confidence": 0.72,
  "actual_outcome": {
    "acos": 20.5,
    "sales_change": "+9%",
    "prediction_accuracy": 0.89
  }
}
```

## Performance Tracking

```json
{
  "model_name": "bid_optimizer_ensemble",
  "period": "last_7_days",
  "metrics": {
    "predictions_made": 1247,
    "average_confidence": 0.76,
    "actual_accuracy": 0.71,
    "calibration_error": 0.05,
    "trend": "stable"
  },
  "diagnosis": "Model is well-calibrated. Slight overconfidence (+5%) but within acceptable range.",
  "recommendation": "No action needed"
}
```

## Self-Diagnostic Alert

```json
{
  "alert_id": "diag_202602051100",
  "severity": "warning",
  "component": "keyword_recommender",
  "issue": "User rejection rate increased to 78% (baseline: 20%)",
  "context": {
    "recent_changes": ["Updated to more aggressive strategy 3 days ago"],
    "affected_campaigns": 5,
    "user_feedback_pattern": "Consistently rejecting bid increases >20%"
  },
  "hypothesis": "Recent strategy update misaligned with user's conservative preferences",
  "suggested_action": "Revert to previous strategy or consult memory-palace for user preferences"
}
```

## API Operations

### Get Decision Audit

```json
{
  "action": "get_decision_audit",
  "decision_id": "dec_20260205_001"
}
```

### Track Model Performance

```json
{
  "action": "track_performance",
  "model_name": "bid_optimizer_ensemble",
  "prediction": {...},
  "actual_outcome": {...}
}
```

### Run Self-Diagnostic

```json
{
  "action": "run_diagnostic",
  "components": ["all"] // or specific: ["bid_optimizer", "anomaly_detector"]
}
```

**Response**:
```json
{
  "overall_health": "good",
  "issues_found": 1,
  "diagnostics": [
    {
      "component": "keyword_recommender",
      "status": "degraded",
      "details": {...}
    }
  ]
}
```

## Usage Patterns

### Pattern 1: Post-Decision Learning

```
AFTER DECISION:
1. Log decision with full context
2. Wait for actual outcome (24-48 hours)
3. Compare prediction vs. reality
4. Update confidence calibration
5. Store in memory-palace if significant pattern
```

### Pattern 2: Proactive Health Check

```
DAILY CRON:
1. Run self-diagnostic on all components
2. Check model accuracy trends
3. Identify degradation early
4. Alert user if critical
5. Auto-adjust if within parameters
```

### Pattern 3: "Why Did You Do That?"

```
USER: "Why did you decrease this bid?"
CONSCIOUSNESS ENGINE:
1. Retrieve decision audit for that bid change
2. Extract reasoning and context
3. Present in natural language
4. Include confidence score and alternatives considered
```

## Database Schema

```sql
-- From server/updates/04_meta_skills_tables.sql

decision_audit (
  decision_type,
  options_considered,
  chosen_option,
  reasoning,
  confidence,
  actual_outcome,
  created_at
)

model_performance_tracking (
  model_name,
  prediction_accuracy,
  usage_count,
  avg_confidence,
  last_calibrated,
  performance_trend
)
```

## Integration with Other Skills

**Feeds data to**:
- **memory-palace**: Store successful decision patterns
- **evolution-engine**: Identify underperforming strategies
- **meta-learner**: Adjust learning rates based on accuracy
- **narrative-architect**: Generate explanations for users

## Files

```
.agent/skills/consciousness-engine/
├── SKILL.md
├── scripts/
│   ├── decision_logger.py           # Audit trail generation
│   ├── confidence_calibrator.py     # Accuracy tracking
│   └── performance_monitor.py       # Self-diagnostics
└── tests/
    └── test_decision_logger.py
```

## Example Invocation

```
USER: "Why have my conversions been dropping?"

CONSCIOUSNESS ENGINE ACTION:
1. Run diagnostic on conversion-related components
2. Check recent decisions affecting conversion rate
3. Compare prediction accuracy vs. baseline
4. FIND: Bid optimization model accuracy dropped from 0.82 to 0.68
5. FIND: Recent market condition changes not captured in features
6. Report: "I detected that my bid predictions have been less accurate recently (68% vs. baseline 82%). This coincides with increased market competition that my current features don't fully capture. Recommend: Update feature engineering to include competitive density metrics."
```

## Key Principle

**The system that knows what it doesn't know is more trustworthy than one that doesn't.**

Consciousness Engine ensures Optimus Pryme:
- Never silently fails
- Learns from mistakes
- Admits uncertainty
- Self-corrects proactively

---

**This skill transforms Optimus Pryme from a "black box" into a transparent, self-aware system you can trust.**
