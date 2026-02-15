# ML-Integrated Optimization Engine

## 🎯 Overview

The Advanced Optimization Engine integrates all specialized ML capabilities into a unified system for intelligent PPC optimization. It combines ensemble predictions, anomaly detection, keyword health analysis, market intelligence, and more into a single coherent optimization workflow.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ADVANCED OPTIMIZATION ENGINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │   Feature     │  │    Model      │  │   Specialized │  │   Market     │ │
│  │  Engineering  │──│   Ensemble    │──│      ML       │──│ Intelligence │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
│         │                  │                  │                  │          │
│         v                  v                  v                  v          │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    INTELLIGENT PLAN GENERATION                         │ │
│  ├───────────────────────────────────────────────────────────────────────┤ │
│  │  • Anomaly Detection      • Keyword Health Analysis                   │ │
│  │  • Performance Forecasting • Market Condition Analysis                │ │
│  │  • Keyword Segmentation   • Competitor Bid Estimation                 │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         v                          v                          v            │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│  │    Bid      │           │   Budget    │           │  Keyword    │       │
│  │  Actions    │           │  Actions    │           │  Actions    │       │
│  └─────────────┘           └─────────────┘           └─────────────┘       │
│                                    │                                        │
│                                    v                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                   INTELLIGENT EXECUTION                                │ │
│  │  • Auto-approve by confidence  • Respect anomaly alerts              │ │
│  │  • ML-aware decision making    • Priority-based execution            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Intelligence Levels

The system supports four intelligence levels, each progressively more sophisticated:

### **BASIC**
- Single ML model (Gradient Boosting)
- Rule-based decisions
- Fast, deterministic

### **STANDARD**
- Model ensemble (GB + RL + Deep + Bandit)
- Weighted voting
- Confidence-based decisions

### **ADVANCED**
- Full ensemble + voting
- Anomaly detection
- Keyword health analysis
- Performance forecasting
- Segment-based optimization

### **AUTONOMOUS**
- All advanced features
- Market intelligence analysis
- Competitor bid estimation
- Opportunity detection
- Self-adaptive learning

---

## 📊 Components Integrated

### **Model Ensemble**
- `BidOptimizer` (Gradient Boosting)
- `PPCRLAgent` (Reinforcement Learning)
- `DeepBidOptimizer` (Neural Network)
- `BidBanditOptimizer` (Multi-Armed Bandits)
- `StackingEnsemble` (Meta-learner)
- `VotingEnsemble` (Consensus voting)

### **Anomaly Detection**
- `IsolationForest` (Unsupervised)
- `ZScoreDetector` (Statistical)
- `ChangePointDetector` (CUSUM)

### **Keyword Health**
- `KeywordHealthAnalyzer` (Multi-factor scoring)
- `KeywordLifecyclePredictor` (Stage prediction)

### **Market Intelligence**
- `MarketAnalyzer` (Market conditions)
- `CompetitorBidEstimator` (Bid inference)
- `AuctionSimulator` (Monte Carlo)

### **Segmentation**
- `KeywordSegmenter` (Performance clusters)
- `PerformanceSegmenter` (Campaign tiers)

### **Forecasting**
- `LSTMForecaster` (Deep learning)
- `PerformanceForecaster` (Holt-Winters)

### **Budget Optimization**
- `BayesianBudgetOptimizer` (Gaussian Process)
- `SpendPacer` (Hourly scheduling)

---

## 🔌 API Endpoints

### Core Optimization

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/optimization/advanced/generate-plan` | Generate intelligent plan |
| POST | `/api/v1/optimization/advanced/execute` | Execute with ML decision-making |
| POST | `/api/v1/optimization/advanced/quick-optimize/{id}` | Quick optimization |

### Alerts & Rules

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/optimization/advanced/alerts/{id}` | Campaign alerts |
| GET | `/api/v1/optimization/advanced/alerts` | All alerts |
| GET | `/api/v1/optimization/advanced/rules` | List rules |
| POST | `/api/v1/optimization/advanced/rules/toggle` | Toggle rule |

### Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/optimization/intelligence-levels` | Available levels |
| GET | `/api/v1/optimization/strategies` | Available strategies |

---

## 📝 Request/Response Examples

### Generate Intelligent Plan

**Request:**
```json
POST /api/v1/optimization/advanced/generate-plan
{
  "campaign_id": 1,
  "strategy": "balanced",
  "target_acos": 25.0,
  "target_roas": 4.0,
  "intelligence_level": "advanced"
}
```

**Response:**
```json
{
  "campaign_id": 1,
  "campaign_name": "Summer Sale Campaign",
  "strategy": "balanced",
  "intelligence_level": "advanced",
  "summary": {
    "total_actions": 15,
    "bid_increases": 5,
    "bid_decreases": 8,
    "keywords_to_pause": 2,
    "high_priority_actions": 3,
    "avg_confidence": 0.78,
    "anomalies_found": 2,
    "at_risk_keywords": 4
  },
  "anomalies_detected": [
    {
      "type": "spend_spike",
      "severity": "critical",
      "metric": "spend",
      "expected": 100,
      "actual": 250,
      "deviation": 3.2,
      "message": "Unusual spend increase detected"
    }
  ],
  "keyword_health": {
    "total_analyzed": 50,
    "status_distribution": {
      "excellent": 10,
      "good": 25,
      "at_risk": 10,
      "declining": 4,
      "critical": 1
    },
    "at_risk_count": 15,
    "avg_health_score": 72.5
  },
  "segment_analysis": {
    "segments": [
      {"name": "Stars", "count": 8, "action": "Scale aggressively"},
      {"name": "Potential", "count": 12, "action": "Increase bids 10-20%"},
      {"name": "Underperformers", "count": 15, "action": "Decrease bids"}
    ]
  },
  "forecast": {
    "horizon_days": 7,
    "sales_forecast": [450, 480, 460, 490, 500, 480, 470]
  },
  "actions": [
    {
      "action_type": "bid_decrease",
      "entity_type": "keyword",
      "entity_id": 123,
      "current_value": 1.50,
      "recommended_value": 1.20,
      "change_percent": -20.0,
      "confidence": 0.85,
      "reasoning": "Ensemble: $1.20, Voting: $1.18 (decrease) [AT RISK - Conservative adjustment]",
      "priority": 9
    }
  ]
}
```

### Execute with ML-Aware Decision Making

**Request:**
```json
POST /api/v1/optimization/advanced/execute
{
  "campaign_id": 1,
  "strategy": "balanced",
  "intelligence_level": "advanced"
}

// Body (separate):
{
  "dry_run": false,
  "auto_approve_confidence": 0.8,
  "respect_anomalies": true
}
```

**Response:**
```json
{
  "dry_run": false,
  "executed": [
    {
      "action_type": "bid_decrease",
      "entity_id": 123,
      "from": 1.50,
      "to": 1.20,
      "confidence": 0.85,
      "status": "executed"
    }
  ],
  "skipped": [
    {
      "entity_id": 456,
      "reason": "Critical anomaly detected - skipping bid increases"
    }
  ],
  "summary": {
    "total": 15,
    "executed": 10,
    "skipped": 5,
    "errors": 0
  },
  "ml_insights": {
    "anomalies_considered": 2,
    "keyword_health_considered": 50,
    "forecast_used": true
  }
}
```

---

## 🔧 ML-Enhanced Rules

The Advanced Rule Engine includes these smart rules:

### Threshold Rules
- **High ACoS Alert**: ACoS > 50% → Bid decrease
- **Low ROAS Warning**: ROAS < 2.0 → Bid decrease
- **Budget Utilization**: > 95% → Budget increase

### ML-Based Rules
- **Spend Anomaly Detection**: Isolation Forest + Z-Score
- **Performance Anomaly**: Multi-dimensional detection
- **Keyword Health Decline**: Health score monitoring
- **Negative Forecast Alert**: Sales decline predicted

### Trend Rules
- **Declining CTR Trend**: > 20% drop
- **Rising CPC Trend**: > 30% increase

---

## 🛡️ Safety Features

### Anomaly Respect
When anomalies are detected:
- Bid increases are blocked during critical anomalies
- Budget changes require higher confidence
- Human review is recommended

### At-Risk Keyword Protection
- Keywords flagged as "declining" or "critical" get conservative adjustments
- Automatic bid reduction for at-risk keywords

### Confidence Thresholds
- Actions below confidence threshold require explicit approval
- Higher intelligence levels have higher confidence requirements

---

## 📁 Files

```
server/app/amazon_ppc_optimizer/optimization/
├── engine.py              # Base optimization engine
├── advanced_engine.py     # ML-integrated engine (NEW)
├── rules.py               # Base rule engine
├── advanced_rules.py      # ML-enhanced rules (NEW)
├── scheduler.py           # Background scheduling
├── router.py              # API endpoints (UPDATED)
└── __init__.py            # Exports (UPDATED)
```

---

## 🚀 Usage Recommendations

1. **Start with STANDARD** intelligence for balanced performance
2. **Use ADVANCED** when you have 30+ days of data
3. **Enable AUTONOMOUS** for mature campaigns with stable performance
4. **Always respect anomalies** in production
5. **Set auto_approve_confidence to 0.85+** for live execution
6. **Review at-risk keywords** before bulk optimization

---

**Status: Integration Complete ✅**
