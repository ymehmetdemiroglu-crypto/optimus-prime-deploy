# Phase 5: ML Model Training - Implementation Summary

## ✅ Completed Components

### 1. Bid Optimizer (`ml/bid_optimizer.py`)

**Gradient Boosting Regressor** for bid prediction:

**Features Used (28 total):**
- Rolling metrics (CTR, conversion rate, ACoS, ROAS, CPC - 7/14/30d)
- Trend indicators (spend, sales, CTR, momentum)
- Competition features (CPC/impression volatility)
- Seasonality (day_of_week, is_weekend, month, quarter, is_q4)
- Keyword-specific (current_bid, revenue_per_click, data_maturity)

**Methods:**
- `train()` - Train from historical data
- `predict_bid()` - Get optimal bid prediction
- `batch_predict()` - Optimize multiple keywords
- `get_feature_importance()` - Explain model decisions

**Rule-Based Fallback:**
When no trained model exists:
1. RPC × Target ACoS = optimal bid
2. Reduce bid if ACoS > target
3. Increase bid if outperforming

---

### 2. RL Agent (`ml/rl_agent.py`)

**Q-Learning Agent** for real-time bid adjustments:

**State Space:**
| Bucket | Values | Meaning |
|--------|--------|---------|
| ACoS | 0-4 | Very low → Very high |
| Trend | 0-2 | Declining → Improving |
| Budget | 0-2 | Underspent → Overspent |
| Competition | 0-2 | Low → High |

**Action Space:**
| Action | Multiplier | Description |
|--------|------------|-------------|
| 0 | 0.80 | Decrease 20% |
| 1 | 0.90 | Decrease 10% |
| 2 | 0.95 | Decrease 5% |
| 3 | 1.00 | Maintain |
| 4 | 1.05 | Increase 5% |
| 5 | 1.10 | Increase 10% |
| 6 | 1.20 | Increase 20% |

**Reward Function:**
- +1.0: Achieving target ACoS
- +0.5: Sales improvement
- +0.3: ACoS improvement
- -0.5: Missing ACoS target
- -0.5: Sales decline

---

### 3. Performance Forecaster (`ml/forecaster.py`)

**Holt's Linear Trend Method** for time series:

**Capabilities:**
- 7-day horizon forecasting
- Confidence intervals (95%)
- Trend detection (up/down/stable)
- Budget pacing analysis
- Anomaly detection (z-score)

**Metrics Forecasted:**
- Impressions, Clicks, Spend, Sales, Orders
- Derived: ACoS (from spend/sales ratio)

---

### 4. Training Pipeline (`ml/training.py`)

**Orchestrates model training from database:**

**Methods:**
- `prepare_training_data()` - Extract labeled training set
- `train_bid_optimizer()` - Train GB model
- `train_rl_agent()` - Train RL from history
- `evaluate_models()` - Compare model predictions
- `get_campaign_forecast()` - Generate forecasts

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ml/train/bid-optimizer` | POST | Train GB model |
| `/ml/train/rl-agent` | POST | Train RL agent |
| `/ml/predict/bid/{keyword_id}` | GET | Get bid predictions |
| `/ml/predict/campaign/{id}/keywords` | GET | Batch predictions |
| `/ml/forecast/campaign/{id}` | GET | Performance forecast |
| `/ml/evaluate/{campaign_id}` | GET | Evaluate all models |
| `/ml/model-status` | GET | Check model status |
| `/ml/feature-importance` | GET | View feature weights |

---

## 📊 Example API Responses

### Bid Prediction
```json
{
  "keyword_id": 123,
  "current_bid": 1.50,
  "gradient_boosting": {
    "predicted_bid": 1.25,
    "confidence": 0.85,
    "expected_acos": 22.5,
    "expected_roas": 4.44,
    "reasoning": "ML model prediction based on historical patterns"
  },
  "reinforcement_learning": {
    "recommended_bid": 1.35,
    "action": "Decrease 10%",
    "confidence": "high"
  },
  "recommended_bid": 1.30
}
```

### Campaign Forecast
```json
{
  "campaign_id": 1,
  "historical_days": 60,
  "forecast_horizon": 7,
  "metrics": {
    "sales": {
      "current": 1250.00,
      "forecast": [1280, 1310, 1295, 1340, 1365, 1390, 1420],
      "trend": "up"
    },
    "acos": {
      "current": 24.5,
      "forecast": [24.2, 23.9, 24.0, 23.7, 23.5, 23.3, 23.0],
      "trend": "down"
    }
  }
}
```

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────────────┐
│                 ENSEMBLE SYSTEM                  │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌───────────────┐    ┌───────────────┐        │
│  │   Gradient    │    │      RL       │        │
│  │   Boosting    │    │    Agent      │        │
│  │  (Strategic)  │    │  (Tactical)   │        │
│  └───────┬───────┘    └───────┬───────┘        │
│          │                    │                 │
│          └────────┬───────────┘                │
│                   │                             │
│           ┌───────▼───────┐                     │
│           │   Ensemble    │                     │
│           │   Average     │                     │
│           └───────┬───────┘                     │
│                   │                             │
│           ┌───────▼───────┐                     │
│           │  Final Bid    │                     │
│           │ Recommendation│                     │
│           └───────────────┘                     │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 📁 New Files Created

```
server/app/amazon_ppc_optimizer/ml/
├── __init__.py
├── bid_optimizer.py    # Gradient Boosting model
├── rl_agent.py         # Q-Learning agent
├── forecaster.py       # Time series forecasting
├── training.py         # Training pipeline
└── router.py           # API endpoints
```

---

## 🔧 Dependencies Added

```
scikit-learn>=1.3.0
numpy>=1.24.0
```

---

## 🚀 Usage Workflow

1. **Ingest Data** → `/ingestion/sync-all`
2. **Compute Features** → `/features/batch-compute`
3. **Train Models** → `/ml/train/bid-optimizer` + `/ml/train/rl-agent`
4. **Get Predictions** → `/ml/predict/bid/{keyword_id}`
5. **Apply Optimizations** → (Phase 6)

---

**Status: Phase 5 Complete ✅**  
**Ready for: Phase 6 - Real-time Optimization Engine**
