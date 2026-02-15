# Advanced ML Models - Implementation Summary

## 🧠 Model Architecture

### 1. Deep Neural Network Optimizer (`deep_optimizer.py`)

**Architecture: Input → 128 → 64 → 32 → 16 → 1**

Custom implementation with:
- Leaky ReLU activations (prevents dead neurons)
- He weight initialization
- Feature normalization
- Batch gradient descent

**Features:**
- 28 input features (same as Gradient Boosting)
- Uncertainty estimation based on data maturity
- Model persistence (pickle saves)

```python
prediction, uncertainty = deep_optimizer.predict(features)
```

---

### 2. Multi-Armed Bandits (`bandits.py`)

**Three Bandit Algorithms:**

| Algorithm | Method | Use Case |
|-----------|--------|----------|
| **Thompson Sampling** | Beta distribution sampling | Fast exploration |
| **UCB (Upper Confidence Bound)** | Confidence-based selection | Optimal regret |
| **Contextual Bandit** | LinUCB with features | Personalized recommendations |

**Arm Space:** 11 bid multipliers from 0.5x to 1.5x

**Key Features:**
- Keyword-specific bandits for personalization
- Ensemble of all three algorithms
- Online learning from outcomes

```python
recommendation = bandit.select_bid_multiplier(features, keyword_id)
# Returns: {ensemble_multiplier, thompson, ucb, contextual}
```

---

### 3. LSTM Forecaster (`lstm_forecaster.py`)

**Architecture: 2-Layer LSTM → Dense Output**

- Sequence length: 14 days
- Hidden size: 32 units
- Multi-metric forecasting

**Metrics Forecasted:**
- Impressions, Clicks, Spend, Sales, Orders
- Derived: ACoS

**Additional Features:**
- `SeasonalDecomposer`: Trend + Seasonality + Residual decomposition
- 7-day weekly patterns

```python
forecast = lstm.forecast(historical_data, horizon=7)
# Returns predictions for all metrics
```

---

### 4. Bayesian Budget Optimizer (`bayesian_budget.py`)

**Gaussian Process Regression** for budget optimization:
- RBF (Radial Basis Function) kernel
- Expected Improvement acquisition function
- Uncertainty quantification

**Components:**

| Class | Purpose |
|-------|---------|
| `BayesianBudgetOptimizer` | Single campaign budget suggestions |
| `SpendPacer` | Hourly budget pacing (dayparting) |

```python
allocation = optimizer.suggest_budget(campaign_id, current_budget)
# Returns: {recommended_budget, expected_roi, confidence}
```

---

### 5. Ensemble System (`ensemble.py`)

**Three Ensemble Methods:**

#### ModelEnsemble (Weighted Average)
```
Final = Σ(weight_i × prediction_i)
```
- Gradient Boosting: 30%
- Deep NN: 25%
- RL Agent: 25%
- Bandit: 20%

#### StackingEnsemble (Meta-Learner)
- Base predictions → Linear regression → Final prediction
- Learns optimal combining weights from outcomes

#### VotingEnsemble (Majority Vote)
- Vote on direction (increase/decrease/maintain)
- Median for magnitude
- Consensus-based confidence

---

## 🔌 New API Endpoints

### Predictions

| Endpoint | Description |
|----------|-------------|
| `GET /ml/predict/ensemble/{keyword_id}` | Full ensemble prediction |
| `GET /ml/predict/voting/{keyword_id}` | Voting ensemble |
| `GET /ml/predict/bandit/{keyword_id}` | Multi-armed bandit |

### Training

| Endpoint | Description |
|----------|-------------|
| `POST /ml/train/deep-optimizer` | Train deep neural network |

### Forecasting

| Endpoint | Description |
|----------|-------------|
| `GET /ml/forecast/lstm/{campaign_id}` | LSTM forecast |
| `GET /ml/decompose/{campaign_id}/{metric}` | Time series decomposition |

### Budget Optimization

| Endpoint | Description |
|----------|-------------|
| `POST /ml/optimize/budget` | Portfolio optimization |
| `GET /ml/optimize/budget/{campaign_id}` | Single campaign suggestion |
| `POST /ml/dayparting/learn` | Learn hourly patterns |
| `GET /ml/dayparting/{campaign_id}` | Get dayparting schedule |

---

## 📊 Example API Responses

### Ensemble Prediction
```json
{
  "keyword_id": 123,
  "current_bid": 1.50,
  "ensemble_prediction": 1.32,
  "confidence": 0.78,
  "model_predictions": {
    "gradient_boost": 1.25,
    "deep_nn": 1.40,
    "rl_agent": 1.35,
    "bandit": 1.28
  },
  "model_weights": {
    "gradient_boost": 0.30,
    "deep_nn": 0.25,
    "rl_agent": 0.25,
    "bandit": 0.20
  },
  "reasoning": "Ensemble of 4 models..."
}
```

### LSTM Forecast
```json
{
  "campaign_id": 1,
  "model": "lstm",
  "historical_days": 60,
  "horizon": 7,
  "metrics": {
    "sales": {
      "forecast": [1250, 1280, 1310, 1295, 1340, 1365, 1390],
      "trend": "up"
    },
    "acos": {
      "forecast": [24.5, 24.2, 23.9, 24.0, 23.7, 23.5, 23.3],
      "trend": "down"
    }
  }
}
```

### Budget Optimization
```json
{
  "campaign_id": 1,
  "current_budget": 100.00,
  "recommended_budget": 115.00,
  "expected_roi": 3.2,
  "confidence": 0.72,
  "reasoning": "Bayesian optimization suggests..."
}
```

### Dayparting Schedule
```json
{
  "campaign_id": 1,
  "daily_budget": 100.00,
  "schedule": [
    {"hour": 0, "budget": 2.50, "multiplier": 0.60},
    {"hour": 9, "budget": 6.25, "multiplier": 1.50},
    {"hour": 12, "budget": 5.00, "multiplier": 1.20},
    {"hour": 20, "budget": 7.50, "multiplier": 1.80}
  ]
}
```

---

## 📁 Files Created

```
server/app/amazon_ppc_optimizer/ml/
├── __init__.py          # Updated with all exports
├── bid_optimizer.py     # Gradient Boosting (Phase 5)
├── rl_agent.py          # Q-Learning (Phase 5)
├── forecaster.py        # Holt's method (Phase 5)
├── training.py          # Training pipeline (Phase 5)
├── router.py            # Updated with new endpoints
│
│ # 🆕 Advanced Models
├── deep_optimizer.py    # Deep Neural Network
├── bandits.py           # Thompson, UCB, Contextual
├── lstm_forecaster.py   # LSTM + Seasonal Decomposition
├── bayesian_budget.py   # GP Budget Optimizer + SpendPacer
└── ensemble.py          # Model, Stacking, Voting Ensembles
```

---

## 🔧 Model Comparison

| Model | Type | Pros | Cons |
|-------|------|------|------|
| Gradient Boosting | Tree-based | Interpretable, robust | Static predictions |
| Deep NN | Neural | Pattern recognition | Needs more data |
| RL Agent | Reinforcement | Adapts to change | Slow learning |
| Bandits | Exploration | Fast adaptation | Simple actions |
| LSTM | Sequence | Temporal patterns | Complex training |
| Bayesian GP | Probabilistic | Uncertainty | Computationally heavy |

---

## 🚀 Usage Recommendations

1. **Default**: Use `ModelEnsemble` for balanced predictions
2. **Low Data**: Use `Gradient Boosting` with rule-based fallback
3. **High Volume**: Use `Deep NN` for complex patterns
4. **Real-time**: Use `Multi-Armed Bandit` for quick adaptation
5. **Forecasting**: Use `LSTM` for time series prediction
6. **Budget**: Use `Bayesian GP` for portfolio optimization

---

**Status: Advanced ML Models Complete ✅**
