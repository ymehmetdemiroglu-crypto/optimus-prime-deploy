# AI Integration - Quick Start Guide

## What Was Implemented

The Grok AdMaster now includes **4 Advanced ML Models** for precision optimization:

1.  **`ModelEnsemble`**: Combines Gradient Boosting, Deep Neural Networks, and Reinforcement Learning for precise bid predictions
2.  **`BidBanditOptimizer`**: Uses Thompson Sampling + UCB for rapid exploration of new keywords
3.  **`LSTMForecaster`**: Predicts future sales trends for proactive budget planning
4.  **`BayesianBudgetOptimizer`**: Suggests optimal daily budget using Bayesian optimization

## Database Tables Created

Run the SQL migration manually via **Supabase Dashboard → SQL Editor**:

```sql
-- Located at: server/updates/01_ai_tables.sql
-- Tables: model_registry, bandit_arms, prediction_logs, training_jobs
```

## How to Activate

### Option 1: Via Campaign Settings (Recommended)
Set a campaign's `ai_mode` to one of:
- `"advanced"` - Enables AI predictions with ensemble models
- `"autonomous"` - Full AI control with logging

### Option 2: Direct API Call
```python
from app.services.ai_integration import get_ai_bid_recommendation

# Example keyword data
keyword = {
    "id": 123,
    "keyword_text": "wireless headphones",
    "campaign_id": 1,
    "current_bid": 1.50,
    "impressions": 5000,
    "clicks": 150,
    "spend": 225,
    "sales": 900
}

# Get AI recommendation
result = await get_ai_bid_recommendation(
    keyword_data=keyword,
    target_acos=25.0,
    db_conn=db_connection  # asyncpg connection
)

print(result)
# Output:
# {
#   "recommended_bid": 1.34,
#   "confidence": 0.82,
#   "reasoning": "Ensemble: decrease by 10.7% based on GB, NN, RL consensus",
#   "model": "ModelEnsemble"
# }
```

## Files Modified

| File | Change |
|------|--------|
| `server/app/services/ai_integration.py` | NEW: Complete AI service with DB persistence |
| `server/app/services/ppc_optimizer.py` | MODIFIED: Routes `advanced`/`autonomous` modes to AI |
| `server/updates/01_ai_tables.sql` | NEW: Migration for 4 ML tables |
| `server/apply_ai_migration.py` | NEW: Python script to apply migration (needs manual DB access) |

## Next Steps

1.  **Execute Migration**: Run `01_ai_tables.sql` via Supabase Dashboard
2.  **Test**: Set a test campaign to `ai_mode = "advanced"`
3.  **Monitor**: Check `prediction_logs` table to see AI decisions being recorded

## Troubleshooting

**Q: Permission denied when running migration**
A: The `app_user` doesn't have CREATE TABLE privileges. Use Supabase Dashboard SQL Editor (runs as privileged user).

**Q: Models are slow to load**
A: Models use lazy initialization. First prediction will take ~2-3 seconds, then cached.

**Q: Want to use sync code without async/await**
A: For synchronous contexts, remove the `db_conn` parameter - predictions will work but won't be logged.
