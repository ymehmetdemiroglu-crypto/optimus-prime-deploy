# Quick Start — Anomaly Detection Integration

## Prerequisites
- ✅ PostgreSQL database running
- ✅ Phase 6 ML modules implemented (`advanced_anomaly.py`)
- ✅ Tests passing (17/17)

---

## Installation Steps

### 1. Apply Database Migration

```bash
cd server
python apply_migration_anomaly.py
```

**Expected Output:**
```
[Migration] Creating anomaly detection tables...
[Migration] Creating indexes for anomaly_alerts...
[Migration] Creating indexes for anomaly_history...
[Migration] Creating indexes for anomaly_training_data...
[Migration] Creating RLS policies...
[Migration] Creating helper functions...
[Migration] ✅ Anomaly detection tables created successfully!
```

---

### 2. Register API Router

**Edit:** `server/app/main.py`

```python
# Add import
from app.modules.amazon_ppc.anomaly.router import router as anomaly_router

# Register router with other routers
app.include_router(anomaly_router, prefix="/api")
```

---

### 3. (Optional) Setup Background Monitoring

**Edit:** `server/app/main.py`

```python
from app.modules.amazon_ppc.anomaly.tasks import setup_anomaly_monitoring_tasks

@app.on_event("startup")
async def startup_event():
    # ... existing startup code ...
    
    # Setup anomaly monitoring
    scheduler = setup_anomaly_monitoring_tasks()
    if scheduler:
        logger.info("Anomaly monitoring tasks scheduled")
```

**Install APScheduler:**
```bash
pip install apscheduler
```

---

### 4. Test the Integration

#### Test API Endpoints

```bash
# Start server
cd server
uvicorn app.main:app --reload

# Test anomaly detection (use actual profile_id from your database)
curl -X POST "http://localhost:8000/api/anomaly/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "keyword",
    "profile_id": 1,
    "detector_type": "ensemble",
    "include_explanation": true,
    "include_root_cause": true
  }'

# Get active alerts
curl "http://localhost:8000/api/anomaly/alerts/active?profile_id=1"

# Get statistics
curl "http://localhost:8000/api/anomaly/statistics?profile_id=1"
```

#### Test Background Tasks (Manual)

```bash
cd server
python -m app.modules.amazon_ppc.anomaly.tasks
```

---

## API Usage Examples

### 1. Detect Anomalies for All Keywords

```python
import requests

response = requests.post(
    "http://localhost:8000/api/anomaly/detect",
    json={
        "entity_type": "keyword",
        "profile_id": 1,
        "detector_type": "ensemble",
        "include_explanation": True,
        "include_root_cause": True,
    }
)

result = response.json()
print(f"Checked: {result['total_entities_checked']} keywords")
print(f"Anomalies: {result['anomalies_detected']}")
print(f"Critical: {result['critical_count']}")

# View first anomaly
if result['alerts']:
    alert = result['alerts'][0]
    print(f"\nAnomaly Details:")
    print(f"  Entity: {alert['entity_name']}")
    print(f"  Score: {alert['anomaly_score']:.2f}")
    print(f"  Severity: {alert['severity']}")
    print(f"  Root Causes: {alert['root_causes']}")
    print(f"  Feature Contributions:")
    for feature, contrib in alert['explanation'].items():
        print(f"    {feature}: {contrib:.2f}")
```

### 2. Get Unresolved Critical Alerts

```python
response = requests.get(
    "http://localhost:8000/api/anomaly/alerts/active",
    params={
        "profile_id": 1,
        "severity": "critical",
        "limit": 10,
    }
)

alerts = response.json()
print(f"Found {len(alerts)} critical alerts")

for alert in alerts:
    print(f"\n⚠️ {alert['entity_name']}")
    print(f"   Detected: {alert['detection_timestamp']}")
    print(f"   Score: {alert['anomaly_score']:.2f}")
```

### 3. Acknowledge and Resolve Alert

```python
alert_id = 1

# Acknowledge
requests.patch(
    f"http://localhost:8000/api/anomaly/alerts/{alert_id}/acknowledge",
    params={"acknowledged_by": "user@example.com"}
)

# Resolve
requests.patch(
    f"http://localhost:8000/api/anomaly/alerts/{alert_id}/resolve",
    json={
        "is_resolved": True,
        "resolution_notes": "Reduced bid from $2.50 to $1.80"
    }
)
```

---

## Environment Variables (Optional)

Add to `.env`:

```bash
# Anomaly Detection Settings
ANOMALY_LSTM_MODEL_PATH=models/anomaly/lstm_autoencoder.pth
ANOMALY_DETECTION_SCHEDULE=0 * * * *  # Hourly at :00
ANOMALY_ALERT_EMAIL=alerts@example.com
ANOMALY_SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

---

## Monitoring & Observability

### Database Queries

```sql
-- Get anomaly summary
SELECT 
    severity,
    COUNT(*) as count,
    AVG(anomaly_score) as avg_score
FROM anomaly_alerts
WHERE is_resolved = FALSE
GROUP BY severity;

-- Top anomalous entities
SELECT 
    entity_type,
    entity_name,
    COUNT(*) as anomaly_count,
    MAX(anomaly_score) as max_score
FROM anomaly_history
WHERE detection_timestamp >= NOW() - INTERVAL '7 days'
GROUP BY entity_type, entity_name
ORDER BY anomaly_count DESC
LIMIT 10;

-- Resolution performance
SELECT 
    AVG(resolution_time_minutes) as avg_resolution_time,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_time_minutes) as median_resolution_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY resolution_time_minutes) as p95_resolution_time
FROM anomaly_history
WHERE was_resolved = TRUE;
```

### Log Monitoring

```bash
# Watch anomaly detection logs
tail -f logs/app.log | grep "AnomalyMonitor"

# Expected output:
# [AnomalyMonitor] Starting hourly anomaly check
# [AnomalyMonitor] Found 5 active profiles
# [AnomalyMonitor] Starting detection for profile 1
# [AnomalyMonitor] Profile 1: Checked 150 entities, found 3 anomalies (critical=1, high=2) in 2341ms
```

---

## Troubleshooting

### Issue: "Table anomaly_alerts does not exist"
**Solution:** Run migration script:
```bash
python apply_migration_anomaly.py
```

### Issue: "No module named 'torch'"
**Solution:** Install PyTorch:
```bash
pip install torch
```

### Issue: "LSTM model not found"
**Solution:** Train initial model or use without LSTM:
```python
# The ensemble gracefully degrades without LSTM
# Streaming + Isolation Forest still work
```

### Issue: "No anomalies detected"
**Solution:** Check if PerformanceMetric table has data:
```sql
SELECT COUNT(*) FROM performance_metrics;
```

If empty, you need to populate historical data first.

---

## Performance Tuning

### For High-Volume Accounts (10,000+ keywords)

**1. Batch Processing:**
```python
# Process in batches of 1000
batch_size = 1000
for i in range(0, total_keywords, batch_size):
    entity_ids = keyword_ids[i:i+batch_size]
    await anomaly_service.detect_anomalies(
        db,
        AnomalyDetectionRequest(
            entity_type=EntityType.KEYWORD,
            entity_ids=entity_ids,
            profile_id=profile_id,
        )
    )
```

**2. Async Parallelization:**
```python
import asyncio

tasks = [
    run_anomaly_detection_for_profile(db, pid)
    for pid in profile_ids
]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**3. Database Connection Pooling:**
```python
# .env
SQLALCHEMY_POOL_SIZE=20
SQLALCHEMY_MAX_OVERFLOW=40
```

---

## Next Steps

1. ✅ Run migration
2. ✅ Register router
3. ✅ Test API endpoints
4. ⏭️ Integrate with dashboard UI
5. ⏭️ Setup email/Slack notifications
6. ⏭️ Train initial LSTM model on historical data
7. ⏭️ Enable background monitoring tasks

---

## Support

For issues or questions:
- Check logs: `logs/app.log`
- Review test suite: `pytest tests/test_advanced_anomaly.py`
- See integration summary: `phase6_integration_summary.md`
