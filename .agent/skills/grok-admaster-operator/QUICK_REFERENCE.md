# Grok AdMaster Quick Reference

## 🚀 Quick Start

### 1. Start the Server
```bash
cd c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server
uvicorn app.main:app --reload --port 8000
```

### 2. Verify Health
```bash
python .agent/skills/grok-admaster-operator/scripts/operator.py health
```

## 📊 Common Operations

### Dashboard & Overview
```bash
# Get KPI summary
python operator.py dashboard

# List all campaigns
python operator.py campaigns

# Get specific campaign
python operator.py campaign 1
```

### Campaign Optimization
```bash
# Generate optimization plan
python operator.py optimize 1 balanced

# Execute (dry-run)
python operator.py execute 1 balanced

# Execute (live) - BE CAREFUL!
python operator.py execute 1 balanced --live
```

### Anomaly Detection
```bash
# Check for anomalies
python operator.py anomalies

# Run GPT-4 investigation (use examples.py)
python examples.py 3
```

### Creative Generation
```bash
# Generate headlines
python operator.py headlines "Product Name" "keyword1,keyword2" "usp1,usp2"

# Full creative workflow
python examples.py 4
```

### Feature Analysis
```bash
# Get campaign features
python operator.py features 1

# Full ML analysis
python examples.py 6
```

### Automation
```bash
# Create daily schedule
python operator.py schedule 1 balanced daily

# Run daily workflow
python examples.py 5
```

## 🎯 Optimization Strategies

| Strategy | When to Use | Max Bid ↑ | Max Bid ↓ |
|----------|-------------|-----------|-----------|
| `aggressive` | New launches, rapid growth | 30% | 15% |
| `balanced` | Steady growth | 20% | 20% |
| `conservative` | Risk averse, mature | 10% | 25% |
| `profit` | Maximize margins | 15% | 30% |
| `volume` | Brand awareness | 35% | 10% |

## 🔥 Power User Workflows

### Complete Campaign Audit
```bash
python examples.py 1
```
Analyzes performance, features, anomalies, and generates recommendations.

### Batch Optimize All Campaigns
```bash
python examples.py 2
```
Optimizes all campaigns with intelligent strategy selection.

### Daily Automation
```bash
python examples.py 5
```
Complete daily workflow: health check → alerts → optimization → report.

## 📡 Direct API Calls

### Using curl
```bash
# Dashboard
curl http://localhost:8000/api/v1/dashboard/summary

# Campaigns
curl http://localhost:8000/api/v1/campaigns

# Generate plan
curl -X POST http://localhost:8000/api/v1/optimization/generate-plan \
  -H "Content-Type: application/json" \
  -d '{"campaign_id": 1, "strategy": "balanced", "target_acos": 25.0}'

# GPT-4 anomaly explanation
curl -X POST http://localhost:8000/api/v1/anomalies/explain \
  -H "Content-Type: application/json" \
  -d '{"anomaly": {"type": "acos_spike", "severity": "high"}}'

# Claude headlines
curl -X POST http://localhost:8000/api/v1/creative/headlines \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Headphones", "keywords": ["wireless"], "unique_selling_points": ["40hr battery"]}'
```

### Using Python
```python
from scripts.operator import GrokAdMasterOperator
import asyncio

async def main():
    op = GrokAdMasterOperator()
    
    # Dashboard
    summary = await op.get_dashboard_summary()
    
    # Campaigns
    campaigns = await op.list_campaigns()
    
    # Optimize
    plan = await op.generate_optimization_plan(1, "balanced")
    
    # Anomalies
    alerts = await op.get_alerts()
    
    # Creative
    headlines = await op.generate_headlines(
        "Product", ["keyword"], ["usp"]
    )

asyncio.run(main())
```

## 🎨 Creative AI Examples

### Generate Headlines
```bash
python operator.py headlines \
  "Wireless Bluetooth Headphones" \
  "wireless,bluetooth,noise cancelling" \
  "40hr battery,premium sound,comfortable"
```

### Enhance Description
```python
from scripts.operator import GrokAdMasterOperator
import asyncio

async def enhance():
    op = GrokAdMasterOperator()
    result = await op.enhance_description(
        "Basic product description",
        ["keyword1", "keyword2"]
    )
    print(result['enhanced_description'])

asyncio.run(enhance())
```

## 🤖 ML & Features

### Get Features
```bash
# Campaign features
python operator.py features 1

# Rolling metrics (7, 14, 30 day)
curl http://localhost:8000/api/v1/features/campaign/1/rolling?windows=7,14,30

# Keyword bid recommendations
curl http://localhost:8000/api/v1/features/keyword/123/bid-recommendations?target_acos=25.0
```

### ML Predictions
```bash
# Full ML analysis
python examples.py 6

# Direct API
curl -X POST http://localhost:8000/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"campaign_id": 1, "features": {...}}'
```

## ⚡ Automation Patterns

### Daily Cron Job
```bash
# Add to crontab (Linux/Mac) or Task Scheduler (Windows)
0 6 * * * cd /path/to/optimus-pryme && python .agent/skills/grok-admaster-operator/scripts/examples.py 5
```

### Continuous Monitoring
```python
import asyncio
from scripts.operator import GrokAdMasterOperator

async def monitor():
    op = GrokAdMasterOperator()
    while True:
        alerts = await op.get_alerts()
        if alerts['critical_count'] > 0:
            # Take action
            print(f"⚠️  {alerts['critical_count']} critical alerts!")
        await asyncio.sleep(3600)  # Check every hour

asyncio.run(monitor())
```

## 🔧 Troubleshooting

### Server Not Running
```bash
# Check
curl http://localhost:8000/health

# Start
cd c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server
uvicorn app.main:app --reload --port 8000
```

### Database Issues
```bash
cd c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server
python check_db_permissions.py
```

### API Errors
- Check `/docs` for API documentation
- Verify request payload format
- Review server logs

## 📚 Resources

- **Full Documentation**: `.agent/skills/grok-admaster-operator/SKILL.md`
- **Script Documentation**: `.agent/skills/grok-admaster-operator/scripts/README.md`
- **API Docs**: `http://localhost:8000/docs`
- **Project Docs**: `grok-admaster/docs/`

## 💡 Pro Tips

1. **Always dry-run first**: Test before executing live
2. **Monitor confidence**: Only auto-execute actions with confidence > 0.7
3. **Use batch operations**: More efficient for multiple campaigns
4. **Review GPT-4 insights**: Understand anomalies before acting
5. **A/B test creative**: Generate multiple variations
6. **Schedule regular audits**: Automate daily workflows
7. **Track execution history**: Review logs regularly
8. **Set alert thresholds**: Customize based on your risk tolerance

## 🎯 Success Metrics

Track these KPIs to measure optimization success:
- **ACoS Improvement**: Target reduction of 10-20%
- **ROAS Increase**: Target improvement of 15-25%
- **Conversion Rate**: Monitor for 5-10% lift
- **Wasted Spend Reduction**: Identify and eliminate 20-30%
- **High-Confidence Actions**: Aim for >75% confidence average

---

**Remember**: The goal is to make data-driven decisions with AI assistance, not to blindly automate everything. Always review recommendations before executing live changes!
