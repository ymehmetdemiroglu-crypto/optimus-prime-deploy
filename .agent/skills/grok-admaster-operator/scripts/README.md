# Grok AdMaster Operator Scripts

This directory contains Python scripts for operating the Grok AdMaster platform through the command line and programmatically.

## Scripts

### 1. `operator.py` - Command-Line Interface

Main CLI tool for common operations.

**Installation:**
```bash
pip install httpx
```

**Usage:**
```bash
python operator.py <command> [options]
```

**Available Commands:**

| Command | Description | Example |
|---------|-------------|---------|
| `dashboard` | Get dashboard summary | `python operator.py dashboard` |
| `campaigns` | List all campaigns | `python operator.py campaigns` |
| `campaign <id>` | Get campaign details | `python operator.py campaign 1` |
| `optimize <id> [strategy]` | Generate optimization plan | `python operator.py optimize 1 balanced` |
| `execute <id> [strategy] [--live]` | Execute optimization | `python operator.py execute 1 balanced --live` |
| `anomalies` | Get all anomalies | `python operator.py anomalies` |
| `features <id>` | Get campaign features | `python operator.py features 1` |
| `headlines "<product>" "<keywords>" "<usps>"` | Generate headlines | `python operator.py headlines "Headphones" "wireless,bluetooth" "40hr battery"` |
| `schedule <account_id> [strategy] [freq]` | Create schedule | `python operator.py schedule 1 balanced daily` |
| `sync` | Sync all accounts | `python operator.py sync` |
| `health` | Check system health | `python operator.py health` |

**Examples:**

```bash
# Get dashboard summary
python operator.py dashboard

# List all campaigns
python operator.py campaigns

# Optimize campaign 1 with balanced strategy
python operator.py optimize 1 balanced

# Execute optimization (dry-run)
python operator.py execute 1 balanced

# Execute optimization (live)
python operator.py execute 1 balanced --live

# Generate ad headlines
python operator.py headlines "Wireless Headphones" "bluetooth,wireless,noise cancelling" "40hr battery,premium sound"

# Create daily optimization schedule
python operator.py schedule 1 balanced daily

# Check system health
python operator.py health
```

### 2. `examples.py` - Advanced Usage Examples

Demonstrates complex workflows and automation patterns.

**Usage:**
```bash
# Run all examples
python examples.py

# Run specific example
python examples.py <1-6>
```

**Available Examples:**

1. **Complete Campaign Audit**
   - Fetches campaign details
   - Computes features and metrics
   - Checks for anomalies
   - Generates optimization plan
   - Shows top recommendations

2. **Batch Optimization**
   - Optimizes all campaigns
   - Selects strategy based on performance
   - Generates comprehensive report

3. **Anomaly Investigation**
   - Detects anomalies
   - Uses GPT-4 for explanations
   - Provides actionable recommendations

4. **Creative Content Generation**
   - Generates ad headlines with Claude
   - Enhances product descriptions
   - Creates multiple variations

5. **Automated Daily Workflow**
   - Morning health check
   - Dashboard overview
   - Alert monitoring
   - Campaign optimization
   - Summary report

6. **ML-Powered Predictions**
   - Computes comprehensive features
   - Analyzes trends
   - Generates ML-based recommendations

**Run Examples:**

```bash
# Run all examples interactively
python examples.py

# Run specific example
python examples.py 1  # Complete Campaign Audit
python examples.py 2  # Batch Optimization
python examples.py 3  # Anomaly Investigation
python examples.py 4  # Creative Content Generation
python examples.py 5  # Automated Daily Workflow
python examples.py 6  # ML-Powered Predictions
```

## Requirements

```bash
pip install httpx
```

## Configuration

Scripts connect to the Grok AdMaster API at:
```
http://localhost:8000/api/v1
```

Make sure the server is running before using these scripts:

```bash
cd c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server
uvicorn app.main:app --reload --port 8000
```

## Programmatic Usage

You can also import the `GrokAdMasterOperator` class for custom scripts:

```python
from operator import GrokAdMasterOperator

async def my_custom_workflow():
    operator = GrokAdMasterOperator()
    
    # Get dashboard summary
    summary = await operator.get_dashboard_summary()
    print(f"Total Sales: ${summary['total_sales']:,.2f}")
    
    # List campaigns
    campaigns = await operator.list_campaigns()
    for campaign in campaigns:
        print(f"{campaign['name']}: {campaign['acos']:.1f}% ACoS")
    
    # Generate optimization plan
    plan = await operator.generate_optimization_plan(
        campaign_id=1,
        strategy="balanced",
        target_acos=25.0
    )
    print(f"Recommended actions: {plan['summary']['total_actions']}")

# Run it
import asyncio
asyncio.run(my_custom_workflow())
```

## Optimization Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `aggressive` | Maximize growth (30% max increase) | New product launches |
| `balanced` | Balance growth & efficiency (20% max increase) | Steady growth |
| `conservative` | Minimize risk (10% max increase) | Mature products |
| `profit` | Maximize profit margins (15% max increase) | High-margin products |
| `volume` | Maximize impressions (35% max increase) | Brand awareness |

## Error Handling

All scripts include comprehensive error handling. If you encounter errors:

1. **Check server status:**
   ```bash
   python operator.py health
   ```

2. **Verify API connectivity:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Check server logs:**
   Look at the FastAPI server console output

4. **Verify database connection:**
   ```bash
   cd c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server
   python check_db_permissions.py
   ```

## Tips

- **Always use dry-run first**: Test optimizations before executing live
- **Monitor confidence scores**: Only execute high-confidence actions (>0.7)
- **Use batch operations**: More efficient for multiple campaigns
- **Schedule regular audits**: Run daily workflow example as a cron job
- **Review GPT-4 explanations**: Understand anomalies before taking action
- **A/B test creative**: Generate multiple variations and test

## Support

For issues or questions:
1. Check the main SKILL.md documentation
2. Review API documentation at `http://localhost:8000/docs`
3. Examine server logs for detailed error messages
4. Verify environment variables and API keys
