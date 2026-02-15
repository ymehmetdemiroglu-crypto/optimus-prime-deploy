---
name: grok-admaster-operator
description: Comprehensive autonomous operation of the Grok AdMaster (Optimus Pryme) AI-powered Amazon PPC/DSP optimization platform. Use this skill to manage campaigns, analyze performance, generate creative content, detect anomalies, optimize bids, and execute AI-driven strategies through direct API and database interactions.
---

# Grok AdMaster Operator Skill

This skill enables autonomous operation of the **Grok AdMaster** (Optimus Pryme) platform - an AI-powered War Room for Amazon Sellers that automates PPC, SEO, and DSP strategies.

## System Overview

**Grok AdMaster** is a full-stack application with:
- **Backend**: Python FastAPI with SQLAlchemy ORM, PostgreSQL database
- **AI Integration**: GPT-4 (anomaly detection), Claude 3.5 Sonnet (creative generation), custom ML models
- **Core Capabilities**: Campaign management, bid optimization, anomaly detection, creative AI, DSP integration, real-time optimization

## Core Components

### 1. **Campaign Management**
- View all campaigns with performance metrics
- Update campaign strategies (AI modes)
- Monitor campaign status and budgets
- Track spend, sales, and ACoS

### 2. **Dashboard & Analytics**
- High-level KPI summaries (sales, spend, ACoS, ROAS)
- Time-series performance data
- AI-driven action recommendations
- Trend analysis

### 3. **Anomaly Detection (GPT-4 Powered)**
- Detect ACoS spikes, CTR drops, spend anomalies
- AI-powered explanations with context
- Batch anomaly analysis for pattern detection
- Actionable recommendations

### 4. **Creative AI (Claude 3.5 Sonnet)**
- Generate high-converting ad headlines
- Enhance product descriptions
- SEO-optimized copy generation
- Tone and audience customization

### 5. **PPC Optimization Engine**
- 5 optimization strategies (aggressive, balanced, conservative, profit, volume)
- ML-powered bid recommendations
- Automated scheduling (hourly, daily, weekly)
- Rule-based alerts and triggers
- Dry-run simulation mode

### 6. **Feature Engineering & ML**
- Rolling metrics computation (7, 14, 30-day windows)
- Seasonality features
- Trend indicators
- Competition analysis
- Keyword-level feature extraction

### 7. **Data Ingestion**
- Amazon Ads API integration
- Profile and account synchronization
- Campaign and keyword data collection
- Performance report automation

### 8. **DSP Integration**
- Display advertising management
- Audience targeting
- Cross-channel optimization

## API Endpoints Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Dashboard Endpoints
```
GET  /dashboard/summary              # KPI summary
GET  /dashboard/chart-data?range=7d  # Time-series data (7d, 30d, ytd)
GET  /dashboard/ai-actions            # AI recommendations
```

### Campaign Endpoints
```
GET    /campaigns                     # List all campaigns
GET    /campaigns/{id}                # Get specific campaign
PATCH  /campaigns/{id}/strategy       # Update AI strategy
```

### Anomaly Detection Endpoints
```
POST /anomalies/explain               # Explain single anomaly (GPT-4)
POST /anomalies/explain-batch         # Batch anomaly analysis
GET  /anomalies/test-gpt4             # Test GPT-4 connection
```

### Creative AI Endpoints
```
POST /creative/headlines              # Generate ad headlines (Claude)
POST /creative/description            # Enhance product description
```

### Optimization Endpoints
```
POST   /optimization/generate-plan    # Generate optimization plan
POST   /optimization/execute          # Execute optimization
GET    /optimization/strategies       # List strategies
POST   /optimization/quick-optimize/{id}  # Quick optimize
GET    /optimization/alerts/{id}      # Get campaign alerts
GET    /optimization/alerts           # Get all alerts
POST   /optimization/schedule         # Create schedule
DELETE /optimization/schedule/{id}    # Remove schedule
GET    /optimization/scheduler/status # Scheduler status
POST   /optimization/scheduler/start  # Start scheduler
POST   /optimization/scheduler/stop   # Stop scheduler
```

### Feature Engineering Endpoints
```
GET  /features/campaign/{id}          # Get campaign features
GET  /features/campaign/{id}/rolling  # Rolling metrics
GET  /features/campaign/{id}/trends   # Trend features
GET  /features/seasonality            # Seasonality features
GET  /features/keyword/{id}           # Keyword features
GET  /features/keyword/{id}/bid-recommendations  # Bid recommendations
GET  /features/campaign/{id}/keywords # All keyword features
POST /features/batch-compute          # Batch compute features
DELETE /features/cleanup              # Cleanup old features
```

### ML Endpoints
```
POST /ml/train                        # Train ML models
POST /ml/predict                      # Get predictions
GET  /ml/model-info                   # Model information
POST /ml/advanced/ensemble-predict    # Ensemble predictions
GET  /ml/advanced/feature-importance  # Feature importance
```

### Ingestion Endpoints
```
POST /ingestion/sync-all              # Sync all accounts
POST /ingestion/sync-account/{id}     # Sync specific account
GET  /ingestion/status                # Ingestion status
```

### Account Management Endpoints
```
GET    /accounts                      # List accounts
POST   /accounts                      # Create account
GET    /accounts/{id}                 # Get account
PUT    /accounts/{id}                 # Update account
DELETE /accounts/{id}                 # Delete account
POST   /accounts/{id}/credentials     # Add credentials
GET    /accounts/{id}/profiles        # Get profiles
```

## Usage Patterns

### Pattern 1: Campaign Performance Analysis

**Objective**: Analyze campaign performance and get AI recommendations

**Steps**:
1. Get dashboard summary for KPIs
2. Fetch campaign list with performance metrics
3. Get AI-driven action recommendations
4. Analyze specific campaigns for detailed insights

**Example Commands**:
```bash
# Get KPI summary
curl http://localhost:8000/api/v1/dashboard/summary

# Get all campaigns
curl http://localhost:8000/api/v1/campaigns

# Get AI actions
curl http://localhost:8000/api/v1/dashboard/ai-actions

# Get specific campaign
curl http://localhost:8000/api/v1/campaigns/1
```

### Pattern 2: Anomaly Detection & Explanation

**Objective**: Detect and explain performance anomalies using GPT-4

**Steps**:
1. Identify anomalies in campaign/keyword data
2. Prepare anomaly context (metrics, historical data)
3. Call GPT-4 explanation endpoint
4. Review AI-generated insights and recommendations

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/v1/anomalies/explain \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly": {
      "type": "acos_spike",
      "keyword": "wireless headphones",
      "severity": "high",
      "metric": "acos",
      "expected_value": 22.5,
      "actual_value": 32.6,
      "deviation_percent": 45
    },
    "keyword_context": {
      "keyword_text": "wireless headphones",
      "match_type": "exact",
      "bid": 1.50,
      "impressions": 12500,
      "clicks": 450,
      "ctr": 3.6
    }
  }'
```

### Pattern 3: Creative Content Generation

**Objective**: Generate high-converting ad copy using Claude 3.5 Sonnet

**Steps**:
1. Prepare product information and keywords
2. Call headline generation endpoint
3. Optionally enhance product descriptions
4. Review and select best variations

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/v1/creative/headlines \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Wireless Bluetooth Headphones",
    "keywords": ["wireless headphones", "bluetooth earbuds", "noise cancelling"],
    "unique_selling_points": ["40-hour battery", "Premium sound quality", "Comfortable fit"],
    "target_audience": "music lovers",
    "tone": "professional"
  }'
```

### Pattern 4: Bid Optimization

**Objective**: Generate and execute optimization plans

**Steps**:
1. Select optimization strategy (aggressive, balanced, conservative, profit, volume)
2. Generate optimization plan
3. Review recommendations (dry-run mode)
4. Execute approved optimizations
5. Monitor results

**Example Workflow**:
```bash
# Generate plan
curl -X POST http://localhost:8000/api/v1/optimization/generate-plan \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": 1,
    "strategy": "balanced",
    "target_acos": 25.0
  }'

# Execute with dry-run
curl -X POST http://localhost:8000/api/v1/optimization/execute \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": 1,
    "strategy": "balanced",
    "dry_run": true
  }'

# Execute live (after review)
curl -X POST http://localhost:8000/api/v1/optimization/execute \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": 1,
    "strategy": "balanced",
    "dry_run": false
  }'
```

### Pattern 5: Automated Optimization Scheduling

**Objective**: Set up automated optimization schedules

**Steps**:
1. Create optimization schedule
2. Configure frequency and strategy
3. Set confidence thresholds
4. Enable auto-execution (optional)
5. Monitor scheduler status

**Example**:
```bash
# Create daily schedule
curl -X POST http://localhost:8000/api/v1/optimization/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": 1,
    "strategy": "balanced",
    "frequency": "daily",
    "auto_execute": true,
    "min_confidence": 0.7
  }'

# Check scheduler status
curl http://localhost:8000/api/v1/optimization/scheduler/status
```

### Pattern 6: Feature Engineering & ML Predictions

**Objective**: Compute features and get ML-powered predictions

**Steps**:
1. Compute campaign features
2. Get rolling metrics and trends
3. Request ML predictions
4. Use ensemble predictions for higher accuracy

**Example**:
```bash
# Get campaign features
curl http://localhost:8000/api/v1/features/campaign/1

# Get rolling metrics
curl http://localhost:8000/api/v1/features/campaign/1/rolling?windows=7,14,30

# Get keyword bid recommendations
curl http://localhost:8000/api/v1/features/keyword/123/bid-recommendations?target_acos=25.0

# Get ML predictions
curl -X POST http://localhost:8000/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": 1,
    "features": {...}
  }'
```

### Pattern 7: Data Ingestion & Synchronization

**Objective**: Sync data from Amazon Ads API

**Steps**:
1. Configure account credentials
2. Trigger synchronization
3. Monitor ingestion status
4. Verify data integrity

**Example**:
```bash
# Sync all accounts
curl -X POST http://localhost:8000/api/v1/ingestion/sync-all

# Sync specific account
curl -X POST http://localhost:8000/api/v1/ingestion/sync-account/1

# Check status
curl http://localhost:8000/api/v1/ingestion/status
```

## Database Direct Access

### Using Python Scripts

For complex operations, you can directly interact with the database using Python:

```python
# Example: Query campaigns directly
python -c "
from app.core.database import get_db
from app.models.campaign import Campaign
from sqlalchemy import select
import asyncio

async def get_campaigns():
    async for db in get_db():
        result = await db.execute(select(Campaign))
        campaigns = result.scalars().all()
        for c in campaigns:
            print(f'{c.id}: {c.name} - ACoS: {c.current_acos}%')
        break

asyncio.run(get_campaigns())
"
```

### SQL Queries via check_db_permissions.py

Use the existing database utility script for direct SQL:

```bash
cd c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server
python check_db_permissions.py
```

## Server Management

### Starting the Server

```bash
cd c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server
uvicorn app.main:app --reload --port 8000
```

### Health Check

```bash
curl http://localhost:8000/health
```

### API Documentation

Access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Optimization Strategies Reference

| Strategy | Use Case | Max Bid Increase | Max Bid Decrease | ACoS Tolerance |
|----------|----------|------------------|------------------|----------------|
| **aggressive** | Rapid growth, new product launches | 30% | 15% | 130% of target |
| **balanced** | Steady growth with efficiency | 20% | 20% | 115% of target |
| **conservative** | Risk minimization, mature products | 10% | 25% | 105% of target |
| **profit** | Maximize profit margins | 15% | 30% | 95% of target |
| **volume** | Maximize impressions and reach | 35% | 10% | 150% of target |

## Alert Rules Reference

| Condition | Threshold | Recommended Action | Severity |
|-----------|-----------|-------------------|----------|
| `acos_threshold` | ACoS > 50% | Decrease bid | Critical |
| `spend_spike` | Spend > 2x average | Decrease budget | Warning |
| `no_sales` | $50 spend, 0 sales | Pause keyword | Critical |
| `ctr_drop` | CTR drops 50% | Increase bid | Warning |
| `budget_depletion` | 90% budget used | Increase budget | Warning |

## Best Practices

### 1. **Always Use Dry-Run First**
Before executing any optimization, run in dry-run mode to preview changes.

### 2. **Monitor Confidence Scores**
Only execute actions with confidence > 0.7 for automated workflows.

### 3. **Batch Operations**
Use batch endpoints for multiple campaigns to improve efficiency.

### 4. **Feature Caching**
Leverage feature caching to reduce computation time.

### 5. **Anomaly Context**
Always provide full context (keyword, campaign, historical data) for better GPT-4 explanations.

### 6. **Creative Iterations**
Generate multiple headline variations and A/B test.

### 7. **Scheduler Monitoring**
Regularly check scheduler status and review execution logs.

### 8. **Alert Response**
Set up automated responses to critical alerts.

## Troubleshooting

### Server Not Running
```bash
# Check if server is running
curl http://localhost:8000/health

# If not, start server
cd c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server
uvicorn app.main:app --reload --port 8000
```

### Database Connection Issues
```bash
# Verify database connection
python check_db_permissions.py
```

### API Errors
- Check API documentation at `/docs`
- Verify request payload format
- Check server logs for detailed error messages

### GPT-4/Claude API Issues
```bash
# Test GPT-4 connection
curl http://localhost:8000/api/v1/anomalies/test-gpt4

# Check environment variables for API keys
```

## Advanced Operations

### Custom Python Scripts

Create custom scripts in `.agent/skills/grok-admaster-operator/scripts/` for complex operations:

**Example: Bulk Campaign Analysis**
```python
import asyncio
import httpx

async def analyze_all_campaigns():
    async with httpx.AsyncClient() as client:
        # Get all campaigns
        campaigns = await client.get("http://localhost:8000/api/v1/campaigns")
        
        for campaign in campaigns.json():
            # Get features
            features = await client.get(
                f"http://localhost:8000/api/v1/features/campaign/{campaign['id']}"
            )
            
            # Generate optimization plan
            plan = await client.post(
                "http://localhost:8000/api/v1/optimization/generate-plan",
                json={
                    "campaign_id": campaign['id'],
                    "strategy": "balanced",
                    "target_acos": 25.0
                }
            )
            
            print(f"Campaign: {campaign['name']}")
            print(f"Plan: {plan.json()}")

asyncio.run(analyze_all_campaigns())
```

## Integration with Other Skills

### Market Research Integration
Combine with `market-researcher` skill for competitive analysis:

1. Use market-researcher to find competitor ASINs
2. Analyze competitor pricing and positioning
3. Generate creative content targeting competitor weaknesses
4. Optimize bids based on competitive landscape

### Workflow Automation
Integrate with existing workflows:
- `/autonomous-research`: Research + Creative AI
- `/feature-implementation`: Add new optimization strategies
- `/health-check`: Verify system health before operations

## Skill Invocation Examples

### Example 1: Complete Campaign Audit
```
"Analyze campaign ID 1, detect any anomalies, explain them using GPT-4, 
generate an optimization plan, and create new ad headlines"
```

### Example 2: Automated Daily Optimization
```
"Set up daily automated optimization for all campaigns using balanced strategy, 
with minimum confidence of 0.75, and send me a summary of actions taken"
```

### Example 3: Creative Content Sprint
```
"For all active campaigns, generate 5 headline variations for each product, 
enhance product descriptions, and save the best performing options"
```

### Example 4: Anomaly Investigation
```
"Check all campaigns for anomalies in the last 7 days, batch analyze them 
with GPT-4, and create optimization plans to address the issues"
```

### Example 5: Performance Deep Dive
```
"Get dashboard summary, analyze top 5 campaigns by spend, compute their 
features and trends, get ML predictions, and recommend strategy adjustments"
```

## Notes

- **Server must be running** at `http://localhost:8000` for API operations
- **Database must be accessible** for direct database operations
- **API keys required** for GPT-4 (OpenRouter) and Claude (Anthropic)
- **Amazon Ads credentials** needed for data ingestion
- All operations are **asynchronous** - use appropriate async patterns
- **Dry-run mode** is highly recommended before live executions
- **Confidence thresholds** should be tuned based on risk tolerance

## File Locations

- **Server Root**: `c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server`
- **Main App**: `server/app/main.py`
- **API Routes**: `server/app/api/` and `server/app/amazon_ppc_optimizer/*/router.py`
- **Models**: `server/app/models/` and `server/app/amazon_ppc_optimizer/*/models.py`
- **Services**: `server/app/services/`
- **Documentation**: `grok-admaster/docs/`
- **Database Utils**: `server/check_db_permissions.py`

---

**This skill enables complete autonomous operation of Grok AdMaster through conversational AI interface. No UI required.**
