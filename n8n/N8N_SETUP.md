# n8n Workflow Orchestration for Optimus Prime

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    n8n Workflow Orchestrator                     │
│                     (localhost:5678)                             │
├─────────────┬─────────────────┬─────────────────────────────────┤
│  01_Master  │  02_Anomaly     │  03_Daily Optimization          │
│  Orchestr.  │  Response       │  Cycle                          │
│  (30 min)   │  (sub-workflow) │  (6:00 AM daily)                │
└──────┬──────┴────────┬────────┴─────────┬───────────────────────┘
       │               │                  │
       ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│             Optimus Prime Custom n8n Node                       │
│    (OptimusPrime.node.ts + OptimusPrimeApi.credentials.ts)      │
├─────────────────────────────────────────────────────────────────┤
│  Resources: Dashboard | Campaign | Anomaly | Optimization       │
│             Contextual Bidding | RL Budget | Semantic Intel      │
│             Competitive Intel | Ingestion | Health               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                Optimus Prime API (FastAPI)                       │
│                    (localhost:8000)                              │
├──────┬──────┬────────┬────────┬──────────┬──────────────────────┤
│ Dash │ Camp │ Anomal │ ML/TS  │ Semantic │ Autonomous Operator  │
│ board│ aigns│ ies    │ Bidding│ Cortex   │ (6hr patrol cycles)  │
└──────┴──────┴────────┴────────┴──────────┴──────────────────────┘
```

## Workflows

### 1. Master Orchestration (`01_master_orchestration.json`)
**Trigger:** Every 30 minutes

**Flow:**
1. Health check → Is API alive?
2. **If healthy** → Fan out to 3 parallel data fetches:
   - Anomaly Feed → If critical anomalies → triggers Anomaly Response sub-workflow
   - Dashboard KPIs → If ACoS > 25% → checks Contextual Bidding optimizer status
   - Client Matrix → Aggregates system status → Posts to Slack
3. **If unhealthy** → Alert via Telegram

### 2. Anomaly Response Pipeline (`02_anomaly_response.json`)
**Trigger:** Called by Master Orchestration when critical anomalies detected

**Flow:**
1. Fetch full anomaly feed
2. Prioritize (high > medium, skip low)
3. Batch process:
   - **High priority** → GPT-4 explanation → Build action plan → Auto-execute or notify
   - **Medium priority** → Log to Google Sheets for daily review
4. Auto-actions available:
   - `PAUSE_CAMPAIGN` → Switches to Profit Guard mode
   - `ADJUST_BID` → Triggers bid optimization
   - `NOTIFY_OPERATOR` → Slack alert only

### 3. Daily Optimization Cycle (`03_daily_optimization.json`)
**Trigger:** Daily at 6:00 AM (EST)

**Flow:**
1. Health gate check
2. Trigger Amazon data ingestion sync
3. Wait 30s for ingestion to complete
4. Get all clients from Client Matrix
5. Filter active (non-disconnected) clients
6. **Per-client loop** (parallel within each client):
   - **3a.** Contextual Thompson Sampling bid optimization
   - **3b.** Hierarchical RL budget allocation
   - **3c.** Semantic bleed detection
7. Merge results per client
8. Run learning step (posterior updates for TS arms)
9. Run RL policy gradient update
10. Build daily summary report
11. Post to Slack + Email report

## Setup

### 1. Start the Stack

```bash
docker compose up -d
```

n8n will be available at `http://localhost:5678`.

### 2. Configure Credentials in n8n

1. Open n8n at `http://localhost:5678`
2. Go to **Settings → Credentials → Add Credential**
3. Search for "Optimus Prime API"
4. Fill in:
   - **Base URL:** `http://api:8000` (Docker internal) or `http://localhost:8000` (external)
   - **API Token:** Get this by calling `POST /api/v1/auth/login`

### 3. Import Workflows

1. Go to **Workflows → Import from File**
2. Import in order:
   - `workflows/02_anomaly_response.json` (sub-workflow first)
   - `workflows/03_daily_optimization.json`
   - `workflows/01_master_orchestration.json` (master last)

### 4. Configure Notification Channels

Set these environment variables in `.env` or docker-compose:

| Variable | Description |
|----------|-------------|
| `N8N_USER` | n8n basic auth username |
| `N8N_PASSWORD` | n8n basic auth password |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for API-down alerts |
| `GOOGLE_SHEET_ID` | Google Sheet ID for anomaly logging |
| `REPORT_EMAIL` | Email for daily optimization reports |

### 5. Activate Workflows

Toggle each workflow to **Active** in the n8n UI.

## Custom Node API Coverage

The `OptimusPrime` custom node maps to these API endpoints:

| Resource | Operation | Endpoint | Method |
|----------|-----------|----------|--------|
| Dashboard | Get Summary | `/api/v1/dashboard/summary` | GET |
| Dashboard | Get Chart Data | `/api/v1/dashboard/chart-data` | GET |
| Dashboard | Get AI Actions | `/api/v1/dashboard/ai-actions` | GET |
| Dashboard | Get Client Matrix | `/api/v1/dashboard/matrix` | GET |
| Dashboard | Get Client Dashboard | `/api/v1/performance/dashboard/{id}` | GET |
| Campaign | List All | `/api/v1/campaigns` | GET |
| Campaign | Update Strategy | `/api/v1/campaigns/{id}/strategy` | PATCH |
| Anomaly | Get Feed | `/api/v1/anomalies/feed` | GET |
| Anomaly | Explain Single | `/api/v1/anomalies/explain` | POST |
| Anomaly | Explain Batch | `/api/v1/anomalies/explain-batch` | POST |
| Ctx Bidding | Optimize Profile | `/api/v1/contextual-bidding/{id}/optimize` | POST |
| Ctx Bidding | Learn | `/api/v1/contextual-bidding/{id}/learn` | POST |
| Ctx Bidding | Get Arm Stats | `/api/v1/contextual-bidding/keywords/{id}/arms` | GET |
| Ctx Bidding | Get Features | `/api/v1/contextual-bidding/keywords/{id}/features` | GET |
| Ctx Bidding | Feature Importance | `/api/v1/contextual-bidding/keywords/{id}/feature-importance` | GET |
| Ctx Bidding | Select Arm | `/api/v1/contextual-bidding/keywords/{id}/select-arm` | POST |
| Ctx Bidding | Get Status | `/api/v1/contextual-bidding/status` | GET |
| RL Budget | Allocate | `/api/v1/rl-budget/allocate` | POST |
| RL Budget | Learn | `/api/v1/rl-budget/learn` | POST |
| RL Budget | Get State | `/api/v1/rl-budget/state/{id}` | GET |
| Semantic | Detect Bleed | `/api/v1/semantic/bleed` | GET |
| Semantic | Find Opportunities | `/api/v1/semantic/opportunities` | GET |
| Semantic | Health Report | `/api/v1/semantic/health` | GET |
| Competitive | Get Competitors | `/api/v1/competitive/competitors` | GET |
| Ingestion | Trigger Sync | `/api/v1/ingestion/sync` | POST |
| Health | Check | `/health` | GET |
