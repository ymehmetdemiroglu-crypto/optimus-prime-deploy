# API Contracts

**Base URL:** `http://localhost:8000/api/v1`

## 1. Dashboard Data

### `GET /dashboard/summary`
Returns high-level KPI cards for the "War Room".
*   **Response:**
    ```json
    {
      "total_sales": 15400.00,
      "ad_spend": 2300.00,
      "acos": 14.9,
      "roas": 6.7,
      "velocity_trend": "up" // 'up' | 'down' | 'flat'
    }
    ```

### `GET /dashboard/chart-data`
Returns time-series data for the main graph.
*   **Query Params:** `range` (7d, 30d, ytd)
*   **Response:** `List[PerformanceMetric]`

## 2. Campaigns

### `GET /campaigns`
List all campaigns with their AI status.
*   **Response:** `List[Campaign]`

### `PATCH /campaigns/{id}/strategy`
Update the AI strategy for a specific campaign.
*   **Body:** `{ "ai_mode": "aggressive_growth" }`
*   **Response:** `Campaign` (Updated)

## 3. Grok Chat (AI Agent)

### `POST /chat/message`
Send a message to Grok.
*   **Body:**
    ```json
    {
      "message": "Why is my ACoS high on ASIN B08X...",
      "context_asin": "B08X..." // Optional
    }
    ```
*   **Response:**
    ```json
    {
      "id": "msg_123",
      "sender": "grok",
      "content": "ACoS spiked because competitor 'BrandX' increased bids on 'running shoes'. I recommend...",
      "timestamp": "2026-06-15T10:00:00Z"
    }
    ```
