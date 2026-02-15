# Workflow / Automation Document

## 1. Development Workflow
Since this is a prototype, we prioritize speed and "mockability".

### The "Loop"
1.  **Mock Data First:** Define the JSON shape in `04_API_Contracts.md`.
2.  **Backend Stub:** Create the FastAPI endpoint returning that static JSON.
3.  **Frontend Component:** Build the UI to consume that endpoint.
4.  **Refine:** Add "simulated logic" to the backend (e.g., randomizing values slightly) to make the prototype feel alive.

## 2. Simulated "AI" Workflow
Real AI calls are expensive and slow. We simulate the AI workflow:

**Trigger:** User clicks "Optimize Campaign"
1.  **Frontend:** Sends `POST /optimize`
2.  **Backend:**
    *   Calculates a "fake" optimization (e.g., lowers bid by $0.05).
    *   Sleeps for 1.5 seconds (to simulate "thinking").
    *   Returns success message + explanation.
3.  **Frontend:** Show "Grok is analyzing..." -> "Optimization Complete".

## 3. Environment Variables
Local development requires a `.env` file (git-ignored).

```bash
# Backend
PORT=8000
DEBUG=True

# Frontend
VITE_API_URL=http://localhost:8000/api/v1
```

## 4. Build & Run
To start the system:
1.  Terminal 1: `cd server && venv\Scripts\activate && uvicorn app.main:app --reload`
2.  Terminal 2: `cd client && npm run dev`
