# Grok AdMaster

AI-powered "War Room" dashboard for Amazon sellers that automates PPC, SEO, and DSP strategies.

## Features

- 📊 **War Room Dashboard** - Real-time KPIs, sales velocity charts, and AI action feed
- 🎯 **Campaign Manager** - AI strategy toggles (Auto Pilot, Aggressive, Profit Guard)
- 💬 **Grok AI Chat** - Intelligent assistant for optimization recommendations
- 🌙 **Cyber-Professional UI** - Dark mode with neon accents

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+

### Backend Setup
```bash
cd server
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd client
npm install
npm run dev
```

### Access
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Environment Variables

**Backend** (`server/.env`):
- `CORS_ORIGINS` – Comma-separated origins (default: `http://localhost:5173,http://127.0.0.1:5173`)
- `SECRET_KEY` – Required in production (non-default value)
- `POSTGRES_SERVER`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` – Database (defaults for dev; set in production)
- `ENV` – Set to `production` to enforce secret validation
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` – For WebSocket chat (LangChain/OpenRouter)

**Frontend** (`client/.env`):
- `VITE_API_URL` – Backend API base URL (default: `http://localhost:8000/api/v1`)
- `VITE_WS_URL` – WebSocket base URL (default: `ws://localhost:8000`)

## Tech Stack

**Frontend**
- React 18 + TypeScript
- Tailwind CSS (Dark Mode)
- Recharts
- React Router

**Backend**
- FastAPI
- Pydantic
- Uvicorn (ASGI)

## Project Structure

```
grok-admaster/
├── client/           # React Frontend
│   └── src/
│       ├── api/          # API client
│       ├── components/   # UI components
│       ├── pages/        # Route pages
│       └── types/        # TypeScript interfaces
├── server/           # Python Backend
│   └── app/
│       ├── api/          # Route handlers
│       ├── models/       # Pydantic schemas
│       └── services/     # AI simulation
└── docs/             # Documentation
```
