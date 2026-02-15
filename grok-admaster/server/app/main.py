"""
Grok AdMaster API - Main Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import dashboard, anomalies, creative, settings as settings_api, audit, discovery, performance, lockdown, validation
from app.api.dsp import dsp_api
from app.modules.amazon_ppc.accounts import router as accounts_router
from app.modules.amazon_ppc.ingestion import router as ingestion_router
from app.modules.amazon_ppc.entrypoints import campaigns
from app.modules.amazon_ppc.features import router as features_router
from app.modules.amazon_ppc.ml import router as ml_router
from app.modules.amazon_ppc.ml import advanced_router as advanced_ml_router
from app.modules.amazon_ppc.optimization import router as optimization_router
from app.modules.amazon_ppc.anomaly import router as anomaly_detection_router
from app.modules.amazon_ppc.competitive_intel.router import router as competitive_intel_router
from app.api.meta_skills import router as meta_skills_router
from app.api.rl_optimization import router as rl_budget_router
from app.core.config import settings
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.database import engine, Base
    # Import all models to ensure they're registered
    from app.modules.amazon_ppc.accounts.models import Account, Profile, Credential
    from app.modules.amazon_ppc.models.ppc_data import PPCCampaign, PPCKeyword, PerformanceRecord
    from app.modules.amazon_ppc.features.store import FeatureSnapshot
    # Market Intelligence Models (DataForSEO data persistence)
    from app.models.market_intelligence import MarketProduct, CompetitorPrice, KeywordRanking, MarketKeywordVolume
    # Anomaly Detection Models (Phase 6)
    from app.modules.amazon_ppc.anomaly.models import AnomalyAlert, AnomalyHistory, AnomalyTrainingData
    # Competitive Intelligence Models (Phase 7)
    from app.modules.amazon_ppc.competitive_intel.models import (
        CompetitorPriceHistory, PriceChangeEvent, CompetitorForecast, 
        UndercutProbability, StrategicSimulation, KeywordCannibalization
    )
    # Semantic Intelligence Models (Phase 8)
    from app.models.semantic import (
        SearchTermEmbedding, ProductEmbedding,
        SemanticBleedLog, SemanticOpportunityLog, AutonomousPatrolLog
    )
    
    # Only run DB migrations/seed if NOT running tests (or handle via env var)
    # But for now, we follow original logic
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    # Seed campaigns
    from app.db.seed import run_seed
    try:
        seeded = await run_seed()
        if seeded:
            print("Database seeded with initial campaigns.")
    except Exception as e:
        print(f"Seed skipped or failed: {e}")
        
    # Start Persistent Scheduler for background tasks
    from app.core.scheduler import scheduler
    from app.services.orchestrator import orchestrator
    import asyncio
    
    # Define a sample background optimization task
    async def daily_optimization_loop():
        # In production, you would fetch all active accounts from DB
        # For demo, we run for a specific target
        print("Running daily AI optimization loop...")
        # result = await orchestrator.execute_optimization_mission("ACC001", "B0DWK3C1R7")
        # print(f"Mission complete: {result}")
    
    # Schedule it to run every 24 hours (1440 minutes)
    scheduler.schedule_task("daily_ai_optimization", daily_optimization_loop, 1440)
    
    scheduler_task = asyncio.create_task(scheduler.start())
    print("Persistent Scheduler initialized and running in background.")
        
    yield
    # Cleanup (if any)
    scheduler.stop()
    await engine.dispose()

app = FastAPI(
    title="Optimus Pryme API",
    description="AI-powered War Room for Amazon Sellers",
    version="1.0.0",
    lifespan=lifespan
)


# CORS Configuration (origins from env CORS_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.websockets import chat as chat_ws

# Include API Routers
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(campaigns.router, prefix="/api/v1/campaigns", tags=["Campaigns"])
app.include_router(anomalies.router, prefix="/api/v1", tags=["Anomalies"])  # GPT-4 powered
app.include_router(creative.router, prefix="/api/v1", tags=["Creative AI"])  # Claude powered
app.include_router(audit.router, prefix="/api/v1/audit", tags=["Audit"])
app.include_router(discovery.router, prefix="/api/v1/discovery", tags=["Discovery"])
app.include_router(performance.router, prefix="/api/v1/performance", tags=["Performance"])
app.include_router(lockdown.router, prefix="/api/v1/lockdown", tags=["Lockdown"])
app.include_router(validation.router, prefix="/api/v1/validation", tags=["Validation"])
app.include_router(dsp_api.router, prefix="/api/v1/dsp", tags=["DSP"])
app.include_router(accounts_router.router, prefix="/api/v1/accounts", tags=["Accounts"])
app.include_router(ingestion_router.router, prefix="/api/v1/ingestion", tags=["Ingestion"])
app.include_router(features_router.router, prefix="/api/v1/features", tags=["Features"])
app.include_router(ml_router.router, prefix="/api/v1/ml", tags=["ML"])
app.include_router(advanced_ml_router.router, prefix="/api/v1/ml/advanced", tags=["Advanced ML"])
app.include_router(optimization_router.router, prefix="/api/v1/optimization", tags=["Optimization"])
app.include_router(anomaly_detection_router.router, prefix="/api/v1/anomaly-detection", tags=["Anomaly Detection"])
app.include_router(competitive_intel_router, prefix="/api/v1/competitive", tags=["Competitive Intelligence"])
app.include_router(rl_budget_router, prefix="/api/v1/rl-budget", tags=["RL Budget Allocation"])
app.include_router(meta_skills_router, prefix="/api/v1/meta-skills", tags=["Meta-Skills"])
app.include_router(settings_api.router, prefix="/api/v1/settings", tags=["Settings"])
app.include_router(chat_ws.router, tags=["WebSockets"])

# Register Domain Skills (Financial Analyst, Campaign Strategist)
from app.api.domain_skills import router as domain_skills_router
app.include_router(domain_skills_router, prefix="/api/v1/skills", tags=["Domain Skills"])

# Register Market Intelligence API (DataForSEO data persistence)
from app.api.market_intelligence import router as market_intel_router
app.include_router(market_intel_router, prefix="/api/v1/market-intelligence", tags=["Market Intelligence"])

# Register Semantic Intelligence API (Vector Analytics)
from app.api.semantic import router as semantic_router
app.include_router(semantic_router, prefix="/api/v1/semantic", tags=["Semantic Intelligence"])



@app.get("/")
async def root():
    return {
        "message": "Grok AdMaster API",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
