"""
Optimus Prime API - Main Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

# Initialize logging FIRST, before any other imports
from app.core.logging_config import setup_logging
setup_logging(env=settings.ENV, level="DEBUG" if settings.ENV == "development" else "INFO")

from app.api import dashboard, anomalies, creative, settings as settings_api, audit, discovery, performance, lockdown, validation
from app.api.dsp import dsp_api
from app.modules.auth.router import router as auth_router
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
from contextlib import asynccontextmanager
import sqlalchemy as sa

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.database import engine, Base
    # Import all models to ensure they're registered
    from app.modules.auth.models import User  # noqa: F811
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

    # Initialize logging
    from app.core.logging_config import get_logger
    logger = get_logger(__name__)

    # ===== Phase 3: Initialize Dependency Injection Container =====
    from app.core.container import init_container, shutdown_container
    try:
        await init_container()
    except Exception as e:
        logger.error(f"Failed to initialize DI container: {e}")
        if settings.ENV == "production":
            raise

    # ===== Phase 3: Initialize Cache Client =====
    from app.core.cache import cache_client
    try:
        await cache_client.connect()
    except Exception as e:
        logger.warning(f"Cache client initialization failed: {e}")

    # ===== Phase 3: Initialize Feature Store =====
    from app.ml.feature_store import feature_store
    from app.ml.feature_store.definitions import register_all_features
    try:
        # Set Redis client for feature caching
        if cache_client.is_enabled:
            feature_store._redis = cache_client._client

        # Register all feature definitions
        register_all_features(feature_store)
        logger.info("[OK] Feature store initialized with all feature groups")
    except Exception as e:
        logger.warning(f"Feature store initialization warning: {e}")

    # Verify database migrations have been run
    # Instead of auto-creating tables (unsafe in production), we check if they exist
    try:
        async with engine.begin() as conn:
            # Check if the accounts table exists (basic migration check)
            result = await conn.execute(
                sa.text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'accounts')"
                )
            )
            tables_exist = result.scalar()

            if not tables_exist:
                logger.warning(
                    "Database tables not found. Please run migrations: python manage_db.py migrate"
                )
                if settings.ENV == "development":
                    logger.info("Auto-creating tables in development mode...")
                    await conn.run_sync(Base.metadata.create_all)
                else:
                    raise RuntimeError(
                        "Database not initialized. Run 'python manage_db.py migrate' before starting the application."
                    )
            else:
                logger.info("Database connection verified")
    except Exception as e:
        logger.error(f"Database initialization check failed: {e}")
        if settings.ENV == "production":
            raise
        
    # Seed campaigns
    from app.db.seed import run_seed
    try:
        seeded = await run_seed()
        if seeded:
            logger.info("Database seeded with initial campaigns.")
    except Exception as e:
        logger.warning(f"Seed skipped or failed: {e}")

    # Start Persistent Scheduler for background tasks
    from app.core.scheduler import scheduler
    from app.services.orchestrator import orchestrator
    import asyncio

    # Background optimization task — runs every 24 hours
    async def daily_optimization_loop():
        logger.info("Running daily AI optimization loop...")
        # result = await orchestrator.execute_optimization_mission("ACC001", "B0DWK3C1R7")
        # logger.info(f"Mission complete: {result}")

    # Schedule it to run every 24 hours (1440 minutes)
    scheduler.schedule_task("daily_ai_optimization", daily_optimization_loop, 1440)

    # Keep a reference on app.state so the task is not garbage-collected and
    # can be properly cancelled during shutdown.
    app.state.scheduler_task = asyncio.create_task(scheduler.start())
    logger.info("Persistent Scheduler initialized and running in background.")

    yield

    # ===== Cleanup Phase 3 Components =====
    logger.info("Shutting down application...")

    # Stop scheduler: set the flag first so the loop exits cleanly, then
    # cancel the asyncio task so we don't have to wait for the next 30-second
    # sleep tick and avoid "Task was destroyed but it is pending!" warnings.
    scheduler.stop()
    scheduler_task = getattr(app.state, "scheduler_task", None)
    if scheduler_task and not scheduler_task.done():
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass

    # Shutdown cache client
    from app.core.cache import cache_client
    try:
        await cache_client.disconnect()
    except Exception as e:
        logger.error(f"Error disconnecting cache: {e}")

    # Shutdown DI container
    from app.core.container import shutdown_container
    try:
        await shutdown_container()
    except Exception as e:
        logger.error(f"Error shutting down DI container: {e}")

    # Dispose database engine
    await engine.dispose()

    logger.info("[OK] Application shutdown complete")

app = FastAPI(
    title="Optimus Prime API",
    description="AI-powered War Room for Amazon Sellers",
    version="1.0.0",
    lifespan=lifespan
)


# CORS Configuration - Restrictive whitelist approach
# Only allow specific origins, methods, and headers needed for the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,  # Whitelist from environment
    allow_credentials=True,  # Required for cookie-based auth
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS"  # Required for preflight requests
    ],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-CSRF-Token"
    ],
    max_age=600  # Cache preflight requests for 10 minutes
)

# Add correlation ID tracking and request logging
from app.core.middleware import CorrelationIDMiddleware, SecurityHeadersMiddleware
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

from app.websockets import chat as chat_ws

# Include API Routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
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
        "message": "Optimus Prime API",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
