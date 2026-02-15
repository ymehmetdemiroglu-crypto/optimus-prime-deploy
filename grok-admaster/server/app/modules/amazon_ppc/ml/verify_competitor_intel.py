
import asyncio
import sys
import os
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Dict, Any

# Add server root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_competitor_intel")

async def verify_flow():
    """
    Verify the data flow from ingestion to trend detection.
    Mocks DB and External APIs to isolate logic.
    """
    logger.info("Initializing Competitor Intel Verification...")
    
    # MOCK DB Session
    mock_session = AsyncMock()
    # When await session.execute() is called, it returns a MagicMock (the result object)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None # Default: product not found
    mock_result.scalars.return_value.all.return_value = [] # Default: empty list
    mock_session.execute.return_value = mock_result
    
    mock_session.commit = AsyncMock()
    
    # Mock return values for DB queries
    # Scenario: We have some internal performance data and some competitor price data
    
    # Mocking TrendDetector dependencies
    # We need to import classes AFTER sys.path modification
    try:
        from app.services.market_intelligence_ingester import MarketIntelligenceIngester
        from app.services.meta_skills.trend_detector import TrendDetector
        from app.models.market_intelligence import MarketProduct, CompetitorPrice
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return

    # 1. Verify Ingestion Logic
    logger.info("Step 1: Verifying Data Ingestion...")
    ingester = MarketIntelligenceIngester()
    
    # Dummy product data from DataForSEO
    test_products = [
        {"asin": "B01DUMMY001", "title": "Competitor X", "price": 19.99},
        {"asin": "B01DUMMY002", "title": "Competitor Y", "price": 24.99}
    ]
    
    # Patch the DB context manager
    with patch('app.services.market_intelligence_ingester.AsyncSessionLocal') as mock_db_ctx:
        mock_db_ctx.return_value.__aenter__.return_value = mock_session
        
        # Test ingestion
        summary = await ingester.ingest_amazon_products("test keyword", test_products)
        logger.info(f"Ingestion Summary: {summary}")
        
        # Check if DB execute was called (Upsert products + record prices)
        if mock_session.execute.called:
            logger.info("SUCCESS: DB operations triggered during ingestion.")
        else:
            logger.error("FAILURE: No DB operations detected.")

    # 2. Verify Trend Detection Integration
    logger.info("Step 2: Verifying Trend Detector Logic...")
    
    detector = TrendDetector() # No args
    
    # Simulate Data: Price War Scenario
    # Internal CPC Rising: [1.0, 1.1, 1.2, 1.3, 1.5] -> Momentum > 0
    internal_cpc = [1.0, 1.05, 1.10, 1.25, 1.40] 
    
    # Competitor Price Falling: [20, 19, 18, 17, 15] -> Momentum < 0
    competitor_prices = [20.0, 19.5, 18.0, 17.0, 15.5]
    
    context = detector.analyze_market_context(
        internal_cpc_history=internal_cpc,
        competitor_price_history=competitor_prices
    )
    
    logger.info(f"Market Context Result: {context}")
    
    if context['status'] == 'price_war_risk':
         logger.info("SUCCESS: Detected 'price_war_risk' scenario correctly.")
    else:
         logger.warning(f"Expected 'price_war_risk', got '{context.get('status')}'")

    # 3. Verify Data Retrieval (Service Layer)
    logger.info("Step 3: Verifying Data Retrieval Service...")
    
    # Mock get_price_history on the ingester instance we created
    # We need to test that the method constructs the query correctly
    
    # Just checking if the method exists and runs is a good start for "flow" verification
    # mocking session.execute result for get_price_history
    mock_history_result = MagicMock()
    # It returns a list of CompetitorPrice objects
    mock_price_record = MagicMock()
    mock_price_record.price = 19.99
    mock_price_record.date = '2023-01-01'
    mock_history_result.scalars.return_value.all.return_value = [mock_price_record]
    
    mock_session.execute.return_value = mock_history_result
    
    with patch('app.services.market_intelligence_ingester.AsyncSessionLocal') as mock_db_ctx:
        mock_db_ctx.return_value.__aenter__.return_value = mock_session
        
        history = await ingester.get_price_history("B01DUMMY001", days=30)
        logger.info(f"Retrieved {len(history)} price records.")
        if len(history) > 0:
            logger.info("SUCCESS: Price history retrieval functional.")
        else:
             logger.error("FAILURE: Price history retrieval returned empty.")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_flow())
