import logging
from typing import Dict, Any
from app.core.database import get_db_session
from app.core.credentials import CredentialManager
from app.services.openrouter_service import OpenRouterClient
# Assuming these services exist or will be refactored to accept injected creds
from app.modules.amazon_ppc.services.sp_api import SPAPIService
from app.modules.amazon_ppc.services.ads_api import AdsAPIService

class DirectOrchestrator:
    """Deterministic, non-MCP orchestrator for multi-tenant AI automation."""
    
    def __init__(self):
        self.logger = logging.getLogger("orchestrator")
        self.ai_client = OpenRouterClient()
        self.sp_service = SPAPIService()
        self.ads_service = AdsAPIService()

    async def execute_optimization_mission(self, account_id: str, target_asin: str):
        """
        Runs a complete optimization loop for a specific client:
        Data Ingestion -> AI Analysis -> Direct Execution
        """
        self.logger.info(f"Starting mission for Account: {account_id} | ASIN: {target_asin}")
        
        async with get_db_session() as db:
            try:
                # 1. Inject Credentials
                creds = await CredentialManager.get_client_credentials(db, account_id)
                
                # 2. Fetch Live Market Data (Direct API)
                self.logger.info("Fetching market data...")
                market_context = await self.sp_service.get_competitive_context(
                    creds=creds, 
                    asin=target_asin
                )
                
                # 3. Generate Strategy via OpenRouter (Deterministic AI)
                self.logger.info("Consulting OpenRouter for strategy...")
                strategy = await self.ai_client.generate_strategy(
                    context=market_context,
                    objective="maximize_roas"
                )
                
                # 4. Execute Campaign Actions (Direct API)
                self.logger.info(f"Executing strategy: {strategy['strategy_name']}")
                execution_result = await self.ads_service.apply_strategy(
                    creds=creds,
                    asin=target_asin,
                    strategy=strategy
                )
                
                return {
                    "status": "success",
                    "account_id": account_id,
                    "strategy": strategy['strategy_name'],
                    "actions_taken": execution_result
                }
                
            except Exception as e:
                self.logger.error(f"Mission failed for {account_id}: {str(e)}")
                return {"status": "error", "message": str(e)}

# Global Instance
orchestrator = DirectOrchestrator()
