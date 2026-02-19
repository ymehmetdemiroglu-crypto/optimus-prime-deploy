"""
Autonomous Operator — The Permanent AI Pilot
=============================================

This is the "heartbeat" of the system. It runs as a background daemon
on your server and performs continuous optimization without human input.

Architecture:
    1. INGEST  → Pull latest search terms, generate embeddings
    2. ANALYZE → Run BleedDetector and OpportunityFinder
    3. DECIDE  → Use AI (OpenRouter) to validate and strategize
    4. ACT     → Execute changes (add negatives, expand targets)
    5. LOG     → Record every action in autonomous_patrol_log
    6. SLEEP   → Wait for next cycle (default: 6 hours)

Usage:
    python autonomous_operator.py              # Run forever
    python autonomous_operator.py --once       # Run one cycle and exit
    python autonomous_operator.py --dry-run    # Simulate without executing
"""
import asyncio
import json
import logging
import os
import sys
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db_session, engine, Base
from app.services.analytics.semantic_engine import SemanticIngestor, BleedDetector, OpportunityFinder
from app.services.openrouter_service import OpenRouterClient
from app.models.semantic import AutonomousPatrolLog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("autonomous_operator")


class AutonomousOperator:
    """
    The AI Pilot. Runs continuously on your server.
    
    It does not need user input. It reads the database, thinks,
    and takes action based on semantic intelligence.
    """
    
    def __init__(
        self,
        patrol_interval_hours: float = 6.0,
        dry_run: bool = False,
        bleed_threshold: float = 0.40,
        opportunity_floor: float = 0.70
    ):
        self.patrol_interval = patrol_interval_hours * 3600  # Convert to seconds
        self.dry_run = dry_run
        self.bleed_threshold = bleed_threshold
        self.opportunity_floor = opportunity_floor
        self.patrol_cycle = 0
        self.ai_client = OpenRouterClient()
        
        logger.info("=" * 60)
        logger.info("  OPTIMUS PRIME — AUTONOMOUS OPERATOR INITIALIZED")
        logger.info(f"  Mode: {'DRY RUN (Simulation)' if dry_run else 'LIVE (Actions will execute)'}")
        logger.info(f"  Patrol Interval: {patrol_interval_hours} hours")
        logger.info(f"  Bleed Threshold: {bleed_threshold}")
        logger.info(f"  Opportunity Floor: {opportunity_floor}")
        logger.info("=" * 60)
    
    async def run_forever(self):
        """Main loop: patrol → sleep → repeat."""
        logger.info("Autonomous patrol loop starting...")
        
        while True:
            try:
                self.patrol_cycle += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"  PATROL CYCLE #{self.patrol_cycle}")
                logger.info(f"  Time: {datetime.now(timezone.utc).isoformat()}")
                logger.info(f"{'='*60}")
                
                await self._run_single_patrol()
                
                logger.info(
                    f"Patrol #{self.patrol_cycle} complete. "
                    f"Sleeping for {self.patrol_interval / 3600:.1f} hours..."
                )
                await asyncio.sleep(self.patrol_interval)
                
            except KeyboardInterrupt:
                logger.info("Operator shutdown requested.")
                break
            except Exception as e:
                logger.error(f"Patrol #{self.patrol_cycle} failed: {e}", exc_info=True)
                logger.info("Retrying in 5 minutes...")
                await asyncio.sleep(300)
    
    async def run_once(self):
        """Run a single patrol cycle (for testing or cron jobs)."""
        self.patrol_cycle = 1
        await self._run_single_patrol()
    
    async def _run_single_patrol(self):
        """Execute one full patrol cycle across all accounts."""
        async with get_db_session() as db:
            # --- Step 1: Discover active accounts ---
            accounts = await self._get_active_accounts(db)
            
            if not accounts:
                logger.warning("No active accounts found. Nothing to patrol.")
                await self._log_action(db, "scan", "system", {"message": "No active accounts"})
                return
            
            logger.info(f"Found {len(accounts)} active account(s) to patrol.")
            
            for account in accounts:
                account_id = account["id"]
                account_name = account.get("name", f"Account-{account_id}")
                
                logger.info(f"\n--- Patrolling: {account_name} (ID: {account_id}) ---")
                
                try:
                    # --- Step 2: Ingest new search terms ---
                    ingestor = SemanticIngestor(db)
                    ingested = await ingestor.ingest_search_terms(account_id, limit=200)
                    await self._log_action(db, "ingest", account_name, {
                        "terms_embedded": ingested
                    })
                    
                    # --- Step 3: Get products for this account ---
                    products = await self._get_account_products(db, account_id)
                    
                    if not products:
                        logger.info(f"No product embeddings for {account_name}. Skipping analysis.")
                        continue
                    
                    # --- Step 4: Run analysis for each product ---
                    for product in products:
                        asin = product["asin"]
                        logger.info(f"  Analyzing ASIN: {asin}")
                        
                        # Run Bleed Detection
                        bleed_results = await self._detect_and_act_bleed(
                            db, asin, account_id
                        )
                        
                        # Run Opportunity Discovery
                        opportunity_results = await self._discover_and_act_opportunities(
                            db, asin, account_id
                        )
                        
                        # --- Step 5: AI Strategy Review ---
                        if bleed_results or opportunity_results:
                            await self._ai_strategy_review(
                                db, asin, account_name,
                                bleed_results, opportunity_results
                            )
                    
                except Exception as e:
                    logger.error(f"Error patrolling {account_name}: {e}", exc_info=True)
                    await self._log_action(db, "error", account_name, {"error": str(e)}, "error")
    
    async def _get_active_accounts(self, db: AsyncSession) -> List[Dict]:
        """Fetch all active accounts from the database."""
        from sqlalchemy import text
        result = await db.execute(text(
            "SELECT id, name, amazon_account_id FROM accounts WHERE status = 'active'"
        ))
        return [{"id": r.id, "name": r.name, "amazon_id": r.amazon_account_id} for r in result.fetchall()]
    
    async def _get_account_products(self, db: AsyncSession, account_id: int) -> List[Dict]:
        """Fetch products that have embeddings for this account."""
        from sqlalchemy import text
        result = await db.execute(text(
            "SELECT asin, title FROM product_embeddings WHERE account_id = :aid"
        ), {"aid": account_id})
        return [{"asin": r.asin, "title": r.title} for r in result.fetchall()]
    
    async def _detect_and_act_bleed(
        self,
        db: AsyncSession,
        asin: str,
        account_id: int
    ) -> List[Dict]:
        """Run bleed detection and take action."""
        detector = BleedDetector(db)
        bleeds = await detector.detect_bleed(
            asin=asin,
            account_id=account_id,
            similarity_threshold=self.bleed_threshold,
            min_spend=1.00,
            limit=20
        )
        
        if not bleeds:
            logger.info(f"    No bleed detected for {asin}")
            return []
        
        total_waste = sum(b["spend"] for b in bleeds)
        logger.warning(
            f"    🚨 BLEED DETECTED: {len(bleeds)} terms wasting ${total_waste:.2f} for {asin}"
        )
        
        for bleed in bleeds:
            logger.info(
                f"      → \"{bleed['term']}\" | similarity={bleed['semantic_similarity']:.2f} "
                f"| spend=${bleed['spend']:.2f} | urgency={bleed['urgency']}"
            )
            
            if not self.dry_run:
                # Log the action
                await detector.log_bleed_action(
                    search_term_embedding_id=bleed["embedding_id"],
                    product_embedding_id=bleed["product_embedding_id"],
                    semantic_distance=1.0 - bleed["semantic_similarity"],
                    spend=bleed["spend"],
                    action="negative_added",
                    operator="autonomous"
                )
        
        await self._log_action(db, "bleed_detect", asin, {
            "terms_found": len(bleeds),
            "total_waste": total_waste,
            "top_bleeders": [b["term"] for b in bleeds[:5]]
        })
        
        return bleeds
    
    async def _discover_and_act_opportunities(
        self,
        db: AsyncSession,
        asin: str,
        account_id: int
    ) -> List[Dict]:
        """Run opportunity discovery and log results."""
        finder = OpportunityFinder(db)
        opportunities = await finder.find_opportunities(
            asin=asin,
            account_id=account_id,
            similarity_floor=self.opportunity_floor,
            min_orders=1,
            limit=15
        )
        
        if not opportunities:
            logger.info(f"    No new opportunities for {asin}")
            return []
        
        total_revenue = sum(o["sales"] for o in opportunities)
        logger.info(
            f"    💡 OPPORTUNITIES: {len(opportunities)} high-value terms "
            f"(${total_revenue:.2f} revenue potential) for {asin}"
        )
        
        for opp in opportunities:
            logger.info(
                f"      → \"{opp['term']}\" | similarity={opp['semantic_similarity']:.2f} "
                f"| orders={opp['orders']} | confidence={opp['confidence']}"
            )
            
            if not self.dry_run:
                await finder.log_opportunity(
                    term=opp["term"],
                    asin=asin,
                    similarity=opp["semantic_similarity"],
                    match_type=opp["suggested_match_type"],
                    bid=opp["suggested_bid"]
                )
        
        await self._log_action(db, "opportunity_find", asin, {
            "terms_found": len(opportunities),
            "total_revenue_potential": total_revenue,
            "top_opportunities": [o["term"] for o in opportunities[:5]]
        })
        
        return opportunities
    
    async def _ai_strategy_review(
        self,
        db: AsyncSession,
        asin: str,
        account_name: str,
        bleeds: List[Dict],
        opportunities: List[Dict]
    ):
        """
        Use AI to generate a strategic summary and recommendations.
        This is the "thinking" step where the AI reviews findings.
        """
        context = {
            "asin": asin,
            "account": account_name,
            "patrol_cycle": self.patrol_cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bleed_summary": {
                "count": len(bleeds),
                "total_waste": sum(b["spend"] for b in bleeds),
                "top_3": [{"term": b["term"], "spend": b["spend"], "similarity": b["semantic_similarity"]} for b in bleeds[:3]]
            },
            "opportunity_summary": {
                "count": len(opportunities),
                "total_potential": sum(o["sales"] for o in opportunities),
                "top_3": [{"term": o["term"], "similarity": o["semantic_similarity"], "orders": o["orders"]} for o in opportunities[:3]]
            }
        }
        
        try:
            strategy = await self.ai_client.generate_strategy(
                context=context,
                objective="semantic_optimization_review"
            )
            
            logger.info(f"    🧠 AI STRATEGY: {strategy.get('strategy_name', 'N/A')}")
            logger.info(f"    📝 Reasoning: {strategy.get('reasoning', 'N/A')}")
            
            await self._log_action(db, "ai_review", asin, {
                "strategy": strategy.get("strategy_name"),
                "reasoning": strategy.get("reasoning"),
                "full_strategy": strategy
            })
            
        except Exception as e:
            logger.warning(f"    AI review skipped (non-critical): {e}")
    
    async def _log_action(
        self,
        db: AsyncSession,
        action_type: str,
        target: str,
        details: Dict,
        status: str = "success"
    ):
        """Record an action in the autonomous patrol log."""
        log = AutonomousPatrolLog(
            patrol_cycle=self.patrol_cycle,
            action_type=action_type,
            target_entity=target,
            details=details,
            status=status
        )
        db.add(log)
        await db.commit()


# ===================================================
# Entry Point
# ===================================================

async def main():
    parser = argparse.ArgumentParser(description="Optimus Prime Autonomous Operator")
    parser.add_argument("--once", action="store_true", help="Run one patrol cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without executing actions")
    parser.add_argument("--interval", type=float, default=6.0, help="Hours between patrols (default: 6)")
    parser.add_argument("--bleed-threshold", type=float, default=0.40, help="Bleed similarity threshold (0-1)")
    parser.add_argument("--opportunity-floor", type=float, default=0.70, help="Opportunity similarity floor (0-1)")
    args = parser.parse_args()
    
    # Ensure tables exist
    async with engine.begin() as conn:
        from app.models.semantic import (
            SearchTermEmbedding, ProductEmbedding,
            SemanticBleedLog, SemanticOpportunityLog, AutonomousPatrolLog
        )
        await conn.run_sync(Base.metadata.create_all)
    
    operator = AutonomousOperator(
        patrol_interval_hours=args.interval,
        dry_run=args.dry_run,
        bleed_threshold=args.bleed_threshold,
        opportunity_floor=args.opportunity_floor
    )
    
    if args.once:
        await operator.run_once()
    else:
        await operator.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
