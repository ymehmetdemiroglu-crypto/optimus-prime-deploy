"""
MCP Cortex — Natural Language Interface to Semantic Intelligence

This MCP server exposes semantic analysis tools to Claude/LLMs via
the Model Context Protocol, enabling conversational analytics.

Usage: Runs alongside the main API as a separate MCP server.
"""
from fastmcp import FastMCP
from typing import Dict, List, Optional
import sys
import os
import json
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mcp = FastMCP("Optimus-Cortex")


@mcp.tool()
async def analyze_semantic_bleed(asin: str, account_id: int, threshold: float = 0.40) -> Dict:
    """
    Detect search terms that are wasting budget by being semantically distant
    from the target product. Returns terms that should be added as negatives.

    Args:
        asin: The product ASIN to analyze.
        account_id: The client account ID.
        threshold: Similarity threshold below which terms are flagged (0-1).
    """
    from app.core.database import get_db_session
    from app.services.analytics.semantic_engine import BleedDetector

    async with get_db_session() as db:
        detector = BleedDetector(db)
        results = await detector.detect_bleed(
            asin=asin,
            account_id=account_id,
            similarity_threshold=threshold
        )

    total_waste = sum(r["spend"] for r in results)
    return {
        "asin": asin,
        "bleed_terms_found": len(results),
        "total_wasted_spend": round(total_waste, 2),
        "top_bleeders": results[:10],
        "recommendation": (
            f"Found {len(results)} semantically irrelevant terms costing ${total_waste:.2f}. "
            f"Add these as negative keywords to stop wasted spend."
        )
    }


@mcp.tool()
async def find_untapped_opportunities(asin: str, account_id: int, min_orders: int = 1) -> Dict:
    """
    Discover high-value search terms that are semantically related to the product
    and already converting, but are not being explicitly targeted.

    Args:
        asin: The product ASIN.
        account_id: The client account ID.
        min_orders: Minimum number of orders to qualify as an opportunity.
    """
    from app.core.database import get_db_session
    from app.services.analytics.semantic_engine import OpportunityFinder

    async with get_db_session() as db:
        finder = OpportunityFinder(db)
        results = await finder.find_opportunities(
            asin=asin,
            account_id=account_id,
            min_orders=min_orders
        )

    total_potential = sum(r["sales"] for r in results)
    return {
        "asin": asin,
        "opportunities_found": len(results),
        "total_revenue_potential": round(total_potential, 2),
        "top_opportunities": results[:10],
        "recommendation": (
            f"Found {len(results)} high-converting terms with ${total_potential:.2f} "
            f"in proven revenue. Add these as exact/phrase match targets."
        )
    }


@mcp.tool()
async def get_semantic_health_report(account_id: int) -> Dict:
    """
    Generate a complete semantic health report for an account.
    Includes bleed stats, opportunity count, and patrol activity.

    Args:
        account_id: The client account ID.
    """
    from app.core.database import get_db_session
    from sqlalchemy import text

    async with get_db_session() as db:
        # Count embeddings
        emb_count = await db.execute(text(
            "SELECT COUNT(*) as cnt FROM search_term_embeddings WHERE account_id = :aid"
        ), {"aid": account_id})
        embedding_count = emb_count.scalar() or 0

        # Count products
        prod_count = await db.execute(text(
            "SELECT COUNT(*) as cnt FROM product_embeddings WHERE account_id = :aid"
        ), {"aid": account_id})
        product_count = prod_count.scalar() or 0

        # Recent patrol activity
        patrol_count = await db.execute(text(
            "SELECT COUNT(*) as cnt FROM autonomous_patrol_log WHERE executed_at > NOW() - INTERVAL '7 days'"
        ))
        recent_patrols = patrol_count.scalar() or 0

        # Recent bleeds
        bleed_count = await db.execute(text(
            "SELECT COUNT(*) as cnt FROM semantic_bleed_log WHERE detected_at > NOW() - INTERVAL '7 days'"
        ))
        recent_bleeds = bleed_count.scalar() or 0

    return {
        "account_id": account_id,
        "semantic_coverage": {
            "search_terms_embedded": embedding_count,
            "products_embedded": product_count,
        },
        "last_7_days": {
            "patrol_actions": recent_patrols,
            "bleeds_detected": recent_bleeds,
        },
        "status": "healthy" if embedding_count > 0 and product_count > 0 else "needs_setup"
    }


@mcp.tool()
async def embed_new_product(asin: str, title: str, account_id: int, bullet_points: List[str] = None) -> Dict:
    """
    Generate and store the semantic identity of a product.
    Call this when onboarding a new ASIN to the semantic analysis system.

    Args:
        asin: The Amazon ASIN.
        title: The product title.
        account_id: The client account ID.
        bullet_points: Optional list of bullet points for richer embedding.
    """
    from app.core.database import get_db_session
    from app.services.analytics.semantic_engine import SemanticIngestor

    async with get_db_session() as db:
        ingestor = SemanticIngestor(db)
        product = await ingestor.embed_product(
            asin=asin,
            title=title,
            bullet_points=bullet_points or [],
            account_id=account_id
        )

    return {
        "status": "embedded",
        "asin": asin,
        "title": title,
        "embedding_id": str(product.id),
        "message": f"Product {asin} is now registered in the semantic system."
    }


@mcp.tool()
async def query_patrol_log(limit: int = 20) -> Dict:
    """
    View recent autonomous patrol activity log.
    Shows what the AI operator has been doing automatically.

    Args:
        limit: Number of recent log entries to return.
    """
    from app.core.database import get_db_session
    from sqlalchemy import text

    async with get_db_session() as db:
        result = await db.execute(text(
            "SELECT patrol_cycle, action_type, target_entity, details, status, executed_at "
            "FROM autonomous_patrol_log ORDER BY executed_at DESC LIMIT :limit"
        ), {"limit": limit})
        rows = result.fetchall()

    return {
        "total_entries": len(rows),
        "logs": [
            {
                "cycle": r.patrol_cycle,
                "action": r.action_type,
                "target": r.target_entity,
                "details": r.details,
                "status": r.status,
                "time": r.executed_at.isoformat() if r.executed_at else None
            }
            for r in rows
        ]
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
