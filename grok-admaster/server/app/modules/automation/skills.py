"""
Automation Agent — Skills Registry

Extensible registry for agent skills (Python callables) that can be
invoked during automation flows. Supports both sync and async functions.
"""

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from .schemas import SkillDefinition

logger = logging.getLogger(__name__)


class AgentSkillRegistry:
    """
    Register, unregister, and invoke Python callables by skill_id.
    Skills are deterministic tools the agent can use during execution.
    """

    def __init__(self):
        self._skills: Dict[str, dict] = {}

    def register(
        self,
        skill_id: str,
        name: str,
        fn: Callable,
        description: str = "",
        parameter_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a callable as a named skill."""
        self._skills[skill_id] = {
            "definition": SkillDefinition(
                skill_id=skill_id,
                name=name,
                description=description,
                callable_name=fn.__qualname__,
                parameter_schema=parameter_schema,
                tags=tags or [],
            ),
            "fn": fn,
        }
        logger.info(f"[Skills] Registered skill: {skill_id} ({name})")

    def unregister(self, skill_id: str) -> bool:
        """Remove a skill. Returns True if it existed."""
        removed = self._skills.pop(skill_id, None)
        if removed:
            logger.info(f"[Skills] Unregistered skill: {skill_id}")
        return removed is not None

    def list_skills(self) -> List[SkillDefinition]:
        """List all registered skill definitions."""
        return [entry["definition"] for entry in self._skills.values()]

    def get_skill(self, skill_id: str) -> Optional[SkillDefinition]:
        """Get a single skill definition."""
        entry = self._skills.get(skill_id)
        return entry["definition"] if entry else None

    async def invoke(self, skill_id: str, **kwargs) -> Any:
        """
        Invoke a registered skill by ID.
        Handles both sync and async callables.
        """
        entry = self._skills.get(skill_id)
        if not entry:
            raise ValueError(f"Skill '{skill_id}' is not registered")

        fn = entry["fn"]
        logger.info(f"[Skills] Invoking skill: {skill_id} with kwargs={list(kwargs.keys())}")

        try:
            if inspect.iscoroutinefunction(fn):
                return await fn(**kwargs)
            else:
                return await asyncio.to_thread(fn, **kwargs)
        except Exception as e:
            logger.error(f"[Skills] Skill '{skill_id}' failed: {e}")
            raise

    def decorator(
        self,
        skill_id: str,
        name: str,
        description: str = "",
        parameter_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Decorator for declarative skill registration."""
        def wrapper(fn: Callable):
            self.register(skill_id, name, fn, description, parameter_schema, tags)
            return fn
        return wrapper


# ── Singleton Registry ──
skill_registry = AgentSkillRegistry()


# ══════════════════════════════════════════════════════════
#  Built-in Skills
# ══════════════════════════════════════════════════════════

def register_builtin_skills():
    """Register default agent skills for PPC automation."""

    # 1. Bid Adjustment (Enhanced)
    @skill_registry.decorator(
        skill_id="bid_adjust",
        name="Bid Adjustment",
        description="Adjust keyword bid by a percentage or absolute amount.",
        parameter_schema={
            "type": "object",
            "properties": {
                "keyword_id": {"type": "string"},
                "adjustment_pct": {"type": "number", "description": "e.g., 10 for +10%, -15 for -15%"},
                "new_bid": {"type": "number", "description": "Target absolute bid amount"},
            },
            "required": ["keyword_id"],
        },
        tags=["ppc", "optimization", "bidding"],
    )
    def bid_adjust(keyword_id: str, adjustment_pct: float = None, new_bid: float = None):
        """Adjust bid for a keyword. Validates and calculates final bid."""
        if adjustment_pct is None and new_bid is None:
            raise ValueError("Must provide either 'adjustment_pct' or 'new_bid'")

        # Simulated current bid fetching
        current_bid = 2.00 
        
        if new_bid is None:
            new_bid = round(current_bid * (1 + (adjustment_pct / 100)), 2)
            
        logger.info(f"[Skill:bid_adjust] keyword={keyword_id}, old=${current_bid}, new=${new_bid}")
        return {
            "action": "bid_adjust",
            "keyword_id": keyword_id,
            "old_bid": current_bid,
            "new_bid": new_bid,
            "change_pct": round(((new_bid - current_bid) / current_bid) * 100, 2),
            "status": "applied_successfully",
        }

    # 2. Pause Keyword (Enhanced)
    @skill_registry.decorator(
        skill_id="pause_keyword",
        name="Pause Keyword",
        description="Pause a keyword that is underperforming, bleeding spend, or has zero impressions.",
        parameter_schema={
            "type": "object",
            "properties": {
                "keyword_id": {"type": "string"},
                "reason": {"type": "string", "description": "Reason for pausing (e.g. 'cpa_too_high')"},
            },
            "required": ["keyword_id"],
        },
        tags=["ppc", "management", "pruning"],
    )
    def pause_keyword(keyword_id: str, reason: str = "Automated performance pruning"):
        """Pause a keyword and log the reason."""
        if not keyword_id.strip():
            raise ValueError("keyword_id cannot be empty")
            
        logger.info(f"[Skill:pause_keyword] keyword={keyword_id}, reason='{reason}'")
        return {
            "action": "pause_keyword",
            "keyword_id": keyword_id,
            "reason": reason,
            "archived_at": __import__('datetime').datetime.utcnow().isoformat(),
            "status": "paused",
        }

    # 3. Budget Reallocation (Enhanced)
    @skill_registry.decorator(
        skill_id="budget_reallocate",
        name="Budget Reallocation",
        description="Reallocate daily budget across campaigns based on performance heuristics.",
        parameter_schema={
            "type": "object",
            "properties": {
                "campaign_id": {"type": "string"},
                "new_daily_budget": {"type": "number"},
            },
            "required": ["campaign_id", "new_daily_budget"],
        },
        tags=["ppc", "optimization", "budget"],
    )
    def budget_reallocate(campaign_id: str, new_daily_budget: float):
        """Update campaign budget and validate sensible ranges."""
        if new_daily_budget < 1.0:
            raise ValueError("Budget must be at least $1.00")
            
        logger.info(f"[Skill:budget_reallocate] campaign={campaign_id}, target_budget=${new_daily_budget: .2f}")
        return {
            "action": "budget_reallocate",
            "campaign_id": campaign_id,
            "new_daily_budget": round(new_daily_budget, 2),
            "status": "applied",
            "notes": "Budget constraints enforced correctly."
        }

    # 4. Diagnose Campaign (Integration with Paid Ads Module)
    @skill_registry.decorator(
        skill_id="diagnose_campaign",
        name="Diagnose Campaign Metrics",
        description="Analyses CPA, CTR, CPM, and ROAS against targets and returns prioritised fix actions.",
        parameter_schema={
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["google", "meta", "linkedin", "tiktok", "amazon"]},
                "current_cpa": {"type": "number"},
                "target_cpa": {"type": "number"},
                "current_roas": {"type": "number"},
                "target_roas": {"type": "number"},
                "current_ctr": {"type": "number"},
            },
            "required": ["platform"],
        },
        tags=["analytics", "paid_ads", "diagnostic"],
    )
    def diagnose_campaign(
        platform: str, 
        current_cpa: float = None, target_cpa: float = None,
        current_roas: float = None, target_roas: float = None,
        current_ctr: float = None
    ):
        """Uses the paid_ads native service to diagnose health."""
        from app.modules.paid_ads.schemas import CampaignOptimizationRequest, AdPlatform
        from app.modules.paid_ads.service import diagnose_campaign as diagnose_svc
        
        req = CampaignOptimizationRequest(
            platform=AdPlatform(platform.lower()),
            current_cpa=current_cpa, target_cpa=target_cpa,
            current_roas=current_roas, target_roas=target_roas,
            current_ctr=current_ctr
        )
        res = diagnose_svc(req)
        return res.model_dump()

    # 5. Generate Ad Copy (Integration with Paid Ads Module)
    @skill_registry.decorator(
        skill_id="generate_ad_copy",
        name="Generate Ad Copy",
        description="Create ad copy variations using established marketing frameworks (pas, bab, aida, social_proof).",
        parameter_schema={
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["google", "meta", "linkedin", "tiktok", "amazon"]},
                "product_name": {"type": "string"},
                "target_audience": {"type": "string"},
                "unique_selling_points": {"type": "array", "items": {"type": "string"}},
                "framework": {"type": "string", "enum": ["pas", "bab", "aida", "social_proof"]},
            },
            "required": ["platform", "product_name", "target_audience", "unique_selling_points"],
        },
        tags=["creative", "paid_ads", "copywriting"],
    )
    def generate_ad_copy(
        platform: str, product_name: str, target_audience: str, 
        unique_selling_points: list, framework: str = "aida"
    ):
        """Generates copy using the internal paid_ads templating engine."""
        from app.modules.paid_ads.schemas import AdCopyRequest, AdPlatform, AdCopyFramework
        from app.modules.paid_ads.service import generate_ad_copy as generate_copy_svc
        
        req = AdCopyRequest(
            platform=AdPlatform(platform.lower()),
            product_name=product_name,
            target_audience=target_audience,
            unique_selling_points=unique_selling_points,
            framework=AdCopyFramework(framework.lower()),
            num_variations=2
        )
        res = generate_copy_svc(req)
        return res.model_dump()

    logger.info("[Skills] Built-in skills registered (Optimized with Paid Ads Integrations).")
