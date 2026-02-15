
"""
Bid Optimizer Service orchestrating Thompson Sampling and rule-based optimizations.

Supports two modes:
  - 'static'     : classic Beta-distribution Thompson Sampling
  - 'contextual' : Bayesian linear-regression TS conditioned on features

Implements the full optimise → execute → evaluate → learn cycle.
"""
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timedelta
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.orm import joinedload

from app.modules.amazon_ppc.models.ppc_data import (
    PPCKeyword,
    PPCCampaign,
    PerformanceRecord,
)
from app.modules.amazon_ppc.ml.thompson_sampling_db import (
    ThompsonSamplingOptimizerDB,
    ContextualThompsonSamplingDB,
)

logger = logging.getLogger(__name__)

OptMode = Literal["static", "contextual"]


class BidOptimizerService:
    """
    Orchestrates bid optimization using Multi-Armed Bandits.

    Parameters
    ----------
    db : AsyncSession
    mode : OptMode
        'static'  → ThompsonSamplingOptimizerDB
        'contextual' → ContextualThompsonSamplingDB (Phase 1 upgrade)
    """

    def __init__(self, db: AsyncSession, mode: OptMode = "contextual"):
        self.db = db
        self.mode = mode
        if mode == "contextual":
            self.thompson = ContextualThompsonSamplingDB(db)
        else:
            self.thompson = ThompsonSamplingOptimizerDB(db)

    # ══════════════════════════════════════════════════════════════════
    #  1. Optimise   (Plan → Execute)
    # ══════════════════════════════════════════════════════════════════

    async def optimize_profile(
        self, profile_id: str, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run bid optimization for all AI-enabled keywords in a profile.
        """
        logger.info(
            f"Starting {self.mode} TS optimization for profile "
            f"{profile_id} (dry_run={dry_run})"
        )

        # ── Create Optimization Plan ──────────────────────────────────
        result = await self.db.execute(
            text("""
                INSERT INTO optimization_plans
                    (campaign_id, generated_at, status, intelligence_level)
                VALUES
                    (NULL, NOW(), 'pending', :level)
                RETURNING id
            """),
            {"level": f"thompson_sampling_{self.mode}"},
        )
        plan_id = result.scalar()

        # ── Fetch Keywords ────────────────────────────────────────────
        stmt = (
            select(PPCKeyword)
            .join(PPCKeyword.campaign)
            .where(
                PPCCampaign.profile_id == profile_id,
                PPCCampaign.state == "enabled",
                PPCKeyword.state == "enabled",
            )
        )
        try:
            stmt = stmt.where(text("ppc_campaigns.ai_mode IS NOT NULL"))
        except Exception:
            pass

        keywords = (await self.db.execute(stmt)).scalars().all()
        logger.info(f"Found {len(keywords)} keywords to optimize")

        actions_executed = 0
        actions_details: List[Dict[str, Any]] = []

        for keyword in keywords:
            # ── Select arm ─────────────────────────────────────────────
            arm_id, multiplier, expected_reward = await self.thompson.select_arm(
                keyword.id
            )

            current_bid = float(keyword.bid)
            proposed_bid = current_bid * multiplier

            # ── Constraints ────────────────────────────────────────────
            min_bid = 0.02
            max_bid = 10.00
            final_bid = max(min_bid, min(proposed_bid, max_bid))

            if abs(final_bid - current_bid) < 0.01:
                continue

            action_status = "pending" if not dry_run else "simulated"

            # ── Record action ──────────────────────────────────────────
            action_result = await self.db.execute(
                text("""
                    INSERT INTO optimization_actions (
                        plan_id, action_type, entity_type, entity_id,
                        old_value, new_value, confidence_score,
                        reasoning, status
                    ) VALUES (
                        :plan_id, 'bid_adjustment', 'keyword', :entity_id,
                        :old_value, :new_value, :confidence,
                        :reasoning, :status
                    )
                    RETURNING id
                """),
                {
                    "plan_id": plan_id,
                    "entity_id": keyword.id,
                    "old_value": str(current_bid),
                    "new_value": str(final_bid),
                    "confidence": expected_reward,
                    "reasoning": (
                        f"{self.mode.title()} Thompson Sampling: "
                        f"arm {arm_id}, mult {multiplier:.2f}"
                    ),
                    "status": action_status,
                },
            )
            action_id = action_result.scalar()

            actions_details.append(
                {
                    "keyword_id": keyword.id,
                    "arm_id": arm_id,
                    "old_bid": current_bid,
                    "new_bid": final_bid,
                    "multiplier": multiplier,
                    "confidence": expected_reward,
                }
            )

            if not dry_run:
                try:
                    keyword.bid = final_bid
                    keyword.updated_at = datetime.utcnow()
                    self.db.add(keyword)

                    await self.db.execute(
                        text(
                            "UPDATE optimization_actions "
                            "SET status = 'executed', executed_at = NOW() "
                            "WHERE id = :id"
                        ),
                        {"id": action_id},
                    )

                    # Decision audit trail
                    await self.db.execute(
                        text("""
                            INSERT INTO decision_audit (
                                profile_id, decision_type,
                                options_considered, chosen_option,
                                reasoning, confidence
                            ) VALUES (
                                :pid, 'bid_optimization',
                                :opts, :chosen, :reason, :conf
                            )
                        """),
                        {
                            "pid": profile_id,
                            "opts": str(self.thompson.multipliers),
                            "chosen": str(multiplier),
                            "reason": (
                                f"[{self.mode}] Arm {arm_id} selected "
                                f"for kw {keyword.id}"
                            ),
                            "conf": expected_reward,
                        },
                    )
                    actions_executed += 1
                except Exception as e:
                    logger.error(f"Failed to update keyword {keyword.id}: {e}")
                    await self.db.execute(
                        text(
                            "UPDATE optimization_actions "
                            "SET status = 'failed' WHERE id = :id"
                        ),
                        {"id": action_id},
                    )

        await self.db.commit()

        final_status = "completed" if actions_executed > 0 else "no_changes"
        await self.db.execute(
            text("UPDATE optimization_plans SET status = :s WHERE id = :id"),
            {"s": final_status, "id": plan_id},
        )
        await self.db.commit()

        return {
            "plan_id": plan_id,
            "mode": self.mode,
            "keywords_analyzed": len(keywords),
            "actions_executed": actions_executed,
            "dry_run": dry_run,
            "actions": actions_details if dry_run else [],
        }

    # ══════════════════════════════════════════════════════════════════
    #  2. Evaluate → Learn   (close the feedback loop)
    # ══════════════════════════════════════════════════════════════════

    async def evaluate_and_learn(
        self,
        profile_id: str,
        lookback_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Look at executed bid changes, compare pre/post performance,
        calculate rewards, and update bandit arm posteriors.

        This is the *learning step* that closes the explore→exploit loop.
        Should be called daily (via scheduler) after performance data
        has been synced from the Amazon Ads API.
        """
        logger.info(
            f"[{self.mode}] Evaluating & learning for profile {profile_id}"
        )

        # Fetch executed bid actions in the lookback window
        rows = (
            await self.db.execute(
                text("""
                    SELECT
                        oa.entity_id   AS keyword_id,
                        oa.old_value   AS old_bid,
                        oa.new_value   AS new_bid,
                        oa.reasoning,
                        oa.executed_at,
                        c.target_acos
                    FROM optimization_actions oa
                    JOIN ppc_keywords k  ON oa.entity_id = k.id
                    JOIN ppc_campaigns c ON k.campaign_id = c.id
                    WHERE c.profile_id  = :pid
                      AND oa.action_type = 'bid_adjustment'
                      AND oa.status      = 'executed'
                      AND oa.executed_at >= NOW() - MAKE_INTERVAL(days => :days)
                    ORDER BY oa.executed_at ASC
                """),
                {"pid": profile_id, "days": lookback_days},
            )
        ).mappings().all()

        updated = 0
        skipped = 0

        for row in rows:
            keyword_id = row["keyword_id"]
            executed_at = row["executed_at"]
            target_acos = float(row["target_acos"] or 0.3)

            # Parse arm_id from reasoning  ("…: arm X, mult …")
            arm_id = self._parse_arm_id(row["reasoning"])
            if arm_id is None:
                skipped += 1
                continue

            # ── Fetch pre- and post-execution performance ─────────
            pre_metrics = await self._aggregate_metrics(
                keyword_id,
                end=executed_at,
                days=3,
            )
            post_metrics = await self._aggregate_metrics(
                keyword_id,
                start=executed_at,
                days=3,
            )

            if not pre_metrics or not post_metrics:
                skipped += 1
                continue

            # ── Calculate reward ─────────────────────────────────
            reward = self.thompson.calculate_reward(
                pre_metrics, post_metrics, target_acos
            )

            # ── Update arm posterior ─────────────────────────────
            await self.thompson.update_arm(keyword_id, arm_id, reward)
            updated += 1

            logger.debug(
                f"kw={keyword_id} arm={arm_id} reward={reward:.3f} "
                f"(pre_acos={pre_metrics.get('acos',0):.2%} "
                f"→ post_acos={post_metrics.get('acos',0):.2%})"
            )

        logger.info(
            f"[{self.mode}] Learning complete: "
            f"{updated} arms updated, {skipped} skipped"
        )
        return {"updated": updated, "skipped": skipped, "total": len(rows)}

    # ══════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════

    async def _aggregate_metrics(
        self,
        keyword_id: int,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: int = 3,
    ) -> Optional[Dict[str, float]]:
        """Aggregate spend / sales / clicks / impressions over a window."""
        if start and not end:
            end = start + timedelta(days=days)
        elif end and not start:
            start = end - timedelta(days=days)
        elif not start and not end:
            end = datetime.utcnow()
            start = end - timedelta(days=days)

        result = await self.db.execute(
            text("""
                SELECT
                    COALESCE(SUM(spend), 0)       AS spend,
                    COALESCE(SUM(sales), 0)       AS sales,
                    COALESCE(SUM(clicks), 0)      AS clicks,
                    COALESCE(SUM(impressions), 0) AS impressions,
                    COALESCE(SUM(orders), 0)      AS orders
                FROM performance_records
                WHERE keyword_id = :kid
                  AND date >= :start
                  AND date <  :end
            """),
            {"kid": keyword_id, "start": start, "end": end},
        )
        row = result.mappings().first()
        if row is None or float(row["impressions"]) == 0:
            return None

        spend = float(row["spend"])
        sales = float(row["sales"])
        return {
            "spend": spend,
            "sales": sales,
            "clicks": float(row["clicks"]),
            "impressions": float(row["impressions"]),
            "orders": float(row["orders"]),
            "acos": spend / sales if sales > 0 else 2.0,
        }

    @staticmethod
    def _parse_arm_id(reasoning: str) -> Optional[int]:
        """Extract arm index from reasoning string like 'arm 3, mult 1.10'."""
        try:
            # Pattern: "… arm <N>,"
            idx = reasoning.lower().index("arm ")
            num_str = reasoning[idx + 4 :].split(",")[0].split()[0]
            return int(num_str)
        except (ValueError, IndexError):
            return None

