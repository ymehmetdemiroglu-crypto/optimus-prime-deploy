"""
Hierarchical Reinforcement Learning for Budget Allocation.

Three-tier decision hierarchy:

    PortfolioAgent   ─▶ allocates total budget across campaigns
    CampaignAgent    ─▶ allocates campaign budget across ad groups / keywords
    KeywordAgent     ─▶ wraps Contextual Thompson Sampling for bid multipliers

State is persisted to ``rl_portfolio_state`` and ``rl_budget_actions``
tables so the system can learn across sessions.

Architecture
------------
The PortfolioAgent uses a softmax policy gradient approach:
  - State: per-campaign 7-day metrics, remaining budget, day-of-month
  - Action: budget allocation weights (softmax vector)
  - Reward: portfolio-wide ROAS improvement

The CampaignAgent uses a simple proportional-to-expected-value approach
conditioned on campaign-level signals.

The KeywordAgent delegates to ContextualThompsonSamplingDB.
"""
from __future__ import annotations

import json
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.modules.amazon_ppc.ml.thompson_sampling_db import (
    ContextualThompsonSamplingDB,
    ThompsonSamplingOptimizerDB,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  State / Action data classes
# ═══════════════════════════════════════════════════════════════════════

class PortfolioState:
    """Snapshot of portfolio-level state used by PortfolioAgent."""

    __slots__ = (
        "campaign_metrics", "total_budget", "budget_remaining",
        "day_of_month", "days_in_month", "global_acos", "global_roas",
        "num_campaigns",
    )

    def __init__(
        self,
        campaign_metrics: List[Dict[str, float]],
        total_budget: float,
        budget_remaining: float,
        day_of_month: int = 1,
        days_in_month: int = 30,
        global_acos: float = 0.3,
        global_roas: float = 3.0,
        num_campaigns: int = 0,
    ):
        self.campaign_metrics = campaign_metrics
        self.total_budget = total_budget
        self.budget_remaining = budget_remaining
        self.day_of_month = day_of_month
        self.days_in_month = days_in_month
        self.global_acos = global_acos
        self.global_roas = global_roas
        self.num_campaigns = num_campaigns or len(campaign_metrics)

    def to_vector(self) -> np.ndarray:
        """Flatten to fixed-length feature vector for policy network."""
        # Global scalars (5)
        scalars = np.array([
            self.total_budget / 10000,           # normalise
            self.budget_remaining / max(self.total_budget, 1),
            self.day_of_month / self.days_in_month,
            min(self.global_acos, 2.0),
            min(self.global_roas / 10, 2.0),
        ])
        # Per-campaign summaries (max 20 campaigns × 4 features = 80)
        MAX_CAMPS = 20
        camp_feats = np.zeros(MAX_CAMPS * 4)
        for i, m in enumerate(self.campaign_metrics[:MAX_CAMPS]):
            base = i * 4
            camp_feats[base] = min(m.get("spend_7d", 0) / 1000, 5.0)
            camp_feats[base + 1] = min(m.get("sales_7d", 0) / 5000, 5.0)
            camp_feats[base + 2] = min(m.get("acos_7d", 1.0), 2.0)
            camp_feats[base + 3] = min(m.get("roas_7d", 0) / 10, 5.0)
        return np.concatenate([scalars, camp_feats])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_metrics": self.campaign_metrics,
            "total_budget": self.total_budget,
            "budget_remaining": self.budget_remaining,
            "day_of_month": self.day_of_month,
            "days_in_month": self.days_in_month,
            "global_acos": self.global_acos,
            "global_roas": self.global_roas,
            "num_campaigns": self.num_campaigns,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Level 1 – Portfolio Agent
# ═══════════════════════════════════════════════════════════════════════

class PortfolioAgent:
    """
    Allocates total daily budget across campaigns.

    Uses a softmax policy parameterised by a linear weight vector θ.
    The policy is updated via REINFORCE-style policy gradient after
    each allocation cycle (daily).

    Parameters
    ----------
    n_campaigns : int
    learning_rate : float
    temperature : float   softmax temperature (exploration control)
    """

    # ── constants ──────────────────────────────────────────────────────
    STATE_DIM = 5 + 20 * 4  # must match PortfolioState.to_vector()
    MAX_CAMPAIGNS = 20      # hard cap matching state vector layout
    GRAD_CLIP = 1.0         # gradient clipping threshold

    def __init__(
        self,
        n_campaigns: int,
        learning_rate: float = 0.01,
        temperature: float = 1.0,
    ):
        self.n_campaigns = min(n_campaigns, self.MAX_CAMPAIGNS)
        self.lr = learning_rate
        self.temperature = temperature

        # θ: state_dim × n_campaigns weight matrix (initialised small)
        self.theta = np.random.randn(self.STATE_DIM, self.n_campaigns) * 0.01
        self.baseline = 0.0   # running average reward baseline
        self.t = 0  # Time step for annealing

        # Production enhancements
        self.action_counts = np.zeros(self.n_campaigns)  # Track exploration per campaign
        # Dynamic floor: ensure total floor never exceeds 50% of budget
        # e.g. 2 campaigns → 0.05 each, 20 campaigns → 0.025 each
        self.min_allocation = min(0.05, 0.50 / max(self.n_campaigns, 1))
        # UCB exploration: scales with campaign count. Disabled for small
        # portfolios (<5 campaigns) where softmax + temperature is sufficient.
        # For large portfolios, helps discover underexplored campaigns.
        self.exploration_bonus_coef = 0.02 if self.n_campaigns >= 5 else 0.0
        self.baseline_decay = 0.9  # EMA coefficient for baseline update
        self._recent_rewards: List[float] = []  # Track reward history for change detection

    # ── policy ────────────────────────────────────────────────────────

    def allocate(
        self, state: PortfolioState, explore: bool = True
    ) -> np.ndarray:
        """
        Return a budget allocation vector (sums to 1).

        If explore=True, sample from the softmax distribution.
        If explore=False, return the greedy (argmax) allocation.
        """
        sv = state.to_vector()

        # Validate state/theta dimension compatibility
        if sv.shape[0] != self.theta.shape[0]:
            logger.warning(
                f"[PortfolioAgent] State dim {sv.shape[0]} != theta dim "
                f"{self.theta.shape[0]}, padding/truncating"
            )
            if sv.shape[0] < self.theta.shape[0]:
                sv = np.pad(sv, (0, self.theta.shape[0] - sv.shape[0]))
            else:
                sv = sv[:self.theta.shape[0]]

        logits = sv @ self.theta / self.temperature

        # Add UCB-style exploration bonus (diminishes as actions are explored)
        # Only activate after initial warm-up and when coefficient is non-zero
        if explore and self.t > 0 and self.exploration_bonus_coef > 0:
            total_pulls = self.action_counts.sum()
            exploration_bonus = self.exploration_bonus_coef * np.sqrt(
                np.log(total_pulls + 1) / (self.action_counts + 1)
            )
            logits += exploration_bonus

        # Numerical stability
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        if explore:
            # Sample stochastic allocation, then re-normalise
            allocation = np.random.dirichlet(probs * 100 + 1e-6)
        else:
            allocation = probs

        # Apply minimum allocation floor to maintain monitoring
        # The floor is dynamically scaled so total floors never exceed 50%
        if self.min_allocation > 0 and self.n_campaigns > 0:
            allocation = np.maximum(allocation, self.min_allocation)
            allocation = allocation / allocation.sum()  # Re-normalize

        # Track which actions were selected (discrete step per allocation)
        if explore:
            self.action_counts += 1  # Simple discrete count per step

        return allocation

    # ── learning ──────────────────────────────────────────────────────

    def update(
        self,
        state: PortfolioState,
        allocation: np.ndarray,
        reward: float,
    ):
        """
        REINFORCE update:  θ ← θ + lr * (R - baseline) * ∇ log π(a|s)
        """
        sv = state.to_vector()

        # Pad/truncate for safety
        if sv.shape[0] != self.theta.shape[0]:
            if sv.shape[0] < self.theta.shape[0]:
                sv = np.pad(sv, (0, self.theta.shape[0] - sv.shape[0]))
            else:
                sv = sv[:self.theta.shape[0]]

        logits = sv @ self.theta / self.temperature
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        # Update baseline with Exponential Moving Average (EMA)
        advantage = reward - self.baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward

        # ∇ log softmax w.r.t. θ
        # For each campaign j:  ∂ log π_j / ∂ θ_j = s * (1{j=a} - π_j)
        grad = np.outer(sv, allocation - probs)

        # Gradient clipping to prevent unstable updates
        grad = np.clip(grad, -self.GRAD_CLIP, self.GRAD_CLIP)
        self.theta += self.lr * advantage * grad

        # Anneal temperature
        self.t += 1
        if self.temperature > 0.03:
            self.temperature = max(0.03, self.temperature * 0.92)

        # Track reward history for environment change detection
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 20:
            self._recent_rewards = self._recent_rewards[-20:]

        # Change detection: if reward variance spikes, re-inject exploration
        if len(self._recent_rewards) >= 10:
            recent_var = np.var(self._recent_rewards[-10:])
            older_var = np.var(self._recent_rewards[:10]) if len(self._recent_rewards) >= 20 else recent_var
            if recent_var > older_var * 3.0 and self.temperature < 0.15:
                self.temperature = 0.15  # Re-inject exploration on regime change
                logger.info(
                    f"[PortfolioAgent] Environment change detected "
                    f"(var {recent_var:.4f} vs {older_var:.4f}), "
                    f"resetting temperature to {self.temperature}"
                )

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theta": self.theta.tolist(),
            "baseline": self.baseline,
            "n_campaigns": self.n_campaigns,
            "lr": self.lr,
            "temperature": self.temperature,
            "t": self.t,
            "action_counts": self.action_counts.tolist(),
            "min_allocation": self.min_allocation,
            "exploration_bonus_coef": self.exploration_bonus_coef,
            "baseline_decay": self.baseline_decay,
            "recent_rewards": list(getattr(self, "_recent_rewards", [])),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortfolioAgent":
        n_camps = d["n_campaigns"]
        agent = cls(
            n_camps,
            learning_rate=d.get("lr", 0.01),
            temperature=d.get("temperature", 1.0),
        )
        agent.theta = np.array(d["theta"])
        agent.baseline = d.get("baseline", 0.0)
        agent.t = d.get("t", 0)
        agent.action_counts = np.array(d.get("action_counts", [0.0] * n_camps))
        agent.min_allocation = d.get("min_allocation", min(0.05, 0.50 / max(n_camps, 1)))
        agent.exploration_bonus_coef = d.get("exploration_bonus_coef", 0.02)
        agent.baseline_decay = d.get("baseline_decay", 0.9)
        agent._recent_rewards = d.get("recent_rewards", [])
        return agent


# ═══════════════════════════════════════════════════════════════════════
#  Level 2 – Campaign Agent
# ═══════════════════════════════════════════════════════════════════════

class CampaignAgent:
    """
    Distributes a campaign-level budget across its keyword groups.

    Uses a performance-proportional strategy:
      weight_i ∝ expected_revenue_i / spend_i   (ROAS-weighted)

    This is simpler than the portfolio agent because it operates
    per-campaign and doesn't need cross-campaign co-ordination.
    """

    def __init__(self, min_allocation_pct: float = 0.05):
        self.min_alloc = min_allocation_pct

    async def allocate(
        self,
        db: AsyncSession,
        campaign_id: int,
        campaign_budget: float,
        reference_date: datetime = None,
    ) -> List[Dict[str, Any]]:
        """
        Return per-keyword budget recommendations within a campaign.
        """
        result = await db.execute(
            text("""
                SELECT
                    k.id        AS keyword_id,
                    k.bid       AS current_bid,
                    COALESCE(SUM(p.spend), 0)  AS spend_7d,
                    COALESCE(SUM(p.sales), 0)  AS sales_7d,
                    COALESCE(SUM(p.clicks), 0) AS clicks_7d,
                    COALESCE(SUM(p.orders), 0) AS orders_7d
                FROM ppc_keywords k
                LEFT JOIN performance_records p
                    ON p.keyword_id = k.id
                    AND p.date >= :start_date
                WHERE k.campaign_id = :cid
                  AND k.state = 'ENABLED'
                GROUP BY k.id, k.bid
            """),
            {"cid": campaign_id, "start_date": (reference_date or datetime.utcnow()) - timedelta(days=7)},
        )
        rows = result.mappings().all()

        if not rows:
            return []

        # Calculate ROAS-weighted scores
        scores = []
        for r in rows:
            spend = float(r["spend_7d"])
            sales = float(r["sales_7d"])
            roas = sales / spend if spend > 0 else 1.0
            scores.append(max(roas, 0.1))   # floor to avoid zero

        total_score = sum(scores)
        allocations = []

        for r, score in zip(rows, scores):
            raw_pct = score / total_score
            alloc_pct = max(raw_pct, self.min_alloc)
            keyword_budget = campaign_budget * alloc_pct

            allocations.append({
                "keyword_id": r["keyword_id"],
                "current_bid": float(r["current_bid"]),
                "allocation_pct": alloc_pct,
                "keyword_budget": keyword_budget,
                "score": score,
            })

        # Re-normalise after clamping minimums
        total_alloc = sum(a["allocation_pct"] for a in allocations)
        for a in allocations:
            a["allocation_pct"] /= total_alloc
            a["keyword_budget"] = campaign_budget * a["allocation_pct"]

        return allocations


# ═══════════════════════════════════════════════════════════════════════
#  Level 3 – Keyword Agent   (thin wrapper over Contextual TS)
# ═══════════════════════════════════════════════════════════════════════

class KeywordAgent:
    """
    Selects the bid multiplier for a single keyword using
    ContextualThompsonSamplingDB (Phase 1).
    """

    def __init__(self, db: AsyncSession):
        self.ts = ContextualThompsonSamplingDB(db)

    async def select_bid_multiplier(
        self, keyword_id: int
    ) -> Tuple[int, float, float]:
        return await self.ts.select_arm(keyword_id)

    async def update(
        self, keyword_id: int, arm_id: int, reward: float
    ):
        await self.ts.update_arm(keyword_id, arm_id, reward)


# ═══════════════════════════════════════════════════════════════════════
#  Orchestrator – Hierarchical Budget Controller
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalBudgetController:
    """
    Full-stack orchestration of the three-tier hierarchy:

        1. PortfolioAgent  → campaign budget weights
        2. CampaignAgent   → keyword budget shares
        3. KeywordAgent    → bid multiplier per keyword

    Persists state to ``rl_portfolio_state`` and ``rl_budget_actions``.

    Usage::

        ctrl = HierarchicalBudgetController(db)
        result = await ctrl.run_allocation(
            profile_id="ABC123",
            total_budget=500.0
        )
    """

    def __init__(
        self,
        db: AsyncSession,
        learning_rate: float = 0.01,
        temperature: float = 1.0,
    ):
        self.db = db
        self.lr = learning_rate
        self.temp = temperature
        self.campaign_agent = CampaignAgent()
        self.keyword_agent = KeywordAgent(db)

    # ── main entry point ──────────────────────────────────────────────

    async def run_allocation(
        self,
        profile_id: str,
        total_budget: float,
        dry_run: bool = False,
        reference_date: datetime = None,
    ) -> Dict[str, Any]:
        """
        Execute a full top-down budget allocation cycle.

        Returns plan with allocations at every level.
        """
        logger.info(
            f"[HRL] Starting hierarchical allocation for {profile_id}, "
            f"budget=${total_budget:.2f}"
        )

        # ── 1. Build portfolio state ─────────────────────────────────
        state = await self._build_portfolio_state(profile_id, total_budget, reference_date)

        # ── 2. Load or create portfolio agent ────────────────────────
        portfolio_agent = await self._load_portfolio_agent(
            profile_id, state.num_campaigns
        )

        # ── 3. Portfolio allocation ──────────────────────────────────
        allocation_weights = portfolio_agent.allocate(state, explore=not dry_run)

        # Map weights to campaigns
        campaign_ids = [
            m.get("campaign_id") for m in state.campaign_metrics
        ]
        campaign_budgets = {
            cid: total_budget * w
            for cid, w in zip(campaign_ids, allocation_weights)
        }

        # ── 4. Persist portfolio state ───────────────────────────────
        state_id = await self._save_portfolio_state(
            profile_id, state, total_budget
        )

        # ── 5. Campaign-level allocation ─────────────────────────────
        all_keyword_allocations: List[Dict[str, Any]] = []
        action_records: List[Dict[str, Any]] = []

        for cid, camp_budget in campaign_budgets.items():
            if cid is None:
                continue

            # Record budget action
            action_records.append({
                "portfolio_state_id": state_id,
                "campaign_id": cid,
                "allocated_budget": camp_budget,
                "confidence_score": float(
                    allocation_weights[campaign_ids.index(cid)]
                ),
                "reasoning": (
                    f"Portfolio weight "
                    f"{allocation_weights[campaign_ids.index(cid)]:.3f}"
                ),
            })

            # Campaign → keyword allocation
            kw_allocs = await self.campaign_agent.allocate(
                self.db, cid, camp_budget, reference_date
            )

            # Keyword-level bid selection
            for kw in kw_allocs:
                arm_id, mult, exp_r = (
                    await self.keyword_agent.select_bid_multiplier(
                        kw["keyword_id"]
                    )
                )
                kw["arm_id"] = arm_id
                kw["bid_multiplier"] = mult
                kw["expected_reward"] = exp_r
                kw["proposed_bid"] = kw["current_bid"] * mult
                kw["campaign_id"] = cid
                kw["reference_date"] = reference_date or datetime.utcnow()

            all_keyword_allocations.extend(kw_allocs)

        # ── 6. Save budget actions ───────────────────────────────────
        for action in action_records:
            status = "simulated" if dry_run else "pending"
            await self.db.execute(
                text("""
                    INSERT INTO rl_budget_actions
                        (portfolio_state_id, campaign_id,
                         allocated_budget, confidence_score,
                         reasoning, status)
                    VALUES
                        (:psid, :cid, :budget, :conf, :reason, :status)
                """),
                {
                    "psid": action["portfolio_state_id"],
                    "cid": action["campaign_id"],
                    "budget": action["allocated_budget"],
                    "conf": action["confidence_score"],
                    "reason": action["reasoning"],
                    "status": status,
                },
            )

        # ── 7. Save agent params ─────────────────────────────────────
        await self._save_portfolio_agent(profile_id, portfolio_agent)

        await self.db.commit()

        result = {
            "profile_id": profile_id,
            "state_id": state_id,
            "total_budget": total_budget,
            "num_campaigns": len(campaign_budgets),
            "num_keywords": len(all_keyword_allocations),
            "dry_run": dry_run,
            "campaign_allocations": {
                str(cid): round(b, 2)
                for cid, b in campaign_budgets.items()
            },
            "keyword_allocations": all_keyword_allocations,
        }

        logger.info(
            f"[HRL] Allocation complete: {len(campaign_budgets)} campaigns, "
            f"{len(all_keyword_allocations)} keywords"
        )
        return result

    # ── learning step ─────────────────────────────────────────────────

    async def learn_from_outcomes(
        self,
        profile_id: str,
        lookback_days: int = 1,
        reference_date: datetime = None,
    ) -> Dict[str, Any]:
        """
        Evaluate recent allocations and update portfolio agent policy.

        Should be called daily after performance data is synced.
        """
        # Fetch most recent portfolio state
        result = await self.db.execute(
            text("""
                SELECT id, state_vector, total_budget
                FROM rl_portfolio_state
                WHERE profile_id = :pid
                  AND state_vector ? 'campaign_metrics'
                ORDER BY timestamp DESC
                LIMIT 1
            """),
            {"pid": profile_id},
        )
        row = result.mappings().first()
        if row is None:
            return {"status": "no_prior_state", "updated": False}

        state_dict = (
            row["state_vector"]
            if isinstance(row["state_vector"], dict)
            else json.loads(row["state_vector"])
        )
        state = PortfolioState(**state_dict)

        # Fetch the allocation that was executed
        actions = (
            await self.db.execute(
                text("""
                    SELECT campaign_id, allocated_budget
                    FROM rl_budget_actions
                    WHERE portfolio_state_id = :sid
                    ORDER BY campaign_id
                """),
                {"sid": row["id"]},
            )
        ).mappings().all()

        if not actions:
            return {"status": "no_actions_found", "updated": False}

        # Calculate portfolio reward (ROAS improvement)
        reward = await self._calculate_portfolio_reward(profile_id, reference_date=reference_date)

        # Rebuild allocation vector
        campaign_ids = [m.get("campaign_id") for m in state.campaign_metrics]
        total = float(row["total_budget"])
        alloc_vec = np.zeros(len(campaign_ids))
        for a in actions:
            try:
                idx = campaign_ids.index(a["campaign_id"])
                alloc_vec[idx] = float(a["allocated_budget"]) / max(total, 1)
            except ValueError:
                continue

        # Update portfolio agent
        agent = await self._load_portfolio_agent(
            profile_id, state.num_campaigns
        )
        agent.update(state, alloc_vec, reward)
        await self._save_portfolio_agent(profile_id, agent)
        await self.db.commit()

        logger.info(
            f"[HRL] Portfolio agent updated: reward={reward:.3f}"
        )
        return {
            "status": "updated",
            "updated": True,
            "reward": reward,
            "baseline": agent.baseline,
        }

    # ══════════════════════════════════════════════════════════════════
    #  Private helpers
    # ══════════════════════════════════════════════════════════════════

    async def _build_portfolio_state(
        self, profile_id: str, total_budget: float, reference_date: datetime = None
    ) -> PortfolioState:
        """Query DB to build portfolio state snapshot."""
        now = reference_date or datetime.utcnow()

        result = await self.db.execute(
            text("""
                SELECT
                    c.id AS campaign_id,
                    COALESCE(SUM(p.spend), 0)  AS spend_7d,
                    COALESCE(SUM(p.sales), 0)  AS sales_7d,
                    COALESCE(SUM(p.clicks), 0) AS clicks_7d,
                    COALESCE(SUM(p.orders), 0) AS orders_7d
                FROM ppc_campaigns c
                LEFT JOIN ppc_keywords k ON k.campaign_id = c.id
                LEFT JOIN performance_records p
                    ON p.keyword_id = k.id
                    AND p.date >= :start_date
                WHERE c.profile_id = :pid
                  AND c.state = 'ENABLED'
                GROUP BY c.id
                ORDER BY c.id
            """),
            {"pid": profile_id, "start_date": now - timedelta(days=7)},
        )
        rows = result.mappings().all()

        campaign_metrics = []
        total_spend = 0.0
        total_sales = 0.0

        for r in rows:
            s7 = float(r["spend_7d"])
            sa7 = float(r["sales_7d"])
            total_spend += s7
            total_sales += sa7
            campaign_metrics.append({
                "campaign_id": r["campaign_id"],
                "spend_7d": s7,
                "sales_7d": sa7,
                "clicks_7d": float(r["clicks_7d"]),
                "orders_7d": float(r["orders_7d"]),
                "acos_7d": s7 / sa7 if sa7 > 0 else 1.0,
                "roas_7d": sa7 / s7 if s7 > 0 else 0.0,
            })

        global_acos = total_spend / total_sales if total_sales > 0 else 1.0
        global_roas = total_sales / total_spend if total_spend > 0 else 0.0

        return PortfolioState(
            campaign_metrics=campaign_metrics,
            total_budget=total_budget,
            budget_remaining=total_budget,
            day_of_month=now.day,
            days_in_month=30,
            global_acos=global_acos,
            global_roas=global_roas,
        )

    async def _save_portfolio_state(
        self,
        profile_id: str,
        state: PortfolioState,
        total_budget: float,
    ) -> int:
        """Persist state snapshot and return its ID."""
        result = await self.db.execute(
            text("""
                INSERT INTO rl_portfolio_state
                    (profile_id, state_vector, total_budget, budget_remaining)
                VALUES
                    (:pid, :sv, :tb, :br)
                RETURNING id
            """),
            {
                "pid": profile_id,
                "sv": json.dumps(state.to_dict()),
                "tb": total_budget,
                "br": state.budget_remaining,
            },
        )
        return result.scalar()

    async def _load_portfolio_agent(
        self,
        profile_id: str,
        n_campaigns: int,
    ) -> PortfolioAgent:
        """Load persisted agent params or create new one."""
        result = await self.db.execute(
            text("""
                SELECT state_vector
                FROM rl_portfolio_state
                WHERE profile_id = :pid
                  AND state_vector ? 'agent_params'
                ORDER BY timestamp DESC
                LIMIT 1
            """),
            {"pid": profile_id},
        )
        row = result.mappings().first()

        if row:
            try:
                sv = (
                    row["state_vector"]
                    if isinstance(row["state_vector"], dict)
                    else json.loads(row["state_vector"])
                )
                if "agent_params" in sv:
                    return PortfolioAgent.from_dict(sv["agent_params"])
            except Exception:
                pass

        return PortfolioAgent(
            n_campaigns=max(n_campaigns, 1),
            learning_rate=self.lr,
            temperature=self.temp,
        )

    async def _save_portfolio_agent(
        self,
        profile_id: str,
        agent: PortfolioAgent,
    ):
        """Save agent parameters to latest state row."""
        # We store agent params in a separate lightweight record
        await self.db.execute(
            text("""
                INSERT INTO rl_portfolio_state
                    (profile_id, state_vector, total_budget, budget_remaining,
                     allocation_strategy)
                VALUES
                    (:pid, :sv, 0, 0, 'agent_checkpoint')
            """),
            {
                "pid": profile_id,
                "sv": json.dumps({"agent_params": agent.to_dict()}),
            },
        )

    async def _calculate_portfolio_reward(
        self,
        profile_id: str,
        lookback_days: int = 1,
        reference_date: datetime = None,
    ) -> float:
        """
        Calculate reward as ROAS improvement vs previous period.
        Returns value in [-1, 1].
        """
        result = await self.db.execute(
            text("""
                WITH recent AS (
                    SELECT
                        COALESCE(SUM(p.spend), 0)  AS spend,
                        COALESCE(SUM(p.sales), 0)  AS sales
                    FROM performance_records p
                    JOIN ppc_keywords k ON p.keyword_id = k.id
                    JOIN ppc_campaigns c ON k.campaign_id = c.id
                    WHERE c.profile_id = :pid
                      AND p.date >= :d1_start
                ),
                previous AS (
                    SELECT
                        COALESCE(SUM(p.spend), 0)  AS spend,
                        COALESCE(SUM(p.sales), 0)  AS sales
                    FROM performance_records p
                    JOIN ppc_keywords k ON p.keyword_id = k.id
                    JOIN ppc_campaigns c ON k.campaign_id = c.id
                    WHERE c.profile_id = :pid
                      AND p.date >= :d2_start
                      AND p.date <  :d1_start
                )
                SELECT
                    r.spend AS r_spend, r.sales AS r_sales,
                    p.spend AS p_spend, p.sales AS p_sales
                FROM recent r, previous p
            """),
            {
                "pid": profile_id,
                "d1_start": (reference_date or datetime.utcnow()) - timedelta(days=lookback_days),
                "d2_start": (reference_date or datetime.utcnow()) - timedelta(days=lookback_days * 2),
            },
        )
        row = result.mappings().first()
        if row is None:
            return 0.0

        r_spend = float(row["r_spend"])
        r_sales = float(row["r_sales"])
        p_spend = float(row["p_spend"])
        p_sales = float(row["p_sales"])

        recent_roas = r_sales / r_spend if r_spend > 0 else 0.0
        prev_roas = p_sales / p_spend if p_spend > 0 else 0.0

        if prev_roas == 0:
            return 0.5 if recent_roas > 0 else 0.0

        # Log-ratio reward for better stability and relative improvement tracking
        # reward = log(recent_roas / prev_roas)
        # Add small epsilon to prevent division by zero
        ratio = (recent_roas + 1e-9) / (prev_roas + 1e-9)
        reward = math.log(ratio)
        
        # Clip reward to reasonable bounds [-1, 1] to prevent massive gradients
        return max(-1.0, min(1.0, reward))
