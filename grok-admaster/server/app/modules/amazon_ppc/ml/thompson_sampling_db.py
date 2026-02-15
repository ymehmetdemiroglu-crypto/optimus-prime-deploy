
"""
Database-backed Multi-Armed Bandit using Thompson Sampling.

Two implementations:
  1. ThompsonSamplingOptimizerDB   – classic (non-contextual) TS
  2. ContextualThompsonSamplingDB  – Bayesian linear-regression TS
     conditioned on a context feature vector (Phase 1 upgrade)

The contextual version uses a Gaussian posterior over per-arm weight
vectors so that the expected reward of each arm depends on the current
market/temporal/performance context.  This produces faster convergence
and more accurate bids than static Beta-distribution bandits.
"""
import json
import math
import numpy as np
from scipy import linalg as la
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, text
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

from app.core.database import Base
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Numeric, JSON
from sqlalchemy.orm import relationship

from app.modules.amazon_ppc.ml.contextual_features import (
    ContextFeatureExtractor,
    CONTEXT_DIM,
    CONTEXT_FEATURE_NAMES,
    context_to_json,
    context_from_json,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  ORM Model
# ═══════════════════════════════════════════════════════════════════════

class BanditArm(Base):
    __tablename__ = 'bandit_arms'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword_id = Column(Integer, nullable=False)
    arm_id = Column(Integer, nullable=False)
    multiplier = Column(Numeric, nullable=False)
    alpha = Column(Numeric, default=1.0)
    beta = Column(Numeric, default=1.0)
    pulls = Column(Integer, default=0)
    total_reward = Column(Numeric, default=0.0)
    sum_squared_reward = Column(Numeric, default=0.0)
    context_vector = Column(JSON, nullable=True)  # JSON posterior params
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════
#  1. Classic (non-contextual) Thompson Sampling   (unchanged API)
# ═══════════════════════════════════════════════════════════════════════

class ThompsonSamplingOptimizerDB:
    """
    Database-backed Thompson Sampling optimizer.
    Persists state to `bandit_arms` table.
    """
    
    def __init__(self, db: AsyncSession, multipliers: List[float] = None):
        self.db = db
        self.multipliers = multipliers or [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    
    async def initialize_arms(self, keyword_id: int):
        """Create bandit arms for a keyword if they don't exist."""
        result = await self.db.execute(
            select(BanditArm.arm_id).where(BanditArm.keyword_id == keyword_id)
        )
        existing_arms = result.scalars().all()
        
        if existing_arms:
            return

        for arm_id, multiplier in enumerate(self.multipliers):
            arm = BanditArm(
                keyword_id=keyword_id,
                arm_id=arm_id,
                multiplier=multiplier,
                alpha=1.0,
                beta=1.0,
                pulls=0,
                total_reward=0.0,
                sum_squared_reward=0.0,
                last_updated=datetime.utcnow()
            )
            self.db.add(arm)
        
        await self.db.commit()
        logger.info(f"Initialized {len(self.multipliers)} arms for keyword {keyword_id}")
    
    async def select_arm(self, keyword_id: int) -> Tuple[int, float, float]:
        """
        Select an arm using Thompson Sampling.
        Returns: (arm_id, multiplier, expected_reward)
        """
        result = await self.db.execute(
            select(BanditArm).where(BanditArm.keyword_id == keyword_id)
        )
        arms = result.scalars().all()
        
        if not arms:
            await self.initialize_arms(keyword_id)
            return await self.select_arm(keyword_id)
        
        best_arm_id = None
        best_sample = -1
        best_multiplier = 1.0
        
        for arm in arms:
            sample = np.random.beta(float(arm.alpha), float(arm.beta))
            if sample > best_sample:
                best_sample = sample
                best_arm_id = arm.arm_id
                best_multiplier = float(arm.multiplier)
        
        logger.debug(f"Selected arm {best_arm_id} (mult={best_multiplier}) "
                      f"with expected reward {best_sample:.3f}")
        return best_arm_id, best_multiplier, best_sample

    async def update_arm(self, keyword_id: int, arm_id: int, reward: float):
        """
        Update arm statistics after observing a reward.
        Reward should be normalized [0, 1].
        """
        reward = max(0.0, min(1.0, reward))
        
        stmt = (
            update(BanditArm)
            .where(BanditArm.keyword_id == keyword_id, BanditArm.arm_id == arm_id)
            .values(
                alpha=BanditArm.alpha + reward,
                beta=BanditArm.beta + (1 - reward),
                pulls=BanditArm.pulls + 1,
                total_reward=BanditArm.total_reward + reward,
                sum_squared_reward=BanditArm.sum_squared_reward + (reward * reward),
                last_updated=datetime.utcnow()
            )
        )
        
        await self.db.execute(stmt)
        await self.db.commit()
        logger.info(f"Updated arm {arm_id} for keyword {keyword_id} "
                     f"with reward {reward:.3f}")

    async def get_arm_statistics(self, keyword_id: int) -> List[Dict[str, Any]]:
        """Get statistics for all arms of a keyword."""
        result = await self.db.execute(
            select(BanditArm)
            .where(BanditArm.keyword_id == keyword_id)
            .order_by(BanditArm.arm_id)
        )
        arms = result.scalars().all()
        
        stats = []
        for arm in arms:
            a, b = float(arm.alpha), float(arm.beta)
            avg_reward = float(arm.total_reward) / arm.pulls if arm.pulls > 0 else 0
            expected_val = a / (a + b)
            stats.append({
                'arm_id': arm.arm_id,
                'multiplier': float(arm.multiplier),
                'alpha': a,
                'beta': b,
                'pulls': arm.pulls,
                'total_reward': float(arm.total_reward),
                'avg_reward': avg_reward,
                'expected_value': expected_val,
                'last_updated': arm.last_updated
            })
        return stats

    @staticmethod
    def calculate_reward(
        old_metrics: Dict, new_metrics: Dict, target_acos: float = 0.3
    ) -> float:
        """
        Calculate reward based on ACoS improvement toward target.
        Returns value in [0, 1].
        """
        old_spend = float(old_metrics.get('spend', 0))
        old_sales = float(old_metrics.get('sales', 0))
        old_acos = old_spend / old_sales if old_sales > 0 else 2.0

        new_spend = float(new_metrics.get('spend', 0))
        new_sales = float(new_metrics.get('sales', 0))
        new_acos = new_spend / new_sales if new_sales > 0 else 2.0

        old_distance = abs(old_acos - target_acos)
        new_distance = abs(new_acos - target_acos)

        if target_acos <= 0:
            target_acos = 0.01

        imp_ratio = (old_distance - new_distance) / target_acos
        reward = 0.5 + (imp_ratio * 0.5)
        return max(0.0, min(1.0, reward))


# ═══════════════════════════════════════════════════════════════════════
#  2. Contextual Thompson Sampling  (Bayesian Linear Regression)
# ═══════════════════════════════════════════════════════════════════════

class _ArmPosterior:
    """
    Per-arm Bayesian linear regression posterior.

    Model:  reward = context^T w + noise,  noise ~ N(0, sigma²)
    Prior:  w ~ N(mu_0, sigma² * (B_0)^{-1})

    Posterior after n observations:
        B_n = B_0 + X^T X
        mu_n = B_n^{-1} (B_0 mu_0 + X^T y)

    Thompson sample: w_tilde ~ N(mu_n, sigma² * B_n^{-1})
    """

    def __init__(self, dim: int, regularisation: float = 1.0):
        self.dim = dim
        self.B = np.eye(dim) * regularisation     # precision matrix
        self.f = np.zeros(dim)                     # B @ mu
        self.sigma2 = 1.0                          # reward noise variance
        self.n_obs = 0

    # ── serialisation ─────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "B": self.B.tolist(),
            "f": self.f.tolist(),
            "sigma2": self.sigma2,
            "n_obs": self.n_obs,
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_ArmPosterior":
        dim = d["dim"]
        p = cls(dim)
        p.B = np.array(d["B"])
        p.f = np.array(d["f"])
        p.sigma2 = d["sigma2"]
        p.n_obs = d["n_obs"]
        return p

    # ── Bayesian update ───────────────────────────────────────────────
    def update(self, context: np.ndarray, reward: float):
        """Rank-1 posterior update:  B ← B + x x^T,  f ← f + r x"""
        self.B += np.outer(context, context)
        self.f += reward * context
        self.n_obs += 1

        # Online σ² update (simple running variance)
        mu = self._posterior_mean()
        pred = context @ mu
        resid = reward - pred
        if self.n_obs > 1:
            self.sigma2 += (resid ** 2 - self.sigma2) / self.n_obs

    def _posterior_mean(self) -> np.ndarray:
        return la.solve(self.B, self.f, assume_a="pos")

    # ── Thompson sample ───────────────────────────────────────────────
    def sample(self, context: np.ndarray) -> float:
        """
        Draw w ~ N(mu_n, sigma² B_n^{-1}), return context^T w.
        """
        mu = self._posterior_mean()
        try:
            cov = self.sigma2 * la.inv(self.B)
            # Make numerically symmetric
            cov = (cov + cov.T) / 2
            w_sample = np.random.multivariate_normal(mu, cov)
        except la.LinAlgError:
            # Fallback: diagonal approximation
            var = self.sigma2 / np.diag(self.B)
            w_sample = mu + np.random.randn(self.dim) * np.sqrt(np.abs(var))
        return float(context @ w_sample)

    def expected_reward(self, context: np.ndarray) -> float:
        """Posterior mean reward for a context."""
        return float(context @ self._posterior_mean())


class ContextualThompsonSamplingDB:
    """
    Database-backed **Contextual** Thompson Sampling.

    Each arm maintains a Bayesian linear regression posterior stored
    serialised in the ``context_vector`` column of ``bandit_arms``.

    Usage::

        ctx_ts = ContextualThompsonSamplingDB(db)

        # Select arm given current market context
        arm_id, mult, exp_r = await ctx_ts.select_arm(keyword_id)

        # … observe outcome …
        reward = ctx_ts.calculate_reward(old, new, target_acos=0.30)
        await ctx_ts.update_arm(keyword_id, arm_id, reward)
    """

    def __init__(
        self,
        db: AsyncSession,
        multipliers: Optional[List[float]] = None,
        regularisation: float = 1.0,
    ):
        self.db = db
        self.multipliers = multipliers or [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
        self.reg = regularisation
        self.feature_extractor = ContextFeatureExtractor(db)
        self._dim = CONTEXT_DIM + 1   # +1 for bias term

    # ── arm lifecycle ─────────────────────────────────────────────────

    async def initialize_arms(self, keyword_id: int):
        """Create contextual bandit arms with prior posteriors."""
        result = await self.db.execute(
            select(BanditArm.arm_id).where(BanditArm.keyword_id == keyword_id)
        )
        if result.scalars().all():
            return

        for arm_id, multiplier in enumerate(self.multipliers):
            posterior = _ArmPosterior(self._dim, self.reg)
            arm = BanditArm(
                keyword_id=keyword_id,
                arm_id=arm_id,
                multiplier=multiplier,
                alpha=1.0,
                beta=1.0,
                pulls=0,
                total_reward=0.0,
                sum_squared_reward=0.0,
                context_vector=json.dumps(posterior.to_dict()),
                last_updated=datetime.utcnow(),
            )
            self.db.add(arm)
        await self.db.commit()
        logger.info(
            f"Initialized {len(self.multipliers)} contextual arms "
            f"for keyword {keyword_id} (dim={self._dim})"
        )

    # ── selection ─────────────────────────────────────────────────────

    async def select_arm(
        self,
        keyword_id: int,
        context: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[int, float, float]:
        """
        Select an arm conditioned on context.

        Returns
        -------
        arm_id      index of selected arm
        multiplier  bid multiplier value
        ts_sample   Thompson-sampled expected reward (higher = more confident)
        """
        result = await self.db.execute(
            select(BanditArm).where(BanditArm.keyword_id == keyword_id)
        )
        arms = result.scalars().all()

        if not arms:
            await self.initialize_arms(keyword_id)
            return await self.select_arm(keyword_id, context, timestamp)

        # Build context vector (with bias)
        if context is None:
            raw_ctx = await self.feature_extractor.extract(keyword_id, timestamp)
        else:
            raw_ctx = context
        ctx = np.concatenate([[1.0], raw_ctx])  # prepend bias

        best_arm_id = 0
        best_sample = -np.inf
        best_multiplier = 1.0

        for arm in arms:
            posterior = self._load_posterior(arm)
            sample = posterior.sample(ctx)
            if sample > best_sample:
                best_sample = sample
                best_arm_id = arm.arm_id
                best_multiplier = float(arm.multiplier)

        logger.debug(
            f"[CTX-TS] kw={keyword_id} → arm {best_arm_id} "
            f"(mult={best_multiplier:.2f}, sample={best_sample:.4f})"
        )
        return best_arm_id, best_multiplier, float(best_sample)

    # ── update ────────────────────────────────────────────────────────

    async def update_arm(
        self,
        keyword_id: int,
        arm_id: int,
        reward: float,
        context: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Update arm posterior with observed (context, reward) pair.
        """
        reward = max(0.0, min(1.0, reward))

        # Load arm
        result = await self.db.execute(
            select(BanditArm).where(
                BanditArm.keyword_id == keyword_id,
                BanditArm.arm_id == arm_id,
            )
        )
        arm = result.scalars().first()
        if arm is None:
            logger.warning(f"Arm {arm_id} for keyword {keyword_id} not found")
            return

        # Rebuild context
        if context is None:
            raw_ctx = await self.feature_extractor.extract(keyword_id, timestamp)
        else:
            raw_ctx = context
        ctx = np.concatenate([[1.0], raw_ctx])

        # Bayesian update
        posterior = self._load_posterior(arm)
        posterior.update(ctx, reward)

        # Persist: update posterior JSON + Beta params + counters
        new_alpha = float(arm.alpha) + reward
        new_beta = float(arm.beta) + (1 - reward)

        stmt = (
            update(BanditArm)
            .where(BanditArm.keyword_id == keyword_id, BanditArm.arm_id == arm_id)
            .values(
                alpha=new_alpha,
                beta=new_beta,
                pulls=arm.pulls + 1,
                total_reward=float(arm.total_reward) + reward,
                sum_squared_reward=float(arm.sum_squared_reward) + reward ** 2,
                context_vector=json.dumps(posterior.to_dict()),
                last_updated=datetime.utcnow(),
            )
        )
        await self.db.execute(stmt)
        await self.db.commit()
        logger.info(
            f"[CTX-TS] Updated arm {arm_id} for kw {keyword_id}: "
            f"reward={reward:.3f}, pulls={arm.pulls + 1}"
        )

    # ── diagnostics ───────────────────────────────────────────────────

    async def get_arm_statistics(
        self,
        keyword_id: int,
        context: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return per-arm statistics.  If a context is provided,
        also include the posterior-mean expected reward *for that context*.
        """
        result = await self.db.execute(
            select(BanditArm)
            .where(BanditArm.keyword_id == keyword_id)
            .order_by(BanditArm.arm_id)
        )
        arms = result.scalars().all()

        # Optionally build context
        ctx = None
        if context is not None:
            ctx = np.concatenate([[1.0], context])
        else:
            try:
                raw = await self.feature_extractor.extract(keyword_id)
                ctx = np.concatenate([[1.0], raw])
            except Exception:
                pass

        stats = []
        for arm in arms:
            a, b = float(arm.alpha), float(arm.beta)
            avg_rw = float(arm.total_reward) / arm.pulls if arm.pulls > 0 else 0.0
            expected_beta = a / (a + b)
            entry: Dict[str, Any] = {
                "arm_id": arm.arm_id,
                "multiplier": float(arm.multiplier),
                "alpha": a,
                "beta": b,
                "pulls": arm.pulls,
                "total_reward": float(arm.total_reward),
                "avg_reward": avg_rw,
                "expected_value_beta": expected_beta,
                "last_updated": arm.last_updated,
            }
            # Contextual expected reward
            if ctx is not None:
                posterior = self._load_posterior(arm)
                entry["expected_value_contextual"] = posterior.expected_reward(ctx)
                entry["posterior_observations"] = posterior.n_obs
            stats.append(entry)
        return stats

    async def get_feature_importance(
        self, keyword_id: int
    ) -> Dict[str, List[float]]:
        """
        Return the posterior-mean weight vector for each arm.

        Interpreting these weights reveals which context features
        most influence the reward prediction for each multiplier.
        """
        result = await self.db.execute(
            select(BanditArm)
            .where(BanditArm.keyword_id == keyword_id)
            .order_by(BanditArm.arm_id)
        )
        arms = result.scalars().all()

        importance: Dict[str, List[float]] = {}
        labels = ["bias"] + list(CONTEXT_FEATURE_NAMES)

        for arm in arms:
            posterior = self._load_posterior(arm)
            mu = la.solve(posterior.B, posterior.f, assume_a="pos")
            importance[f"arm_{arm.arm_id}_mult_{float(arm.multiplier):.2f}"] = dict(
                zip(labels, mu.tolist())
            )
        return importance

    # ── helpers ────────────────────────────────────────────────────────

    def _load_posterior(self, arm: BanditArm) -> _ArmPosterior:
        """Deserialise posterior from DB column, or create fresh prior."""
        if arm.context_vector:
            try:
                return _ArmPosterior.from_dict(json.loads(arm.context_vector))
            except Exception:
                logger.warning(
                    f"Corrupt posterior for arm {arm.arm_id}, "
                    f"kw {arm.keyword_id}; resetting"
                )
        return _ArmPosterior(self._dim, self.reg)

    @staticmethod
    def calculate_reward(
        old_metrics: Dict, new_metrics: Dict, target_acos: float = 0.3
    ) -> float:
        """Shared reward calculation (same as static TS)."""
        return ThompsonSamplingOptimizerDB.calculate_reward(
            old_metrics, new_metrics, target_acos
        )
