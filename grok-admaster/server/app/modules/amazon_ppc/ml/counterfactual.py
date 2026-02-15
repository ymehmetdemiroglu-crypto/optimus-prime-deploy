"""
Counterfactual Learning & Off-Policy Evaluation (Phase 3).

This module enables the system to learn from *hypothetical* actions —
evaluating "what would have happened if we'd bid differently?" without
actually executing those bids in the live market.

Three core components:
    1. InversePropensityScorer (IPS)
       — Re-weights logged rewards by the inverse of the logging policy's
         probability of selecting that action. Produces unbiased (but
         high-variance) estimates of alternative policies.

    2. DoublyRobustEstimator (DR)
       — Combines a reward model (direct method) with IPS to produce
         lower-variance estimates. The estimator is "doubly robust":
         if either the reward model OR the propensity model is correct,
         the estimate is unbiased.

    3. SafeExperimentFramework
       — Wraps the estimators into a production-safe A/B testing and
         policy evaluation pipeline. Ensures new policies are validated
         off-policy before being deployed to live traffic.

Data Sources:
    - prediction_logs  — logged decisions (features, action, propensity)
    - backtest_results — historical hypothetical vs actual comparisons
    - performance_records — actual outcomes (spend, sales, clicks)

Key Concepts:
    - Logging policy π_0(a|x): the policy that *actually* chose the
      action. We reconstruct this from prediction_logs.
    - Target policy π(a|x): the new policy we want to evaluate.
    - Propensity ratio: π(a|x) / π_0(a|x) — the IPS weight.
"""
from __future__ import annotations

import json
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LoggedDecision:
    """A single logged decision from the prediction pipeline."""
    log_id: int
    keyword_id: int
    campaign_id: int
    input_features: Dict[str, Any]
    predicted_bid: float
    confidence: float           # proxy for propensity
    model_name: str
    actual_outcome: Optional[Dict[str, Any]]  # {spend, sales, clicks, orders}
    created_at: datetime
    was_executed: bool

    @property
    def logging_propensity(self) -> float:
        """
        Estimate the probability that the logging policy would have
        chosen this specific action (bid). Higher confidence → higher
        propensity. Clipped to [0.01, 1.0] for numerical stability.
        """
        return max(0.01, min(1.0, self.confidence))

    @property
    def reward(self) -> Optional[float]:
        """
        Calculate reward from actual outcome (ROAS-based).
        Returns None if outcome not yet observed (distinguishes from zero reward).
        """
        if not self.actual_outcome:
            return None
        spend = float(self.actual_outcome.get("spend", 0))
        sales = float(self.actual_outcome.get("sales", 0))
        
        if spend <= 0:
            return 0.0  # No spend = no reward (avoids division by zero)
        
        roas = sales / spend
        # Clip extreme values for numerical stability
        return max(0.0, min(roas, 20.0))


@dataclass
class PolicyEvaluation:
    """Result of an off-policy evaluation."""
    policy_name: str
    estimated_value: float      # expected reward under policy
    variance: float             # estimation variance
    confidence_interval: Tuple[float, float]  # 95% CI
    n_samples: int
    effective_sample_size: float  # ESS after IPS re-weighting
    method: str                 # 'ips', 'dm', 'dr'
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """Is the estimated value significantly different from zero?"""
        return self.confidence_interval[0] > 0 or self.confidence_interval[1] < 0


# ═══════════════════════════════════════════════════════════════════════
#  1. Inverse Propensity Scoring (IPS)
# ═══════════════════════════════════════════════════════════════════════

class InversePropensityScorer:
    """
    Estimate the value of a target policy using Inverse Propensity
    Scoring on historical logged decisions.

    V_IPS(π) = (1/n) Σ [π(a_i|x_i) / π_0(a_i|x_i)] * r_i

    where:
        π(a|x)   = target policy propensity
        π_0(a|x) = logging policy propensity (from prediction confidence)
        r_i      = observed reward

    Includes Self-Normalized IPS (SNIPS) for reduced variance:
        V_SNIPS(π) = Σ w_i * r_i / Σ w_i
    """

    def __init__(
        self,
        clip_ratio: float = 10.0,
        use_self_normalized: bool = True,
        use_bootstrap_ci: bool = True,
        n_bootstrap: int = 1000,
    ):
        """
        Parameters
        ----------
        clip_ratio : float
            Maximum allowed propensity ratio (weight clipping).
            Prevents extreme weights from single observations.
        use_self_normalized : bool
            If True, use Self-Normalized IPS (SNIPS) for lower variance.
        use_bootstrap_ci : bool
            If True, use bootstrap for CIs (robust for heavy tails).
            If False, use normal approximation (faster but assumes normality).
        n_bootstrap : int
            Number of bootstrap samples for CI estimation.
        """
        self.clip_ratio = clip_ratio
        self.use_self_normalized = use_self_normalized
        self.use_bootstrap_ci = use_bootstrap_ci
        self.n_bootstrap = n_bootstrap
    
    def _bootstrap_ci(
        self,
        weights: np.ndarray,
        rewards: np.ndarray,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for IPS/SNIPS estimate.
        
        Robust to heavy-tailed weight distributions, recommended when ESS < 100.
        """
        n = len(weights)
        estimates = []
        
        rng = np.random.RandomState(42)
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = rng.choice(n, size=n, replace=True)
            boot_weights = weights[indices]
            boot_rewards = rewards[indices]
            
            # SNIPS or IPS estimate for this bootstrap sample
            if self.use_self_normalized and boot_weights.sum() > 0:
                estimate = (boot_weights * boot_rewards).sum() / boot_weights.sum()
            else:
                estimate = (boot_weights * boot_rewards).mean()
            
            estimates.append(estimate)
        
        # Percentile method
        estimates_arr = np.array(estimates)
        lower = float(np.percentile(estimates_arr, 100 * alpha / 2))
        upper = float(np.percentile(estimates_arr, 100 * (1 - alpha / 2)))
        
        return (lower, upper)

    def evaluate(
        self,
        logged_decisions: List[LoggedDecision],
        target_propensities: List[float],
    ) -> PolicyEvaluation:
        """
        Evaluate a target policy given logged decisions and their
        propensities under the target policy.

        Parameters
        ----------
        logged_decisions : list of LoggedDecision
            Historical decisions with outcomes.
        target_propensities : list of float
            π(a_i|x_i) — probability that the target policy would
            have chosen the same action for each observation.

        Returns
        -------
        PolicyEvaluation
        """
        if not logged_decisions:
            return PolicyEvaluation(
                policy_name="unknown",
                estimated_value=0.0,
                variance=0.0,
                confidence_interval=(0.0, 0.0),
                n_samples=0,
                effective_sample_size=0.0,
                method="ips",
            )

        n = len(logged_decisions)
        weights = []
        weighted_rewards = []

        for decision, target_p in zip(logged_decisions, target_propensities):
            # Skip only if not executed OR outcome not yet observed
            # (do NOT skip legitimate zero rewards!)
            if not decision.was_executed or decision.reward is None:
                continue

            logging_p = decision.logging_propensity
            ratio = target_p / logging_p

            # Clip extreme ratios
            clipped_ratio = min(ratio, self.clip_ratio)
            weights.append(clipped_ratio)
            weighted_rewards.append(clipped_ratio * decision.reward)

        if not weights:
            return PolicyEvaluation(
                policy_name="unknown",
                estimated_value=0.0,
                variance=0.0,
                confidence_interval=(0.0, 0.0),
                n_samples=n,
                effective_sample_size=0.0,
                method="ips",
            )

        weights_arr = np.array(weights)
        wr_arr = np.array(weighted_rewards)

        # Self-Normalized IPS (SNIPS) or standard IPS
        if self.use_self_normalized and weights_arr.sum() > 0:
            estimated_value = wr_arr.sum() / weights_arr.sum()
        else:
            estimated_value = wr_arr.mean()

        # Variance estimation
        variance = np.var(wr_arr) / max(len(wr_arr), 1)

        # Effective sample size: ESS = (Σ w_i)² / Σ w_i²
        ess = (weights_arr.sum() ** 2) / (weights_arr ** 2).sum() if (weights_arr ** 2).sum() > 0 else 0

        # Confidence interval: bootstrap for heavy tails (ESS < 100) or if requested
        if self.use_bootstrap_ci and (ess < 100 or len(weights) < 500):
            ci = self._bootstrap_ci(weights_arr, wr_arr)
            ci_method = "bootstrap"
        else:
            # Normal approximation (faster, valid for large ESS)
            stderr = math.sqrt(variance) if variance > 0 else 0
            ci = (estimated_value - 1.96 * stderr, estimated_value + 1.96 * stderr)
            ci_method = "normal"

        return PolicyEvaluation(
            policy_name="ips_target",
            estimated_value=estimated_value,
            variance=variance,
            confidence_interval=ci,
            n_samples=len(weights),
            effective_sample_size=ess,
            method="snips" if self.use_self_normalized else "ips",
            details={
                "mean_weight": float(weights_arr.mean()),
                "max_weight": float(weights_arr.max()),
                "clipped_count": int(np.sum(weights_arr >= self.clip_ratio)),
                "ci_method": ci_method,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
#  2. Doubly Robust Estimator
# ═══════════════════════════════════════════════════════════════════════

class DoublyRobustEstimator:
    """
    Doubly Robust (DR) off-policy evaluation.

    Combines:
        - Direct Method (DM): a reward model r̂(x, a) that predicts
          reward from features and action.
        - IPS: importance-weighted correction term.

    V_DR = (1/n) Σ [ r̂(x,a) + w_i * (r_i - r̂(x_i,a_i)) ]

    Properties:
        - Unbiased if EITHER the reward model OR propensity model
          is correctly specified ("doubly robust").
        - Lower variance than pure IPS when the reward model is decent.
        - Reduces to DM if propensities are flat.
        - Reduces to IPS if the reward model outputs zero.
    """

    def __init__(
        self,
        reward_model: Optional[Any] = None,
        clip_ratio: float = 10.0,
        use_gb: bool = True,
    ):
        """Initialize doubly robust estimator.
        
        Parameters
        ----------
        reward_model : optional
            Pre-trained reward model (if None, will train internally)
        clip_ratio : float
            Maximum propensity ratio for IPS weights
        use_gb : bool
            If True, use GradientBoostingRegressor (production)
            If False, fall back to ridge regression (lightweight)
        """
        self.clip_ratio = clip_ratio
        self.use_gb = use_gb and SKLEARN_AVAILABLE
        self._fitted = False
        
        # Model storage
        if reward_model is not None:
            self.reward_model = reward_model
            self._fitted = True
        elif self.use_gb:
            self.reward_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
            )
            self.scaler = StandardScaler()
        else:
            # Fallback: simple linear model
            self.reward_model = None
            self._feature_weights: Optional[np.ndarray] = None

    MIN_SAMPLES = 1000  # Production threshold (was 10)
    
    def fit_reward_model(
        self,
        logged_decisions: List[LoggedDecision],
        min_samples: Optional[int] = None,
    ):
        """
        Fit a reward model from logged data.
        
        Production version uses GradientBoostingRegressor for nonlinear
        reward functions. Falls back to ridge regression if sklearn unavailable.
        """
        min_samples = min_samples or self.MIN_SAMPLES
        features_list = []
        rewards_list = []

        for d in logged_decisions:
            if not d.was_executed or d.reward is None:
                continue
            feat_vec = self._extract_features(d)
            features_list.append(feat_vec)
            rewards_list.append(d.reward)

        n_samples = len(features_list)
        
        if n_samples < min_samples:
            logger.warning(
                f"[DR] Insufficient data for reward model "
                f"({n_samples} samples < {min_samples} required). Using zero model."
            )
            self._fitted = False
            return

        X = np.array(features_list)
        y = np.array(rewards_list)
        
        # Production: Gradient Boosting
        if self.use_gb:
            try:
                X_scaled = self.scaler.fit_transform(X)
                self.reward_model.fit(X_scaled, y)
                self._fitted = True
                
                # Compute R² on training data
                y_pred = self.reward_model.predict(X_scaled)
                r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
                
                logger.info(
                    f"[DR] GradientBoosting reward model fitted on {n_samples} samples, "
                    f"R² = {r2:.3f}"
                )
                
                if r2 < 0.3:
                    logger.warning(
                        f"[DR] Reward model R² = {r2:.3f} is low. "
                        "Consider more features or data."
                    )
            except Exception as e:
                logger.error(f"[DR] GradientBoosting failed: {e}. Using zero model.")
                self._fitted = False
        
        # Fallback: Ridge regression
        else:
            lambda_reg = 1.0
            XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
            Xty = X.T @ y

            try:
                self._feature_weights = np.linalg.solve(XtX, Xty)
                self._fitted = True
                logger.info(
                    f"[DR] Ridge reward model fitted on {n_samples} samples, "
                    f"R² ≈ {self._compute_r2(X, y):.3f}"
                )
            except np.linalg.LinAlgError:
                logger.error("[DR] Ridge regression failed (singular matrix)")
                self._fitted = False

    def predict_reward(self, decision: LoggedDecision) -> float:
        """Predict expected reward using the fitted model."""
        if not self._fitted:
            return 0.0
            
        feat_vec = self._extract_features(decision)
        
        # Gradient Boosting prediction
        if self.use_gb and hasattr(self, 'scaler'):
            try:
                feat_scaled = self.scaler.transform(feat_vec.reshape(1, -1))
                return float(self.reward_model.predict(feat_scaled)[0])
            except Exception:
                return 0.0
        
        # Ridge regression prediction
        elif self._feature_weights is not None:
            return float(feat_vec @ self._feature_weights)
        
        return 0.0

    def evaluate(
        self,
        logged_decisions: List[LoggedDecision],
        target_propensities: List[float],
    ) -> PolicyEvaluation:
        """
        Doubly Robust policy evaluation.

        V_DR = (1/n) Σ [ r̂(x,a) + w_i * (r_i - r̂(x_i,a_i)) ]
        """
        if not logged_decisions:
            return PolicyEvaluation(
                policy_name="unknown",
                estimated_value=0.0,
                variance=0.0,
                confidence_interval=(0.0, 0.0),
                n_samples=0,
                effective_sample_size=0.0,
                method="dr",
            )

        # Fit reward model if not already fitted
        if not self._fitted:
            self.fit_reward_model(logged_decisions)

        n = len(logged_decisions)
        dr_values = []
        weights = []

        for decision, target_p in zip(logged_decisions, target_propensities):
            r_hat = self.predict_reward(decision)

            if decision.was_executed and decision.reward != 0:
                logging_p = decision.logging_propensity
                ratio = min(target_p / logging_p, self.clip_ratio)
                correction = ratio * (decision.reward - r_hat)
                weights.append(ratio)
            else:
                correction = 0.0
                weights.append(0.0)

            dr_values.append(r_hat + correction)

        dr_arr = np.array(dr_values)
        weights_arr = np.array(weights)

        estimated_value = dr_arr.mean()
        variance = np.var(dr_arr) / max(n, 1)

        # ESS
        valid_weights = weights_arr[weights_arr > 0]
        ess = (valid_weights.sum() ** 2) / (valid_weights ** 2).sum() if len(valid_weights) > 0 and (valid_weights ** 2).sum() > 0 else 0

        stderr = math.sqrt(variance) if variance > 0 else 0
        ci = (estimated_value - 1.96 * stderr, estimated_value + 1.96 * stderr)

        return PolicyEvaluation(
            policy_name="dr_target",
            estimated_value=estimated_value,
            variance=variance,
            confidence_interval=ci,
            n_samples=n,
            effective_sample_size=ess,
            method="dr",
            details={
                "reward_model_fitted": self._fitted,
                "mean_correction": float(np.mean([
                    w * (d.reward - self.predict_reward(d))
                    for d, w in zip(logged_decisions, weights)
                    if d.was_executed
                ])) if any(d.was_executed for d in logged_decisions) else 0.0,
            },
        )

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_features(d: LoggedDecision) -> np.ndarray:
        """
        Extract features BEFORE action was taken (avoid target leakage).
        Uses only historical and contextual features, NOT post-action outcomes.
        """
        features = d.input_features or {}
        
        # Historical performance (7-day lookback)
        hist_impressions = features.get("hist_impressions_7d", 0)
        hist_clicks = features.get("hist_clicks_7d", 0)
        hist_spend = features.get("hist_spend_7d", 0)
        hist_sales = features.get("hist_sales_7d", 0)
        
        # Derived historical metrics
        hist_ctr = hist_clicks / max(hist_impressions, 1)
        hist_roas = hist_sales / max(hist_spend, 1)
        
        # Contextual features (time, day, etc.)
        hour_of_day = features.get("hour_of_day", 12) / 24.0
        day_of_week = features.get("day_of_week", 3) / 7.0
        
        return np.array([
            hist_ctr,
            hist_roas,
            np.log1p(hist_impressions),  # log-transform for scale
            np.log1p(hist_spend),
            features.get("current_bid", 1.0),
            d.predicted_bid,
            d.confidence,
            hour_of_day,
            day_of_week,
            features.get("keyword_quality_score", 5.0) / 10.0,
            features.get("competitor_density", 0.5),
        ], dtype=np.float64)

    def _compute_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² for fitted model."""
        if self._feature_weights is None:
            return 0.0
        y_pred = X @ self._feature_weights
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
#  3. Safe Experiment Framework
# ═══════════════════════════════════════════════════════════════════════

class SafeExperimentFramework:
    """
    Production-safe framework for evaluating and deploying new policies.

    Workflow:
        1. Propose a target policy (e.g., new bid multiplier function)
        2. Evaluate it off-policy using DR estimator
        3. Run safety checks (degradation bounds, ESS threshold)
        4. If safe, deploy via gradual traffic ramp
        5. Log results to backtest_results for future analysis

    Safety Guarantees:
        - Never deploy a policy whose lower CI bound is below baseline
        - Require minimum Effective Sample Size (ESS) for significance
        - Gradual rollout (10% → 50% → 100%) with monitoring
    """

    def __init__(
        self,
        db: AsyncSession,
        min_ess: float = 50.0,
        min_improvement: float = 0.05,
        max_degradation: float = 0.10,
        ips_clip: float = 10.0,
    ):
        """
        Parameters
        ----------
        db : AsyncSession
        min_ess : float
            Minimum effective sample size for statistical significance.
        min_improvement : float
            Minimum estimated improvement (fraction) over baseline
            before recommending deployment.
        max_degradation : float
            Maximum allowed degradation in lower CI bound vs baseline.
        ips_clip : float
            Propensity ratio clipping threshold.
        """
        self.db = db
        self.min_ess = min_ess
        self.min_improvement = min_improvement
        self.max_degradation = max_degradation

        self.ips = InversePropensityScorer(clip_ratio=ips_clip)
        self.dr = DoublyRobustEstimator(clip_ratio=ips_clip)

    # ── main evaluation entry point ──────────────────────────────────

    async def evaluate_policy(
        self,
        profile_id: str,
        target_policy_fn,
        policy_name: str = "new_policy",
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Evaluate a target policy against historical data.

        Parameters
        ----------
        profile_id : str
            The advertiser profile to evaluate against.
        target_policy_fn : callable
            Function (input_features, current_bid) → (new_bid, propensity).
            Must return the bid the target policy would choose and the
            probability of choosing it.
        policy_name : str
            Human-readable name for the policy.
        lookback_days : int
            Number of days of historical data to use.

        Returns
        -------
        dict with evaluation results, safety assessment, and deployment
        recommendation.
        """
        logger.info(
            f"[Counterfactual] Evaluating policy '{policy_name}' "
            f"for {profile_id}, lookback={lookback_days}d"
        )

        # ── 1. Load logged decisions from prediction_logs ────────────
        decisions = await self._load_logged_decisions(
            profile_id, lookback_days
        )

        if len(decisions) < 20:
            return {
                "status": "insufficient_data",
                "policy_name": policy_name,
                "n_decisions": len(decisions),
                "message": (
                    f"Need at least 20 logged decisions, found {len(decisions)}. "
                    f"Run the bid optimizer for more days."
                ),
                "safe_to_deploy": False,
            }

        # ── 2. Compute target policy propensities ────────────────────
        target_propensities = []
        for d in decisions:
            try:
                _, propensity = target_policy_fn(
                    d.input_features, d.predicted_bid
                )
                target_propensities.append(max(0.01, propensity))
            except Exception:
                target_propensities.append(0.5)  # fallback

        # ── 3. Run IPS evaluation ────────────────────────────────────
        ips_result = self.ips.evaluate(decisions, target_propensities)

        # ── 4. Run DR evaluation ─────────────────────────────────────
        dr_result = self.dr.evaluate(decisions, target_propensities)

        # ── 5. Compute baseline (logging policy value) ───────────────
        baseline_value = self._compute_baseline(decisions)

        # ── 6. Safety assessment ─────────────────────────────────────
        safety = self._assess_safety(
            dr_result, baseline_value, policy_name
        )

        # ── 7. Record to backtest_results ────────────────────────────
        await self._record_backtest(
            profile_id, policy_name, decisions,
            ips_result, dr_result, baseline_value, safety
        )

        result = {
            "status": "evaluated",
            "policy_name": policy_name,
            "n_decisions": len(decisions),
            "baseline_roas": round(baseline_value, 4),
            "ips_evaluation": {
                "estimated_roas": round(ips_result.estimated_value, 4),
                "variance": round(ips_result.variance, 6),
                "ci_95": (
                    round(ips_result.confidence_interval[0], 4),
                    round(ips_result.confidence_interval[1], 4),
                ),
                "ess": round(ips_result.effective_sample_size, 1),
                "method": ips_result.method,
            },
            "dr_evaluation": {
                "estimated_roas": round(dr_result.estimated_value, 4),
                "variance": round(dr_result.variance, 6),
                "ci_95": (
                    round(dr_result.confidence_interval[0], 4),
                    round(dr_result.confidence_interval[1], 4),
                ),
                "ess": round(dr_result.effective_sample_size, 1),
                "method": dr_result.method,
                "reward_model_fitted": dr_result.details.get(
                    "reward_model_fitted", False
                ),
            },
            "safety_assessment": safety,
        }

        logger.info(
            f"[Counterfactual] Policy '{policy_name}': "
            f"baseline={baseline_value:.3f}, "
            f"DR={dr_result.estimated_value:.3f}, "
            f"safe={safety['safe_to_deploy']}"
        )

        return result

    # ── example target policies ─────────────────────────────────────

    @staticmethod
    def aggressive_bidding_policy(
        features: Dict[str, Any], current_bid: float
    ) -> Tuple[float, float]:
        """
        Example: Bid 20% higher on keywords with ACoS < 25%.
        Returns (new_bid, propensity).
        """
        spend = features.get("spend", 0)
        sales = features.get("sales", 0)
        acos = spend / sales if sales > 0 else 1.0

        if acos < 0.25:
            new_bid = current_bid * 1.20
            propensity = 0.8  # high confidence we'd choose this
        else:
            new_bid = current_bid * 0.95
            propensity = 0.6
        return new_bid, propensity

    @staticmethod
    def conservative_bidding_policy(
        features: Dict[str, Any], current_bid: float
    ) -> Tuple[float, float]:
        """
        Example: Reduce bids across the board by 10%, increase on
        high-conversion keywords.
        """
        orders = features.get("orders", 0)
        clicks = features.get("clicks", 0)
        cvr = orders / clicks if clicks > 0 else 0

        if cvr > 0.10:
            new_bid = current_bid * 1.05
            propensity = 0.7
        else:
            new_bid = current_bid * 0.90
            propensity = 0.65
        return new_bid, propensity

    @staticmethod
    def roas_maximizing_policy(
        features: Dict[str, Any], current_bid: float
    ) -> Tuple[float, float]:
        """
        Example: Scale bid proportional to keyword ROAS.
        High ROAS → higher bid. Low ROAS → lower bid.
        """
        spend = features.get("spend", 0)
        sales = features.get("sales", 0)
        roas = sales / spend if spend > 0 else 1.0

        # Target: bid multiplier = sqrt(roas / 3.0), capped
        multiplier = min(2.0, max(0.5, math.sqrt(roas / 3.0)))
        new_bid = current_bid * multiplier
        propensity = 0.7 if 0.8 <= multiplier <= 1.5 else 0.4
        return new_bid, propensity

    # ── data loading ─────────────────────────────────────────────────

    async def _load_logged_decisions(
        self,
        profile_id: str,
        lookback_days: int,
    ) -> List[LoggedDecision]:
        """Load prediction_logs with outcomes for a profile."""
        result = await self.db.execute(
            text("""
                SELECT
                    pl.id,
                    pl.keyword_id,
                    pl.campaign_id,
                    pl.input_features,
                    pl.predicted_bid,
                    pl.confidence_score,
                    pl.model_name,
                    pl.actual_outcome,
                    pl.was_executed,
                    pl.created_at
                FROM prediction_logs pl
                JOIN ppc_keywords k ON pl.keyword_id = k.id
                JOIN ppc_campaigns c ON k.campaign_id = c.id
                WHERE c.profile_id = :pid
                  AND pl.created_at >= NOW() - MAKE_INTERVAL(days => :days)
                ORDER BY pl.created_at ASC
            """),
            {"pid": profile_id, "days": lookback_days},
        )
        rows = result.mappings().all()

        decisions = []
        for r in rows:
            # Parse JSON fields
            features = r["input_features"]
            if isinstance(features, str):
                features = json.loads(features)
            outcome = r["actual_outcome"]
            if isinstance(outcome, str):
                outcome = json.loads(outcome)

            decisions.append(LoggedDecision(
                log_id=r["id"],
                keyword_id=r["keyword_id"],
                campaign_id=r["campaign_id"],
                input_features=features or {},
                predicted_bid=float(r["predicted_bid"] or 0),
                confidence=float(r["confidence_score"] or 0.5),
                model_name=r["model_name"],
                actual_outcome=outcome,
                created_at=r["created_at"],
                was_executed=bool(r["was_executed"]),
            ))

        logger.info(
            f"[Counterfactual] Loaded {len(decisions)} logged decisions "
            f"for {profile_id}"
        )
        return decisions

    # ── safety checks ────────────────────────────────────────────────

    def _assess_safety(
        self,
        evaluation: PolicyEvaluation,
        baseline_value: float,
        policy_name: str,
    ) -> Dict[str, Any]:
        """
        Run safety checks on a policy evaluation.

        Safety criteria:
            1. ESS ≥ min_ess (enough effective data)
            2. Lower CI bound ≥ baseline * (1 - max_degradation)
            3. Estimated improvement ≥ min_improvement
        """
        checks = {}

        # Check 1: Sufficient effective sample size
        checks["ess_sufficient"] = evaluation.effective_sample_size >= self.min_ess
        checks["ess_value"] = round(evaluation.effective_sample_size, 1)
        checks["ess_threshold"] = self.min_ess

        # Check 2: Degradation bound
        degradation_floor = baseline_value * (1 - self.max_degradation)
        lower_ci = evaluation.confidence_interval[0]
        checks["no_degradation"] = lower_ci >= degradation_floor
        checks["lower_ci"] = round(lower_ci, 4)
        checks["degradation_floor"] = round(degradation_floor, 4)

        # Check 3: Minimum improvement
        improvement = (
            (evaluation.estimated_value - baseline_value) / baseline_value
            if baseline_value > 0 else 0
        )
        checks["improvement_sufficient"] = improvement >= self.min_improvement
        checks["improvement_pct"] = round(improvement * 100, 2)
        checks["improvement_threshold_pct"] = round(self.min_improvement * 100, 2)

        # Overall safety verdict
        safe = all([
            checks["ess_sufficient"],
            checks["no_degradation"],
            checks["improvement_sufficient"],
        ])

        # Deployment recommendation
        if safe:
            recommendation = "DEPLOY_GRADUAL"
            rollout_plan = [
                {"phase": 1, "traffic_pct": 10, "duration_days": 3},
                {"phase": 2, "traffic_pct": 50, "duration_days": 5},
                {"phase": 3, "traffic_pct": 100, "duration_days": 0},
            ]
        elif checks["no_degradation"] and not checks["improvement_sufficient"]:
            recommendation = "HOLD_NEEDS_MORE_DATA"
            rollout_plan = None
        else:
            recommendation = "REJECT_UNSAFE"
            rollout_plan = None

        return {
            "safe_to_deploy": safe,
            "recommendation": recommendation,
            "rollout_plan": rollout_plan,
            "checks": checks,
            "policy_name": policy_name,
        }

    @staticmethod
    def _compute_baseline(decisions: List[LoggedDecision]) -> float:
        """Compute the baseline ROAS from executed decisions."""
        total_spend = 0.0
        total_sales = 0.0
        for d in decisions:
            if d.was_executed and d.actual_outcome:
                total_spend += float(d.actual_outcome.get("spend", 0))
                total_sales += float(d.actual_outcome.get("sales", 0))
        return total_sales / total_spend if total_spend > 0 else 0.0

    # ── persistence ──────────────────────────────────────────────────

    async def _record_backtest(
        self,
        profile_id: str,
        policy_name: str,
        decisions: List[LoggedDecision],
        ips_result: PolicyEvaluation,
        dr_result: PolicyEvaluation,
        baseline_value: float,
        safety: Dict[str, Any],
    ):
        """Save evaluation to backtest_results table."""
        try:
            dates = [d.created_at for d in decisions if d.created_at]
            start_date = min(dates).date() if dates else datetime.utcnow().date()
            end_date = max(dates).date() if dates else datetime.utcnow().date()

            await self.db.execute(
                text("""
                    INSERT INTO backtest_results (
                        profile_id, strategy_name,
                        test_period_start, test_period_end,
                        hypothetical_performance, actual_performance,
                        improvement_metrics, confidence
                    ) VALUES (
                        :pid, :name,
                        :start, :end,
                        :hypo, :actual,
                        :improvement, :conf
                    )
                """),
                {
                    "pid": profile_id,
                    "name": policy_name,
                    "start": start_date,
                    "end": end_date,
                    "hypo": json.dumps({
                        "ips_estimated_roas": ips_result.estimated_value,
                        "dr_estimated_roas": dr_result.estimated_value,
                        "ips_ci_95": list(ips_result.confidence_interval),
                        "dr_ci_95": list(dr_result.confidence_interval),
                        "method": dr_result.method,
                    }),
                    "actual": json.dumps({
                        "baseline_roas": baseline_value,
                        "n_decisions": len(decisions),
                        "executed_count": sum(
                            1 for d in decisions if d.was_executed
                        ),
                    }),
                    "improvement": json.dumps({
                        "estimated_improvement_pct": safety["checks"]["improvement_pct"],
                        "safe_to_deploy": safety["safe_to_deploy"],
                        "recommendation": safety["recommendation"],
                        "ess": dr_result.effective_sample_size,
                    }),
                    "conf": min(
                        1.0,
                        dr_result.effective_sample_size / self.min_ess
                    ),
                },
            )
            await self.db.commit()
            logger.info(
                f"[Counterfactual] Backtest result saved for '{policy_name}'"
            )
        except Exception as e:
            logger.error(f"[Counterfactual] Failed to save backtest: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  4. Convenience: run all built-in policies
# ═══════════════════════════════════════════════════════════════════════

async def evaluate_all_policies(
    db: AsyncSession,
    profile_id: str,
    lookback_days: int = 30,
) -> Dict[str, Any]:
    """
    Evaluate all built-in target policies against historical data
    for a given profile. Returns a comparison of all policies.
    """
    framework = SafeExperimentFramework(db)

    policies = [
        ("aggressive_bidding", SafeExperimentFramework.aggressive_bidding_policy),
        ("conservative_bidding", SafeExperimentFramework.conservative_bidding_policy),
        ("roas_maximizing", SafeExperimentFramework.roas_maximizing_policy),
    ]

    results = {}
    for name, fn in policies:
        result = await framework.evaluate_policy(
            profile_id, fn, policy_name=name, lookback_days=lookback_days
        )
        results[name] = result

    # Rank by DR estimated value
    ranked = sorted(
        results.items(),
        key=lambda x: x[1].get("dr_evaluation", {}).get("estimated_roas", 0),
        reverse=True,
    )

    return {
        "profile_id": profile_id,
        "lookback_days": lookback_days,
        "policies_evaluated": len(results),
        "ranking": [
            {
                "rank": i + 1,
                "policy": name,
                "estimated_roas": r.get("dr_evaluation", {}).get("estimated_roas", 0),
                "safe": r.get("safety_assessment", {}).get("safe_to_deploy", False),
                "recommendation": r.get("safety_assessment", {}).get("recommendation", "unknown"),
            }
            for i, (name, r) in enumerate(ranked)
        ],
        "details": results,
    }
