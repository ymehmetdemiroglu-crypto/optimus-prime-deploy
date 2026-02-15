"""
Bayesian Forecast-Driven Optimization (Phase 4).

Implements Bayesian structural time-series (BSTS) forecasting for PPC
metrics with pre-emptive bid adjustments based on predicted demand
shifts.

Components:
    1. **BayesianStateSpace** — Local-linear-trend + weekly seasonality
       state-space model with Kalman filtering. Produces probabilistic
       forecasts (mean + credible intervals) for 24-72 hours ahead.

    2. **DemandPredictor** — DB-backed demand forecasting pipeline that
       loads historical performance_records and produces hourly/daily
       demand forecasts per campaign or keyword.

    3. **PreemptiveBidAdjuster** — Converts demand forecasts into
       concrete bid adjustments *before* demand materialises.

Mathematical Foundation:
    State transition:   α_{t+1} = T α_t + R η_t,   η ~ N(0, Q)
    Observation:        y_t     = Z α_t + ε_t,       ε ~ N(0, H)

    where α = [level, trend, seasonal_1, ..., seasonal_6]

References:
    - Durbin & Koopman, "Time Series Analysis by State Space Methods"
    - Scott & Varian, "Predicting the Present with Bayesian Structural
      Time Series" (Google, 2014)
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BayesianForecast:
    """Result of a Bayesian forecast."""
    metric: str
    horizon: int                       # number of steps forecast
    mean: List[float]                  # point forecasts
    lower_50: List[float]              # 50% credible interval lower
    upper_50: List[float]
    lower_95: List[float]              # 95% credible interval lower
    upper_95: List[float]
    trend_direction: str               # "up", "down", "stable"
    trend_strength: float              # 0–1 confidence in trend
    seasonality_strength: float        # 0–1, how seasonal the series is
    anomaly_probability: float         # P(next value is anomalous)
    timestamps: List[str] = field(default_factory=list)


@dataclass
class BidAdjustment:
    """Pre-emptive bid adjustment recommendation."""
    keyword_id: Optional[int]
    campaign_id: int
    current_bid: float
    recommended_bid: float
    multiplier: float
    reason: str
    confidence: float
    forecast_horizon_hours: int
    expected_demand_change: float      # fractional change vs baseline


# ═══════════════════════════════════════════════════════════════════════
#  1. Bayesian State-Space Model (Kalman Filter)
# ═══════════════════════════════════════════════════════════════════════

class BayesianStateSpace:
    """
    Local-linear-trend + weekly seasonality state-space model.

    State vector α = [level, trend, s1, s2, s3, s4, s5, s6]
    (8-dimensional for weekly seasonality with period 7)

    The Kalman filter provides:
        - Filtering: P(α_t | y_{1:t})
        - Forecasting: P(y_{t+h} | y_{1:t})
        - Smoothing: P(α_t | y_{1:T})  (for retrospective analysis)
    """

    STATE_DIM = 8   # level + trend + 6 seasonal components (period=7)

    def __init__(
        self,
        sigma_level: float = 0.01,
        sigma_trend: float = 0.001,
        sigma_seasonal: float = 0.005,
        sigma_obs: float = 0.05,
    ):
        """
        Parameters
        ----------
        sigma_level : float
            Innovation std for the local level.
        sigma_trend : float
            Innovation std for the trend component.
        sigma_seasonal : float
            Innovation std for seasonal components.
        sigma_obs : float
            Observation noise std.
        """
        d = self.STATE_DIM

        # ── Transition matrix T (8×8) ──────────────────────────────
        self.T = np.eye(d)
        self.T[0, 1] = 1.0   # level += trend
        # Seasonal rotation: s_{t+1} = -s1 - s2 - ... - s6
        for j in range(2, d):
            self.T[2, j] = -1.0
        for j in range(3, d):
            self.T[j, j - 1] = 1.0
            self.T[j, j] = 0.0

        # ── Observation vector Z (1×8) ─────────────────────────────
        self.Z = np.zeros((1, d))
        self.Z[0, 0] = 1.0   # observe level
        self.Z[0, 2] = 1.0   # + current seasonal

        # ── State covariance Q ─────────────────────────────────────
        self.Q = np.diag([
            sigma_level ** 2,
            sigma_trend ** 2,
            sigma_seasonal ** 2,
            *(sigma_seasonal ** 2 * 0.1 for _ in range(5)),  # damped seasonal
        ])

        # ── Observation variance H ─────────────────────────────────
        self.H = np.array([[sigma_obs ** 2]])

        # ── Initial state ──────────────────────────────────────────
        self.alpha = np.zeros(d)          # state mean
        self.P = np.eye(d) * 1.0          # state covariance (diffuse)
        self._fitted = False
        self._residuals: List[float] = []

    def fit(self, observations: List[float]) -> "BayesianStateSpace":
        """
        Run the Kalman filter forward on observed data.

        This updates the state estimate α and covariance P.
        """
        if len(observations) < 3:
            logger.warning("[BSTS] Too few observations to fit")
            return self

        # Initialise level to first observation
        self.alpha[0] = observations[0]
        if len(observations) >= 2:
            self.alpha[1] = observations[1] - observations[0]

        self._residuals = []

        for y in observations:
            self._kalman_update(y)

        self._fitted = True
        return self

    def forecast(self, horizon: int = 7) -> BayesianForecast:
        """
        Produce h-step-ahead probabilistic forecasts.

        Returns BayesianForecast with mean and credible intervals.
        """
        means = []
        vars_ = []

        alpha_f = self.alpha.copy()
        P_f = self.P.copy()

        for h in range(horizon):
            # Predict state
            alpha_f = self.T @ alpha_f
            P_f = self.T @ P_f @ self.T.T + self.Q

            # Predict observation
            y_hat = float(self.Z @ alpha_f)
            y_var = float(self.Z @ P_f @ self.Z.T + self.H)

            means.append(y_hat)
            vars_.append(max(y_var, 1e-10))

        stds = [math.sqrt(v) for v in vars_]

        # Credible intervals
        lower_50 = [m - 0.674 * s for m, s in zip(means, stds)]
        upper_50 = [m + 0.674 * s for m, s in zip(means, stds)]
        lower_95 = [m - 1.96 * s for m, s in zip(means, stds)]
        upper_95 = [m + 1.96 * s for m, s in zip(means, stds)]

        # Trend analysis
        trend_dir, trend_str = self._analyse_trend(means)
        seas_str = self._seasonality_strength()
        anom_p = self._anomaly_probability()

        return BayesianForecast(
            metric="",   # caller fills this in
            horizon=horizon,
            mean=means,
            lower_50=lower_50,
            upper_50=upper_50,
            lower_95=lower_95,
            upper_95=upper_95,
            trend_direction=trend_dir,
            trend_strength=trend_str,
            seasonality_strength=seas_str,
            anomaly_probability=anom_p,
        )

    # ── Kalman filter core ────────────────────────────────────────

    def _kalman_update(self, y: float):
        """Single Kalman filter step (predict + update)."""
        # ── Predict ──
        alpha_pred = self.T @ self.alpha
        P_pred = self.T @ self.P @ self.T.T + self.Q

        # ── Innovation ──
        y_pred = float(self.Z @ alpha_pred)
        v = y - y_pred                                 # innovation
        F = float(self.Z @ P_pred @ self.Z.T + self.H)  # innovation variance
        F = max(F, 1e-10)

        # ── Kalman gain ──
        K = (P_pred @ self.Z.T) / F                    # (d×1)

        # ── Update ──
        self.alpha = alpha_pred + K.flatten() * v
        self.P = P_pred - (K @ self.Z) * F  # Joseph form simplified

        # Ensure P stays symmetric positive-definite
        self.P = (self.P + self.P.T) / 2
        np.fill_diagonal(self.P, np.maximum(np.diag(self.P), 1e-10))

        self._residuals.append(v / math.sqrt(F))

    # ── diagnostics ───────────────────────────────────────────────

    def _analyse_trend(self, forecast_means: List[float]) -> Tuple[str, float]:
        """Determine trend direction and strength from forecast."""
        if len(forecast_means) < 2:
            return "stable", 0.0

        # Use the state trend component for primary signal
        trend = self.alpha[1]
        level = max(abs(self.alpha[0]), 1e-6)
        relative_trend = trend / level

        if abs(relative_trend) < 0.01:
            return "stable", abs(relative_trend) * 10
        elif relative_trend > 0:
            return "up", min(1.0, abs(relative_trend) * 5)
        else:
            return "down", min(1.0, abs(relative_trend) * 5)

    def _seasonality_strength(self) -> float:
        """Estimate how much of the signal is seasonal (0–1)."""
        seasonal = self.alpha[2:8]
        seasonal_var = np.var(seasonal)
        level_var = max(self.P[0, 0], 1e-10)
        return min(1.0, seasonal_var / (seasonal_var + level_var + 1e-10))

    def _anomaly_probability(self) -> float:
        """Probability that the next observation will be anomalous."""
        if len(self._residuals) < 10:
            return 0.0
        recent = np.array(self._residuals[-10:])
        # Fraction of recent standardised residuals exceeding 2σ
        return float(np.mean(np.abs(recent) > 2.0))

    # ── serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha.tolist(),
            "P": self.P.tolist(),
            "T": self.T.tolist(),
            "Z": self.Z.tolist(),
            "Q": np.diag(self.Q).tolist(),
            "H": float(self.H[0, 0]),
            "residuals": self._residuals[-50:],
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BayesianStateSpace":
        model = cls()
        model.alpha = np.array(d["alpha"])
        model.P = np.array(d["P"])
        if "T" in d:
            model.T = np.array(d["T"])
        if "Z" in d:
            model.Z = np.array(d["Z"])
        if "Q" in d:
            model.Q = np.diag(d["Q"])
        if "H" in d:
            model.H = np.array([[d["H"]]])
        model._residuals = d.get("residuals", [])
        model._fitted = d.get("fitted", False)
        return model


# ═══════════════════════════════════════════════════════════════════════
#  2. Demand Predictor (DB-backed)
# ═══════════════════════════════════════════════════════════════════════

class DemandPredictor:
    """
    Multi-metric demand forecasting pipeline.

    For each campaign, fits a BayesianStateSpace model to each metric
    (impressions, clicks, spend, sales) and produces probabilistic
    forecasts. Models are persisted in the database for incremental
    online updating.
    """

    METRICS = ["impressions", "clicks", "spend", "sales", "orders"]

    def __init__(self, db: AsyncSession):
        self.db = db
        self._models: Dict[str, Dict[str, BayesianStateSpace]] = {}

    async def forecast_campaign(
        self,
        campaign_id: int,
        horizon: int = 7,
        lookback_days: int = 60,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, BayesianForecast]:
        """
        Generate demand forecasts for all metrics of a campaign.

        Returns dict mapping metric name → BayesianForecast.
        """
        ref = reference_date or datetime.utcnow()
        history = await self._load_campaign_history(
            campaign_id, lookback_days, ref
        )

        if len(history) < 14:
            logger.warning(
                f"[DemandPredictor] Campaign {campaign_id}: only "
                f"{len(history)} days of history (need ≥14)"
            )
            return {}

        forecasts: Dict[str, BayesianForecast] = {}
        cache_key = str(campaign_id)

        for metric in self.METRICS:
            values = [float(h.get(metric, 0)) for h in history]

            # Skip if all zeros
            if max(values) == 0:
                continue

            # Fit or update model
            model = self._models.get(cache_key, {}).get(metric)
            if model is None:
                model = BayesianStateSpace()

            model.fit(values)

            # Cache model
            if cache_key not in self._models:
                self._models[cache_key] = {}
            self._models[cache_key][metric] = model

            # Forecast
            fc = model.forecast(horizon)
            fc.metric = metric

            # Add timestamps
            base_date = ref.date() if hasattr(ref, 'date') else ref
            fc.timestamps = [
                str(base_date + timedelta(days=i + 1))
                for i in range(horizon)
            ]

            forecasts[metric] = fc

        return forecasts

    async def forecast_portfolio(
        self,
        profile_id: str,
        horizon: int = 7,
        lookback_days: int = 60,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Forecast demand for all campaigns in a portfolio.

        Returns per-campaign forecasts + aggregated portfolio forecast.
        """
        ref = reference_date or datetime.utcnow()

        # Get campaigns
        result = await self.db.execute(
            text("SELECT id FROM ppc_campaigns WHERE profile_id = :pid AND state = 'ENABLED'"),
            {"pid": profile_id},
        )
        campaign_ids = [r["id"] for r in result.mappings().all()]

        campaign_forecasts = {}
        for cid in campaign_ids:
            fc = await self.forecast_campaign(cid, horizon, lookback_days, ref)
            if fc:
                campaign_forecasts[cid] = fc

        # Aggregate portfolio-level forecast
        portfolio_forecast = self._aggregate_forecasts(campaign_forecasts, horizon)

        return {
            "profile_id": profile_id,
            "n_campaigns": len(campaign_ids),
            "n_forecasted": len(campaign_forecasts),
            "horizon": horizon,
            "campaign_forecasts": {
                str(cid): {
                    metric: {
                        "mean": fc.mean,
                        "lower_95": fc.lower_95,
                        "upper_95": fc.upper_95,
                        "trend": fc.trend_direction,
                        "trend_strength": fc.trend_strength,
                        "seasonality": fc.seasonality_strength,
                    }
                    for metric, fc in forecasts.items()
                }
                for cid, forecasts in campaign_forecasts.items()
            },
            "portfolio_forecast": portfolio_forecast,
        }

    # ── data loading ──────────────────────────────────────────────

    async def _load_campaign_history(
        self,
        campaign_id: int,
        lookback_days: int,
        reference_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Load daily aggregated performance for a campaign."""
        start_date = reference_date - timedelta(days=lookback_days)

        result = await self.db.execute(
            text("""
                SELECT
                    p.date,
                    SUM(p.impressions) AS impressions,
                    SUM(p.clicks)      AS clicks,
                    SUM(p.spend)       AS spend,
                    SUM(p.sales)       AS sales,
                    SUM(p.orders)      AS orders
                FROM performance_records p
                JOIN ppc_keywords k ON p.keyword_id = k.id
                WHERE k.campaign_id = :cid
                  AND p.date >= :start
                  AND p.date < :end
                GROUP BY p.date
                ORDER BY p.date
            """),
            {
                "cid": campaign_id,
                "start": start_date,
                "end": reference_date,
            },
        )
        return [dict(r) for r in result.mappings().all()]

    def _aggregate_forecasts(
        self,
        campaign_forecasts: Dict[int, Dict[str, BayesianForecast]],
        horizon: int,
    ) -> Dict[str, Any]:
        """Aggregate individual campaign forecasts into portfolio level."""
        agg: Dict[str, Dict[str, List[float]]] = {}

        for cid, metrics in campaign_forecasts.items():
            for metric, fc in metrics.items():
                if metric not in agg:
                    agg[metric] = {
                        "mean": [0.0] * horizon,
                        "lower_95": [0.0] * horizon,
                        "upper_95": [0.0] * horizon,
                    }
                for i in range(min(horizon, len(fc.mean))):
                    agg[metric]["mean"][i] += fc.mean[i]
                    agg[metric]["lower_95"][i] += fc.lower_95[i]
                    agg[metric]["upper_95"][i] += fc.upper_95[i]

        return agg


# ═══════════════════════════════════════════════════════════════════════
#  3. Pre-emptive Bid Adjuster
# ═══════════════════════════════════════════════════════════════════════

class PreemptiveBidAdjuster:
    """
    Converts demand forecasts into bid adjustments.

    Strategy:
        - If demand (impressions) is forecast to INCREASE and current
          ACoS is healthy → increase bid to capture more volume.
        - If demand is forecast to DECREASE → reduce bid to avoid
          overspending on low-volume periods.
        - If ROAS is forecast to improve → hold steady or increase.
        - If spend is forecast to spike with flat sales → reduce bid.

    All adjustments are conservative and bounded:
        multiplier ∈ [0.7, 1.5]
    """

    MIN_MULTIPLIER = 0.7
    MAX_MULTIPLIER = 1.5

    def __init__(
        self,
        db: AsyncSession,
        target_acos: float = 0.3,
        aggressiveness: float = 0.5,
    ):
        """
        Parameters
        ----------
        db : AsyncSession
        target_acos : float
            Target ACoS to optimise towards.
        aggressiveness : float
            0 = very conservative, 1 = aggressive adjustments.
        """
        self.db = db
        self.target_acos = target_acos
        self.aggressiveness = np.clip(aggressiveness, 0, 1)
        self.predictor = DemandPredictor(db)

    async def get_adjustments(
        self,
        campaign_id: int,
        horizon_hours: int = 48,
        reference_date: Optional[datetime] = None,
    ) -> List[BidAdjustment]:
        """
        Generate pre-emptive bid adjustments for keywords in a campaign
        based on demand forecasts.
        """
        ref = reference_date or datetime.utcnow()
        horizon_days = max(1, horizon_hours // 24)

        # Get demand forecast
        forecasts = await self.predictor.forecast_campaign(
            campaign_id, horizon=horizon_days,
            lookback_days=60, reference_date=ref,
        )

        if not forecasts:
            return []

        # Compute expected demand change
        demand_signal = self._compute_demand_signal(forecasts)

        # Load current keywords with bids
        keywords = await self._load_keywords(campaign_id)

        adjustments: List[BidAdjustment] = []
        for kw in keywords:
            adj = self._compute_adjustment(kw, demand_signal, horizon_hours)
            if adj is not None:
                adjustments.append(adj)

        logger.info(
            f"[PreemptiveBid] Campaign {campaign_id}: "
            f"{len(adjustments)} adjustments from {len(keywords)} keywords, "
            f"demand_signal={demand_signal}"
        )
        return adjustments

    async def get_portfolio_adjustments(
        self,
        profile_id: str,
        horizon_hours: int = 48,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate pre-emptive adjustments for all campaigns in a portfolio.
        """
        ref = reference_date or datetime.utcnow()

        result = await self.db.execute(
            text("SELECT id FROM ppc_campaigns WHERE profile_id = :pid AND state = 'ENABLED'"),
            {"pid": profile_id},
        )
        campaign_ids = [r["id"] for r in result.mappings().all()]

        all_adjustments = []
        for cid in campaign_ids:
            adjs = await self.get_adjustments(cid, horizon_hours, ref)
            all_adjustments.extend(adjs)

        # Summary stats
        if all_adjustments:
            multipliers = [a.multiplier for a in all_adjustments]
            return {
                "profile_id": profile_id,
                "n_campaigns": len(campaign_ids),
                "n_adjustments": len(all_adjustments),
                "avg_multiplier": float(np.mean(multipliers)),
                "min_multiplier": float(np.min(multipliers)),
                "max_multiplier": float(np.max(multipliers)),
                "adjustments": [
                    {
                        "campaign_id": a.campaign_id,
                        "keyword_id": a.keyword_id,
                        "current_bid": a.current_bid,
                        "recommended_bid": round(a.recommended_bid, 4),
                        "multiplier": round(a.multiplier, 4),
                        "reason": a.reason,
                        "confidence": round(a.confidence, 3),
                        "expected_demand_change": round(a.expected_demand_change, 4),
                    }
                    for a in all_adjustments
                ],
            }

        return {
            "profile_id": profile_id,
            "n_campaigns": len(campaign_ids),
            "n_adjustments": 0,
            "message": "No adjustments recommended",
        }

    # ── helpers ────────────────────────────────────────────────────

    def _compute_demand_signal(
        self, forecasts: Dict[str, BayesianForecast]
    ) -> Dict[str, float]:
        """
        Summarise forecasts into a demand signal dict.

        Returns fractional change vs current level for each metric.
        """
        signal = {}
        for metric, fc in forecasts.items():
            if not fc.mean or len(fc.mean) < 2:
                signal[metric] = 0.0
                continue

            # Compare forecast mean to current level (first forecast value
            # is tomorrow, so use the average of next few days vs today)
            current = fc.mean[0]
            future_avg = np.mean(fc.mean[1:min(4, len(fc.mean))])

            if abs(current) < 1e-6:
                signal[metric] = 0.0
            else:
                signal[metric] = (future_avg - current) / abs(current)

        return signal

    def _compute_adjustment(
        self,
        keyword: Dict[str, Any],
        demand_signal: Dict[str, float],
        horizon_hours: int,
    ) -> Optional[BidAdjustment]:
        """Compute a single bid adjustment for a keyword."""
        current_bid = float(keyword.get("bid", 0))
        if current_bid <= 0:
            return None

        kw_id = keyword.get("id")
        campaign_id = keyword.get("campaign_id")

        # Keyword's recent performance
        kw_acos = float(keyword.get("acos", self.target_acos))
        kw_roas = float(keyword.get("roas", 1.0 / self.target_acos if self.target_acos > 0 else 3.0))

        # Demand signals
        imp_change = demand_signal.get("impressions", 0.0)
        click_change = demand_signal.get("clicks", 0.0)
        spend_change = demand_signal.get("spend", 0.0)
        sales_change = demand_signal.get("sales", 0.0)

        # ── Decision logic ────────────────────────────────────────
        multiplier = 1.0
        reasons = []

        # Signal 1: Demand increase + healthy ACoS → bid up
        if imp_change > 0.05 and kw_acos < self.target_acos * 1.1:
            boost = imp_change * self.aggressiveness * 0.5
            multiplier += boost
            reasons.append(
                f"Demand ↑{imp_change:.1%}, ACoS healthy ({kw_acos:.1%})"
            )

        # Signal 2: Demand decrease → bid down to save budget
        elif imp_change < -0.05:
            reduction = abs(imp_change) * self.aggressiveness * 0.3
            multiplier -= reduction
            reasons.append(f"Demand ↓{imp_change:.1%}")

        # Signal 3: Sales forecast up, spend flat → good opportunity
        if sales_change > 0.1 and abs(spend_change) < 0.05:
            multiplier += 0.05 * self.aggressiveness
            reasons.append(f"Sales forecast ↑{sales_change:.1%}, spend stable")

        # Signal 4: Spend forecast spike, sales flat → danger
        if spend_change > 0.15 and sales_change < 0.05:
            multiplier -= 0.1 * self.aggressiveness
            reasons.append(
                f"Spend spike ↑{spend_change:.1%} with flat sales"
            )

        # Signal 5: High ACoS correction
        if kw_acos > self.target_acos * 1.3:
            overshoot = (kw_acos - self.target_acos) / self.target_acos
            multiplier -= overshoot * 0.15 * self.aggressiveness
            reasons.append(f"ACoS correction ({kw_acos:.1%} > target {self.target_acos:.1%})")

        # Signal 6: Under-target ACoS opportunity
        elif kw_acos < self.target_acos * 0.7 and kw_roas > 3.0:
            multiplier += 0.05 * self.aggressiveness
            reasons.append(f"Strong ROAS {kw_roas:.1f}x, room to scale")

        # Clamp
        multiplier = np.clip(multiplier, self.MIN_MULTIPLIER, self.MAX_MULTIPLIER)

        # Skip if negligible change
        if abs(multiplier - 1.0) < 0.02:
            return None

        recommended_bid = round(current_bid * multiplier, 4)

        # Confidence based on forecast strength
        confidence = min(
            1.0,
            0.5 + abs(multiplier - 1.0) * 2 + abs(imp_change) * 0.5
        )

        return BidAdjustment(
            keyword_id=kw_id,
            campaign_id=campaign_id,
            current_bid=current_bid,
            recommended_bid=recommended_bid,
            multiplier=float(multiplier),
            reason="; ".join(reasons) if reasons else "forecast-based",
            confidence=float(confidence),
            forecast_horizon_hours=horizon_hours,
            expected_demand_change=float(imp_change),
        )

    async def _load_keywords(self, campaign_id: int) -> List[Dict[str, Any]]:
        """Load keywords with current bids and recent performance."""
        result = await self.db.execute(
            text("""
                SELECT
                    k.id,
                    k.campaign_id,
                    k.bid,
                    COALESCE(SUM(p.spend), 0) / NULLIF(SUM(p.sales), 0) AS acos,
                    COALESCE(SUM(p.sales), 0) / NULLIF(SUM(p.spend), 0) AS roas
                FROM ppc_keywords k
                LEFT JOIN performance_records p
                    ON p.keyword_id = k.id
                    AND p.date >= NOW() - INTERVAL '7 days'
                WHERE k.campaign_id = :cid
                  AND k.state = 'ENABLED'
                GROUP BY k.id, k.campaign_id, k.bid
            """),
            {"cid": campaign_id},
        )
        return [dict(r) for r in result.mappings().all()]
