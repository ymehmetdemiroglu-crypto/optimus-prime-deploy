"""
Bayesian predictor for CVR (conversion rate) and ACoS per keyword/search term.
Uses Beta-Binomial for CVR and Beta for ACoS; scores keywords for predicted best performers.
"""
from typing import Any, List, Tuple
from scipy import stats


# --- CVR (Conversion Rate): Beta-Binomial ---

def update_cvr_posterior(
    clicks: int,
    orders: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 9.0,
) -> Tuple[float, float]:
    """Posterior Beta(alpha + orders, beta + clicks - orders)."""
    if clicks < 0 or orders < 0 or orders > clicks:
        return (prior_alpha, prior_beta)
    posterior_alpha = prior_alpha + orders
    posterior_beta = prior_beta + (clicks - orders)
    return (posterior_alpha, posterior_beta)


def predict_cvr(
    clicks: int,
    orders: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 9.0,
    threshold: float = 0.05,
) -> dict:
    """Posterior mean, std, and P(CVR > threshold)."""
    a, b = update_cvr_posterior(clicks, orders, prior_alpha, prior_beta)
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    std = var ** 0.5
    prob_above = 1.0 - stats.beta.cdf(threshold, a, b)
    return {"mean": mean, "std": std, "prob_above": prob_above}


# --- ACoS: Beta prior, single-sample update ---

def update_acos_posterior(
    spend: float,
    sales: float,
    prior_alpha: float = 2.0,
    prior_beta: float = 8.0,
    weight: float = 1.0,
) -> Tuple[float, float]:
    """Posterior Beta from observed ACoS (spend/sales). ACoS treated as proportion in (0,1]."""
    if sales <= 0:
        return (prior_alpha, prior_beta)
    acos_obs = spend / sales
    acos_obs = min(max(acos_obs, 1e-6), 1.0)  # clip to (0, 1]
    # Pseudo-counts: one observation with proportion acos_obs, scaled by weight
    posterior_alpha = prior_alpha + weight * acos_obs
    posterior_beta = prior_beta + weight * (1.0 - acos_obs)
    return (posterior_alpha, posterior_beta)


def predict_acos(
    spend: float,
    sales: float,
    target_acos: float,
    prior_alpha: float = 2.0,
    prior_beta: float = 8.0,
    weight: float = 1.0,
) -> dict:
    """Posterior mean ACoS (as proportion 0-1) and P(ACoS < target). target_acos in percent (e.g. 20)."""
    a, b = update_acos_posterior(spend, sales, prior_alpha, prior_beta, weight)
    mean = a / (a + b)
    target_proportion = min(max(target_acos / 100.0, 0.0), 1.0)
    prob_below_target = stats.beta.cdf(target_proportion, a, b)
    return {"mean": mean, "prob_below_target": prob_below_target}


# --- Scoring ---

def _get_term(t: Any, key: str, default: Any = None) -> Any:
    if isinstance(t, dict):
        return t.get(key, default)
    return getattr(t, key, default)


def score_keywords_bayesian(
    terms: List[dict],
    target_acos: float,
    cvr_threshold: float = 0.05,
    prior_cvr_alpha: float = 1.0,
    prior_cvr_beta: float = 9.0,
    prior_acos_alpha: float = 2.0,
    prior_acos_beta: float = 8.0,
) -> List[dict]:
    """
    Score each term by Bayesian CVR and ACoS; rank by score = prob_acos_below_target * ROAS.
    Each term dict must have clicks, orders, spend, sales; optional id, keyword_text, campaign_id, campaign_name.
    """
    if not terms:
        return []
    target_acos = float(target_acos)
    scored: List[dict] = []
    for i, t in enumerate(terms):
        clicks = int(_get_term(t, "clicks", 0) or 0)
        orders = int(_get_term(t, "orders", 0) or 0)
        spend = float(_get_term(t, "spend", 0) or 0)
        sales = float(_get_term(t, "sales", 0) or 0)
        cvr_out = predict_cvr(clicks, orders, prior_cvr_alpha, prior_cvr_beta, cvr_threshold)
        acos_out = predict_acos(spend, sales, target_acos, prior_acos_alpha, prior_acos_beta)
        roas = sales / max(spend, 1e-6)
        score = acos_out["prob_below_target"] * roas
        scored.append({
            "term_id": _get_term(t, "id", _get_term(t, "keyword_text", f"term_{i}")),
            "keyword_text": _get_term(t, "keyword_text", ""),
            "campaign_id": _get_term(t, "campaign_id", ""),
            "campaign_name": _get_term(t, "campaign_name", ""),
            "clicks": clicks,
            "orders": orders,
            "spend": spend,
            "sales": sales,
            "predicted_cvr": cvr_out["mean"],
            "predicted_cvr_std": cvr_out["std"],
            "prob_cvr_above_threshold": cvr_out["prob_above"],
            "predicted_acos": acos_out["mean"] * 100.0,
            "prob_acos_below_target": acos_out["prob_below_target"],
            "score": score,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
