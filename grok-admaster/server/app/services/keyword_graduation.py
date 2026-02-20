"""
Keyword graduation: graduate to exact (harvest) and negate lists using Bayesian predictor.

Supports intent-aware thresholds so Rufus/discovery traffic is not
prematurely negated and is graduated with appropriate match types.
"""
from typing import Any, Dict, List, Optional

from app.services.bayesian_predictor import score_keywords_bayesian


def compute_graduation(
    terms: List[dict],
    target_acos: float,
    min_orders_to_graduate: int = 3,
    min_clicks_to_negate: int = 10,
    acos_ceiling_for_negate: float = 0.50,
    min_spend_for_negate_by_acos: float = 5.0,
    prob_acos_threshold_graduate: float = 0.6,
    intent_aware: bool = True,
) -> dict:
    """
    Returns {"to_graduate": [...], "to_negate": [...]}.

    When intent_aware=True, each term is classified by shopping intent
    and uses intent-specific thresholds for graduation and negation.
    Rufus/discovery terms get higher order requirements and more lenient
    negation rules to account for their longer conversion funnel.

    Args:
        terms: List of dicts with clicks, orders, spend, sales, keyword_text, etc.
        target_acos: Target ACoS in percent (e.g. 25.0).
        min_orders_to_graduate: Default minimum orders (overridden per-intent).
        min_clicks_to_negate: Default min clicks for negation (overridden per-intent).
        acos_ceiling_for_negate: Default ACoS ceiling for negation (overridden per-intent).
        min_spend_for_negate_by_acos: Default min spend for ACoS-based negation.
        prob_acos_threshold_graduate: Default Bayesian P(ACoS<target) threshold.
        intent_aware: Whether to apply intent-specific threshold adjustments.
    """
    if not terms:
        return {"to_graduate": [], "to_negate": []}

    target_acos = float(target_acos)
    scored = score_keywords_bayesian(terms, target_acos=target_acos)

    # Load intent classifier lazily
    classifier = None
    if intent_aware:
        try:
            from app.services.ml.intent_classifier import (
                get_intent_classifier, get_intent_thresholds, ShoppingIntent,
            )
            classifier = get_intent_classifier()
        except Exception:
            classifier = None

    to_graduate: List[dict] = []
    to_negate: List[dict] = []

    for s in scored:
        keyword_text = s.get("keyword_text", "")
        campaign_id = s.get("campaign_id", "")
        campaign_name = s.get("campaign_name", "")
        orders = s.get("orders", 0)
        clicks = s.get("clicks", 0)
        spend = s.get("spend", 0.0)
        sales = s.get("sales", 0.0)
        prob_below = s.get("prob_acos_below_target", 0.0)
        observed_acos = (spend / sales) if sales and sales > 0 else 1.0

        # Determine intent-specific thresholds
        intent_str = "transactional"
        if classifier is not None:
            result = classifier.classify(keyword_text)
            intent_str = result.intent.value
            thresholds = get_intent_thresholds(result.intent)
            grad_orders = thresholds["min_orders_to_graduate"]
            grad_prob = thresholds["prob_acos_threshold"]
            grad_match = thresholds["suggested_match_type"]
            neg_clicks = thresholds["min_clicks_to_negate"]
            neg_acos_ceil = thresholds["acos_ceiling_for_negate"]
            neg_min_spend = thresholds["min_spend_for_negate_by_acos"]
        else:
            grad_orders = min_orders_to_graduate
            grad_prob = prob_acos_threshold_graduate
            grad_match = "exact"
            neg_clicks = min_clicks_to_negate
            neg_acos_ceil = acos_ceiling_for_negate
            neg_min_spend = min_spend_for_negate_by_acos

        # Graduate: enough orders and Bayesian P(ACoS < target) >= threshold
        if orders >= grad_orders and prob_below >= grad_prob:
            to_graduate.append({
                "keyword_text": keyword_text,
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "suggested_match_type": grad_match,
                "intent_type": intent_str,
                "reason": (
                    f"Orders={orders}, P(ACoS<target)={prob_below:.2f}; "
                    f"ready for {grad_match} match ({intent_str})."
                ),
            })

        # Negate: many clicks and zero orders, or observed ACoS above ceiling
        if clicks >= neg_clicks and orders == 0:
            to_negate.append({
                "keyword_text": keyword_text,
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "intent_type": intent_str,
                "reason": (
                    f"{clicks} clicks, 0 orders "
                    f"(threshold={neg_clicks} for {intent_str}); add as negative."
                ),
            })
        elif spend >= neg_min_spend and observed_acos > neg_acos_ceil:
            to_negate.append({
                "keyword_text": keyword_text,
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "intent_type": intent_str,
                "reason": (
                    f"Observed ACoS {observed_acos*100:.1f}% > "
                    f"{neg_acos_ceil*100:.0f}% ceiling for {intent_str}; add as negative."
                ),
            })

    return {"to_graduate": to_graduate, "to_negate": to_negate}
