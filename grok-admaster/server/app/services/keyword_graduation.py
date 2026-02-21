"""
Keyword graduation: graduate to exact (harvest) and negate lists using Bayesian predictor.
"""
from typing import Any, List

from app.services.bayesian_predictor import score_keywords_bayesian


def compute_graduation(
    terms: List[dict],
    target_acos: float,
    min_orders_to_graduate: int = 3,
    min_clicks_to_negate: int = 10,
    acos_ceiling_for_negate: float = 0.50,
    min_spend_for_negate_by_acos: float = 5.0,
    prob_acos_threshold_graduate: float = 0.6,
) -> dict:
    """
    Returns {"to_graduate": [...], "to_negate": [...]}.
    to_graduate: list of {keyword_text, campaign_id, campaign_name, suggested_match_type: "exact", reason}.
    to_negate: list of {keyword_text, campaign_id, campaign_name, reason}.
    """
    if not terms:
        return {"to_graduate": [], "to_negate": []}
    target_acos = float(target_acos)
    scored = score_keywords_bayesian(terms, target_acos=target_acos)
    to_graduate: List[dict] = []
    to_negate: List[dict] = []
    for s in scored:
        keyword_text = s.get("keyword_text", "")
        campaign_id = s.get("campaign_id", "")
        campaign_name = s.get("campaign_name", "")
        query_source = s.get("query_source", "organic")
        orders = s.get("orders", 0)
        clicks = s.get("clicks", 0)
        spend = s.get("spend", 0.0)
        sales = s.get("sales", 0.0)
        prob_below = s.get("prob_acos_below_target", 0.0)
        observed_acos = (spend / sales) if sales and sales > 0 else 1.0
        
        # Adaptive thresholds depending on intent source
        if query_source == "rufus":
            # Rufus converts deeper in the funnel; requires higher proof of volume
            effective_min_orders = 5
            effective_prob_threshold = 0.50
        else:
            effective_min_orders = min_orders_to_graduate
            effective_prob_threshold = prob_acos_threshold_graduate
            
        # Graduate: enough orders and Bayesian P(ACoS < target) >= threshold
        if orders >= effective_min_orders and prob_below >= effective_prob_threshold:
            to_graduate.append({
                "keyword_text": keyword_text,
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "suggested_match_type": "exact",
                "reason": f"Orders={orders}, P(ACoS<target)={prob_below:.2f} (source: {query_source}); ready for exact match.",
            })
        # Negate: many clicks and zero orders, or observed ACoS above ceiling with enough spend
        if clicks >= min_clicks_to_negate and orders == 0:
            to_negate.append({
                "keyword_text": keyword_text,
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "reason": f"{clicks} clicks, 0 orders; add as negative.",
            })
        elif spend >= min_spend_for_negate_by_acos and observed_acos > acos_ceiling_for_negate:
            to_negate.append({
                "keyword_text": keyword_text,
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "reason": f"Observed ACoS {observed_acos*100:.1f}% > {acos_ceiling_for_negate*100:.0f}%; add as negative.",
            })
    return {"to_graduate": to_graduate, "to_negate": to_negate}
