"""
Paid Ads Module — Strategy Service

Pure-Python strategy engine that encodes the "paid-ads" skill
knowledge into deterministic recommendation logic.

No external API calls — this is a rules engine, not an LLM wrapper.
"""

import logging
from typing import List, Dict

from .schemas import (
    AdPlatform, CampaignObjective, FunnelStage, AdCopyFramework,
    BidStrategyType,
    PlatformRecommendationRequest, PlatformRecommendationResponse, PlatformScore,
    CampaignStructureRequest, CampaignStructureResponse, CampaignBlueprint, AdSetStructure,
    AdCopyRequest, AdCopyResponse, AdCopyVariation,
    BudgetAllocationRequest, BudgetAllocationResponse, BudgetSplit,
    RetargetingStrategyRequest, RetargetingStrategyResponse, RetargetingLayer,
    CampaignOptimizationRequest, CampaignOptimizationResponse, OptimizationAction,
    SetupChecklistRequest, SetupChecklistResponse, ChecklistItem,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  1. Platform Recommendation Engine
# ═══════════════════════════════════════════════════════════════════

def recommend_platforms(req: PlatformRecommendationRequest) -> PlatformRecommendationResponse:
    """Score each platform against the user's profile and return ranked list."""

    scores: List[PlatformScore] = []

    # ------ Google ------
    g_score = 60
    g_types = ["Search"]
    if req.objective in (CampaignObjective.SALES, CampaignObjective.LEADS):
        g_score += 20
        g_types.append("Performance Max")
    if req.monthly_budget >= 3000:
        g_score += 10
        g_types.append("Display")
    scores.append(PlatformScore(
        platform=AdPlatform.GOOGLE, score=min(g_score, 100),
        reasoning="Google captures existing high-intent demand via search.",
        recommended_campaign_types=g_types,
        estimated_cpc_range="$0.50 – $5.00",
    ))

    # ------ Meta ------
    m_score = 50
    m_types = ["Conversions"]
    if req.has_visual_assets:
        m_score += 20
        m_types.append("Advantage+ Shopping")
    if req.objective == CampaignObjective.AWARENESS:
        m_score += 15
        m_types.append("Reach")
    if not req.b2b:
        m_score += 10
    scores.append(PlatformScore(
        platform=AdPlatform.META, score=min(m_score, 100),
        reasoning="Meta excels at demand generation with visual creative.",
        recommended_campaign_types=m_types,
        estimated_cpc_range="$0.30 – $3.00",
    ))

    # ------ LinkedIn ------
    l_score = 30
    l_types = ["Sponsored Content"]
    if req.b2b:
        l_score += 40
        l_types.extend(["Lead Gen Forms", "Message Ads"])
    if req.monthly_budget >= 5000:
        l_score += 10
    scores.append(PlatformScore(
        platform=AdPlatform.LINKEDIN, score=min(l_score, 100),
        reasoning="LinkedIn offers precise B2B job-title and company targeting.",
        recommended_campaign_types=l_types,
        estimated_cpc_range="$5.00 – $15.00",
    ))

    # ------ TikTok ------
    t_score = 35
    t_types = ["In-Feed"]
    if req.has_visual_assets:
        t_score += 15
    if req.objective == CampaignObjective.AWARENESS:
        t_score += 20
    scores.append(PlatformScore(
        platform=AdPlatform.TIKTOK, score=min(t_score, 100),
        reasoning="TikTok delivers low CPMs for brand awareness with younger audiences.",
        recommended_campaign_types=t_types,
        estimated_cpc_range="$0.20 – $2.00",
    ))

    # ------ Amazon ------
    a_score = 40
    a_types = ["Sponsored Products"]
    if "amazon" in req.product_description.lower():
        a_score += 30
        a_types.extend(["Sponsored Brands", "Sponsored Display"])
    scores.append(PlatformScore(
        platform=AdPlatform.AMAZON, score=min(a_score, 100),
        reasoning="Amazon Ads capture high-intent purchase traffic on the marketplace.",
        recommended_campaign_types=a_types,
        estimated_cpc_range="$0.20 – $6.00",
    ))

    # Sort descending
    scores.sort(key=lambda s: s.score, reverse=True)

    # Budget split — proportional to score for top-3
    top = scores[:3]
    total_score = sum(s.score for s in top)
    budget_split = {
        s.platform.value: round(s.score / total_score * req.monthly_budget, 2)
        for s in top
    }

    return PlatformRecommendationResponse(
        recommended_platforms=scores,
        budget_split=budget_split,
        notes=[
            "Start with 2 platforms max, then expand once you hit CPA targets.",
            "Reserve 20-30% of budget for testing new audiences and creative.",
        ],
    )


# ═══════════════════════════════════════════════════════════════════
#  2. Campaign Structure Generator
# ═══════════════════════════════════════════════════════════════════

def generate_campaign_structure(req: CampaignStructureRequest) -> CampaignStructureResponse:
    """Build a recommended account/campaign/ad-set hierarchy."""

    naming = f"{req.platform.value.upper()}_{req.objective.value}_{'{audience}'}_{'{offer}'}_{'{date}'}"

    campaigns: List[CampaignBlueprint] = []

    if req.platform == AdPlatform.GOOGLE:
        # Search campaign
        campaigns.append(CampaignBlueprint(
            campaign_name=f"GOOG_Search_{req.product_name}_Ongoing",
            objective="Search — capture high-intent traffic",
            ad_sets=[
                AdSetStructure(name="Exact Match – Core", targeting_description="Exact match on primary keywords", budget_percentage=50, ad_count=3),
                AdSetStructure(name="Phrase Match – Expansion", targeting_description="Phrase match on secondary keywords", budget_percentage=30, ad_count=3),
                AdSetStructure(name="Brand Defense", targeting_description="Exact brand terms", budget_percentage=20, ad_count=2),
            ],
            naming_convention=naming,
            bid_strategy="Manual CPC → switch to Target CPA after 50+ conversions",
            notes=["Build negative keyword lists from day 1", "Add sitelink, callout, and structured snippet extensions"],
        ))
        # Performance Max
        campaigns.append(CampaignBlueprint(
            campaign_name=f"GOOG_PMax_{req.product_name}_Ongoing",
            objective="Performance Max — AI-driven cross-channel",
            ad_sets=[
                AdSetStructure(name="Asset Group – Primary", targeting_description="All signals (audience, search themes)", budget_percentage=100, ad_count=5),
            ],
            naming_convention=naming,
            bid_strategy="Maximize Conversions with Target CPA",
            notes=["Provide at least 5 images, 5 headlines, 5 descriptions", "Exclude brand search if running separate brand campaign"],
        ))

    elif req.platform == AdPlatform.META:
        campaigns.append(CampaignBlueprint(
            campaign_name=f"META_Conv_{req.product_name}_Prospecting",
            objective="Conversions — prospecting cold audiences",
            ad_sets=[
                AdSetStructure(name="Interest – Core", targeting_description="Interest targeting: primary interests", budget_percentage=40, ad_count=3),
                AdSetStructure(name="Lookalike – 1%", targeting_description="1% lookalike of best customers", budget_percentage=35, ad_count=3),
                AdSetStructure(name="Broad – Open", targeting_description="Broad targeting, let algorithm optimise", budget_percentage=25, ad_count=3),
            ],
            naming_convention=naming,
            bid_strategy="Cost Cap (start with 2× your target CPA, tighten over time)",
            notes=["Exclude existing purchasers", "Start CBO at campaign level", "Wait 3-5 days before judging performance"],
        ))
        campaigns.append(CampaignBlueprint(
            campaign_name=f"META_Conv_{req.product_name}_Retargeting",
            objective="Conversions — retargeting warm audiences",
            ad_sets=[
                AdSetStructure(name="Web Visitors 7d", targeting_description="Website visitors last 7 days", budget_percentage=50, ad_count=2),
                AdSetStructure(name="Engagers 30d", targeting_description="Page/post engagers last 30 days", budget_percentage=50, ad_count=2),
            ],
            naming_convention=naming,
            bid_strategy="Lowest Cost",
            notes=["Frequency cap 3–5 per week", "Refresh creative every 2 weeks"],
        ))

    elif req.platform == AdPlatform.LINKEDIN:
        campaigns.append(CampaignBlueprint(
            campaign_name=f"LI_LeadGen_{req.product_name}_DecisionMakers",
            objective="Lead Gen — targeting decision makers",
            ad_sets=[
                AdSetStructure(name="C-Suite + VP", targeting_description="Job seniority: VP/CXO + industry", budget_percentage=60, ad_count=3),
                AdSetStructure(name="Directors", targeting_description="Job seniority: Director + company size 51+", budget_percentage=40, ad_count=3),
            ],
            naming_convention=naming,
            bid_strategy="Manual Bidding (LinkedIn CPCs are high, control is key)",
            notes=["Budget min $50/day per campaign", "Use Lead Gen Forms for lower CPA than website conversions"],
        ))

    else:
        # Generic fallback
        campaigns.append(CampaignBlueprint(
            campaign_name=f"{req.platform.value.upper()}_Conv_{req.product_name}",
            objective=f"{req.objective.value} on {req.platform.value}",
            ad_sets=[
                AdSetStructure(name="Primary Audience", targeting_description="Core targeting", budget_percentage=70, ad_count=3),
                AdSetStructure(name="Test Audience", targeting_description="Experimental targeting", budget_percentage=30, ad_count=2),
            ],
            naming_convention=naming,
            bid_strategy="Start manual, switch to automated after 50+ conversions",
            notes=["Test one variable at a time", "Review weekly"],
        ))

    return CampaignStructureResponse(
        platform=req.platform,
        campaigns=campaigns,
        naming_convention_template=naming,
        budget_allocation_notes="Testing: 70% proven / 30% testing. Scaling: consolidate into winners, increase 20-30% at a time.",
    )


# ═══════════════════════════════════════════════════════════════════
#  3. Ad Copy Generator
# ═══════════════════════════════════════════════════════════════════

def generate_ad_copy(req: AdCopyRequest) -> AdCopyResponse:
    """Generate ad copy variations using the specified framework."""

    variations: List[AdCopyVariation] = []
    usps = req.unique_selling_points
    pains = req.pain_points or ["wasting time", "losing money", "falling behind competitors"]

    for i in range(req.num_variations):
        usp = usps[i % len(usps)]
        pain = pains[i % len(pains)]

        if req.framework == AdCopyFramework.PAS:
            headline = f"Stop {pain.title()}"
            primary = (
                f"Tired of {pain}?\n"
                f"Every day you wait, your competitors pull further ahead.\n"
                f"{req.product_name} gives you {usp} — so you can focus on growth.\n"
            )
            cta = "Start Free Trial →"
        elif req.framework == AdCopyFramework.BAB:
            headline = f"{usp} — Made Simple"
            primary = (
                f"Before: {pain.capitalize()} slowing you down.\n"
                f"After: {usp} working for you automatically.\n"
                f"{req.product_name} bridges the gap."
            )
            cta = "See How It Works"
        elif req.framework == AdCopyFramework.SOCIAL_PROOF:
            headline = f"Trusted by 1,000+ Teams"
            primary = (
                f'"We achieved {usp} in just 30 days." — Happy Customer\n'
                f"{req.product_name} eliminates {pain} for teams like yours."
            )
            cta = "Join Them Today →"
        else:  # AIDA
            headline = f"Discover {usp}"
            primary = (
                f"Attention: Are you still {pain}?\n"
                f"Interest: {req.product_name} offers {usp}.\n"
                f"Desire: Imagine having that solved by next week.\n"
                f"Action: Get started now."
            )
            cta = "Get Started Free"

        variations.append(AdCopyVariation(
            headline=headline,
            primary_text=primary,
            description=f"{req.product_name}: {usp}. Built for {req.target_audience}.",
            cta=cta,
            framework_used=req.framework.value,
        ))

    return AdCopyResponse(
        platform=req.platform,
        variations=variations,
        testing_tips=[
            "Test concept/angle first (biggest impact), then hooks, then CTAs.",
            "Kill losers fast (3–5 days with sufficient spend).",
            "Need 100+ conversions per variant for statistical significance.",
            "Iterate on winners rather than creating entirely new ads.",
        ],
    )


# ═══════════════════════════════════════════════════════════════════
#  4. Budget Allocation Engine
# ═══════════════════════════════════════════════════════════════════

def allocate_budget(req: BudgetAllocationRequest) -> BudgetAllocationResponse:
    """Split budget across platforms using objective-weighted heuristics."""

    weights: Dict[AdPlatform, float] = {}
    for p in req.platforms:
        if p == AdPlatform.GOOGLE:
            w = 40 if req.objective in (CampaignObjective.SALES, CampaignObjective.LEADS) else 25
        elif p == AdPlatform.META:
            w = 35 if req.objective == CampaignObjective.AWARENESS else 30
        elif p == AdPlatform.LINKEDIN:
            w = 30 if req.objective == CampaignObjective.LEADS else 15
        elif p == AdPlatform.TIKTOK:
            w = 25 if req.objective == CampaignObjective.AWARENESS else 10
        elif p == AdPlatform.AMAZON:
            w = 35 if req.objective == CampaignObjective.SALES else 20
        else:
            w = 15
        weights[p] = w

    total_w = sum(weights.values())
    testing_reserve = 0.3 if req.is_testing_phase else 0.15
    spendable = req.total_monthly_budget * (1 - testing_reserve)

    splits = [
        BudgetSplit(
            platform=p,
            amount=round(spendable * (w / total_w), 2),
            percentage=round(w / total_w * 100, 1),
            rationale=f"{'Testing' if req.is_testing_phase else 'Scaling'} allocation for {p.value}",
        )
        for p, w in weights.items()
    ]

    return BudgetAllocationResponse(
        total_budget=req.total_monthly_budget,
        splits=splits,
        testing_reserve_percentage=testing_reserve * 100,
        scaling_guidelines=[
            "Increase budgets 20-30% at a time.",
            "Wait 3-5 days between increases for algorithm learning.",
            "Consolidate budget into winning campaigns before scaling.",
        ],
    )


# ═══════════════════════════════════════════════════════════════════
#  5. Retargeting Strategy Builder
# ═══════════════════════════════════════════════════════════════════

def build_retargeting_strategy(req: RetargetingStrategyRequest) -> RetargetingStrategyResponse:
    """Generate a funnel-based retargeting plan."""

    layers = [
        RetargetingLayer(
            funnel_stage=FunnelStage.TOP,
            audience_description="Blog readers, video viewers, social engagers",
            window_days="30–90 days",
            frequency_cap="1–2× per week",
            message_theme="Educational content, social proof, brand story",
            exclusions=["Existing customers", "Recent converters (14 days)"],
        ),
        RetargetingLayer(
            funnel_stage=FunnelStage.MIDDLE,
            audience_description="Pricing page visitors, feature page visitors, demo page viewers",
            window_days="7–30 days",
            frequency_cap="3–5× per week",
            message_theme="Case studies, product demos, comparisons",
            exclusions=["Existing customers", "Bounced visitors (<10s on site)"],
        ),
        RetargetingLayer(
            funnel_stage=FunnelStage.BOTTOM,
            audience_description="Cart abandoners, trial users, demo no-shows",
            window_days="1–7 days",
            frequency_cap="Higher OK (urgency)",
            message_theme="Urgency, objection handling, special offers, testimonials",
            exclusions=["Converted users (last 7 days)"],
        ),
    ]

    return RetargetingStrategyResponse(
        platform=req.platform,
        layers=layers,
        exclusion_rules=[
            "Always exclude existing customers (unless upsell campaign).",
            "Exclude recent converters for 7–14 days.",
            "Exclude bounced visitors (<10 seconds on site).",
            "Exclude irrelevant-page visitors (careers, support, legal).",
        ],
        general_notes=[
            "Ensure pixel / CAPI is firing correctly before launching.",
            "Start with BOF retargeting first — highest ROAS.",
            "Use dynamic creative to match the product the user viewed.",
        ],
    )


# ═══════════════════════════════════════════════════════════════════
#  6. Campaign Optimisation Diagnostics
# ═══════════════════════════════════════════════════════════════════

def diagnose_campaign(req: CampaignOptimizationRequest) -> CampaignOptimizationResponse:
    """Analyse metrics and return prioritised fix actions."""

    actions: List[OptimizationAction] = []
    health = 70  # baseline

    # ---- CPA diagnosis ----
    if req.current_cpa and req.target_cpa and req.current_cpa > req.target_cpa:
        gap_pct = (req.current_cpa - req.target_cpa) / req.target_cpa * 100
        health -= min(int(gap_pct / 5), 30)
        actions.append(OptimizationAction(
            priority=1,
            area="CPA",
            diagnosis=f"CPA is {gap_pct:.0f}% above target (${req.current_cpa:.2f} vs ${req.target_cpa:.2f}).",
            action="Check landing page conversion rate first. Then tighten audience targeting and test new creative angles.",
            expected_impact=f"Potential {min(gap_pct, 40):.0f}% CPA reduction",
        ))

    # ---- CTR diagnosis ----
    if req.current_ctr is not None and req.current_ctr < 1.0:
        health -= 15
        actions.append(OptimizationAction(
            priority=2,
            area="CTR",
            diagnosis=f"CTR is low ({req.current_ctr:.2f}%). Ads are not resonating with the audience.",
            action="Test new hooks and value propositions. Consider audience mismatch — refine targeting. Refresh creative if running >2 weeks.",
            expected_impact="2–3× CTR improvement possible with the right angle",
        ))

    # ---- CPM diagnosis ----
    if req.current_cpm is not None and req.current_cpm > 30:
        health -= 10
        actions.append(OptimizationAction(
            priority=3,
            area="CPM",
            diagnosis=f"CPM is elevated (${req.current_cpm:.2f}). Audience may be too narrow or competitive.",
            action="Expand audience size. Try different placements or dayparting. Improve relevance score via better creative fit.",
            expected_impact="15–30% CPM reduction",
        ))

    # ---- ROAS diagnosis ----
    if req.current_roas and req.target_roas and req.current_roas < req.target_roas:
        health -= 15
        actions.append(OptimizationAction(
            priority=1,
            area="ROAS",
            diagnosis=f"ROAS is below target ({req.current_roas:.1f}× vs {req.target_roas:.1f}×).",
            action="Shift budget to top-performing ad sets. Test higher-AOV products. Review attribution window settings.",
            expected_impact="Move toward target by consolidating spend",
        ))

    # ---- Bid strategy recommendation ----
    bid_rec = "Manual CPC to start"
    if req.campaign_age_days and req.campaign_age_days > 30:
        bid_rec = "Switch to Target CPA or Target ROAS (you have enough data)"
    elif req.campaign_age_days and req.campaign_age_days > 14:
        bid_rec = "Consider Cost Cap — enough data for constrained automation"

    if not actions:
        actions.append(OptimizationAction(
            priority=3, area="General",
            diagnosis="Metrics look healthy. Focus on scaling.",
            action="Increase budget 20-30% and expand to new audience segments.",
            expected_impact="Incremental volume at similar efficiency",
        ))
        health = min(health + 10, 95)

    actions.sort(key=lambda a: a.priority)

    return CampaignOptimizationResponse(
        platform=req.platform,
        health_score=max(health, 10),
        diagnosis_summary=f"{len(actions)} optimisation action(s) identified.",
        actions=actions,
        bid_strategy_recommendation=bid_rec,
    )


# ═══════════════════════════════════════════════════════════════════
#  7. Platform Setup Checklists
# ═══════════════════════════════════════════════════════════════════

_CHECKLISTS: Dict[AdPlatform, List[dict]] = {
    AdPlatform.GOOGLE: [
        {"task": "Conversion tracking installed and tested", "critical": True},
        {"task": "Google Analytics 4 linked", "critical": True},
        {"task": "Audience lists created (remarketing, customer match)", "critical": False},
        {"task": "Negative keyword lists built", "critical": True},
        {"task": "Ad extensions set up (sitelinks, callouts, structured snippets)", "critical": False},
        {"task": "Brand campaign running (protect branded terms)", "critical": False},
        {"task": "Location and language targeting set", "critical": True},
        {"task": "Ad schedule aligned with business hours (if B2B)", "critical": False},
    ],
    AdPlatform.META: [
        {"task": "Pixel installed and all events firing", "critical": True},
        {"task": "Conversions API (CAPI) set up (server-side tracking)", "critical": True},
        {"task": "Custom audiences created", "critical": False},
        {"task": "Product catalog connected (if e-commerce)", "critical": False},
        {"task": "Domain verified", "critical": True},
        {"task": "Business Manager properly configured", "critical": True},
        {"task": "Aggregated event measurement prioritised", "critical": False},
        {"task": "Creative assets in correct sizes (1080×1080, 1080×1920)", "critical": False},
        {"task": "UTM parameters in all destination URLs", "critical": True},
    ],
    AdPlatform.LINKEDIN: [
        {"task": "Insight Tag installed on all pages", "critical": True},
        {"task": "Conversion tracking configured", "critical": True},
        {"task": "Matched audiences created (retargeting, ABM)", "critical": False},
        {"task": "Company page connected", "critical": True},
        {"task": "Lead gen form templates created", "critical": False},
        {"task": "Audience size validated (not too narrow — min 50k)", "critical": False},
        {"task": "Budget realistic for LinkedIn CPCs ($8–$15+)", "critical": True},
    ],
    AdPlatform.TIKTOK: [
        {"task": "TikTok Pixel installed", "critical": True},
        {"task": "Events API configured", "critical": False},
        {"task": "Creative in vertical 9:16 format", "critical": True},
        {"task": "Captions added to all videos (85% watch muted)", "critical": True},
    ],
    AdPlatform.AMAZON: [
        {"task": "Seller Central account verified", "critical": True},
        {"task": "Brand Registry enrolled", "critical": False},
        {"task": "Product listings optimised (title, bullets, images)", "critical": True},
        {"task": "Sponsored Products campaign launched", "critical": True},
        {"task": "Negative keyword lists built from search term reports", "critical": True},
    ],
}


def get_setup_checklist(req: SetupChecklistRequest) -> SetupChecklistResponse:
    """Return a platform-specific pre-launch checklist."""

    items = _CHECKLISTS.get(req.platform, [
        {"task": "Conversion tracking installed", "critical": True},
        {"task": "Audience lists created", "critical": False},
        {"task": "Creative assets prepared", "critical": True},
    ])

    return SetupChecklistResponse(
        platform=req.platform,
        checklist=[ChecklistItem(**item) for item in items],
    )
