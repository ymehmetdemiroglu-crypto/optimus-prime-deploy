---
name: campaign-strategist
description: AI-driven strategic campaign planning that designs campaign architectures, segment strategies, launch plans, and long-term advertising roadmaps based on business objectives.
---

# Campaign Strategist Skill

The **Campaign Strategist** is your AI advertising director. It doesn't just optimize existing campaigns - it designs comprehensive campaign architectures and strategic plans aligned with business objectives.

## Core Capabilities

### 1. **Campaign Architecture Design**
- Full account structure planning
- Campaign type recommendations
- Segmentation strategy
- Keyword grouping logic
- Budget hierarchy design

### 2. **Product Launch Planning**
- Pre-launch checklist
- Launch phase strategy (0-30-60-90 days)
- Budget ramp plans
- Success metrics definition
- Pivot triggers

### 3. **Seasonal Strategy**
- Q4 / Prime Day preparation
- Pre-event budget planning
- Peak period bid strategies
- Post-season optimization
- Year-round calendar

### 4. **Portfolio Strategy**
- Multi-product coordination
- Cross-sell / upsell campaigns
- Brand portfolio balance
- Cannibalization prevention
- Portfolio-level budgeting

### 5. **Goal-Based Planning**
- Objective to tactics translation
- KPI target setting
- Milestone planning
- Strategy adjustment triggers

## Strategic Planning

### New Product Launch Strategy

```json
{
  "action": "plan_product_launch",
  "asin": "B0NEWPROD001",
  "product_type": "Wireless Earbuds",
  "price_point": 49.99,
  "launch_date": "2026-03-01",
  "budget_available": 10000
}
```

**Response**:
```json
{
  "launch_strategy": {
    "product": "B0NEWPROD001",
    "launch_date": "2026-03-01",
    "strategy_name": "Aggressive Market Entry",
    
    "pre_launch": {
      "duration": "14 days before",
      "checklist": [
        {"task": "Listing optimization complete", "owner": "amazon-listing-optimizer"},
        {"task": "Main image A/B test running", "status": "pending"},
        {"task": "Backend keywords optimized", "owner": "amazon-listing-optimizer"},
        {"task": "Initial review generation plan", "status": "pending"},
        {"task": "Competitor price positioning confirmed", "owner": "competitive-intelligence"}
      ]
    },
    
    "phase_1_launch": {
      "duration": "Days 1-14",
      "objective": "Velocity & Reviews",
      "budget": 3500,
      "strategy": {
        "campaign_types": ["SP Auto", "SP Manual Exact", "SB Video"],
        "bid_strategy": "Aggressive (+40% above suggested)",
        "targeting": "High-intent exact match keywords only",
        "dayparting": "All hours initially for data collection"
      },
      "success_metrics": {
        "daily_sales_target": 20,
        "review_target": 10,
        "acos_tolerance": 50
      },
      "tactics": [
        "Run Auto campaign for keyword discovery",
        "Launch Brand Video ad for visibility",
        "Target competitor ASINs"
      ]
    },
    
    "phase_2_optimize": {
      "duration": "Days 15-30",
      "objective": "Efficiency & Ranking",
      "budget": 3500,
      "strategy": {
        "campaign_evolution": "Graduate top keywords from Auto to Manual",
        "bid_strategy": "Data-driven adjustment",
        "new_campaigns": ["SP Product Targeting", "SB Headline"],
        "negative_keywords": "Apply learnings from Phase 1"
      },
      "success_metrics": {
        "organic_rank_improvement": "Top 50 for 3+ keywords",
        "acos_target": 35,
        "daily_sales_target": 30
      }
    },
    
    "phase_3_scale": {
      "duration": "Days 31-60",
      "objective": "Profitable Growth",
      "budget": 3000,
      "strategy": {
        "campaign_expansion": "Add broad match exploratory campaigns",
        "bid_strategy": "ROAS-optimized bidding",
        "budget_shift": "Move budget to proven performers"
      },
      "success_metrics": {
        "organic_rank": "Page 1 for main keywords",
        "acos_target": 25,
        "break_even_status": "Profitable"
      }
    },
    
    "pivot_triggers": [
      {
        "condition": "ACoS > 60% after Day 14",
        "action": "Pause low performers, review listing competitiveness"
      },
      {
        "condition": "Sales < 50% of target after Day 7",
        "action": "Check listing, increase bids 20%, review pricing"
      },
      {
        "condition": "CTR < 0.3%",
        "action": "Urgent main image review - likely visual issue"
      }
    ],
    
    "campaign_structure": {
      "campaigns": [
        {"name": "[Launch] Earbuds - Auto Discovery", "type": "SP Auto", "budget": 30},
        {"name": "[Launch] Earbuds - Exact High Intent", "type": "SP Manual", "budget": 50},
        {"name": "[Launch] Earbuds - Competitor ASIN", "type": "SP Product", "budget": 20},
        {"name": "[Brand] Earbuds - Video Ad", "type": "SB Video", "budget": 25}
      ]
    }
  }
}
```

### Campaign Architecture Audit

```json
{
  "action": "audit_campaign_architecture",
  "account_id": "ACC_123"
}
```

**Response**:
```json
{
  "architecture_audit": {
    "account_id": "ACC_123",
    
    "current_state": {
      "total_campaigns": 45,
      "campaign_types": {
        "SP_Auto": 15,
        "SP_Manual": 22,
        "SB": 5,
        "SD": 3
      },
      "structure_score": 62,
      "grade": "C+"
    },
    
    "issues_identified": [
      {
        "issue": "Keyword Cannibalization",
        "severity": "high",
        "details": "Same keyword targets in 3+ campaigns competing against each other",
        "affected_keywords": ["wireless earbuds", "bluetooth earbuds"],
        "impact": "Wasted spend, inflated CPCs",
        "recommendation": "Consolidate to single campaign per keyword"
      },
      {
        "issue": "Missing Campaign Types",
        "severity": "medium",
        "details": "No Sponsored Brands Video campaigns",
        "impact": "Missing high-engagement placement opportunity",
        "recommendation": "Add 2-3 SB Video campaigns for top products"
      },
      {
        "issue": "Poor Naming Convention",
        "severity": "low",
        "details": "Inconsistent naming makes analysis difficult",
        "example": "Campaign names: 'test123', 'new campaign', 'asdf'",
        "recommendation": "Implement naming: [Type] [Product] [Strategy] [Match]"
      }
    ],
    
    "recommended_structure": {
      "naming_convention": "[Type]_[Product]_[Strategy]_[Match/Target]",
      "examples": [
        "SP_Earbuds_Brand_Exact",
        "SP_Earbuds_Competitor_ASIN",
        "SB_Earbuds_Video_Category"
      ],
      "ideal_campaign_count": 35,
      "consolidation_savings": "$450/month estimated"
    }
  }
}
```

### Prime Day Strategy

```json
{
  "action": "plan_prime_day",
  "prime_day_dates": ["2026-07-15", "2026-07-16"],
  "budget_available": 25000
}
```

**Response**:
```json
{
  "prime_day_strategy": {
    "event": "Prime Day 2026",
    "dates": "July 15-16, 2026",
    
    "pre_event": {
      "period": "July 1-14",
      "objectives": ["Build momentum", "Improve organic rank", "Stockpile reviews"],
      "budget_allocation": 5000,
      "tactics": [
        "Increase bids 15% on top performers",
        "Launch deal-specific campaigns",
        "Ramp up brand awareness ads"
      ]
    },
    
    "event_days": {
      "period": "July 15-16",
      "budget_allocation": 15000,
      "hourly_strategy": {
        "00:00-06:00": "Standard bids (low competition)",
        "06:00-12:00": "Aggressive bids (+50%)",
        "12:00-18:00": "Maximum bids (+80%)",
        "18:00-24:00": "Aggressive bids (+50%)"
      },
      "deal_support": [
        "Create specific campaigns for each deal ASIN",
        "Activate Sponsored Display retargeting",
        "Maximum budget on brand defense"
      ]
    },
    
    "post_event": {
      "period": "July 17-24",
      "budget_allocation": 5000,
      "objectives": ["Capture deal-miss shoppers", "Maintain rank gains"],
      "tactics": [
        "Retarget cart abandoners",
        "Maintain elevated bids for 3 days",
        "Gradual return to normal by Day 7"
      ]
    }
  }
}
```

## Integration Points

**Consumes from**:
- **competitive-intelligence**: Market landscape
- **financial-analyst**: Budget constraints and profitability
- **memory-palace**: Historical campaign performance
- **simulation-lab**: Scenario projections

**Feeds to**:
- **grok-admaster-operator**: Campaign creation commands
- **executive-reporter**: Strategy summaries
- **orchestrator-maestro**: Multi-step execution plans

## Files

```
.agent/skills/campaign-strategist/
├── SKILL.md
├── scripts/
│   ├── architecture_designer.py
│   ├── launch_planner.py
│   ├── seasonal_strategist.py
│   └── portfolio_optimizer.py
└── templates/
    ├── launch_templates.json
    └── prime_day_playbook.json
```

---

**This skill elevates Optimus Pryme from a campaign optimizer to a strategic advertising partner.**
