---
name: competitive-intelligence
description: Real-time competitive monitoring and strategic intelligence gathering for Amazon sellers, tracking competitor pricing, advertising activity, market share, and strategic moves.
---

# Competitive Intelligence Skill

The **Competitive Intelligence** skill provides deep visibility into competitor activities, enabling data-driven strategic decisions. It monitors pricing, advertising, product launches, and market positioning in real-time.

## Core Capabilities

### 1. **Competitor Price Monitoring**
- Real-time price tracking
- Historical price trend analysis
- Price change alerts
- Buy Box monitoring
- Promotion/deal detection

### 2. **Advertising Intelligence**
- Competitor ad spend estimation
- Keyword overlap analysis
- Sponsored Products tracking
- Sponsored Brands monitoring
- Share of Voice calculation

### 3. **Market Share Analysis**
- Category market share estimation
- Trend direction (growing/declining)
- Revenue estimation per competitor
- Rank tracking over time

### 4. **Product Launch Detection**
- New ASIN alerts
- Variation tracking
- Bundle detection
- Category expansion monitoring

### 5. **Review & Rating Intelligence**
- Review velocity tracking
- Sentiment analysis
- Common complaints identification
- Rating trend analysis

## Intelligence Reports

### Competitor Snapshot

```json
{
  "action": "competitor_snapshot",
  "competitor_brand": "RivalBrand",
  "category": "Electronics > Headphones"
}
```

**Response**:
```json
{
  "competitor_profile": {
    "brand": "RivalBrand",
    "category": "Electronics > Headphones",
    "analysis_date": "2026-02-05",
    
    "market_position": {
      "estimated_market_share": 18.5,
      "share_trend": "growing",
      "share_change_30d": "+1.2pp",
      "category_rank": 2
    },
    
    "product_portfolio": {
      "total_asins": 24,
      "new_last_90d": 3,
      "discontinued_last_90d": 1,
      "top_seller": {
        "asin": "B0RIVAL001",
        "title": "RivalBrand Pro Headphones",
        "bsr": 45,
        "estimated_monthly_revenue": 285000
      }
    },
    
    "pricing_strategy": {
      "average_price": 79.99,
      "price_range": {"min": 29.99, "max": 199.99},
      "pricing_position": "mid-market",
      "promotion_frequency": "weekly",
      "current_deals": [
        {"asin": "B0RIVAL002", "discount": "20%", "ends": "2026-02-07"}
      ]
    },
    
    "advertising_activity": {
      "estimated_monthly_spend": 45000,
      "spend_trend": "increasing",
      "primary_ad_types": ["Sponsored Products", "Sponsored Brands"],
      "share_of_voice": {
        "organic": 15.2,
        "paid": 22.8,
        "combined": 19.0
      },
      "top_targeted_keywords": [
        "wireless headphones",
        "noise cancelling headphones",
        "bluetooth headphones"
      ]
    },
    
    "customer_perception": {
      "average_rating": 4.3,
      "total_reviews": 12500,
      "review_velocity": 850,
      "sentiment_breakdown": {
        "positive": 78,
        "neutral": 12,
        "negative": 10
      },
      "common_complaints": [
        "Battery life shorter than advertised",
        "Uncomfortable after 2 hours",
        "Bluetooth connectivity issues"
      ],
      "common_praise": [
        "Great sound quality",
        "Good value for money",
        "Stylish design"
      ]
    }
  }
}
```

### Share of Voice Report

```json
{
  "action": "share_of_voice",
  "keywords": ["wireless headphones", "bluetooth headphones", "noise cancelling"],
  "competitors": ["RivalBrand", "CompetitorX", "BrandY"]
}
```

**Response**:
```json
{
  "share_of_voice": {
    "analysis_period": "Last 7 days",
    "keywords_analyzed": 3,
    
    "overall_sov": {
      "your_brand": 12.5,
      "RivalBrand": 22.8,
      "CompetitorX": 18.2,
      "BrandY": 15.1,
      "others": 31.4
    },
    
    "by_keyword": [
      {
        "keyword": "wireless headphones",
        "search_volume": 450000,
        "your_position": {"organic": 8, "paid": 3},
        "sov": {
          "your_brand": {"organic": 8.2, "paid": 15.5},
          "RivalBrand": {"organic": 12.1, "paid": 25.0}
        }
      }
    ],
    
    "opportunities": [
      {
        "keyword": "noise cancelling headphones",
        "gap": "CompetitorX dominates paid (30% SOV), you have 5%",
        "recommendation": "Increase bids on this high-value keyword",
        "estimated_cost_to_compete": 2500
      }
    ],
    
    "threats": [
      {
        "keyword": "wireless headphones",
        "threat": "RivalBrand increased paid SOV by 8pp this week",
        "your_impact": "Your CTR dropped 12%",
        "recommendation": "Defensive bid increase or differentiate messaging"
      }
    ]
  }
}
```

### Price War Detection

```json
{
  "action": "detect_price_changes",
  "category": "Electronics > Headphones",
  "threshold_pct": 10
}
```

**Response**:
```json
{
  "price_alerts": {
    "period": "Last 24 hours",
    "significant_changes": [
      {
        "competitor": "RivalBrand",
        "asin": "B0RIVAL001",
        "product": "RivalBrand Pro Headphones",
        "old_price": 99.99,
        "new_price": 79.99,
        "change_pct": -20,
        "change_type": "price_drop",
        "likely_reason": "Clearing inventory or price war initiation",
        "your_comparable_asin": "B0YOUR001",
        "your_current_price": 89.99,
        "recommendation": "Monitor for 48h. If sustained, consider matching to $84.99"
      }
    ],
    
    "market_trend": {
      "direction": "deflationary",
      "avg_price_change_7d": -3.2,
      "competitors_with_drops": 4,
      "competitors_with_increases": 1
    },
    
    "strategic_options": [
      {"option": "Match price", "impact": "Maintain share, reduce margin 11%"},
      {"option": "Hold price", "impact": "Risk 8-12% volume loss"},
      {"option": "Value differentiation", "impact": "Emphasize unique features in ads"}
    ]
  }
}
```

## Alert System

### Configurable Alerts

```json
{
  "alerts": [
    {
      "type": "price_drop",
      "competitor": "RivalBrand",
      "threshold": 15,
      "channel": "slack",
      "urgency": "high"
    },
    {
      "type": "new_product",
      "category": "Electronics > Headphones",
      "channel": "email",
      "urgency": "medium"
    },
    {
      "type": "sov_loss",
      "keyword": "wireless headphones",
      "threshold": 5,
      "channel": "sms",
      "urgency": "high"
    }
  ]
}
```

## Integration Points

**Consumes from**:
- Amazon Product Advertising API
- Keepa / CamelCamelCamel (price history)
- Web scraping services
- **market-researcher**: Deep research queries

**Feeds to**:
- **campaign-strategist**: Competitive positioning
- **executive-reporter**: Competitive landscape section
- **simulation-lab**: Competitive scenarios
- **evolution-engine**: Competitive fitness factors

## Files

```
.agent/skills/competitive-intelligence/
├── SKILL.md
├── scripts/
│   ├── price_tracker.py
│   ├── sov_calculator.py
│   ├── review_analyzer.py
│   └── alert_dispatcher.py
└── resources/
    └── competitor_registry.json
```

---

**Know your enemy, know yourself - this skill ensures you're never surprised by competitor moves.**
