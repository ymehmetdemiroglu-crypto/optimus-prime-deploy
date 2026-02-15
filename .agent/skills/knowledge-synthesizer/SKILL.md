---
name: knowledge-synthesizer
description: Cross-domain learning and insight generation. Synthesizes insights from multiple data sources, detects market trends early, applies academic research, and discovers cross-product opportunities.
---

# Knowledge Synthesizer Skill

The **Knowledge Synthesizer** connects dots across different data sources, products, and domains to generate insights that wouldn't be visible from any single perspective.

## Core Capabilities

### 1. **Cross-Product Insights**
- Identify bundling opportunities across your catalog
- Detect cross-selling patterns
- Portfolio-level optimization
- Category performance correlation
- Complementary product discovery

### 2. **External Knowledge Integration**
- Monitor industry blogs and forums
- Track competitor announcements
- Academic research application
- Market trend reports
- Amazon policy changes

### 3. **Trend Detection**
- Early identification of emerging trends
- Search term momentum analysis
- Category growth/decline signals
- Seasonal pattern prediction
- Consumer behavior shifts

### 4. **Multi-Source Data Synthesis**
- Combine PPC data + reviews + search trends
- Social media sentiment + sales correlation
- Competitor pricing + your performance
- Supply chain + demand forecasting
- Economic indicators + category performance

### 5. **Insight Generation**
- Non-obvious pattern discovery
- Causal relationship identification
- Opportunity scoring
- Risk signal aggregation
- Actionable recommendation synthesis

## Cross-Product Analysis

### Bundling Opportunity Detection

```json
{
  "action": "find_bundling_opportunities",
  "product_catalog": ["ASIN_A", "ASIN_B", "ASIN_C", "ASIN_D"],
  "min_co_purchase_rate": 0.15
}
```

**Output**:
```json
{
  "bundling_opportunities": [
    {
      "products": ["ASIN_A", "ASIN_C"],
      "co_purchase_rate": 0.28,
      "estimated_bundle_demand": 450,
      "pricing_recommendation": {
        "individual_total": 49.98,
        "suggested_bundle": 44.99,
        "discount": "10%"
      },
      "confidence": 0.82,
      "insight": "Customers who buy ASIN_A frequently search for ASIN_C within 7 days"
    }
  ]
}
```

### Portfolio Optimization

```json
{
  "action": "optimize_portfolio",
  "products": ["ASIN_A", "ASIN_B", "ASIN_C"],
  "total_budget": 5000,
  "objective": "maximize_profit"
}
```

**Output**:
```json
{
  "recommended_allocation": {
    "ASIN_A": {"budget": 2500, "reason": "Highest ROAS, growing category"},
    "ASIN_B": {"budget": 1500, "reason": "Stable performer, defensive position"},
    "ASIN_C": {"budget": 1000, "reason": "Emerging product, test allocation"}
  },
  "expected_outcomes": {
    "total_sales": 22500,
    "total_profit": 5600,
    "portfolio_acos": 0.22
  },
  "insights": [
    "ASIN_A and ASIN_C share customer base - coordinate campaigns",
    "ASIN_B cannibalizes ASIN_A by 8% - consider differentiation"
  ]
}
```

## External Knowledge Integration

### Market Trend Monitoring

```json
{
  "action": "monitor_market_trends",
  "category": "electronics",
  "sources": ["amazon_search_trends", "google_trends", "industry_blogs"],
  "lookback_days": 30
}
```

**Output**:
```json
{
  "emerging_trends": [
    {
      "trend": "USB-C charging cables",
      "momentum": "+45% search volume",
      "stage": "early_growth",
      "opportunity_score": 0.78,
      "recommendation": "Consider expanding USB-C product line",
      "supporting_data": {
        "google_trends": "+52% (30d)",
        "amazon_search_rank": "Rising to #3 in category",
        "competitor_activity": "3 new launches this month"
      }
    }
  ],
  "declining_trends": [
    {
      "trend": "Micro-USB accessories",
      "momentum": "-22% search volume",
      "recommendation": "Reduce inventory, phase out campaigns"
    }
  ]
}
```

### GitHub Repository Search

```json
{
  "action": "search_github_repos",
  "topic": "amazon advertising optimization",
  "limit": 5
}
```

**Output**:
```json
{
  "repositories": [
    {
      "title": "amazon-ads-api-python",
      "url": "https://github.com/amzn/amazon-ads-api-python",
      "description": "Official Python SDK for Amazon Ads API",
      "source": "github"
    },
    {
      "title": "ads-optimizer",
      "url": "https://github.com/example/ads-optimizer",
      "description": "ML-based bid optimization tool",
      "source": "github"
    }
  ]
}
```

### Competitive Intelligence Synthesis

```json
{
  "action": "synthesize_competitive_intel",
  "your_asins": ["ASIN_A"],
  "competitor_asins": ["COMP_1", "COMP_2", "COMP_3"],
  "data_sources": ["pricing", "reviews", "sponsored_ads", "search_rank"]
}
```

**Output**:
```json
{
  "competitive_insights": [
    {
      "insight": "COMP_1 dropped price by 15% and increased ad spend by 40%",
      "impact_on_you": "Your conversion rate dropped 12% in same period",
      "recommendation": "Consider price adjustment or differentiation messaging",
      "urgency": "high"
    },
    {
      "insight": "COMP_2 has 3.2x more reviews mentioning 'fast shipping'",
      "opportunity": "Emphasize your FBA Prime advantage in ad copy",
      "estimated_impact": "+8% CTR"
    }
  ],
  "market_position": {
    "price_rank": 2,
    "review_rank": 4,
    "sponsored_visibility": 3,
    "organic_rank": 2
  }
}
```

## Trend Detection

### Search Term Momentum

```json
{
  "action": "detect_search_momentum",
  "keywords": ["wireless charger", "fast charging", "USB-C cable"],
  "timeframe": "last_90_days"
}
```

**Output**:
```json
{
  "momentum_analysis": [
    {
      "keyword": "wireless charger",
      "trend": "accelerating",
      "velocity": "+3.2% per week",
      "current_volume": "high",
      "forecast_30d": "+12%",
      "recommendation": "Increase bids by 15-20%",
      "confidence": 0.84
    },
    {
      "keyword": "fast charging",
      "trend": "plateauing",
      "velocity": "+0.5% per week",
      "recommendation": "Maintain current strategy",
      "confidence": 0.91
    }
  ]
}
```

## Multi-Source Synthesis

### Review Sentiment + Sales Correlation

```json
{
  "action": "correlate_sentiment_sales",
  "asin": "ASIN_A",
  "timeframe": "last_180_days"
}
```

**Output**:
```json
{
  "correlation_analysis": {
    "sentiment_sales_correlation": 0.67,
    "key_findings": [
      {
        "finding": "Negative reviews mentioning 'durability' spike 3 days before sales drops",
        "correlation": 0.72,
        "lag_days": 3,
        "recommendation": "Monitor 'durability' mentions as early warning signal"
      },
      {
        "finding": "Positive reviews mentioning 'value' correlate with +15% sales within 7 days",
        "recommendation": "Encourage reviews highlighting value proposition"
      }
    ],
    "sentiment_breakdown": {
      "positive": 0.78,
      "neutral": 0.15,
      "negative": 0.07
    },
    "actionable_insights": [
      "Address durability concerns in product description",
      "Highlight value in ad copy to match positive review themes"
    ]
  }
}
```

## Insight Generation

### Non-Obvious Pattern Discovery

```json
{
  "action": "discover_patterns",
  "data_sources": ["campaigns", "products", "market_data"],
  "min_confidence": 0.7
}
```

**Output**:
```json
{
  "discovered_patterns": [
    {
      "pattern": "Products with 'Prime' badge in title have 23% higher CTR",
      "confidence": 0.89,
      "sample_size": 1247,
      "recommendation": "Add 'Prime' to titles where applicable",
      "estimated_impact": "+$850/month"
    },
    {
      "pattern": "Campaigns paused on Sundays recover 18% slower than weekday pauses",
      "confidence": 0.76,
      "recommendation": "Avoid Sunday campaign changes",
      "insight": "Weekend shoppers have different behavior patterns"
    },
    {
      "pattern": "Products in 'Electronics > Accessories' perform 2.3x better with video ads",
      "confidence": 0.82,
      "recommendation": "Prioritize video creative for accessories category"
    }
  ]
}
```

## Usage Patterns

### Pattern 1: Portfolio Strategy

```
USER: "How should I allocate my $10K budget across my 5 products?"

KNOWLEDGE SYNTHESIZER:
1. Analyze cross-product relationships
2. Check market trends for each category
3. Review competitor activity
4. Synthesize insights:
   - Product A & C share 40% of customers → coordinate timing
   - Product B category declining -15% → reduce allocation
   - Product D in emerging trend → increase allocation
5. Recommend allocation with reasoning
```

### Pattern 2: Early Trend Detection

```
DAILY SCAN:
1. Monitor search trends across categories
2. Check competitor launches
3. Analyze review sentiment shifts
4. Detect: "Eco-friendly" mentions up 35% in your category
5. Alert: "Emerging trend detected: sustainability focus"
6. Recommend: "Consider eco-friendly messaging in ads"
```

### Pattern 3: Competitive Response

```
TRIGGER: Competitor price drop detected

KNOWLEDGE SYNTHESIZER:
1. Analyze: Competitor dropped price 20%
2. Correlate: Your sales down 15% in same period
3. Check: Competitor reviews mention "great value"
4. Synthesize: Price is key differentiator for this product
5. Recommend: "Match price or emphasize quality/features"
```

## Database Schema

```sql
-- From server/updates/05_tier2_meta_skills_tables.sql

synthesized_insights (
  insight_type,
  source_data,           -- Which data sources contributed
  insight_description,
  confidence,
  actionable_recommendations,
  created_at
)

external_knowledge (
  source,                -- 'blog', 'forum', 'research_paper', 'trend_report'
  content,
  relevance_score,
  fetched_at
)
```

## Integration with Other Skills

**Feeds from**:
- **market-researcher**: Product and competitor data
- **grok-admaster-operator**: Campaign performance
- **memory-palace**: Historical patterns
- **External APIs**: Google Trends, social media, news

**Feeds to**:
- **evolution-engine**: Insights for strategy evolution
- **simulation-lab**: Market scenarios to test
- **narrative-architect**: Insights for reporting

## Files

```
.agent/skills/knowledge-synthesizer/
├── SKILL.md
├── scripts/
│   ├── cross_product_analyzer.py     # Portfolio insights
│   ├── trend_detector.py              # Momentum analysis
│   └── external_knowledge_scraper.py  # Web scraping
└── resources/
    └── data_sources.json              # External source configs
```

## Example Invocation

```
USER: "Why are my sales down this week?"

KNOWLEDGE SYNTHESIZER:
1. Analyze your campaign data: Spend stable, CTR down 8%
2. Check competitor activity: 2 competitors launched new products
3. Review search trends: Category search volume down 5%
4. Sentiment analysis: Your reviews stable, no quality issues
5. External factors: Amazon Prime Day announced (customers waiting)
6. SYNTHESIS: "Sales down due to combination of:
   - New competition (40% impact)
   - Market-wide slowdown pre-Prime Day (35% impact)
   - Seasonal dip (25% impact)
   Recommendation: Maintain current spend, prepare Prime Day strategy, monitor new competitors closely."
```

## Notes

- Synthesizer runs daily scans automatically
- High-confidence insights trigger proactive alerts
- All insights include confidence scores
- External data sources are rate-limited
- Privacy-compliant data collection only

---

**This skill transforms data into wisdom, revealing opportunities hidden in plain sight.**
