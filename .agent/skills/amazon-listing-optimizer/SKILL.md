---
name: amazon-listing-optimizer
description: AI-powered Amazon listing optimization that improves both organic ranking and paid ad performance through title, bullet, backend keyword, and image optimization.
---

# Amazon Listing Optimizer Skill

The **Amazon Listing Optimizer** maximizes product visibility by optimizing every element of an Amazon listing. It impacts both organic search ranking AND paid advertising performance, creating a compounding effect on sales.

## Core Capabilities

### 1. **Title Optimization**
- Keyword-rich title generation
- Character limit compliance (200 chars)
- Brand + key features + benefits structure
- A/B testing recommendations
- Mobile-first optimization (first 80 chars critical)

### 2. **Bullet Point Enhancement**
- Benefit-focused copy
- Keyword integration (natural, not stuffed)
- Scannable formatting
- Emotional triggers + feature callouts
- 5 bullets × 500 chars optimization

### 3. **Backend Search Terms**
- 250-byte keyword extraction
- Competitor keyword gap analysis
- Misspelling inclusion strategy
- Synonym and related term coverage
- Prohibited term filtering

### 4. **Product Description / A+ Content**
- SEO-optimized descriptions
- A+ Content module recommendations
- Brand story integration
- Cross-sell opportunities
- Comparison charts

### 5. **Image Optimization Guidance**
- Main image requirements check
- Lifestyle image recommendations
- Infographic content suggestions
- Video placeholder identification
- Image SEO (alt text, file names)

## Optimization Workflow

### Full Listing Audit

```json
{
  "action": "audit_listing",
  "asin": "B0EXAMPLE123",
  "marketplace": "US"
}
```

**Response**:
```json
{
  "listing_audit": {
    "asin": "B0EXAMPLE123",
    "overall_score": 72,
    "grade": "B-",
    "potential_uplift": "+25-40% visibility",
    
    "title": {
      "score": 65,
      "current": "Wireless Bluetooth Headphones",
      "issues": [
        "Too short (28 chars vs 200 max)",
        "Missing key features (noise cancelling, battery life)",
        "No brand name"
      ],
      "recommendation": "ACME Wireless Bluetooth Headphones - Active Noise Cancelling, 40H Battery, Hi-Fi Sound, Comfortable Over-Ear Design for Travel, Work, Gaming - Black"
    },
    
    "bullets": {
      "score": 70,
      "issues": [
        "Bullet 3 is feature-only, no benefit",
        "Missing emotional triggers",
        "Keyword 'noise cancelling' not in bullets"
      ],
      "recommendations": [
        {
          "bullet": 1,
          "current": "40 hour battery life",
          "improved": "⚡ 40-HOUR MARATHON BATTERY - Travel cross-country or work all week without recharging. Quick charge gives 3 hours playback from just 10 minutes."
        }
      ]
    },
    
    "backend_keywords": {
      "score": 55,
      "bytes_used": 180,
      "bytes_available": 250,
      "missing_opportunities": [
        "headset", "earphones", "audiophile", "podcast", "zoom calls",
        "wireless headset", "bluetooth 5.0", "foldable headphones"
      ],
      "recommended_additions": "headset earphones audiophile podcast zoom calls wireless over ear foldable travel work gaming music"
    },
    
    "images": {
      "score": 80,
      "count": 6,
      "issues": [
        "No lifestyle image showing person wearing headphones",
        "No size comparison image",
        "Video slot empty"
      ],
      "recommendations": [
        "Add lifestyle image: person on airplane using headphones",
        "Add infographic: battery life comparison chart",
        "Add video: 30-second feature highlight"
      ]
    },
    
    "a_plus_content": {
      "has_a_plus": true,
      "score": 75,
      "recommendations": [
        "Add comparison chart module vs competitors",
        "Include brand story module",
        "Add FAQ section"
      ]
    }
  }
}
```

### Title Generator

```json
{
  "action": "generate_title",
  "asin": "B0EXAMPLE123",
  "product_type": "headphones",
  "brand": "ACME",
  "key_features": ["noise cancelling", "40h battery", "bluetooth 5.0"],
  "target_keywords": ["wireless headphones", "noise cancelling headphones"]
}
```

**Response**:
```json
{
  "title_options": [
    {
      "title": "ACME Wireless Noise Cancelling Headphones - 40H Battery, Bluetooth 5.0, Hi-Fi Stereo Sound, Comfortable Over-Ear Design for Travel & Work",
      "length": 142,
      "keyword_density": "optimal",
      "mobile_preview": "ACME Wireless Noise Cancelling Headphones - 40H Battery..."
    },
    {
      "title": "ACME Active Noise Cancelling Bluetooth Headphones, 40 Hour Playtime, Over Ear Wireless Headset with Hi-Fi Sound for Travel, Work, Gaming",
      "length": 147,
      "keyword_density": "optimal",
      "mobile_preview": "ACME Active Noise Cancelling Bluetooth Headphones, 40 Hour..."
    }
  ],
  "a_b_test_recommendation": "Test option 1 vs option 2 for 2 weeks, measure CTR and conversion rate"
}
```

### Backend Keyword Optimizer

```json
{
  "action": "optimize_backend_keywords",
  "asin": "B0EXAMPLE123",
  "current_keywords": "wireless bluetooth headphones noise",
  "competitors": ["B0COMP001", "B0COMP002"]
}
```

**Response**:
```json
{
  "backend_optimization": {
    "current_bytes": 45,
    "max_bytes": 250,
    "utilization": "18%",
    
    "competitor_keywords_missing": [
      "headset", "earbuds", "over ear", "on ear", "studio",
      "gaming", "podcast", "audiophile", "commute", "office"
    ],
    
    "high_volume_missing": [
      {"keyword": "wireless headset", "search_volume": 135000},
      {"keyword": "bluetooth headset", "search_volume": 98000},
      {"keyword": "noise cancelling earbuds", "search_volume": 74000}
    ],
    
    "recommended_backend": "headset earbuds over ear studio gaming podcast audiophile commute office travel work music calls zoom meeting comfortable foldable portable lightweight long battery hi-fi stereo bass",
    "recommended_bytes": 248,
    
    "excluded_terms": [
      {"term": "best", "reason": "Subjective claim - prohibited"},
      {"term": "Sony", "reason": "Competitor brand - prohibited"},
      {"term": "cheap", "reason": "Devalues product perception"}
    ]
  }
}
```

## Impact Metrics

| Element | Impact on Organic | Impact on PPC |
|---------|-------------------|---------------|
| Title Optimization | +20-40% impressions | +15-25% CTR |
| Bullet Enhancement | +10-20% conversion | +10-15% conversion |
| Backend Keywords | +30-50% keyword reach | +20-30% relevance score |
| Image Optimization | +15-25% conversion | +10-20% CTR |
| A+ Content | +5-15% conversion | Indirect (brand trust) |

## Integration Points

**Consumes from**:
- **market-researcher**: Competitor listings, keyword volumes
- **knowledge-synthesizer**: Trend data, seasonal keywords
- **grok-admaster-operator**: PPC keyword performance data

**Feeds to**:
- Amazon Seller Central (via API)
- **campaign-strategist**: New keyword opportunities
- **evolution-engine**: A/B test results for optimization

## Compliance Checks

Built-in Amazon policy compliance:
- ✅ No promotional language ("sale", "free shipping")
- ✅ No time-sensitive claims ("new", "limited time")
- ✅ No competitor brand names
- ✅ No subjective claims ("best", "#1")
- ✅ Character/byte limits enforced
- ✅ HTML tag restrictions

## Files

```
.agent/skills/amazon-listing-optimizer/
├── SKILL.md
├── scripts/
│   ├── title_generator.py
│   ├── bullet_optimizer.py
│   ├── backend_keyword_extractor.py
│   └── listing_scorer.py
└── resources/
    ├── prohibited_terms.json
    └── category_templates.json
```

---

**This skill bridges the gap between organic and paid - optimizing listings to maximize BOTH channels.**
