---
name: financial-analyst
description: Deep financial analysis for Amazon advertising with unit economics, profitability modeling, budget optimization, and ROI attribution - rare capabilities that differentiate from standard PPC tools.
---

# Financial Analyst Skill

The **Financial Analyst** brings Wall Street-level financial rigor to Amazon advertising. It goes beyond simple ROAS to calculate true profitability, unit economics, customer lifetime value, and optimal budget allocation.

## Core Capabilities

### 1. **True Profitability Analysis**
- Revenue minus ALL costs (not just ad spend)
- Product cost (COGS) integration
- Amazon fees calculation (FBA, referral, storage)
- Net profit per unit and per order
- Contribution margin analysis

### 2. **Unit Economics Modeling**
- Customer Acquisition Cost (CAC)
- Lifetime Value (LTV) estimation
- LTV:CAC ratio optimization
- Break-even analysis
- Payback period calculation

### 3. **Budget Optimization**
- Optimal budget allocation across campaigns
- Marginal ROAS analysis
- Diminishing returns detection
- Budget reallocation recommendations
- Scenario modeling for budget changes

### 4. **ROI Attribution**
- Multi-touch attribution
- New vs returning customer value
- Brand halo effect measurement
- Organic lift from paid ads
- Cross-product attribution

### 5. **Financial Forecasting**
- Revenue projections
- Cash flow impact
- Seasonal budget planning
- What-if scenario analysis

## Financial Analysis Reports

### Product Profitability Deep Dive

```json
{
  "action": "analyze_product_profitability",
  "asin": "B0YOUR001",
  "period": "last_30_days"
}
```

**Response**:
```json
{
  "profitability_analysis": {
    "asin": "B0YOUR001",
    "product_name": "Premium Wireless Headphones",
    "period": "Jan 6 - Feb 5, 2026",
    
    "revenue_breakdown": {
      "gross_revenue": 45000,
      "units_sold": 500,
      "average_selling_price": 90.00,
      "organic_revenue": 27000,
      "paid_revenue": 18000
    },
    
    "cost_structure": {
      "product_cost_per_unit": 22.00,
      "total_cogs": 11000,
      "amazon_referral_fee": 6750,
      "fba_fulfillment_fee": 3500,
      "fba_storage_fee": 180,
      "advertising_cost": 3600,
      "total_costs": 25030
    },
    
    "profitability_metrics": {
      "gross_profit": 19970,
      "gross_margin_pct": 44.4,
      "net_profit_per_unit": 39.94,
      "contribution_margin": 68.00,
      "contribution_margin_pct": 75.6,
      "tacos": 8.0,
      "true_acos": 20.0,
      "roas": 5.0
    },
    
    "advertising_efficiency": {
      "ad_spend": 3600,
      "ad_attributed_revenue": 18000,
      "ad_attributed_profit": 8994,
      "profit_per_ad_dollar": 2.50,
      "break_even_acos": 68.0,
      "current_vs_break_even": "52% headroom"
    },
    
    "insights": [
      "Product is highly profitable with 52% buffer to break-even ACoS",
      "Recommend increasing ad budget - marginal profit available",
      "Storage fees are minimal - inventory levels healthy"
    ]
  }
}
```

### Unit Economics Dashboard

```json
{
  "action": "calculate_unit_economics",
  "account_level": true
}
```

**Response**:
```json
{
  "unit_economics": {
    "period": "Last 90 days",
    
    "customer_metrics": {
      "total_customers": 4500,
      "new_customers": 3200,
      "returning_customers": 1300,
      "repeat_rate": 28.9,
      "average_orders_per_customer": 1.42
    },
    
    "acquisition": {
      "total_ad_spend": 48000,
      "new_customers_from_ads": 2400,
      "cac_blended": 10.67,
      "cac_new_only": 20.00,
      "cac_trend": "improving (-8% MoM)"
    },
    
    "lifetime_value": {
      "average_order_value": 75.00,
      "orders_per_customer_12mo": 2.1,
      "gross_margin_pct": 45,
      "estimated_ltv": 70.88,
      "ltv_calculation": "75 × 2.1 × 0.45 = $70.88"
    },
    
    "ltv_cac_analysis": {
      "ltv_cac_ratio": 3.54,
      "benchmark": "3.0+ is excellent",
      "status": "healthy",
      "payback_period_days": 45,
      "recommendation": "Ratio supports increased acquisition spend"
    },
    
    "customer_segmentation": {
      "high_value": {
        "count": 450,
        "avg_ltv": 180.00,
        "acquisition_source": "Brand campaigns (65%)"
      },
      "medium_value": {
        "count": 2700,
        "avg_ltv": 65.00,
        "acquisition_source": "Product campaigns (55%)"
      },
      "low_value": {
        "count": 1350,
        "avg_ltv": 25.00,
        "acquisition_source": "Generic campaigns (70%)"
      }
    }
  }
}
```

### Budget Optimization Analysis

```json
{
  "action": "optimize_budget_allocation",
  "total_budget": 50000,
  "objective": "maximize_profit"
}
```

**Response**:
```json
{
  "budget_optimization": {
    "total_budget": 50000,
    "objective": "maximize_profit",
    
    "current_allocation": {
      "brand_campaigns": {"budget": 10000, "roas": 8.5, "marginal_roas": 6.2},
      "product_campaigns": {"budget": 25000, "roas": 4.2, "marginal_roas": 3.1},
      "generic_campaigns": {"budget": 15000, "roas": 2.8, "marginal_roas": 1.5}
    },
    
    "recommended_allocation": {
      "brand_campaigns": {"budget": 15000, "change": "+50%", "reason": "High marginal ROAS, under-invested"},
      "product_campaigns": {"budget": 28000, "change": "+12%", "reason": "Solid returns, moderate increase"},
      "generic_campaigns": {"budget": 7000, "change": "-53%", "reason": "Below profit threshold, reallocate"}
    },
    
    "projected_impact": {
      "current_revenue": 185000,
      "projected_revenue": 215000,
      "revenue_change": "+16%",
      "current_profit": 42000,
      "projected_profit": 58000,
      "profit_change": "+38%"
    },
    
    "marginal_analysis": {
      "last_dollar_efficiency": [
        {"campaign": "Brand", "next_1000_roas": 5.8, "recommendation": "invest"},
        {"campaign": "Product", "next_1000_roas": 2.9, "recommendation": "invest"},
        {"campaign": "Generic", "next_1000_roas": 1.2, "recommendation": "cut"}
      ]
    },
    
    "scenario_comparison": [
      {"scenario": "Aggressive Growth", "budget": 70000, "proj_profit": 72000, "risk": "medium"},
      {"scenario": "Efficient Growth", "budget": 50000, "proj_profit": 58000, "risk": "low"},
      {"scenario": "Profit Maximization", "budget": 35000, "proj_profit": 48000, "risk": "very_low"}
    ]
  }
}
```

## Advanced Financial Features

### Break-Even Calculator

```json
{
  "break_even": {
    "product_cost": 22.00,
    "selling_price": 89.99,
    "amazon_fees_pct": 35,
    "break_even_acos": 65,
    "formula": "(Price - COGS - Fees) / Price × 100",
    "interpretation": "Any ACoS below 65% is profitable"
  }
}
```

### TACoS vs ACoS Analysis

```json
{
  "tacos_acos_comparison": {
    "acos": 22.5,
    "tacos": 8.2,
    "organic_ratio": 63.5,
    "interpretation": "Strong organic presence - ads driving halo effect",
    "benchmark": "TACoS < 10% is excellent for established products",
    "trend": "TACoS improving as organic grows"
  }
}
```

## Integration Points

**Consumes from**:
- Amazon Seller Central API (orders, fees)
- Inventory management systems (COGS)
- **grok-admaster-operator**: Campaign spend data
- **simulation-lab**: Forecast scenarios

**Feeds to**:
- **executive-reporter**: Financial sections
- **campaign-strategist**: Budget constraints
- **evolution-engine**: Profit-based fitness functions

## Files

```
.agent/skills/financial-analyst/
├── SKILL.md
├── scripts/
│   ├── profitability_calculator.py
│   ├── unit_economics_engine.py
│   ├── budget_optimizer.py
│   └── ltv_modeler.py
└── resources/
    └── amazon_fee_structure.json
```

---

**This skill transforms advertising from a cost center to a strategic profit lever with full financial transparency.**
