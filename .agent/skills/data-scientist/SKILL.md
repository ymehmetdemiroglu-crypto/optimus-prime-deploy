---
name: data-scientist
description: Personalized machine learning models per seller account, including custom bid prediction, conversion forecasting, anomaly detection, and automated model retraining pipelines.
---

# Data Scientist Skill

The **Data Scientist** provides enterprise-grade machine learning capabilities personalized for each seller's unique data. Unlike generic tools, it trains custom models on YOUR data for YOUR products.

## Core Capabilities

### 1. **Custom Bid Prediction Models**
- Per-account trained models
- Keyword-level bid optimization
- Time-series aware predictions
- Confidence intervals included
- Continuous model improvement

### 2. **Conversion Rate Forecasting**
- Product-specific conversion models
- Feature engineering (price, reviews, rank)
- External factor integration (seasonality, competition)
- Real-time prediction updates

### 3. **Anomaly Detection**
- Multi-dimensional anomaly scoring
- Automatic alert generation
- Root cause hypothesis
- Historical anomaly patterns

### 4. **Customer Segmentation**
- RFM (Recency, Frequency, Monetary) analysis
- Behavioral clustering
- Value-based segments
- Segment-specific strategies

### 5. **Automated ML Pipeline**
- Data ingestion automation
- Feature engineering
- Model training scheduling
- Performance monitoring
- Automatic retraining triggers

## ML Model Operations

### Train Custom Bid Model

```json
{
  "action": "train_bid_model",
  "account_id": "ACC_123",
  "training_data_days": 90,
  "model_type": "gradient_boosting"
}
```

**Response**:
```json
{
  "model_training": {
    "model_id": "bid_model_acc123_v12",
    "account_id": "ACC_123",
    "training_completed": "2026-02-05T11:30:00Z",
    
    "training_summary": {
      "training_samples": 45000,
      "features_used": 24,
      "training_duration_seconds": 180,
      "cross_validation_folds": 5
    },
    
    "features": {
      "keyword_features": [
        "keyword_length", "match_type", "historical_ctr", 
        "historical_cvr", "competition_score", "search_volume"
      ],
      "temporal_features": [
        "hour_of_day", "day_of_week", "is_weekend",
        "days_since_last_sale", "season_index"
      ],
      "product_features": [
        "price", "review_count", "rating", "bsr_rank",
        "category_competitiveness"
      ],
      "competitive_features": [
        "competitor_bid_estimate", "share_of_voice", "price_position"
      ]
    },
    
    "performance_metrics": {
      "mae": 0.12,
      "rmse": 0.18,
      "r_squared": 0.84,
      "mape": 8.5,
      "interpretation": "Model predicts optimal bids within 8.5% of actual optimal on average"
    },
    
    "feature_importance": [
      {"feature": "historical_cvr", "importance": 0.22},
      {"feature": "competition_score", "importance": 0.18},
      {"feature": "search_volume", "importance": 0.15},
      {"feature": "hour_of_day", "importance": 0.12},
      {"feature": "keyword_length", "importance": 0.08}
    ],
    
    "deployment": {
      "status": "deployed",
      "endpoint": "/api/v1/ml/predict_bid",
      "next_retrain": "2026-02-12",
      "performance_monitoring": "active"
    }
  }
}
```

### Get Bid Predictions

```json
{
  "action": "predict_optimal_bids",
  "keywords": [
    {"keyword_id": "KW001", "keyword": "wireless earbuds", "match_type": "exact"},
    {"keyword_id": "KW002", "keyword": "bluetooth headphones", "match_type": "phrase"}
  ],
  "target_acos": 25
}
```

**Response**:
```json
{
  "bid_predictions": [
    {
      "keyword_id": "KW001",
      "keyword": "wireless earbuds",
      "current_bid": 1.25,
      "predicted_optimal_bid": 1.42,
      "confidence_interval": {"low": 1.35, "high": 1.50},
      "expected_acos_at_bid": 23.5,
      "expected_clicks": 145,
      "expected_conversions": 12,
      "recommendation": "increase",
      "reasoning": "High CVR keyword underinvested - model sees +15% conversion opportunity"
    },
    {
      "keyword_id": "KW002",
      "keyword": "bluetooth headphones",
      "current_bid": 1.80,
      "predicted_optimal_bid": 1.45,
      "confidence_interval": {"low": 1.38, "high": 1.52},
      "expected_acos_at_bid": 24.8,
      "expected_clicks": 210,
      "expected_conversions": 18,
      "recommendation": "decrease",
      "reasoning": "Diminishing returns at current bid level - can maintain performance at lower cost"
    }
  ],
  
  "model_info": {
    "model_id": "bid_model_acc123_v12",
    "last_trained": "2026-02-05",
    "prediction_confidence": "high"
  }
}
```

### Multi-Dimensional Anomaly Detection

```json
{
  "action": "detect_anomalies",
  "scope": "account",
  "lookback_days": 7
}
```

**Response**:
```json
{
  "anomaly_detection": {
    "scan_period": "Jan 29 - Feb 5, 2026",
    "total_data_points_analyzed": 125000,
    
    "anomalies_detected": [
      {
        "anomaly_id": "ANM001",
        "severity": "high",
        "type": "performance_drop",
        "dimension": "CTR",
        "affected_entity": "Campaign: Summer Collection",
        "detected_value": 0.18,
        "expected_range": {"low": 0.35, "high": 0.45},
        "deviation_sigma": 4.2,
        "detection_time": "2026-02-04T14:30:00Z",
        "duration": "18 hours",
        "estimated_impact": "-$340 in lost sales",
        "probable_causes": [
          {"cause": "Main image changed", "confidence": 0.75},
          {"cause": "Competitor launched competing product", "confidence": 0.60}
        ],
        "recommended_actions": [
          "Review recent listing changes",
          "Check competitor activity",
          "A/B test current vs previous image"
        ]
      },
      {
        "anomaly_id": "ANM002",
        "severity": "medium",
        "type": "cost_spike",
        "dimension": "CPC",
        "affected_entity": "Keyword: wireless earbuds",
        "detected_value": 2.85,
        "expected_range": {"low": 1.40, "high": 1.80},
        "deviation_sigma": 3.8,
        "probable_causes": [
          {"cause": "Competitor bid increase", "confidence": 0.80},
          {"cause": "Seasonal demand spike", "confidence": 0.45}
        ]
      }
    ],
    
    "summary": {
      "high_severity": 1,
      "medium_severity": 1,
      "low_severity": 3,
      "auto_resolved": 2
    }
  }
}
```

### Customer Segmentation Analysis

```json
{
  "action": "segment_customers",
  "account_id": "ACC_123",
  "method": "rfm"
}
```

**Response**:
```json
{
  "customer_segmentation": {
    "method": "RFM Analysis",
    "total_customers": 8500,
    "analysis_period": "Last 12 months",
    
    "segments": [
      {
        "segment_name": "Champions",
        "count": 850,
        "percentage": 10,
        "rfm_score": "5-5-5",
        "characteristics": {
          "avg_recency_days": 5,
          "avg_frequency": 8.2,
          "avg_monetary": 450
        },
        "recommended_strategy": "VIP treatment, early access to new products, loyalty rewards",
        "ad_targeting": "Exclude from acquisition ads, retarget for upsells"
      },
      {
        "segment_name": "Loyal Customers",
        "count": 1275,
        "percentage": 15,
        "rfm_score": "4-4-4",
        "characteristics": {
          "avg_recency_days": 18,
          "avg_frequency": 5.1,
          "avg_monetary": 280
        },
        "recommended_strategy": "Upsell premium products, request reviews",
        "ad_targeting": "Cross-sell campaigns, new product announcements"
      },
      {
        "segment_name": "At Risk",
        "count": 1020,
        "percentage": 12,
        "rfm_score": "2-3-3",
        "characteristics": {
          "avg_recency_days": 75,
          "avg_frequency": 3.2,
          "avg_monetary": 180
        },
        "recommended_strategy": "Win-back campaigns, special offers",
        "ad_targeting": "Heavy retargeting, discount messaging"
      },
      {
        "segment_name": "New Customers",
        "count": 2550,
        "percentage": 30,
        "rfm_score": "5-1-2",
        "characteristics": {
          "avg_recency_days": 10,
          "avg_frequency": 1.1,
          "avg_monetary": 65
        },
        "recommended_strategy": "Nurture sequence, encourage second purchase",
        "ad_targeting": "Welcome campaigns, bundled offers"
      }
    ],
    
    "actionable_insights": [
      "Champions (10%) drive 35% of revenue - prioritize retention",
      "1,020 customers at risk of churning - launch win-back campaign",
      "New customer 2nd purchase rate is low (22%) - opportunity for improvement"
    ]
  }
}
```

## ML Pipeline Architecture

```
Data Collection → Feature Engineering → Model Training → Validation → Deployment
      ↑                                                                    |
      └──────────── Monitoring & Automatic Retraining ←────────────────────┘
```

### Model Monitoring

```json
{
  "model_health": {
    "bid_model": {
      "status": "healthy",
      "performance_trend": "stable",
      "days_since_training": 5,
      "drift_score": 0.02,
      "retrain_recommended": false
    },
    "conversion_model": {
      "status": "degraded",
      "performance_trend": "declining",
      "days_since_training": 25,
      "drift_score": 0.18,
      "retrain_recommended": true,
      "reason": "Significant feature drift detected in competitor pricing"
    }
  }
}
```

## Integration Points

**Consumes from**:
- **grok-admaster-operator**: Raw campaign data
- **competitive-intelligence**: Market features
- **memory-palace**: Historical patterns
- Amazon Advertising API: Training data

**Feeds to**:
- **evolution-engine**: ML-optimized fitness functions
- **simulation-lab**: Enhanced predictions
- **campaign-strategist**: Data-driven recommendations

## Files

```
.agent/skills/data-scientist/
├── SKILL.md
├── scripts/
│   ├── bid_predictor.py
│   ├── conversion_forecaster.py
│   ├── anomaly_detector.py
│   ├── customer_segmenter.py
│   └── model_pipeline.py
└── models/
    └── model_registry.json
```

---

**This skill gives every seller access to personalized AI that learns and improves specifically for their business.**
