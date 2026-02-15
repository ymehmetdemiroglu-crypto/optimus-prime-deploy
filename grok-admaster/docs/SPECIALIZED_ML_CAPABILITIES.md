# Specialized ML Capabilities - Complete Reference

## 🎯 Overview

This document describes the specialized machine learning capabilities added to Optymus Pryme for advanced PPC optimization.

---

## 1. Keyword Clustering & Segmentation (`clustering.py`)

### Purpose
Groups keywords by performance patterns for bulk optimization strategies.

### Components

#### **KeywordSegmenter**
- **K-Means Clustering**: Groups similar keywords using ML
- **K-Means++ Initialization**: Optimal centroid selection
- **Rule-Based Segmentation**: Predefined performance tiers

#### **Performance Segments**
| Segment | Description | Action |
|---------|-------------|--------|
| **Stars** | High volume, high performance | Increase bids to maximize |
| **Potential** | Good performance, low volume | Scale with higher bids |
| **Workhorses** | High volume, average performance | Optimize carefully |
| **Underperformers** | Poor performance, still active | Reduce bids or pause |
| **Zombies** | Very low activity | Pause or test bid increase |

#### **PerformanceSegmenter**
Campaign tier segmentation:
- **Platinum**: ROAS ≥ 5.0, ACoS ≤ 20%
- **Gold**: ROAS ≥ 3.5, ACoS ≤ 28%
- **Silver**: ROAS ≥ 2.5, ACoS ≤ 40%
- **Bronze**: ROAS ≥ 1.5, ACoS ≤ 60%
- **Needs Attention**: Everything else

### API Endpoints
```
POST /api/v1/ml/advanced/segment/keywords
POST /api/v1/ml/advanced/segment/campaigns
```

---

## 2. Anomaly Detection (`anomaly_detection.py`)

### Purpose
Identifies unusual patterns and potential issues in campaign performance.

### Algorithms

#### **Isolation Forest**
- Unsupervised anomaly detection
- Randomly isolates observations
- Anomalies are isolated quickly (shorter path length)
- Configurable: 50-100 trees, 256 sample size

#### **Z-Score Detector**
- Statistical outlier detection
- Rolling window approach (30 days default)
- Configurable threshold (default: 2.5σ)

#### **Change Point Detector (CUSUM)**
- Cumulative Sum algorithm
- Detects sudden trend changes
- Bidirectional (increases and decreases)

### Anomaly Types
| Type | Description |
|------|-------------|
| `spend_spike` | Unusual increase in spend |
| `spend_drop` | Unusual decrease in spend |
| `ctr_anomaly` | Click-through rate outlier |
| `conversion_anomaly` | Conversion rate change |
| `acos_spike` | ACoS jumped significantly |
| `impression_drop` | Impressions fell unexpectedly |
| `performance_deterioration` | Overall decline |
| `unusual_pattern` | Multi-dimensional anomaly |

### Severity Levels
- **LOW**: Minor fluctuation
- **MEDIUM**: Worth investigating
- **HIGH**: Requires action
- **CRITICAL**: Immediate attention needed

### API Endpoints
```
POST /api/v1/ml/advanced/anomaly/detect
POST /api/v1/ml/advanced/anomaly/trend-changes
```

---

## 3. Search Term Analysis (`search_term_analysis.py`)

### Purpose
Extracts insights from search terms using NLP and text mining.

### Components

#### **TextPreprocessor**
- Tokenization
- Stop word removal (English + Amazon-specific)
- N-gram extraction

#### **TFIDFVectorizer**
- Term Frequency-Inverse Document Frequency
- Feature extraction from text
- Top term identification

#### **SearchTermAnalyzer**
- Word-level performance analysis
- N-gram performance analysis
- Category assignment (winners, losers, etc.)
- Pattern detection
- Recommendation generation

### Analysis Output
```json
{
  "summary": {
    "total_terms": 1500,
    "unique_words": 800,
    "avg_term_length": 3.2
  },
  "word_performance": {...},
  "ngram_performance": {...},
  "categories": {
    "winners": [...],
    "potential": [...],
    "test": [...],
    "losers": [...],
    "expensive": [...]
  },
  "patterns": {
    "single_word_percentage": 15.2,
    "long_tail_percentage": 35.8,
    "question_based": 42
  },
  "recommendations": [...]
}
```

### API Endpoints
```
POST /api/v1/ml/advanced/search-terms/analyze
POST /api/v1/ml/advanced/search-terms/negatives
POST /api/v1/ml/advanced/search-terms/exact-match
```

---

## 4. Competitor Analysis (`competitor_analysis.py`)

### Purpose
Infers competitor bids and market conditions from auction data.

### Components

#### **CompetitorBidEstimator**
Estimates competitor bids using:
- Your bid and position data
- Impression share
- CPC analysis
- First-price auction modeling

#### **MarketAnalyzer**
Generates market intelligence:
- Search volume estimation
- Competition intensity (0-1 scale)
- CPC trend analysis (rising/stable/declining)
- Opportunity scoring
- Optimal bid recommendations

#### **AuctionSimulator**
Monte Carlo simulation of auction outcomes:
- Win rate prediction
- Position forecasting
- CPC estimation
- Optimal bid finding

### API Endpoints
```
POST /api/v1/ml/advanced/competitor/estimate-bids
POST /api/v1/ml/advanced/competitor/market-intelligence
POST /api/v1/ml/advanced/competitor/opportunities
POST /api/v1/ml/advanced/competitor/simulate-auction
POST /api/v1/ml/advanced/competitor/optimal-bid
```

---

## 5. Attribution Modeling (`attribution.py`)

### Purpose
Assigns credit to different touchpoints in the customer journey.

### Attribution Models

| Model | Description |
|-------|-------------|
| **Last Click** | 100% to last touchpoint |
| **First Click** | 100% to first touchpoint |
| **Linear** | Equal credit to all |
| **Time Decay** | More credit closer to conversion |
| **Position Based** | 40% first, 20% middle, 40% last |
| **Data Driven** | Based on engagement and position |

### Components

#### **AttributionEngine**
- Multi-model attribution
- Configurable decay rates
- Customizable position weights

#### **ConversionPathAnalyzer**
- Path length analysis
- Time to conversion
- Common path patterns
- Channel sequence analysis

#### **MarkovAttribution**
- Markov Chain-based
- Removal effect calculation
- Transition probability matrix

### API Endpoints
```
GET  /api/v1/ml/advanced/attribution/models
```

---

## 6. A/B Testing Framework (`ab_testing.py`)

### Purpose
Statistical testing for bid and campaign experiments.

### Components

#### **StatisticalTester**
- Z-test for proportions (CTR, CVR)
- Welch's t-test for means (ACoS, ROAS)
- Confidence intervals
- Effect size calculation (Cohen's d)
- Power analysis

#### **SampleSizeCalculator**
- For proportion tests
- For mean tests
- Multiple power levels (80%, 90%, 95%)

#### **ExperimentManager**
- Create experiments
- Manage variants
- Record results
- Statistical analysis
- Recommendations

### Experiment Types
- `bid_test`: Testing different bid levels
- `budget_test`: Budget allocation experiments
- `targeting_test`: Audience/keyword targeting
- `ad_copy_test`: Creative testing

### API Endpoints
```
POST /api/v1/ml/advanced/experiments/create
POST /api/v1/ml/advanced/experiments/{id}/start
POST /api/v1/ml/advanced/experiments/{id}/stop
POST /api/v1/ml/advanced/experiments/{id}/results
GET  /api/v1/ml/advanced/experiments/{id}/analyze
GET  /api/v1/ml/advanced/experiments/sample-size
```

---

## 7. Keyword Health & Churn Prediction (`keyword_health.py`)

### Purpose
Predicts which keywords are likely to underperform or become inactive.

### Health Factors
1. **Performance Score** (35%): ACoS and ROAS based
2. **Trend Score** (25%): Historical trajectory
3. **Efficiency Score** (25%): Conversion efficiency
4. **Engagement Score** (15%): CTR quality

### Health Status
| Status | Score Range | Action |
|--------|-------------|--------|
| **Excellent** | 85-100 | Maintain strategy |
| **Good** | 70-84 | Minor optimizations |
| **At Risk** | 50-69 | Attention needed |
| **Declining** | 30-49 | Urgent action required |
| **Critical** | 0-29 | Immediate intervention |

### Components

#### **KeywordHealthAnalyzer**
- Multi-factor health scoring
- Risk factor identification
- Recommendation generation
- Decline prediction
- Improvement potential calculation

#### **KeywordLifecyclePredictor**
- Lifecycle stage identification
- Future trajectory prediction
- Stage-specific recommendations

### Lifecycle Stages
1. **New** (0-14 days): Learning phase
2. **Growing** (14-60 days): Scaling up
3. **Mature** (60-180 days): Peak performance
4. **Declining** (180-365 days): Optimization needed
5. **End of Life** (365+ days): Consider retirement

### API Endpoints
```
POST /api/v1/ml/advanced/keyword-health/analyze
POST /api/v1/ml/advanced/keyword-health/bulk
POST /api/v1/ml/advanced/keyword-health/lifecycle
```

---

## 📊 Complete API Reference

### Clustering
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/segment/keywords` | Segment keywords by performance |
| POST | `/segment/campaigns` | Segment campaigns by tier |

### Anomaly Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/anomaly/detect` | Detect campaign anomalies |
| POST | `/anomaly/trend-changes` | Detect trend change points |

### Search Term Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search-terms/analyze` | Full analysis |
| POST | `/search-terms/negatives` | Find negative keywords |
| POST | `/search-terms/exact-match` | Find exact match candidates |

### Competitor Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/competitor/estimate-bids` | Estimate competitor bids |
| POST | `/competitor/market-intelligence` | Market analysis |
| POST | `/competitor/opportunities` | Find opportunities |
| POST | `/competitor/simulate-auction` | Auction simulation |
| POST | `/competitor/optimal-bid` | Find optimal bid |

### A/B Testing
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/experiments/create` | Create experiment |
| POST | `/experiments/{id}/start` | Start experiment |
| POST | `/experiments/{id}/stop` | Stop experiment |
| POST | `/experiments/{id}/results` | Record results |
| GET | `/experiments/{id}/analyze` | Analyze results |
| GET | `/experiments/sample-size` | Calculate sample size |

### Keyword Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/keyword-health/analyze` | Analyze single keyword |
| POST | `/keyword-health/bulk` | Analyze multiple keywords |
| POST | `/keyword-health/lifecycle` | Predict lifecycle stage |

### Attribution
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/attribution/models` | List available models |

---

## 📁 Files Created

```
server/app/amazon_ppc_optimizer/ml/
├── clustering.py           # Keyword clustering
├── anomaly_detection.py    # Anomaly detection
├── search_term_analysis.py # NLP for search terms
├── competitor_analysis.py  # Market intelligence
├── attribution.py          # Multi-touch attribution
├── ab_testing.py           # Statistical testing
├── keyword_health.py       # Health & churn prediction
└── advanced_router.py      # API endpoints
```

---

## 🚀 Usage Examples

### Keyword Segmentation
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/ml/advanced/segment/keywords",
    json={
        "keywords": [
            {"keyword_id": 1, "acos": 15, "impressions": 5000, "clicks": 100},
            {"keyword_id": 2, "acos": 45, "impressions": 8000, "clicks": 50}
        ],
        "target_acos": 25.0
    }
)
```

### Anomaly Detection
```python
response = requests.post(
    "http://localhost:8000/api/v1/ml/advanced/anomaly/detect",
    json={
        "campaign_id": 1,
        "historical_data": [...],  # Last 30 days
        "current_data": {"spend": 500, "sales": 100, ...}
    }
)
```

### Search Term Analysis
```python
response = requests.post(
    "http://localhost:8000/api/v1/ml/advanced/search-terms/analyze",
    json={
        "search_terms": [
            {"term": "wireless headphones", "spend": 50, "sales": 200},
            ...
        ],
        "target_acos": 25.0
    }
)
```

---

**Status: Specialized ML Capabilities Complete ✅**
