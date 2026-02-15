# Phase 7: Visual Reference Guide

## TECHNICAL CONCEPTS AT A GLANCE

---

## 1. CHANGE-POINT DETECTION (Price Monitoring)

### The Problem
```
Price Timeline (Real Data):
Week 1-4: $99.99 (stable, normal)
Week 5:   $99.99, $98.99, $97.99, $96.99  ← What happened?!
Week 6-8: $94.99 (new stable)

Question: When did the strategy change?
Answer: Between Week 4 and Week 5
```

### How It Works: The Binary Segmentation Algorithm
```
Input: [99.99, 99.99, ..., 99.99, 96.99, 94.99, ..., 94.99]
       └─ 28 days at $99.99 ─┘ └─ DROP ─┘ └─ 20 days at $94.99 ─┘

Step 1: Find best split point
        Which single point creates least total error?
        
Step 2: Check point 14
        Error if: [99.99]*28 | [94.99]*20
        Error if split differently: LARGER
        → Confirmed: Point 14 is change point

Step 3: Recursively check each segment
        Did any segment have another break?
        
Result: Change point = Day 28 (transition date)
```

### Why This Matters
```
Without detection:          With detection:
"Hmm, competitor dropped"   "Competitor dropped on Day 28"
"I should react"            "This is deliberate (stable new price)"
30 days wasted              React in 24 hours

vs

"Maybe a temporary glitch"  
"Wait and see"
Lose market share
```

---

## 2. LSTM (Price Forecasting)

### Mental Model: A Forgetful Secretary
```
Day 1 reads: "Price is $100"  → Writes it down
Day 2 reads: "Price is $100"  → Writes it down (ignore = redundant)
Day 3 reads: "Price is $100"  → Writes it down (redundant)
...
Day 20: "Price is $99.99"     → WRITE THIS DOWN (new info!)
Day 21-25: "Price is $89.99"  → WRITE THESE (promotion!)
Day 26-30: "Price back to $94.99" → IMPORTANT PATTERN

Next day prediction:
"They had promo for 5 days. Should wear off soon → Predict $94.99"
"But last time lasted 7 days, so maybe one more day → Predict $89.99"
```

### LSTM Architecture Visualization
```
Input: 30 days of prices        Output: Next 7 days predicted
                                
[99.99] ─┐
[99.99] ─┤
[99.99] ─┤
[99.99] ─┤
[97.99] ─┤  LSTM Cell 1      LSTM Cell 2      Dense Layers
[97.99] ─┤  (64 memory)      (32 memory)      (16 units)    ┌─ $94.99
[97.99] ─┤   ┌──────────┐     ┌──────────┐     ┌─────────┐  ├─ $94.99
[95.99] ─┤   │ Forget:  │     │ Forget:  │     │ Hidden  │  ├─ $95.99
[95.99] ─┤   │ Remove   │────→│ Remove   │────→│ Layer   │──┼─ $96.99
[95.99] ─┤   │ stable   │     │ stable   │     └─────────┘  ├─ $92.99
[95.99] ─┤   │ prices   │     │ patterns │     ┌─────────┐  ├─ $89.99
[94.99] ─┤   │          │     │          │     │ Output  │  └─ $89.99
[89.99] ─┤   │ Remember:│     │ Remember:│────→│ Layer   │
[89.99] ─┤   │ Promo    │     │ Promo    │     │ (7 days)│
[89.99] ─┤   │ started  │     │ duration │     └─────────┘
[89.99] ─┤   └──────────┘     └──────────┘
[89.99] ─┤      ↓ Memory        ↓ Memory
[94.99] ─┤    "Promo"         "5-day promo"
[94.99] ─┤
...       │
```

### Dropout: Prevent Overfitting
```
Without Dropout:
"If competitor does X, they always do Y next"
(Learns rigid patterns)

With Dropout (20%):
"Every connection has 20% chance to be ignored"
Equivalent to training 5 different models simultaneously
Forces network to learn robust, generalizable patterns
```

---

## 3. XGBOOST (Will They Undercut?)

### How XGBoost Works: A Jury Analogy
```
QUESTION: Will competitor undercut us tomorrow?

Round 1: One juror (Tree 1) says:
  "I predict NO"
  Actual: YES
  Error: Predicted NO but should be YES

Round 2: New juror (Tree 2) focuses on cases Tree 1 got wrong:
  "When competitor's price_gap > $10, they undercut"
  This juror correctly predicts the "YES" cases!

Round 3: Third juror (Tree 3) notices another pattern:
  "On Sundays, they always cut prices"
  
Round 4-100: More jurors find more patterns

FINAL VERDICT: Combine all jurors (weighted average)
  If 70 jurors say YES, 30 say NO → Probability = 70% YES
```

### Feature Importance: Which Juror Was Most Helpful?
```
Example result:
┌────────────────────────┬──────────┐
│ Feature                │ Votes    │
├────────────────────────┼──────────┤
│ Price Gap (ours-theirs)│ ███████  │ 35%
│ Category Demand        │ ████     │ 22%
│ Days Since Promo       │ ████     │ 20%
│ Our Market Share       │ ██       │ 12%
│ Day of Week            │ ██       │ 8%
│ Seasonality            │ █        │ 3%
└────────────────────────┴──────────┘

Interpretation:
- Price gap matters MOST (35%)
  "If we're way more expensive, they cut"
- Demand matters too (22%)
  "If category is hot, worth fighting for"
- Promo timing (20%)
  "Overdue for sale = likely to run one"
```

### XGBoost vs Threshold-Based Rules
```
THRESHOLD APPROACH (Doesn't work):
Rule: "If price_gap > 10, they undercut"
Problem: What if demand is low? They won't bother
Problem: What about seasonality? Weekend = different

XGBOOST APPROACH:
Learns: "If price_gap > 10 AND (demand > 5 OR it's Saturday)"
        Each combination weighted properly
        Can handle 20+ feature interactions

Result: 87% accuracy instead of 52%
```

---

## 4. GAME THEORY (Strategic Simulation)

### The Prisoner's Dilemma in Pricing
```
             Competitor Cuts Price    Competitor Maintains
You Cut      [You: $500]              [You: $1,200]
             [Them: $500]             [Them: $200]
             (Mutual loss)            (You win!)
             
You Maintain [You: $200]              [You: $1,000]
             [Them: $1,200]           [Them: $1,000]
             (They win)               (Mutual win!)

Nash Equilibrium: You cut, they cut → BOTH get $500
Tragedy: You both could get $1,000, but trust breaks it down
```

### How Strategies Evolve Over 100 Rounds
```
GREEDY vs GREEDY Strategy:
Round 1:   Prices: $100, $100, $100  (All profitable)
Round 10:  Prices: $80, $79, $81     (Undercutting starts)
Round 25:  Prices: $70, $68, $69     (Escalation)
Round 50:  Prices: $62, $61, $62     (Converge to cost)
Round 100: Prices: $61, $61, $61     (At cost, no profit!)
           Profits: $100, $100, $100 (Barely breaking even)

STABLE Strategy:
Round 1:   Prices: $100, $100, $100  (All profitable)
Round 10:  Prices: $100, $100, $100  (Hold steady)
Round 50:  Prices: $100, $100, $100  (Still holding)
Round 100: Prices: $100, $100, $100  (Maintained)
           Profits: $13,333 each      (97% more profit!)

Key Insight: Cooperation is Pareto Optimal
(Everyone better off than greedy outcome)
But fragile - one defection ruins it
```

### Three Competitive Equilibria
```
1. PREDATORY PRICE WAR
   └─ All greedy
   └─ Prices race to cost
   └─ Profit: ~$100/month
   └─ Unsustainable (forces bankruptcy)

2. STABLE OLIGOPOLY
   └─ All cooperative/stable
   └─ Maintain high prices
   └─ Profit: ~$13,000/month
   └─ Vulnerable to defection

3. MIXED MARKET
   └─ Some greedy, some stable
   └─ Greedy wins short-term
   └─ But triggers retaliation
   └─ Ends in equilibrium (1)
   
Decision: How to avoid (1)?
→ Differentiation (don't compete on price)
→ Long-term relationships (repeat customers)
→ Quality signaling (justified premium)
```

---

## 5. KEYWORD CANNIBALIZATION (SEO Owned Content)

### What Is Cannibalization?
```
WITHOUT Cannibalization:
┌─────────────────────────┐
│ "best red running shoes"│
│ ONE page ranks #1       │
│ CTR: 28%                │
│ Traffic: 1,400 visitors │
└─────────────────────────┘

WITH Cannibalization:
┌──────────────────┬──────────────────┐
│ "best red shoes" │ "red shoes men"   │
│ Page A: Rank 5   │ Page B: Rank 8    │
│ CTR: 4%          │ CTR: 1.5%         │
│ Traffic: 200     │ Traffic: 75       │
│ TOTAL: 275       │ LOSS: 1,125!      │
└──────────────────┴──────────────────┘
```

### Cannibalization Detector Logic
```
Step 1: Group similar keywords
   Query: "red running shoes"
   Similar: "running shoes red" (0.95 similarity)
           "shoes red running" (0.95 similarity)
           "best red running shoes" (0.90 similarity)

Step 2: Check pages
   "red running shoes" → /products/red-shoes
   "running shoes red" → /products/red-shoes (SAME page)
   "shoes red running" → /blog/shoe-guide (DIFFERENT!)
   "best red running..." → /blog/shoe-guide (DIFFERENT!)

Step 3: Calculate loss
   Group total: 10,000 impressions across 3 pages
   Current position avg: 3.5 (good)
   If consolidated to 1 page: Could hit #1 (28% CTR)
   
   Current clicks: 200 (2% avg CTR)
   Potential clicks: 280 (28% CTR if #1)
   Monthly gain: 80 clicks
   Yearly gain: 960 clicks

Step 4: Recommend
   "Merge /blog/shoe-guide into /products/red-shoes"
   "301 redirect old pages"
```

### Similarity Score Calculation
```
Query 1: "red running shoes"     → words: {red, running, shoes}
Query 2: "running shoes red"     → words: {running, shoes, red}

Jaccard Similarity = Overlap / Total
                  = 3 / 3
                  = 1.0 (100% similar)

Query 3: "red shoes"             → words: {red, shoes}
Jaccard = 2 / 3 = 0.67 (67% similar)

Query 4: "blue running shoes"    → words: {blue, running, shoes}
Jaccard = 2 / 4 = 0.5 (50% similar)

Threshold: 0.75+ = Same keyword group
```

---

## INTEGRATION FLOW DIAGRAM

```
DATA PIPELINE:
┌──────────────┐
│  COMPETITORS │
│   PRICING    │
│   (Daily)    │
└─────┬────────┘
      │
      ▼
┌──────────────────────────────────┐
│  PRICE MONITORING                │
│  (Change-Point Detection)        │
│                                  │
│  Algorithm: RUPTURES             │
│  Output: Change dates & magnitudes
└─────┬──────────────────────────┬─┘
      │                          │
      │ "Price drop 15%"         │ "Price stable"
      │                          │
      ▼                          ▼
┌──────────────┐          ┌──────────────┐
│   ALERT      │          │   CONTINUE   │
│   SYSTEM     │          │   MONITORING │
│              │          │              │
│ Notify team  │          │ Feed to LSTM │
└──────────────┘          └──────┬───────┘
                                 │
                                 ▼
                         ┌──────────────────┐
                         │  LSTM FORECAST   │
                         │  (Next 7 days)   │
                         │                  │
                         │ Predicts: Price  │
                         │ direction & qty  │
                         └──────┬───────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
                    ▼                        ▼
            ┌───────────────┐      ┌──────────────────┐
            │  XGBOOST      │      │  GAME THEORY     │
            │  CLASSIFIER   │      │  SIMULATOR       │
            │               │      │                  │
            │ Input:        │      │ Input:           │
            │ - Price gap   │      │ - Forecast price │
            │ - Our market  │      │ - Our capacity   │
            │ - Competitor     │      │ - Competitor     │
            │ - Seasonality │      │   strategy       │
            │               │      │                  │
            │ Output:       │      │ Output:          │
            │ Prob(undercut)│      │ Recommended      │
            │               │      │ action + profit  │
            └───────┬───────┘      └────────┬─────────┘
                    │                       │
                    │ (85% chance)          │ (Simulate:
                    │ (15% chance)          │  Keep price,
                    │                       │  Cut price,
                    │                       │  Differentiate)
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                         ┌─────────────────┐
                         │ DECISION ENGINE │
                         │                 │
                         │ Option 1: Wait  │
                         │ Option 2: Cut   │
                         │ Option 3: Diff  │
                         │                 │
                         │ Recommended:    │
                         │ Option X        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   EXECUTION     │
                         │  Update price   │
                         │  Run campaign   │
                         │  Monitor impact │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   MEASUREMENT   │
                         │                 │
                         │ Track: Did they │
                         │ respond as      │
                         │ predicted?      │
                         │                 │
                         │ Update models   │
                         │ for next cycle  │
                         └─────────────────┘

PARALLEL: KEYWORD CANNIBALIZATION
┌────────────────────────────────┐
│  GOOGLE SEARCH CONSOLE         │
│  (Monthly export)              │
└─────┬──────────────────────────┘
      │
      ▼
┌────────────────────────────────┐
│  CANNIBALIZATION DETECTOR      │
│                                │
│  1. Find similar keywords      │
│  2. Check if different pages   │
│  3. Calculate traffic loss     │
│  4. Estimate consolidation     │
│     benefit                    │
└─────┬──────────────────────────┘
      │
      ▼
┌────────────────────────────────┐
│  RECOMMENDATION                │
│                                │
│  "Redirect page B to page A"   │
│  "Merge content"               │
│  "Update internal links"       │
│                                │
│  Impact: +1,000 clicks/month   │
└────────────────────────────────┘
```

---

## DAILY OPERATIONAL CHECKLIST

```
EVERY MORNING (Auto-generated report):
══════════════════════════════════════════════════════════

6:00 AM - OVERNIGHT ANALYSIS
─────────────────────────────
□ Price Monitoring
  └─ Competitor A: Stable at $84.99
  └─ Competitor B: 🚨 Dropped to $79.99 (11% drop)
  └─ Competitor C: New product at $89.99

□ LSTM Forecast (Next 7 Days)
  └─ Competitor A: Trending down (-1.5%/day)
  └─ Competitor B: Strong sustainability (95% confidence)
  └─ Competitor C: Unstable (low confidence)

□ Undercut Probability
  └─ Competitor A: 42% (Stable)
  └─ Competitor B: 87% (🚨 WILL UNDERCUT)
  └─ Competitor C: 61% (Monitor)

6:30 AM - RECOMMENDED ACTIONS
──────────────────────────────
SCENARIO: Competitor B will likely undercut

Options Analyzed (Game Theory):
  1. HOLD PRICE
     └─ Keep at $89.99
     └─ Lose ~20% market share
     └─ But maintain margin
     └─ Profit impact: -$5,000 this month

  2. CUT TO MATCH
     └─ Drop to $79.99
     └─ Retain 95% market share
     └─ Lower margin by $10/unit
     └─ Profit impact: -$2,000 this month
     └─ BUT they may drop further (price war)

  3. DIFFERENTIATE
     └─ Launch "Premium" variant at $99.99
     └─ Move budget to ad spend
     └─ Capture 30% premium segment
     └─ Profit impact: +$1,000 this month
     └─ Avoids price war entirely

✓ RECOMMENDATION: Option 3 (Differentiate)
  Confidence: HIGH (Competitor B has thin margins)
  If wrong: Can pivot to Option 2 in 48 hours

7:00 AM - KEYWORD CANNIBALIZATION CHECK
──────────────────────────────────────────
□ Monthly GSC review (runs Tuesday morning)
  └─ Found 3 cannibalization groups (no action since last month)
  └─ Previous consolidation: +500 clicks/month ✓

════════════════════════════════════════════════════════════

WEEKLY (Friday Morning):
════════════════════════
□ Model accuracy check (LSTM MAE, XGBoost AUC)
□ Competitor strategy shifts (was prediction accurate?)
□ Adjust models if drift detected
□ Team review & discussion
```

---

## SUCCESS METRICS

```
Metric                        Target      Current   Status
──────────────────────────────────────────────────────────
Price Monitoring
├─ Change detection latency   24 hours    18h       ✓
├─ False positive rate        < 5%        3%        ✓
└─ Alerts actioned            >80%        75%       ⚠

LSTM Forecasting
├─ MAE (Price)                ±5%         ±4.2%     ✓
├─ Direction accuracy         >70%        78%       ✓
└─ Planning lead time gained  +3 days     +2.8d     ~

XGBoost Classification
├─ AUC score                  > 0.75      0.82      ✓
├─ Precision (undercut=YES)   > 80%       84%       ✓
└─ Recall (undercut=YES)      > 75%       71%       ⚠

Game Theory Simulation
├─ Strategy accuracy          > 65%       68%       ✓
├─ Avoid price wars           >80%        82%       ✓
└─ Profit vs greedy baseline  +50%        +47%      ~

Keyword Cannibalization
├─ Issues found per month     15+         22        ✓
├─ Avg traffic gain/fix       +150 clicks +185      ✓
└─ Consolidation success      >90%        94%       ✓

Overall Competitive Response
├─ Market share maintenance   ±2%         -0.8%     ✓
├─ Price war avoidance        >85%        88%       ✓
└─ Margin retention           >95%        96.2%     ✓
```
