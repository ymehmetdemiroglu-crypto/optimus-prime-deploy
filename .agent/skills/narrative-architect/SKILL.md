---
name: narrative-architect
description: Storytelling and strategic communication. Converts data into compelling narratives, frames recommendations persuasively, adapts messaging for different stakeholders, and tracks progress with milestone visualization.
---

# Narrative Architect Skill

The **Narrative Architect** transforms raw data and complex strategies into compelling stories that humans can understand and act on. It's the voice of Optimus Pryme, making AI insights accessible and persuasive.

## Core Capabilities

### 1. **Performance Narratives**
- Convert metrics into stories ("Your sales grew 23% because...")
- Highlight key insights, not just numbers
- Cause-and-effect explanations
- Context-aware storytelling
- Emotional resonance (celebrate wins, empathize with challenges)

### 2. **Scenario Communication**
- Explain complex strategies simply
- "What-if" scenario storytelling
- Risk vs. reward framing
- Visual metaphors for abstract concepts
- Decision trees made human-readable

### 3. **Recommendation Framing**
- Persuasive presentation of suggestions
- Pros/cons balanced view
- Social proof ("Similar sellers saw...")
- Urgency indicators (when appropriate)
- Clear action steps

### 4. **Stakeholder Adaptation**
- CEO view: High-level strategy, ROI focus
- CFO view: Financial metrics, risk analysis
- Manager view: Tactical execution, daily actions
- Technical view: Detailed methodology
- Adjust complexity and terminology per audience

### 5. **Progress Tracking**
- Milestone visualization
- Journey mapping ("You're here on the path to...")
- Achievement celebration
- Goal proximity indicators
- Motivational framing

## Performance Narratives

### Data-to-Story Conversion

**Input**:
```json
{
  "action": "create_performance_narrative",
  "data": {
    "sales": {"current": 15420, "previous": 12500, "change": 0.234},
    "acos": {"current": 0.22, "previous": 0.28, "change": -0.214},
    "key_events": [
      {"date": "2026-01-15", "event": "Launched dayparting"},
      {"date": "2026-01-22", "event": "Added 15 negative keywords"}
    ]
  },
  "tone": "professional"
}
```

**Output**:
```json
{
  "narrative": {
    "headline": "Strong Growth: Sales Up 23% with Improved Efficiency",
    "story": "Your campaign performance improved significantly this month, driven by two strategic optimizations. After implementing dayparting on January 15th, we saw an immediate 12% increase in conversion rate during peak hours (8-10am and 6-8pm). Then, by adding 15 carefully selected negative keywords on January 22nd, we eliminated $340 in wasted spend while maintaining impression volume. The combination resulted in 23% sales growth ($15,420 vs $12,500) while simultaneously reducing ACoS from 28% to 22%. This means you're not just selling more—you're selling more efficiently.",
    "key_insights": [
      "Dayparting delivered 12% CVR boost",
      "Negative keywords saved $340/month",
      "Efficiency improved: 22% ACoS (was 28%)"
    ],
    "emotional_tone": "celebratory",
    "call_to_action": "Consider expanding dayparting to your other campaigns to replicate this success."
  }
}
```

## Scenario Communication

### Complex Strategy Simplification

**Input**:
```json
{
  "action": "explain_strategy",
  "strategy": {
    "name": "genetic_algorithm_bid_optimization",
    "technical_details": "Multi-objective fitness function with Pareto optimization...",
    "expected_outcome": "+18% ROAS"
  },
  "audience": "non_technical"
}
```

**Output**:
```json
{
  "explanation": {
    "simple_version": "Think of this like breeding racehorses. We start with 20 different bidding strategies, test them all, keep the best performers, and 'breed' them together to create even better strategies. After 10 generations of this process, we end up with a strategy that's 18% more effective than what we started with.",
    "how_it_works": [
      "Start: 20 different bid strategies",
      "Test: Run each for 3 days, measure performance",
      "Select: Keep top 5 performers",
      "Combine: Mix the best parts of each",
      "Repeat: 10 times until we find the optimal strategy"
    ],
    "benefit": "Discovers strategies humans might never think of",
    "risk": "Requires 30 days of testing, minimal downside",
    "analogy": "Like evolution in nature, but for your ad campaigns"
  }
}
```

## Recommendation Framing

### Persuasive Presentation

**Input**:
```json
{
  "action": "frame_recommendation",
  "recommendation": {
    "action": "increase_budget",
    "from": 100,
    "to": 150,
    "expected_impact": "+$1200_monthly_profit"
  },
  "user_risk_tolerance": "moderate"
}
```

**Output**:
```json
{
  "framed_recommendation": {
    "headline": "Opportunity: Scale Your Profitable Campaign",
    "opening": "Your campaign is performing exceptionally well (18% ACoS, 5.5 ROAS). The data suggests you're leaving money on the table.",
    "recommendation": "Increase daily budget from $100 to $150 (+50%)",
    "why_now": "Your campaign has been consistently profitable for 21 days. Historical data shows campaigns with your metrics can scale 40-60% before hitting diminishing returns.",
    "expected_outcome": {
      "best_case": "+$1,500 monthly profit",
      "likely_case": "+$1,200 monthly profit",
      "worst_case": "+$800 monthly profit"
    },
    "pros": [
      "Capture more market share while competition is stable",
      "Your ACoS has room (18% vs 25% target)",
      "Proven profitability reduces risk"
    ],
    "cons": [
      "Requires +$1,500 monthly budget commitment",
      "May see slight ACoS increase (18% → 20%) due to broader reach",
      "Takes 7-10 days to stabilize at new budget"
    ],
    "social_proof": "87% of sellers with similar metrics who scaled saw positive ROI",
    "risk_mitigation": "We can test with +25% for 1 week first, then scale to +50% if results are positive",
    "call_to_action": "Approve budget increase to $150/day?",
    "urgency": "moderate",
    "confidence": 0.84
  }
}
```

## Stakeholder Adaptation

### Multi-Audience Views

**Same Data, Different Audiences**:

**CEO View**:
```
"Q1 Performance: Revenue up 34%, profit margins improved 6 points. 
Strategic win: Our AI-driven optimization reduced customer acquisition 
cost by 22% while scaling volume. Recommend: Expand to 3 new product 
categories using proven playbook. ROI projection: +$45K annually."
```

**CFO View**:
```
"Financial Summary:
- Revenue: $125K (+34% YoY)
- Ad Spend: $28K (+12% YoY)
- ROAS: 4.46 (was 3.67)
- Profit Margin: 28% (was 22%)
- CAC: $18.50 (-22%)
- Payback Period: 14 days (was 21 days)
Risk: Increased budget exposure (+$3K/mo). Mitigation: 30-day test period."
```

**Manager View**:
```
"This Week's Actions:
1. Increased bids on 12 high-performing keywords (avg +15%)
2. Paused 8 keywords with ACoS > 35%
3. Added dayparting: 8-10am, 6-8pm (peak conversion hours)
4. Launched A/B test: headline variation
Next Week: Review A/B results, expand negative keyword list"
```

**Technical View**:
```
"Optimization Details:
- Algorithm: Ensemble (XGBoost + LSTM + Bandit)
- Features: 47 (bid history, time-of-day, competition density, etc.)
- Model Accuracy: 82% (bid prediction)
- Confidence Calibration: 0.91
- Hyperparameters: learning_rate=0.03, max_depth=6
- A/B Test: Bayesian sequential testing, 95% confidence threshold"
```

## Progress Tracking

### Journey Visualization

**Input**:
```json
{
  "action": "visualize_progress",
  "goal": "reach_10k_monthly_sales",
  "current": 6500,
  "started": 3200,
  "timeline_days": 45
}
```

**Output**:
```json
{
  "progress_narrative": {
    "headline": "You're 67% of the Way to Your Goal!",
    "journey_map": {
      "start": "$3,200/month (Day 0)",
      "current": "$6,500/month (Day 45) ← You are here",
      "goal": "$10,000/month (Target: Day 90)"
    },
    "progress_percentage": 67,
    "velocity": "+$73/day average",
    "projection": "At current pace, you'll reach $10K by Day 93 (3 days past target)",
    "milestones_achieved": [
      "✅ $5,000/month (Day 28)",
      "✅ Profitable ACoS (Day 14)",
      "✅ First $1K day (Day 35)"
    ],
    "next_milestone": "$7,500/month (estimated Day 59)",
    "motivational_message": "Excellent progress! You've more than doubled sales in 45 days. Three more weeks of this momentum and you'll hit your goal.",
    "acceleration_opportunity": "Increasing budget by 30% could get you to $10K by Day 75 (15 days early)"
  }
}
```

## Usage Patterns

### Pattern 1: Weekly Performance Report

```
USER: "How did my campaigns do this week?"

NARRATIVE ARCHITECT:
1. Gather data: Sales, spend, ACoS, key events
2. Identify story: Sales up 12%, efficiency improved
3. Find cause: Dayparting + negative keywords
4. Create narrative: "Strong week driven by two optimizations..."
5. Add context: "This puts you 72% toward monthly goal"
6. Recommend: "Consider expanding dayparting to Campaign B"
7. Deliver: Compelling, actionable story
```

### Pattern 2: Strategy Proposal

```
EVOLUTION ENGINE: "I evolved a new strategy with 22% better fitness"

NARRATIVE ARCHITECT:
1. Translate: "Fitness 0.88" → "22% more profitable"
2. Explain: How the strategy works (simple terms)
3. Frame: Pros, cons, risk, reward
4. Add social proof: "Similar strategies succeeded 84% of time"
5. Visualize: "Here's what next 30 days could look like..."
6. Call to action: "Approve deployment?"
```

### Pattern 3: Crisis Communication

```
CONSCIOUSNESS ENGINE: "Alert: ACoS spiked to 45%"

NARRATIVE ARCHITECT:
1. Acknowledge: "I detected an ACoS spike today"
2. Explain cause: "Competitor launched aggressive campaign"
3. Contextualize: "This is temporary, happened before in Q3"
4. Show response: "I've already reduced bids by 12%"
5. Set expectations: "Should recover in 3-5 days"
6. Reassure: "Your profit is protected, monitoring closely"
```

## Integration with Other Skills

**Receives data from**:
- **All skills**: Converts their outputs into narratives
- **consciousness-engine**: Decision explanations
- **simulation-lab**: Scenario outcomes
- **knowledge-synthesizer**: Insights and trends

**Delivers to**:
- **User**: All communications
- **memory-palace**: Stores effective narrative patterns

## Files

```
.agent/skills/narrative-architect/
├── SKILL.md
└── scripts/
    ├── story_generator.py          # Data-to-narrative conversion
    ├── stakeholder_adapter.py      # Audience-specific views
    └── progress_visualizer.py      # Journey mapping
```

## Example Invocation

```
USER: "Explain why you want to increase my budget"

NARRATIVE ARCHITECT:
"Your campaign is crushing it right now. Here's why I'm recommending a budget increase:

**The Situation**: You're currently spending $100/day and generating $450/day in sales (4.5 ROAS). Your ACoS is 22%—well below your 28% target. This means you have room to grow.

**The Opportunity**: When campaigns perform this well, they can usually scale 40-60% before hitting diminishing returns. Based on your data, I estimate a 50% budget increase ($100 → $150/day) would generate an additional $1,200/month in profit.

**The Risk**: You'd be committing an extra $1,500/month in ad spend. There's a small chance (15%) that performance doesn't scale as expected. To mitigate this, I recommend starting with +25% for one week, then scaling to +50% if results are positive.

**The Numbers**:
- Best case: +$1,500/month profit
- Likely case: +$1,200/month profit  
- Worst case: +$800/month profit

**Why Now**: Your campaign has been stable and profitable for 21 days. Competition is currently low. Waiting could mean missing this window.

**What Others Did**: 87% of sellers with similar metrics who scaled saw positive ROI.

Should we test a +25% increase this week?"
```

## Notes

- All narratives include confidence scores
- Tone adapts to situation (celebratory, urgent, empathetic)
- Avoids jargon unless audience is technical
- Always provides clear next steps
- Balances optimism with realism

---

**This skill ensures Optimus Pryme communicates like a trusted advisor, not a black box.**
