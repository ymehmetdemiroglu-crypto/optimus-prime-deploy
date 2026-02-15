---
name: executive-reporter
description: Automated executive reporting that transforms complex advertising data into boardroom-ready insights, presentations, and strategic summaries tailored to C-suite audiences.
---

# Executive Reporter Skill

The **Executive Reporter** transforms raw advertising metrics into compelling executive communications. It leverages the existing data infrastructure to produce automated daily briefings, weekly summaries, monthly board reports, and custom presentations.

## Core Capabilities

### 1. **Daily Executive Briefing**
- Morning summary of yesterday's performance
- Key wins and concerns highlighted
- Action items requiring attention
- Comparison to targets and benchmarks

### 2. **Weekly Performance Report**
- Week-over-week trend analysis
- Top/bottom performing campaigns
- Budget utilization and pacing
- Strategic recommendations

### 3. **Monthly Board Report**
- Executive summary (1-page)
- Financial performance vs targets
- Strategic initiative progress
- Competitive positioning update
- Forward-looking projections

### 4. **Custom Presentation Generator**
- Dynamic slide generation
- Chart and graph creation
- Talking points for each slide
- Export to PDF/PowerPoint format

### 5. **Alert Escalation**
- Threshold-based executive alerts
- Priority classification (Critical/High/Medium)
- Recommended actions included
- Historical context provided

## Report Templates

### Daily Briefing Format

```json
{
  "action": "generate_daily_briefing",
  "date": "2026-02-05",
  "account_id": "ACC_123"
}
```

**Response**:
```json
{
  "briefing": {
    "headline": "Strong Day: Revenue +12% vs Target",
    "executive_summary": "Yesterday exceeded revenue targets by 12% while maintaining ACoS within acceptable range. Two campaigns require attention.",
    "key_metrics": {
      "revenue": {"value": 15423, "vs_target": "+12%", "vs_yesterday": "+8%"},
      "ad_spend": {"value": 2347, "vs_budget": "-5%", "efficiency": "good"},
      "acos": {"value": 15.2, "vs_target": "-0.8pp", "trend": "stable"},
      "roas": {"value": 6.57, "benchmark": 5.5, "status": "exceeding"}
    },
    "wins": [
      "Campaign 'Summer Sale' hit 150% of daily target",
      "New keyword batch achieving 8.2% CTR"
    ],
    "concerns": [
      "Campaign 'Electronics' ACoS creeping up (18% vs 15% target)",
      "Competitor X launched aggressive promotion"
    ],
    "action_items": [
      {"priority": "high", "action": "Review Electronics campaign bids", "owner": "AI"},
      {"priority": "medium", "action": "Monitor competitor pricing", "owner": "Team"}
    ]
  }
}
```

### Weekly Report Format

```json
{
  "action": "generate_weekly_report",
  "week_ending": "2026-02-05",
  "comparison": "previous_week"
}
```

**Response**:
```json
{
  "weekly_report": {
    "period": "Jan 30 - Feb 5, 2026",
    "headline": "Record Revenue Week with Improved Efficiency",
    "performance_summary": {
      "total_revenue": 98500,
      "total_spend": 14200,
      "blended_acos": 14.4,
      "wow_revenue_change": "+15%",
      "wow_efficiency_change": "+2.1pp"
    },
    "top_performers": [
      {"campaign": "Brand Defense", "revenue": 25000, "roas": 8.2},
      {"campaign": "Summer Collection", "revenue": 18500, "roas": 7.1}
    ],
    "underperformers": [
      {"campaign": "New Product Launch", "revenue": 2300, "roas": 2.1, "recommendation": "Increase budget, optimize keywords"}
    ],
    "budget_status": {
      "allocated": 15000,
      "spent": 14200,
      "utilization": "94.7%",
      "pacing": "on_track"
    },
    "strategic_recommendations": [
      "Shift 10% budget from General to Brand campaigns",
      "Expand Summer Collection keywords",
      "Test video ads for New Product Launch"
    ]
  }
}
```

### Monthly Board Report

```json
{
  "action": "generate_board_report",
  "month": "2026-01",
  "include_projections": true
}
```

**Response**:
```json
{
  "board_report": {
    "period": "January 2026",
    "executive_summary": {
      "one_liner": "January exceeded all targets with 23% YoY revenue growth and best-ever efficiency metrics.",
      "highlights": [
        "Revenue: $425,000 (+23% YoY, +8% vs target)",
        "Ad Spend: $58,000 (under budget by 12%)",
        "ROAS: 7.3x (company best, +1.2x vs last year)",
        "Market Share: Est. 12.4% (+1.1pp)"
      ]
    },
    "financial_performance": {
      "revenue": {"actual": 425000, "target": 393500, "variance_pct": 8.0},
      "spend": {"actual": 58000, "budget": 66000, "variance_pct": -12.1},
      "profit_contribution": {"actual": 367000, "margin_pct": 86.4}
    },
    "strategic_initiatives": [
      {"initiative": "DSP Expansion", "status": "on_track", "progress": 75},
      {"initiative": "International Launch", "status": "delayed", "progress": 40, "note": "Awaiting legal approval"}
    ],
    "competitive_landscape": {
      "market_position": 3,
      "share_trend": "growing",
      "key_competitor_moves": ["Competitor A increased spend 20%", "Competitor B launched new product line"]
    },
    "forward_outlook": {
      "february_projection": {"revenue": 440000, "confidence": 0.85},
      "q1_projection": {"revenue": 1280000, "vs_target": "+5%"},
      "risks": ["Competitor price war", "Seasonal slowdown in March"],
      "opportunities": ["Prime Day prep", "New category expansion"]
    }
  }
}
```

## Stakeholder Adaptation

The Executive Reporter automatically adjusts content based on the audience:

| Audience | Focus | Detail Level | Metrics Emphasized |
|----------|-------|--------------|-------------------|
| CEO | Strategic, high-level | Minimal | Revenue, Market Share, Growth |
| CFO | Financial, efficiency | Moderate | ROI, Spend, Margins, Budgets |
| CMO | Marketing effectiveness | Moderate | Brand metrics, Customer acquisition |
| Board | Governance, risk | High-level | YoY comparison, Projections, Risks |
| Investors | Growth story | Minimal | TAM, Growth rates, Unit economics |

## Integration Points

**Consumes from**:
- **grok-admaster-operator**: Raw campaign metrics
- **knowledge-synthesizer**: Market trends and insights
- **simulation-lab**: Projections and forecasts
- **narrative-architect**: Story templates and tone adaptation

**Outputs to**:
- Email distribution lists
- Slack/Teams channels
- Dashboard widgets
- Scheduled report delivery

## Automation Schedules

```json
{
  "schedules": [
    {"report": "daily_briefing", "time": "07:00", "recipients": ["ceo@company.com"]},
    {"report": "weekly_summary", "day": "Monday", "time": "08:00", "recipients": ["leadership@company.com"]},
    {"report": "monthly_board", "day": 1, "time": "09:00", "recipients": ["board@company.com"]}
  ]
}
```

## Files

```
.agent/skills/executive-reporter/
├── SKILL.md
├── scripts/
│   ├── report_generator.py
│   ├── chart_builder.py
│   └── email_dispatcher.py
└── templates/
    ├── daily_briefing.html
    ├── weekly_report.html
    └── board_presentation.pptx
```

---

**This skill turns data into decisions by speaking the language executives understand.**
