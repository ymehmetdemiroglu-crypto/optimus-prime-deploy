# Optimus Pryme Skills Library

A comprehensive collection of AI-powered skills for Amazon advertising dominance.

---

## 📊 Skills Overview

| Category | Skills Count | Status |
|----------|--------------|--------|
| **Meta-Skills (Foundation)** | 9 | ✅ Complete |
| **Domain Skills** | 6 | ✅ Complete |
| **Operator Skills** | 2 | ✅ Complete |
| **Total** | **17** | **Production Ready** |

---

## 🧠 Meta-Skills (Tier 1-3)

These are the "thinking" skills that power the intelligence layer.

### Tier 1: Foundation
| Skill | Purpose |
|-------|---------|
| **orchestrator-maestro** | Multi-skill coordination & workflow execution |
| **memory-palace** | Long-term pattern learning & historical knowledge |
| **consciousness-engine** | Self-awareness, decision auditing & diagnostics |
| **skill-creator** | Autonomous skill generation (user-approved) |

### Tier 2: High-Impact
| Skill | Purpose |
|-------|---------|
| **evolution-engine** | Genetic algorithm optimization for strategies |
| **simulation-lab** | Monte Carlo simulation & risk-free testing |
| **knowledge-synthesizer** | Cross-domain insights & trend detection |

### Tier 3: Differentiators
| Skill | Purpose |
|-------|---------|
| **meta-learner** | Adaptive learning & exploration optimization |
| **narrative-architect** | Data-to-story generation & stakeholder communication |

---

## 💼 Domain Skills

These are specialized business function skills.

| Skill | Impact Area | Key Differentiator |
|-------|-------------|-------------------|
| **executive-reporter** | Leadership Communication | Automated C-suite reports & briefings |
| **amazon-listing-optimizer** | Organic + Paid Synergy | Listing optimization that compounds both channels |
| **competitive-intelligence** | Strategic Decisions | Real-time competitor monitoring & market share |
| **financial-analyst** | Profitability | True unit economics & LTV:CAC analysis |
| **campaign-strategist** | Planning | AI-driven campaign architecture & launch planning |
| **data-scientist** | Personalized ML | Custom models trained per seller account |

---

## 🎯 Operator Skills

Core operational skills for platform interaction.

| Skill | Purpose |
|-------|---------|
| **grok-admaster-operator** | Direct Amazon PPC/DSP campaign management |
| **market-researcher** | Internet & Amazon product research |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      OPTIMUS PRYME BRAIN                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 ORCHESTRATOR-MAESTRO                     │   │
│  │            (Workflow Coordination Layer)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│           ┌──────────────────┼──────────────────┐              │
│           ▼                  ▼                  ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   MEMORY    │    │ EVOLUTION   │    │ SIMULATION  │        │
│  │   PALACE    │    │   ENGINE    │    │     LAB     │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│           │                  │                  │              │
│           └──────────────────┼──────────────────┘              │
│                              ▼                                  │
│           ┌─────────────────────────────────────┐              │
│           │         DOMAIN SKILLS LAYER          │              │
│           ├─────────────────────────────────────┤              │
│           │ • Executive Reporter                 │              │
│           │ • Amazon Listing Optimizer           │              │
│           │ • Competitive Intelligence           │              │
│           │ • Financial Analyst                  │              │
│           │ • Campaign Strategist                │              │
│           │ • Data Scientist                     │              │
│           └─────────────────────────────────────┘              │
│                              │                                  │
│                              ▼                                  │
│           ┌─────────────────────────────────────┐              │
│           │      OPERATOR SKILLS (APIs)          │              │
│           ├─────────────────────────────────────┤              │
│           │ • Grok AdMaster Operator             │              │
│           │ • Market Researcher                  │              │
│           └─────────────────────────────────────┘              │
│                              │                                  │
│                              ▼                                  │
│           ┌─────────────────────────────────────┐              │
│           │      EXTERNAL SYSTEMS                │              │
│           ├─────────────────────────────────────┤              │
│           │ • Amazon Advertising API             │              │
│           │ • Supabase Database                  │              │
│           │ • OpenRouter AI Gateway              │              │
│           └─────────────────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
.agent/skills/
├── README.md                      # This file
│
├── # Meta-Skills (Foundation)
├── orchestrator-maestro/
├── memory-palace/
├── consciousness-engine/
├── skill-creator/
├── evolution-engine/
├── simulation-lab/
├── knowledge-synthesizer/
├── meta-learner/
├── narrative-architect/
│
├── # Domain Skills
├── executive-reporter/
├── amazon-listing-optimizer/
├── competitive-intelligence/
├── financial-analyst/
├── campaign-strategist/
├── data-scientist/
│
├── # Operator Skills
├── grok-admaster-operator/
└── market-researcher/
```

---

## 🚀 Quick Start

### Invoke a Skill
```python
# Example: Get executive briefing
from skills.executive_reporter import generate_daily_briefing

briefing = generate_daily_briefing(account_id="ACC_123")
print(briefing["headline"])
```

### Chain Skills via Orchestrator
```python
# Example: Full product launch workflow
workflow = {
    "name": "Product Launch",
    "steps": [
        {"skill": "amazon-listing-optimizer", "action": "audit_listing"},
        {"skill": "competitive-intelligence", "action": "competitor_snapshot"},
        {"skill": "campaign-strategist", "action": "plan_product_launch"},
        {"skill": "simulation-lab", "action": "forecast_launch"},
        {"skill": "executive-reporter", "action": "generate_launch_report"}
    ]
}
orchestrator.execute(workflow)
```

---

## 📈 Skill Capabilities Matrix

| Skill | Reads DB | Writes DB | Uses AI | External API | Real-time |
|-------|----------|-----------|---------|--------------|-----------|
| orchestrator-maestro | ✅ | ✅ | ❌ | ❌ | ✅ |
| memory-palace | ✅ | ✅ | ❌ | ❌ | ❌ |
| consciousness-engine | ✅ | ✅ | ❌ | ❌ | ✅ |
| skill-creator | ❌ | ✅ | ✅ | ❌ | ❌ |
| evolution-engine | ✅ | ✅ | ❌ | ❌ | ❌ |
| simulation-lab | ✅ | ✅ | ❌ | ❌ | ❌ |
| knowledge-synthesizer | ✅ | ✅ | ✅ | ✅ | ❌ |
| meta-learner | ✅ | ✅ | ❌ | ❌ | ✅ |
| narrative-architect | ✅ | ❌ | ✅ | ❌ | ❌ |
| executive-reporter | ✅ | ❌ | ✅ | ✅ | ❌ |
| amazon-listing-optimizer | ✅ | ❌ | ✅ | ✅ | ❌ |
| competitive-intelligence | ✅ | ✅ | ✅ | ✅ | ✅ |
| financial-analyst | ✅ | ✅ | ❌ | ✅ | ❌ |
| campaign-strategist | ✅ | ✅ | ✅ | ❌ | ❌ |
| data-scientist | ✅ | ✅ | ✅ | ❌ | ✅ |
| grok-admaster-operator | ✅ | ✅ | ✅ | ✅ | ✅ |
| market-researcher | ❌ | ✅ | ✅ | ✅ | ❌ |

---

## 🔮 Roadmap

### Phase 4 (Current): Core Implementation
- [x] Meta-Skills Architecture
- [x] Domain Skills Definitions
- [x] API Layer
- [x] Core Scripts

### Phase 5: Deep Integration
- [ ] Live Amazon API connections
- [ ] Real-time data pipelines
- [ ] ML model training infrastructure

### Phase 6: Advanced Features
- [ ] Voice interface integration
- [ ] Mobile app support
- [ ] White-label capabilities

---

**Total Skills: 17 | APIs Exposed: 5 Engines | Database Tables: 14**

*Optimus Pryme: The most intelligent Amazon advertising platform ever built.*
