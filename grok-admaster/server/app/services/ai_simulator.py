"""
AI Simulator Service - Generates realistic Grok responses.
"""
import random
from typing import Optional


def generate_grok_response(message: str, context_asin: Optional[str] = None) -> str:
    """
    Generate a context-aware simulated response based on user message.
    Uses template matching and heuristics for realistic responses.
    """
    message_lower = message.lower()
    
    # ACoS-related questions
    if any(word in message_lower for word in ["acos", "advertising cost", "ad cost"]):
        return _get_acos_response(context_asin)
    
    # Competitor analysis
    if any(word in message_lower for word in ["competitor", "competition", "ranking", "compare"]):
        return _get_competitor_response(context_asin)
    
    # Bid suggestions
    if any(word in message_lower for word in ["bid", "bidding", "increase bid", "lower bid"]):
        return _get_bid_response()
    
    # Keyword questions
    if any(word in message_lower for word in ["keyword", "search term", "target"]):
        return _get_keyword_response()
    
    # Sales/Performance questions
    if any(word in message_lower for word in ["sales", "revenue", "performance", "how am i doing"]):
        return _get_performance_response()
    
    # Optimization requests
    if any(word in message_lower for word in ["optimize", "improve", "better", "help"]):
        return _get_optimization_response()
    
    # Default response
    return _get_default_response()


def _get_acos_response(context_asin: Optional[str]) -> str:
    responses = [
        """**ACoS Analysis Complete** 📊

Your current ACoS of **15.2%** is within healthy range for your category. Here's the breakdown:

• **Top performing keywords**: "premium widgets" (8.2% ACoS), "quality gadgets" (11.5% ACoS)
• **Needs attention**: "cheap widgets" is at 42% ACoS — I recommend adding as negative keyword
• **Opportunity**: Search term "widget gift set" is converting at 6% ACoS but not yet targeted

**Recommendation**: I can automatically pause keywords exceeding 30% ACoS and harvest high-performers. Shall I proceed?""",
        
        """**Understanding your ACoS spike** 📈

I noticed ACoS increased from 12.8% to 15.2% over the last 7 days. Here's why:

1. **Competitor "BrandX" increased bids** on your top 3 keywords by an estimated 20%
2. **Conversion rate dropped** 8% (likely seasonal — happens every Q4)
3. **New campaign "Summer Collection"** is still in learning phase

**Action Items**:
- Switch "Blue Widgets" campaign to **Profit Guard** mode to defend margin
- Let Summer Collection run 5 more days before optimization
- Consider matching competitor bids on "premium widget" — high intent keyword""",
    ]
    return random.choice(responses)


def _get_competitor_response(context_asin: Optional[str]) -> str:
    responses = [
        """**Competitor Intelligence Report** 🎯

Analyzing top 5 competitors in your main keyword space:

| Rank | Competitor | Est. Ad Spend | Key Strength |
|------|-----------|---------------|--------------|
| 1 | BrandX | $2,400/day | Aggressive exact match |
| 2 | GadgetPro | $1,800/day | Strong product targeting |
| 3 | **You** | $350/day | High conversion rate |
| 4 | ValueDeals | $900/day | Price advantage |
| 5 | PremiumCo | $600/day | Brand loyalty |

**Key Insight**: You're outperforming on ROAS despite lower spend. BrandX is bleeding money with 28% ACoS.

**Strategy**: Maintain current efficiency. Target BrandX's ASINs with product targeting — their reviews are dropping.""",
        
        """**Competitive Landscape Scan** 🔍

Your organic rank has improved on 12 of 15 tracked keywords this week!

• **#1 position gained**: "best widget 2024" (was #4)
• **Holding strong**: "premium gadget set" at #2
• **Watch out**: Slipping on "widget gift" — competitor launched variation

I'm adjusting ad bids to defend your #1 positions while the organic ranking solidifies. Estimated time to natural rank lock-in: **14 days**.""",
    ]
    return random.choice(responses)


def _get_bid_response() -> str:
    responses = [
        """**Bid Optimization Recommendations** 💰

Based on last 14 days of data, here are my suggested adjustments:

**Increase Bids** (High conversion, low visibility):
- "premium widget" → $1.85 → **$2.15** (+16%)
- "quality gadget pro" → $0.95 → **$1.20** (+26%)

**Decrease Bids** (Overspending):
- "widget amazon" → $2.40 → **$1.90** (-21%)
- "cheap gadgets" → $1.10 → **$0.65** (-41%)

**Expected Impact**: -$45/day spend, +$120/day attributed sales.

Want me to apply these changes now?""",
        
        """**Smart Bidding Analysis** 📊

Current bid strategy performance:

• **Dynamic - Down Only**: Working great on competitive keywords ✅
• **Fixed Bids**: Underperforming on 4 campaigns — switch to auto recommended
• **Top of Search Modifier**: +50% is too aggressive, recommend +35%

If I optimize bids using profit-first algorithm: **Projected 23% reduction in wasted spend** while maintaining sales velocity.""",
    ]
    return random.choice(responses)


def _get_keyword_response() -> str:
    return """**Keyword Opportunity Report** 🔑

**Harvesting from Search Terms** (Last 7 days):
• "widget organizer deluxe" — 12 orders, 4.2% ACoS ⭐ READY TO EXACT
• "premium gadget case" — 8 orders, 6.1% ACoS ⭐ READY TO EXACT
• "best widget for home" — 6 orders, 8.9% ACoS

**Negative Keyword Candidates**:
• "widget repair" — 145 clicks, 0 orders ❌
• "free widget sample" — 89 clicks, 0 orders ❌
• "widget wholesale bulk" — 67 clicks, 1 order (wrong customer) ❌

**Auto-Actions Available**:
1. Add top 5 search terms to exact match
2. Add negative keywords to all campaigns  
3. Create new campaign with harvested keywords

Select an option or say "do all" for full optimization."""


def _get_performance_response() -> str:
    return """**Performance Dashboard Summary** 📈

**This Week vs Last Week**:
| Metric | This Week | Last Week | Change |
|--------|-----------|-----------|--------|
| Sales | $15,424 | $13,892 | +11% ✅ |
| Ad Spend | $2,348 | $2,156 | +9% |
| ROAS | 6.57x | 6.44x | +2% ✅ |
| Orders | 487 | 441 | +10% ✅ |
| ACoS | 15.2% | 15.5% | -0.3% ✅ |

**What's Working**:
- Campaign "Blue Widgets - Exact" is your star performer (9.2% ACoS)
- Organic rank improvements driving 23% of new orders

**Attention Needed**:
- "Summer Collection" campaign needs 3 more days before I can optimize
- Inventory alert: "Premium Widget Pro" will stockout in ~8 days

Overall: **You're on track for your best month yet!** 🚀"""


def _get_optimization_response() -> str:
    return """**Grok Optimization Suite** ⚡

I can help optimize in several ways. What's your priority?

**1. 🎯 Quick Wins** (5 min)
- Pause 8 keywords with ACoS > 35%
- Add 12 negative keywords from search terms
- Adjust 5 bids based on conversion data

**2. 🚀 Growth Mode** 
- Launch attack campaigns on competitor ASINs
- Expand to 25 new long-tail keywords
- Increase budgets on winners by 20%

**3. 🛡️ Profit Protection**
- Switch to profit-first bidding
- Reduce spend on low-margin products  
- Defensive positioning against competitors

**4. 🔄 Full Audit**
- Complete account restructure recommendation
- Campaign consolidation analysis
- Keyword architecture review

Which would you like me to execute? Or tell me your specific goal!"""


def _get_default_response() -> str:
    responses = [
        """I'm here to help with your Amazon advertising! I can assist with:

• **ACoS Analysis** — Understanding and optimizing your ad costs
• **Competitor Intel** — Tracking and outmaneuvering competition  
• **Bid Optimization** — Smart adjustments for better ROAS
• **Keyword Strategy** — Harvesting winners, eliminating waste
• **Performance Reports** — Clear insights on your metrics

What would you like to explore? Or just tell me about a challenge you're facing!""",
        
        """**Grok at your service!** 🤖

I'm continuously monitoring your campaigns. Here's what I'm seeing right now:

• 6 active campaigns running smoothly
• Made 12 optimizations in the last 24 hours
• Your best performing hour: 8-9 PM EST
• Saved an estimated $47 from wasted clicks today

Ask me anything about your Amazon ads, or tell me to "optimize aggressively" if you want me to go into attack mode!""",
    ]
    return random.choice(responses)
