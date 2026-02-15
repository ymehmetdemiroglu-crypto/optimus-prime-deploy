---
name: market-researcher
description: Conducts market research using Internet search, Amazon product search, and deep AI research analysis. Use this when the user needs information about competitors, product trends, or specific Amazon ASINs.
---

# Market Researcher

This skill allows you to gather intelligence on the Amazon marketplace.

## Tools

### 1. Internet Search
Performs a general web search using Tavily. Good for news, broader trends, and off-Amazon signals.
**Command:**
```bash
python .agent/skills/market-researcher/scripts/research.py internet_search --query "<query>"
```

### 2. Amazon Product Search
Searches Amazon directly for products matching a query. Returns price, reviews, and ranking info.
**Command:**
```bash
python .agent/skills/market-researcher/scripts/research.py amazon_search --query "<query>"
```

### 3. Product Details (ASIN)
Fetches detailed specifications, features, and descriptions for a specific ASIN.
**Command:**
```bash
python .agent/skills/market-researcher/scripts/research.py product_details --query "<ASIN>"
```

### 4. Deep Research (Agent)
Runs a multi-step AI research agent (Deep Research) to answer complex questions.
**Command:**
```bash
python .agent/skills/market-researcher/scripts/research.py deep_research --query "<complex question>"
```

## Examples

**User:** "Find me the top selling yoga mats."
**Action:**
```bash
python .agent/skills/market-researcher/scripts/research.py amazon_search --query "yoga mats"
```

**User:** "What are the specs for ASIN B0DWK3C1R7?"
**Action:**
```bash
python .agent/skills/market-researcher/scripts/research.py product_details --query "B0DWK3C1R7"
```

**User:** "Research the latest trends in sustainable packaging for 2026."
**Action:**
```bash
python .agent/skills/market-researcher/scripts/research.py internet_search --query "sustainable packaging trends 2026"
```
