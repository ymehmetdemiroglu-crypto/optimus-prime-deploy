---
description: A workflow for autonomous research on Amazon DSP, PPC, and API updates.
---

# Autonomous Research Workflow

This workflow is used when the requirements involve external knowledge or specific Amazon strategies that are not in the current codebase.

## 1. Define Research Goals
1. Identify specific keywords or technologies (e.g., "Amazon DSP bidding strategies 2025", "FastAPI background tasks best practices").

## 2. Execute Search
1. Use `search_web` to find the latest documentation and articles.
2. Select high-quality sources (documentation, professional blogs, GitHub repos).

## 3. Extract & Synthesize
1. Use `read_url_content` or `browser_subagent` to extract relevant details from the sources.
2. Summarize the findings into a `docs/research/` file to maintain a record of the discovered knowledge.

## 4. Integrate findings
1. Apply the learned information to the implementation plan or directly into the code.
