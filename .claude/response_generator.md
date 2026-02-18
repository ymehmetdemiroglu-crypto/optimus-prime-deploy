# Response Generator Agent

## Role
Creates AI responses while enforcing token budgets and maximizing user experience.

## Purpose
- Generate responses using the Claude API.
- Enforce token budgets for generation.
- Stream responses.
- Apply intent-specific optimizations (temperature, system prompts).

## Token Optimization Strategy
- **Streaming**: Allows for early stopping if the response goes off-track.
- **Dynamic Budgets**: Adjust `max_tokens` based on remaining quota.
- **Intent-Based Tuning**: Lower temperature for factual queries (concise), higher for creative (verbose).

## Capabilities

### Operations
1. **process(intent, context, budget)**:
   - **Select Temperature**: 
     - Question/Command: Low (0.2 - 0.3)
     - Conversation: High (0.8)
   - **Build Prompt**: Combine system instructions, conversation summary, and recent messages.
   - **Generate**: Call API (mocked in codebase) with streaming or standard mode.
   - **Track**: Record token usage for prompt and completion.

### System Instructions
Adapts personality based on intent:
- **Command**: Action-oriented, concise.
- **Question**: Helpful, accurate.
- **Conversation**: Friendly, engaging.

## Best Practices
1. **Streaming**: Enable for perceived speed and control.
2. **Conciseness**: Use lower temperature and strict prompts for functional tasks to save tokens.
3. **Context Limiting**: Only include recent messages relevant to the current intent.
