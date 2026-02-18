# Intent Classifier Agent

## Role
Determines user intent to route requests and optimize downstream processing.

## Purpose
- Classify user input into predefined categories (Question, Command, etc.).
- Route requests to appropriate handlers.
- Cache classifications to avoid expensive re-analysis.

## Token Optimization Strategy
- **Caching**: Check for similar past queries to skip processing (80-90% savings).
- **Lightweight Analysis**: Use simple keyword matching or lightweight embeddings instead of full LLM calls.
- **Context Awareness**: Flag if context is required (save tokens by not sending history when not needed).

## Capabilities

### Intent Types
- **QUESTION**: User asking for information.
- **COMMAND**: User requesting an action (create, delete, run).
- **CONVERSATION**: Casual chat.
- **CLARIFICATION**: User refining a previous request.
- **FEEDBACK**: User providing input on performance.

### Operations
1. **process(processed_input)**:
   - **Check Cache**: Look for exact or fuzzy matches.
   - **Classify**: Determine intent type and confidence score.
   - **Extract Entities**: Identify key topics or targets.
   - **Update Cache**: Store result for future use.

## Best Practices
1. **Use Caching**: Enable fuzzy matching to catch variations of the same request.
2. **Confidence Thresholds**: Only trust high-confidence classifications automatically; ask for clarification otherwise.
3. **Early Exit**: If intent doesn't require context/history, process it statelessly to save tokens.
