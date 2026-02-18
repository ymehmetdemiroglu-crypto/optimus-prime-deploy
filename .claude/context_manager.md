# Context Manager Agent

## Role
Maintains conversation context efficiently, preserving important information while minimizing token usage.

## Purpose
- Track conversation history.
- Compress old context progressively.
- Maintain a sliding window of recent messages.
- Preserve important information through scoring.

## Token Optimization Strategy
- **Progressive Summarization**: Summarize older messages to save 60-80% tokens on long conversations.
- **Sliding Window**: Keep only the last N messages (e.g., 20) in full detail.
- **Importance Scoring**: Retain critical messages based on utility (e.g., user requests, questions).
- **Auto-Pruning**: Remove low-value messages when approaching token limits.

## Capabilities

### Core Objects
- **Message**: Represents a single interaction (role, content, timestamp, tokens, importance).
- **Context**: The complete state (messages, total tokens, summary).

### Operations
1. **add_message(role, content)**:
   - Adds a message to the history.
   - Calculates importance score (User > Assistant, Questions > Statements).
   - Auto-compresses if threshold reached.
2. **get_context()**:
   - Retrieves the current state for prompt generation.
3. **compress_context()**:
   - Splits messages into "recent" (window) and "history".
   - Summarizes history into a concise text block.
   - Replaces old messages with the summary.
4. **prune_context(target_tokens)**:
   - Removes least important messages until the target token count is met.

## Best Practices
1. **Checkpoints**: Create summaries for long conversations.
2. **Preserve Recent Context**: Always keep the immediate conversation window intact (10-20 messages).
3. **Score Usage**: Use importance scores to decide what to prune, not just age.
