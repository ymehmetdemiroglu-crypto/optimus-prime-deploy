# Agents System for Claude-based AI

This module implements a specialized multi-agent system designed for token usage efficiency and intelligent conversation handling.

## Core Hierarchy

All agents inherit from `BaseAgent`, which provides:
- **Consistent interface** (`process(input_data)`) - Uniform API across all agents
- **Shared token tracking** (`track_tokens()`) - Centralized monitoring
- **Configuration dependency injection** - Flexible, testable design
- **Logging and monitoring** - Detailed observability

### BaseAgent Interface

```python
class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None)
    
    @abstractmethod
    def process(self, input_data: Any) -> Any
    
    def track_tokens(self, tokens_used: int) -> None
    def get_stats(self) -> Dict[str, Any]
    def reset_stats(self) -> None
```

## Common Properties

Every agent in this system must:

1. **Track Token Usage**: Monitor tokens consumed per operation
   - Count both input and output tokens
   - Report to TokenOptimizer for centralized tracking
   
2. **Log Operations**: Maintain detailed logs of activities
   - Debug: Detailed internal state
   - Info: Key operations and results
   - Warning: Budget alerts and anomalies
   
3. **Be Configurable**: Accept a configuration dictionary during initialization
   - Max tokens, cache settings, feature flags
   - Environment-specific overrides
   
4. **Implement Process Method**: Define specific logic in the `process` method
   - Input validation
   - Core processing logic
   - Output formatting

## Token Optimization Strategy

The system is built with token economy in mind:

- **Track**: Measure usage at every step
  - Real-time monitoring via TokenOptimizer
  - Per-agent and per-operation granularity
  
- **Log**: Identify expensive operations
  - Threshold alerts (e.g., >5000 tokens)
  - Trend analysis over time
  
- **Enforce**: Provide hooks for budget enforcement
  - Pre-flight budget checks
  - Graceful degradation when approaching limits
  
- **Optimize**: Specific agents (like TokenOptimizer) provide recommendations
  - Caching opportunities
  - Compression strategies
  - Response truncation

## Agent Pipeline Architecture

### Standard Processing Flow

```
User Input
    ↓
InputProcessor (sanitize, normalize)
    ↓
IntentClassifier (categorize, cache lookup)
    ↓
ContextManager (retrieve relevant history)
    ↓
ResponseGenerator (create response with budget)
    ↓
QAAgent (validate output quality)
    ↓
Output to User

← TokenOptimizer (monitors all steps) →
```

### Agent Interaction Patterns

**Sequential Processing**:
```python
input_processor = InputProcessor(config)
intent_classifier = IntentClassifier(config)
context_manager = ContextManager(config)

# Process sequentially
processed = input_processor.process(raw_input)
intent = intent_classifier.process(processed)
context = context_manager.get_context()
```

**Parallel Validation**:
```python
code_reviewer = CodeReviewer(config)
qa_agent = QAAgent(config)

# Both can validate the same code independently
review_issues = code_reviewer.review_code(code, file_path)
test_results = qa_agent.test_agent(target_agent, test_cases)
```

## Configuration Management

### Standard Config Structure

```python
agent_config = {
    # Token management
    'max_tokens': 50000,
    'cache_enabled': True,
    
    # Logging
    'log_level': 'INFO',
    
    # Agent-specific settings
    'window_size': 20,  # ContextManager
    'confidence_threshold': 0.7,  # IntentClassifier
    'auto_fix': False,  # CodeReviewer
}
```

### Environment-Specific Overrides

```python
# Development: More verbose, mock APIs
dev_config = base_config.copy()
dev_config.update({
    'log_level': 'DEBUG',
    'mock_mode': True,
})

# Production: Strict budgets, real APIs
prod_config = base_config.copy()
prod_config.update({
    'max_tokens': 100000,
    'alert_threshold': 0.8,
    'mock_mode': False,
})
```

## Integration Patterns

### 1. Single-Agent Usage

```python
# Simple review task
reviewer = CodeReviewer(config, logger)
issues = reviewer.review_code(code, "main.py")
report = reviewer.generate_report()
```

### 2. Multi-Agent Pipeline

```python
# Full conversation pipeline
def process_user_message(raw_message: str) -> str:
    # 1. Process input
    processed = input_processor.process(raw_message)
    
    # 2. Classify intent
    intent = intent_classifier.process(processed)
    
    # 3. Add to context if needed
    if intent.requires_context:
        context_manager.add_message('user', processed.text)
        context = context_manager.get_context()
    else:
        context = None
    
    # 4. Generate response
    response = response_generator.process(intent, context)
    
    # 5. Track usage
    token_optimizer.track_usage(
        'pipeline',
        'process_message',
        response.tokens_used
    )
    
    return response.content
```

### 3. Quality Assurance Integration

```python
# Continuous validation
def deploy_with_qa(agent, test_suite):
    qa = QAAgent(config)
    report = qa.test_agent(agent, test_suite)
    
    if report.success_rate < 95:
        raise DeploymentError(
            f"Agent failed QA: {report.success_rate}% pass rate"
        )
    
    return True
```

## Performance Metrics

### Key Performance Indicators (KPIs)

1. **Token Efficiency**: Tokens per conversation turn
   - Target: <5000 tokens/turn
   - Alert: >10000 tokens/turn

2. **Cache Hit Rate**: % of cached classifications
   - Target: >60% for IntentClassifier
   - Monitor: Daily trends

3. **Compression Ratio**: Context size reduction
   - Target: 60-80% for long conversations
   - Method: Progressive summarization

4. **Response Quality**: QA pass rate
   - Target: >95% accuracy
   - Monitor: Per-agent statistics

## Best Practices

### 1. Always Initialize TokenOptimizer First

```python
# Initialize optimizer before other agents
optimizer = TokenOptimizer(config)

# Pass optimizer reference to other agents
config['token_optimizer'] = optimizer
```

### 2. Implement Graceful Degradation

```python
# Check budget before expensive operations
budget_check = optimizer.check_budget(estimated_tokens)

if not budget_check['within_budget']:
    # Fallback to cheaper alternative
    context_manager.compress_context()
    response_generator.max_tokens = 2000  # Reduce
```

### 3. Log Token Usage Consistently

```python
# Standard pattern
tokens_before = optimizer.total_usage
result = some_expensive_operation()
tokens_used = optimizer.total_usage - tokens_before

logger.info(f"Operation used {tokens_used} tokens")
```

### 4. Test Agent Pipelines

```python
# Integration test example
def test_full_pipeline():
    test_input = "Create a function to sort a list"
    
    qa = QAAgent(config)
    result = qa.run_integration_test(
        [input_processor, intent_classifier, response_generator],
        test_input
    )
    
    assert all(r['status'] == 'success' for r in result.values())
```

## Troubleshooting

### High Token Usage
- **Check**: Which agent is consuming most tokens?
- **Action**: Review `optimizer.get_stats()` breakdown
- **Solution**: Enable caching, compression, or reduce context window

### Cache Misses
- **Check**: Cache hit rate in IntentClassifier
- **Action**: Review fuzzy matching threshold
- **Solution**: Adjust similarity threshold or improve normalization

### Context Overflow
- **Check**: ContextManager token count
- **Action**: Trigger `compress_context()` earlier
- **Solution**: Lower `summarization_threshold` or reduce `window_size`

## Cross-Reference Guide

- **Input Processing** → See `input_processor.md`
- **Intent Classification** → See `intent_classifier.md`
- **Context Management** → See `context_manager.md`
- **Response Generation** → See `response_generator.md`
- **Code Review** → See `code_reviewer.md`
- **Quality Assurance** → See `qa_agent.md`
- **Token Optimization** → See `token_optimizer.md`
