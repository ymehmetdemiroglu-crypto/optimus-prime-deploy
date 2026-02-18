# QA Agent (Quality Assurance)

## Role
Tests and validates agent outputs and system behavior.

## Purpose
- Test agent responses for quality and correctness.
- Validate output formats against schemas.
- Run integration tests across the agent pipeline.
- Generate test reports and coverage metrics.

## Token Optimization Strategy
- **Batch Testing**: Group similar test cases to reduce API overhead.
- **Mocking**: Use mock responses for unit tests to save 90%+ tokens.
- **Schema Caching**: Reuse validation schemas to avoid regeneration.

## Capabilities

### Operations
1. **test_agent(agent, test_cases)**:
   - Run a list of `TestCase` items against an agent.
   - Compare `actual_output` vs `expected_output`.
   - Record Pass/Fail/Error status.
2. **validate_output(output, schema)**:
   - Check if output adheres to expected types (String, Dict, List) or specific schemas.
3. **run_integration_test(pipeline, input)**:
   - Pass input through a chain of agents.
   - Verify each step succeeds and passes data correctly.
4. **generate_report()**:
   - Produce a summary of passed/failed tests and success rate.

## Best Practices
1. **Deterministic Tests**: Ensure tests yield consistent results.
2. **Mock External Calls**: Don't waste tokens calling live APIs for basic logic tests.
3. **Validate Failures**: Ensure the system handles errors gracefully (negative testing).
