# Code Reviewer Agent

## Role
Analyzes code quality, identifies bugs, and suggests improvements with minimal token usage through intelligent static analysis.

## Purpose
- Review code for bugs, anti-patterns, and issues
- Suggest improvements and optimizations
- Check code style and best practices
- Generate actionable review reports
- Integrate into CI/CD pipelines for automated quality gates

## Token Optimization Strategy

- **Focus on Changes**: Prioritize reviewing changed code sections to save tokens
  - Git diff integration: Only review modified lines
  - Estimated savings: 70-90% on incremental reviews
  
- **AST Parsing**: Use static analysis (AST) instead of full AI review for simple checks
  - Pattern matching: `print()`, bare `except`, long lines
  - Savings: 70-80% vs full AI analysis
  
- **Cache Rules**: Reuse common patterns and rules to avoid redundant processing
  - Rule templates stored in config
  - Savings: 20-30% through rule reuse
  
- **Batch Processing**: Review multiple files in a single pass
  - Aggregate issues before generating report
  - Savings: 30-40% vs individual file reviews

## Capabilities

### Issue Severity Levels

| Level | Priority | Fix Timeline | Examples |
|-------|----------|--------------|----------|
| **CRITICAL** | Must fix | Immediate | `eval()`, SQL injection, hardcoded secrets |
| **HIGH** | Should fix | This sprint | Bare `except`, memory leaks, race conditions |
| **MEDIUM** | Nice to fix | Next sprint | Long functions, missing docstrings |
| **LOW** | Optional | Backlog | Line length, minor style issues |
| **INFO** | FYI | N/A | Code metrics, complexity scores |

### Core Operations

#### 1. review_code(code, file_path)

Performs comprehensive automated code analysis:

**Static Checks**:
```python
# Examples of detected issues:

# ❌ CRITICAL: Security risk
eval(user_input)  # Use ast.literal_eval() instead

# ❌ HIGH: Error handling
try:
    risky_operation()
except:  # Catches all exceptions, including SystemExit
    pass

# ⚠️ MEDIUM: Debugging code left in
print("Debug:", user_data)  # Use logger.debug() instead

# 💡 LOW: Style issue
def very_long_function_name_that_exceeds_recommended_length():
    pass  # Line > 100 characters
```

**Complexity Analysis**:
```python
# Cyclomatic complexity score: 15 (threshold: 10)
def complex_function(x, y, z):
    if x and y or z:  # +2
        for item in items:  # +1
            if condition1 and condition2:  # +2
                while loop_condition:  # +1
                    # ... more nesting
```

**Security Checks**:
```python
# Detected vulnerabilities:
exec(code_from_api)  # Code injection risk
password = "admin123"  # Hardcoded credential
query = f"SELECT * FROM users WHERE id={user_id}"  # SQL injection
```

**Performance Checks**:
```python
# String concatenation in loop (slow)
result = ""
for item in large_list:
    result += str(item)  # O(n²) time complexity

# Better: Use join()
result = "".join(str(item) for item in large_list)  # O(n)
```

#### 2. suggest_improvements(code)

Identifies maintainability and quality issues:

**Technical Debt Detection**:
```python
# TODO: Implement proper error handling
# FIXME: This breaks on edge cases
# HACK: Temporary workaround for bug #123
```

**Function Length Analysis**:
```python
# Function with 150 lines (threshold: 50)
def do_everything():
    # ... 150 lines of mixed responsibilities
    
# Suggestion: Extract into smaller, focused functions
def validate_input():
    pass

def process_data():
    pass

def generate_output():
    pass
```

**Code Duplication**:
```python
# Duplicated pattern detected 5 times:
user_data = json.loads(request.body)
if 'name' not in user_data:
    return {"error": "Missing name"}
# ... repeated across multiple endpoints

# Suggestion: Extract to utility function
def validate_required_fields(data, fields):
    for field in fields:
        if field not in data:
            return {"error": f"Missing {field}"}
    return None
```

#### 3. generate_report()

Creates structured, actionable reports:

```
======================================================================
CODE REVIEW REPORT
======================================================================

Total Issues Found: 23

🔴 CRITICAL: 2
🟡 HIGH: 5
🟢 MEDIUM: 11
🟢 LOW: 5

======================================================================
DETAILED FINDINGS:
======================================================================

CRITICAL ISSUES:
----------------------------------------------------------------------

📁 File: auth/login.py
📍 Line: 45
🏷️  Category: security
❗ Issue: Use of eval() detected - security risk
💡 Suggestion: Avoid eval(). Use ast.literal_eval() or safer alternatives
   Code:
   > result = eval(user_input)

----------------------------------------------------------------------

HIGH ISSUES:
----------------------------------------------------------------------

📁 File: utils/parser.py
📍 Line: 32
🏷️  Category: error_handling
❗ Issue: Bare except clause catches all exceptions
💡 Suggestion: Catch specific exceptions instead of bare except
   Code:
   > except:

======================================================================
```

## Advanced Features

### Custom Rule Definition

```python
# Define custom project-specific rules
custom_rules = [
    {
        'name': 'no_global_state',
        'severity': IssueSeverity.HIGH,
        'check': lambda line, num, path: (
            CodeIssue(
                file_path=path,
                line_number=num,
                severity=IssueSeverity.HIGH,
                category='architecture',
                description='Global variable detected',
                suggestion='Use dependency injection or class attributes',
                code_snippet=line.strip()
            ) if line.strip().startswith('global ') else None
        )
    },
    {
        'name': 'enforce_type_hints',
        'severity': IssueSeverity.MEDIUM,
        'check': lambda line, num, path: (
            CodeIssue(
                file_path=path,
                line_number=num,
                severity=IssueSeverity.MEDIUM,
                category='maintainability',
                description='Function missing type hints',
                suggestion='Add type hints for better IDE support',
                code_snippet=line.strip()
            ) if 'def ' in line and '->' not in line and '__init__' not in line else None
        )
    }
]

# Initialize with custom rules
reviewer = CodeReviewer({
    'rules_enabled': True,
    'custom_rules': custom_rules
})
```

### Incremental Review (Git Integration)

```python
import subprocess

def review_git_diff(branch='main'):
    """Review only changed files since branch."""
    # Get modified files
    result = subprocess.run(
        ['git', 'diff', '--name-only', branch],
        capture_output=True,
        text=True
    )
    
    changed_files = result.stdout.strip().split('\n')
    
    reviewer = CodeReviewer(config)
    
    for file_path in changed_files:
        if file_path.endswith('.py'):
            with open(file_path, 'r') as f:
                code = f.read()
            
            issues = reviewer.review_code(code, file_path)
            
    return reviewer.generate_report()
```

### CI/CD Integration

#### GitHub Actions

```yaml
name: Code Review

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run Code Reviewer
        run: |
          python -m agents.code_reviewer --target ${{ github.event.pull_request.head.sha }}
      
      - name: Comment on PR
        if: failure()
        uses: actions/github-script@v5
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: 'Code review found issues. Check the logs.'
            })
```

#### Pre-commit Hook

```python
#!/usr/bin/env python
# .git/hooks/pre-commit

from agents.code_reviewer import CodeReviewer

reviewer = CodeReviewer({'rules_enabled': True})

# Get staged files
staged_files = get_staged_python_files()

for file_path in staged_files:
    with open(file_path) as f:
        code = f.read()
    
    issues = reviewer.review_code(code, file_path)
    critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
    
    if critical_issues:
        print(f"❌ CRITICAL issues found in {file_path}")
        print(reviewer.generate_report())
        exit(1)

print("✅ Code review passed")
```

## Integration Patterns

### With QA Agent

```python
# Validate that review doesn't miss known issues
qa = QAAgent(config)
reviewer = CodeReviewer(config)

def test_reviewer_catches_security_issues():
    code_with_eval = "result = eval(user_input)"
    issues = reviewer.review_code(code_with_eval, "test.py")
    
    assert any(i.severity == IssueSeverity.CRITICAL for i in issues)
    assert any('eval' in i.description for i in issues)

qa.add_test_case(
    "Security Check",
    "Reviewer catches eval() usage",
    test_reviewer_catches_security_issues
)
```

### Batch Review Pipeline

```python
def review_entire_codebase(root_dir):
    """Review all Python files in a directory."""
    reviewer = CodeReviewer(config)
    
    for root, dirs, files in os.walk(root_dir):
        # Skip virtual environments and build dirs
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', 'build']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    code = f.read()
                
                reviewer.review_code(code, file_path)
    
    report = reviewer.generate_report()
    
    # Export to JSON for analysis
    stats = {
        'total_files': len(files),
        'total_issues': len(reviewer.issues),
        'critical': reviewer.critical_count,
        'high': reviewer.high_count,
    }
    
    return report, stats
```

## Best Practices

### 1. Prioritize Impact
```python
# Sort issues by severity and potential impact
issues_sorted = sorted(
    reviewer.issues,
    key=lambda i: (
        ['info', 'low', 'medium', 'high', 'critical'].index(i.severity.value),
        -i.line_number  # Higher line numbers last
    ),
    reverse=True
)

# Address critical and high first
for issue in issues_sorted:
    if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]:
        print(f"⚠️  {issue.description} at {issue.file_path}:{issue.line_number}")
```

### 2. Be Specific
```python
# ✅ Good suggestion
CodeIssue(
    description="String concatenation in loop is O(n²)",
    suggestion="Use ''.join() for O(n) performance:\nresult = ''.join(items)",
    code_snippet=bad_code
)

# ❌ Vague suggestion
CodeIssue(
    description="Performance issue",
    suggestion="Make it faster"
)
```

### 3. Safety First
```python
// Security issues take precedence
priority_order = [
    IssueSeverity.CRITICAL,  # Security, data loss
    IssueSeverity.HIGH,      # Reliability, performance
    IssueSeverity.MEDIUM,    # Maintainability
    IssueSeverity.LOW,       # Style
]
```

### 4. Balanced Approach
```python
# Combine static analysis with targeted AI review
def hybrid_review(code, file_path):
    # 1. Fast static checks (no tokens)
    static_issues = reviewer.review_code(code, file_path)
    
    # 2. Only use AI for complex analysis if needed
    if has_complex_logic(code):
        ai_issues = deep_ai_review(code)  # Expensive
        static_issues.extend(ai_issues)
    
    return static_issues
```

## Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Review Speed | <100ms per file | Static analysis only |
| Token Usage | <200 tokens/file | For simple files |
| False Positive Rate | <10% | Manual validation |
| Critical Issue Detection | >95% | Security test suite |

## Common Patterns

### Pattern 1: Pre-deployment Gate

```python
def deployment_check():
    reviewer = CodeReviewer(config)
    report = review_entire_codebase('./src')
    
    if reviewer.critical_count > 0:
        raise DeploymentBlockedError(
            f"Found {reviewer.critical_count} critical issues"
        )
```

### Pattern 2: Progressive Review

```python
# Review in waves: critical first, then cleanup
def review_in_phases():
    reviewer = CodeReviewer(config)
    
    # Phase 1: Security only
    reviewer.enabled_categories = ['security']
    security_report = reviewer.generate_report()
    
    # Phase 2: Add performance
    reviewer.enabled_categories = ['security', 'performance']
    full_report = reviewer.generate_report()
```

## Troubleshooting

### High False Positive Rate
- **Cause**: Over-aggressive rules
- **Solution**: Tune rule thresholds, add exception patterns
```python
# Add exceptions for specific patterns
reviewer.ignored_patterns = [
    r'# noqa',  # Explicitly ignored
    r'test_.*\.py',  # Test files
]
```

### Missing Issues
- **Cause**: Incomplete rule set
- **Solution**: Add custom rules, enable AI-assisted review for edge cases

### Slow Performance
- **Cause**: Reviewing too many files with AI
- **Solution**: Use static analysis first, batch AI reviews
```python
config['ai_review_threshold'] = 100  # Only use AI for files >100 lines
```

## Related Agents

- Use **QAAgent** to validate reviewer accuracy
- Use **TokenOptimizer** to monitor review costs
- Integrate with **ContextManager** for historical issue tracking
