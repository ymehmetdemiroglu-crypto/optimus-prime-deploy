# Input Processor Agent

## Role
Sanitizes and normalizes user input to ensure consistency, efficiency, and safety before downstream processing.

## Purpose
- Remove unnecessary whitespace and formatting
- Normalize text encoding (Unicode, special characters)
- Detect and handle edge cases (empty input, malformed data)
- Minimize token count while preserving semantic meaning
- Protect against injection attacks through sanitization

## Token Optimization Strategy

- **Whitespace Normalization**: Strip redundant spaces/newlines (save 5-15%)
  - Multiple consecutive spaces → Single space
  - Leading/trailing whitespace → Removed
  - `"Hello    world  "` → `"Hello world"`
  
- **Unicode Normalization**: Convert special characters to standard forms (save 2-5%)
  - Ligatures, accents, combining characters → Standard equivalents
  - `"café"` (é as two chars) → `"café"` (é as one char)
  
- **Filler Removal**: Optionally remove conversational fillers (save 3-8%)
  - "um", "uh", "like", "you know", "basically" → Removed
  - **Warning**: May alter meaning in certain contexts

## Input/Output Contract

### Input
```python
raw_input: str  # Unprocessed user input
```

### Output
```python
ProcessedInput(
    text: str,              # Cleaned text
    metadata: Dict,         # Processing statistics
    original_tokens: int,   # Tokens before processing
    processed_tokens: int   # Tokens after processing
)
```

### Example Transformation

```python
# Input
"""
  Hello    there!  
  
  
  How   are you doing today?     
"""

# Output after processing
"Hello there! How are you doing today?"

# Metadata
{
    'original_length': 68,
    'processed_length': 39,
    'tokens_saved': 5
}
```

## Capabilities

### 1. Whitespace Normalization

```python
# Before
input = "Hello\n\n\n    world\t\tfoo    bar   "

# After
output = "Hello world foo bar"

# Savings: ~40% reduction in characters
```

**Implementation Details**:
- Regex: `r'\s+'` → `' '`
- `.strip()` for edges
- Preserves single spaces between words

### 2. Unicode Normalization

```python
# Before (NFC - Canonical Composition)
input = "café"  # c + a + f + é (4 chars)

# After (NFKC - Compatibility Composition)
output = "café"  # c + a + f + e + ´ → café (4 chars normalized)

# Benefits:
# - Consistent representation
# - Better tokenization
# - Cross-platform compatibility
```

**Normalization Forms**:
| Form | Description | Use Case |
|------|-------------|----------|
| NFC | Canonical Composition | General text |
| NFD | Canonical Decomposition | Sorting |
| NFKC | Compatibility Composition | **Recommended** |
| NFKD | Compatibility Decomposition | Search |

### 3. Filler Word Removal

```python
# Before
input = "Um, like, I basically need, you know, help with this"

# After (filler_words enabled)
output = "I need help with this"

# Tokens saved: ~7 tokens (35% reduction)
```

**Configurable Filler List**:
```python
config = {
    'remove_filler_words': True,
    'filler_words': [
        'um', 'uh', 'er', 'ah',
        'like', 'you know', 'i mean',
        'basically', 'actually', 'literally',
        'kind of', 'sort of'
    ]
}
```

**⚠️ Caution**: Some "fillers" may be semantically important:
```python
# Problem case
"I like apples" → "I apples"  # Removed verb!

# Solution: Use word boundaries
r'\blike\b(?! to| that)'  # Only remove standalone "like"
```

### 4. Edge Case Handling

```python
# Empty input
process("") → ProcessedInput(text="", metadata={'empty': True}, ...)

# Only whitespace
process("    \n\n   ") → ProcessedInput(text="", ...)

# Very long input
process("a" * 100000) → ProcessedInput(text="a" * 100000, ...) # No crash

# Special characters
process("@#$%^&*()") → Normalized and preserved

# Mixed encodings
process("Hello™®©") → Normalized to ASCII equivalents if possible
```

## Advanced Features

### Input Validation

```python
class InputProcessor(BaseAgent):
    def validate(self, text: str) -> tuple[bool, str]:
        """Validate input before processing."""
        
        # Check length limits
        if len(text) > config['max_input_length']:
            return False, "Input exceeds maximum length"
        
        # Check for suspicious patterns (injection)
        if self._contains_injection_attempt(text):
            return False, "Potentially malicious input detected"
        
        # Check encoding
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            return False, "Invalid encoding"
        
        return True, "OK"
    
    def _contains_injection_attempt(self, text: str) -> bool:
        """Detect potential injection attacks."""
        suspicious_patterns = [
            r'<script>', r'javascript:', r'onerror=',
            r'SELECT.*FROM', r'DROP TABLE', r'--',
            r'\beval\b', r'\bexec\b'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
```

### Language Detection

```python
def detect_language(text: str) -> str:
    """Detect input language for appropriate processing."""
    # Use langdetect or similar
    try:
        from langdetect import detect
        return detect(text)
    except:
        return 'en'  # Default to English

# Use case: Different filler words per language
filler_words_by_lang = {
    'en': ['um', 'like', 'you know'],
    'es': ['pues', 'bueno', 'entonces'],
    'fr': ['euh', 'ben', 'quoi'],
}
```

### Smart Truncation

```python
def smart_truncate(text: str, max_tokens: int) -> str:
    """Truncate while preserving sentence boundaries."""
    
    # Estimate tokens
    estimated_tokens = len(text) // 4
    
    if estimated_tokens <= max_tokens:
        return text
    
    # Truncate at sentence boundary
    sentences = text.split('. ')
    result = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence) // 4
        if current_tokens + sentence_tokens > max_tokens:
            break
        result.append(sentence)
        current_tokens += sentence_tokens
    
    return '. '.join(result) + '...'
```

## Integration Patterns

### With IntentClassifier

```python
# Sequential processing
processor = InputProcessor(config)
classifier = IntentClassifier(config)

# Process then classify
processed = processor.process(raw_input)
intent = classifier.process(processed)

# Benefits:
# - Classifier receives clean input
# - Better cache hits due to normalization
# - Fewer misclassifications from formatting issues
```

### With ContextManager

```python
# Add processed messages to context
processed = processor.process(user_message)
context_manager.add_message('user', processed.text)

# Benefits:
# - Cleaner conversation history
# - Better compression ratios
# - Reduced storage costs
```

### Validation Pipeline

```python
def safe_process_input(raw_input: str) -> ProcessedInput:
    """Process with validation and error handling."""
    
    processor = InputProcessor(config)
    
    # 1. Validate
    is_valid, error_msg = processor.validate(raw_input)
    if not is_valid:
        raise ValueError(f"Invalid input: {error_msg}")
    
    # 2. Process
    processed = processor.process(raw_input)
    
    # 3. Log savings
    if processed.token_savings > 0:
        logger.info(
            f"Saved {processed.token_savings} tokens "
            f"({processed.token_savings / processed.original_tokens:.1%})"
        )
    
    return processed
```

## Configuration Examples

### Minimal Processing (Fast)

```python
config = {
    'remove_filler_words': False,
    'normalize_unicode': False,  # Skip for speed
    'log_level': 'WARNING'
}

# Use case: Pre-processed input, performance critical
```

### Aggressive Optimization (Max Savings)

```python
config = {
    'remove_filler_words': True,
    'filler_words': COMPREHENSIVE_FILLER_LIST,  # 100+ words
    'normalize_unicode': True,
    'smart_truncate': True,
    'max_input_length': 10000
}

# Use case: Token budget constraints, conversational input
```

### Balanced (Recommended)

```python
config = {
    'remove_filler_words': True,
    'filler_words': ['um', 'uh', 'like'],  # Common ones only
    'normalize_unicode': True,
    'log_level': 'INFO'
}

# Use case: Production default
```

## Best Practices

### 1. Always Pre-process

```python
# ✅ Good: Process before any agent
raw = get_user_input()
processed = input_processor.process(raw)
intent = intent_classifier.process(processed)

# ❌ Bad: Skip processing
raw = get_user_input()
intent = intent_classifier.process(raw)  # Unprocessed!
```

### 2. Log Savings

```python
# Track cumulative impact
total_original = 0
total_saved = 0

for message in user_messages:
    processed = processor.process(message)
    total_original += processed.original_tokens
    total_saved += processed.token_savings

logger.info(
    f"Total savings: {total_saved} tokens "
    f"({total_saved/total_original:.1%})"
)
```

### 3. Preserve Meaning

```python
# Test that processing doesn't alter semantics
def test_semantic_preservation():
    test_cases = [
        ("I like apples", "I like apples"),  # Don't remove "like" verb
        ("  Hello  ", "Hello"),  # Remove whitespace
        ("It's $100", "It's $100"),  # Preserve special chars
    ]
    
    for input_text, expected in test_cases:
        processed = processor.process(input_text)
        assert processed.text == expected
```

### 4. Handle Encoding Issues

```python
# Gracefully handle bad encoding
def safe_decode(raw_bytes: bytes) -> str:
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # Fallback: Replace errors
    return raw_bytes.decode('utf-8', errors='replace')
```

## Performance Metrics

| Operation | Avg Time | Token Savings |
|-----------|----------|---------------|
| Whitespace norm | ~0.1ms | 5-15% |
| Unicode norm | ~0.5ms | 2-5% |
| Filler removal | ~1ms | 3-8% |
| Full processing | ~2ms | 10-25% |

## Common Patterns

### Pattern 1: Streaming Input

```python
# Process as user types (real-time)
buffer = ""

for char in user_input_stream:
    buffer += char
    
    # Process when user pauses (debounce)
    if is_pause_detected():
        processed = processor.process(buffer)
        buffer = ""
```

### Pattern 2: Batch Processing

```python
# Process multiple inputs efficiently
def process_batch(inputs: List[str]) -> List[ProcessedInput]:
    return [processor.process(inp) for inp in inputs]

# Parallel processing
from multiprocessing import Pool

with Pool(4) as p:
    results = p.map(processor.process, inputs)
```

### Pattern 3: Cached Processing

```python
# Cache processed inputs
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_process(input_hash: str) -> ProcessedInput:
    return processor.process(unhash(input_hash))

# Use hash to enable caching
import hashlib

input_hash = hashlib.md5(raw_input.encode()).hexdigest()
processed = cached_process(input_hash)
```

## Troubleshooting

### Over-aggressive Filler Removal
- **Symptom**: Important words removed
- **Solution**: Reduce filler list, use word boundaries
```python
# From: r'like' (matches "likely")
# To: r'\blike\b' (matches only "like")
```

### Unicode Errors
- **Symptom**: `UnicodeDecodeError` or garbled text
- **Solution**: Use error handling, try multiple encodings
```python
text.encode('utf-8', errors='ignore')  # Skip bad chars
```

### Performance Bottleneck
- **Symptom**: Slow processing for long inputs
- **Solution**: Reduce normalization steps, use C-based regex
```python
import regex  # Drop-in replacement, faster
```

## Related Agents

- **IntentClassifier**: Receives processed input for better classification accuracy
- **ContextManager**: Stores processed messages for cleaner history
- **TokenOptimizer**: Tracks savings from input processing
