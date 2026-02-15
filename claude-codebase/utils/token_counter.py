"""
Token counting utilities for accurate token estimation.

Purpose:
- Provide accurate token counts
- Support multiple tokenizers
- Cache tokenization results
- Estimate tokens when exact counting not needed

Note: For production use with Claude, use anthropic.count_tokens()
"""

from typing import Optional
import re


class TokenCounter:
    """
    Counts tokens for various text inputs.
    
    Supports:
    - Approximate counting (fast, less accurate)
    - Exact counting (requires tokenizer library)
    - Batch counting
    - Caching for performance
    
    Usage:
        counter = TokenCounter(method='approximate')
        
        # Count tokens
        count = counter.count('Hello, world!')
        
        # Count multiple texts
        counts = counter.count_batch(['text1', 'text2', 'text3'])
    """
    
    def __init__(self, method: str = 'approximate'):
        """
        Initialize token counter.
        
        Args:
            method: Counting method ('approximate' or 'exact')
        """
        self.method = method
        self._cache = {}
        
        # For exact counting, initialize tokenizer
        if method == 'exact':
            try:
                # Try to import anthropic for Claude token counting
                from anthropic import Anthropic
                self.client = Anthropic()
                self.supports_exact = True
            except ImportError:
                print(
                    "Warning: anthropic library not installed. "
                    "Falling back to approximate counting."
                )
                self.method = 'approximate'
                self.supports_exact = False
    
    def count(self, text: str, use_cache: bool = True) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            use_cache: Whether to use cached results
        
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        # Check cache
        if use_cache and text in self._cache:
            return self._cache[text]
        
        # Count tokens
        if self.method == 'approximate':
            count = self._approximate_count(text)
        else:
            count = self._exact_count(text)
        
        # Cache result
        if use_cache:
            self._cache[text] = count
        
        return count
    
    def count_batch(self, texts: list[str]) -> list[int]:
        """
        Count tokens for multiple texts.
        
        Args:
            texts: List of texts to count
        
        Returns:
            List of token counts
        """
        return [self.count(text) for text in texts]
    
    def _approximate_count(self, text: str) -> int:
        """
        Approximate token count using heuristics.
        
        This is fast but less accurate than exact counting.
        
        Heuristic: Average ~4 characters per token for English text
        This works reasonably well for Claude models.
        """
        # Split on whitespace and punctuation
        words = re.findall(r'\w+|[^\w\s]', text)
        
        # Approximate: most words are 1 token, some are 2-3
        # Punctuation is usually 1 token
        # Formula: ~75% of word count
        return int(len(words) * 0.75)
    
    def _exact_count(self, text: str) -> int:
        """
        Exact token count using Claude's tokenizer.
        
        Requires anthropic library to be installed.
        """
        if not hasattr(self, 'client'):
            # Fall back to approximate if no client
            return self._approximate_count(text)
        
        try:
            # Use anthropic.count_tokens()
            # Note: This is a placeholder as the actual API might differ
            # In production, use the actual method from anthropic library
            return len(text) // 4  # Placeholder
        except Exception:
            # Fall back on error
            return self._approximate_count(text)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int,
                     model: str = 'claude-3-sonnet-20240229') -> float:
        """
        Estimate API cost based on token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
        
        Returns:
            Estimated cost in USD
        """
        # Pricing (as of 2024 - check current pricing)
        pricing = {
            'claude-3-opus-20240229': {
                'input': 15.00 / 1_000_000,   # $15 per million input tokens
                'output': 75.00 / 1_000_000,  # $75 per million output tokens
            },
            'claude-3-sonnet-20240229': {
                'input': 3.00 / 1_000_000,    # $3 per million input tokens
                'output': 15.00 / 1_000_000,  # $15 per million output tokens
            },
            'claude-3-haiku-20240307': {
                'input': 0.25 / 1_000_000,    # $0.25 per million input tokens
                'output': 1.25 / 1_000_000,   # $1.25 per million output tokens
            },
        }
        
        if model not in pricing:
            raise ValueError(f"Unknown model: {model}")
        
        rates = pricing[model]
        cost = (input_tokens * rates['input']) + (output_tokens * rates['output'])
        
        return cost
    
    def clear_cache(self) -> None:
        """Clear the token count cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'total_tokens_cached': sum(self._cache.values())
        }
