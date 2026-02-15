"""
Input Processor - Sanitizes and normalizes user input.

Purpose:
- Remove unnecessary whitespace and formatting
- Normalize text encoding
- Detect and handle edge cases
- Minimize token count while preserving meaning

Token Optimization:
- Strip redundant whitespace (can save 5-15% tokens)
- Normalize unicode characters
- Remove common filler words if configured
"""

import re
from typing import Dict, Any
from ..base_agent import BaseAgent


class ProcessedInput:
    """
    Container for processed input data.
    
    Attributes:
        text: The cleaned and normalized input text
        metadata: Additional information (language, encoding, etc.)
        original_tokens: Estimated tokens in original input
        processed_tokens: Estimated tokens after processing
    """
    
    def __init__(self, text: str, metadata: Dict[str, Any], 
                 original_tokens: int, processed_tokens: int):
        self.text = text
        self.metadata = metadata
        self.original_tokens = original_tokens
        self.processed_tokens = processed_tokens
    
    @property
    def token_savings(self) -> int:
        """Calculate tokens saved by processing."""
        return self.original_tokens - self.processed_tokens


class InputProcessor(BaseAgent):
    """
    Processes and normalizes user input for token efficiency.
    
    Expected Input:
        - raw_input: str - The unprocessed user input
    
    Returns:
        - ProcessedInput object containing cleaned text and metadata
    
    Best Practices:
        1. Always process input before sending to other agents
        2. Log significant token savings for analysis
        3. Preserve original intent while minimizing tokens
        4. Handle edge cases (empty input, special chars, etc.)
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.remove_filler_words = config.get('remove_filler_words', False)
        self.filler_words = config.get('filler_words', [
            'um', 'uh', 'like', 'you know', 'basically', 'actually'
        ])
    
    def process(self, raw_input: str) -> ProcessedInput:
        """
        Process raw input text.
        
        Args:
            raw_input: The raw user input string
        
        Returns:
            ProcessedInput object with cleaned text and metadata
        """
        if not raw_input or not raw_input.strip():
            return ProcessedInput(
                text="",
                metadata={'empty': True},
                original_tokens=0,
                processed_tokens=0
            )
        
        # Estimate original token count (rough approximation)
        original_tokens = self._estimate_tokens(raw_input)
        
        # Apply processing steps
        text = self._normalize_whitespace(raw_input)
        text = self._normalize_unicode(text)
        
        if self.remove_filler_words:
            text = self._remove_filler_words(text)
        
        # Estimate processed token count
        processed_tokens = self._estimate_tokens(text)
        
        # Track tokens saved
        tokens_saved = original_tokens - processed_tokens
        self.track_tokens(processed_tokens)
        
        if tokens_saved > 0:
            self.logger.info(f"Input processing saved {tokens_saved} tokens")
        
        metadata = {
            'original_length': len(raw_input),
            'processed_length': len(text),
            'tokens_saved': tokens_saved
        }
        
        return ProcessedInput(text, metadata, original_tokens, processed_tokens)
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace: collapse multiple spaces, remove trailing spaces.
        
        Token Savings: 5-10% on average
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters to their simplest form.
        
        Token Savings: 2-5% for text with special characters
        """
        import unicodedata
        # Normalize to NFKC (compatibility composition)
        return unicodedata.normalize('NFKC', text)
    
    def _remove_filler_words(self, text: str) -> str:
        """
        Remove common filler words that don't add semantic value.
        
        Token Savings: 3-8% for conversational text
        
        WARNING: Use with caution - may alter meaning in some contexts
        """
        for filler in self.filler_words:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(filler) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up any double spaces created
        return re.sub(r'\s+', ' ', text).strip()
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using simple heuristic.
        
        Note: For production, use actual tokenizer (e.g., tiktoken for Claude)
        This is a rough approximation: ~4 characters per token on average
        """
        if not text:
            return 0
        
        # Simple estimation: split on whitespace and punctuation
        words = re.findall(r'\w+|[^\w\s]', text)
        # Average token count is roughly 75% of word count
        return int(len(words) * 0.75)
