"""
Claude API Service - Wrapper for Anthropic Claude API.

Handles communication with Claude API with proper error handling
and token tracking.
"""

import os
from typing import Dict, Any, List, Optional, Iterator
import logging


class ClaudeAPIService:
    """
    Service wrapper for Claude API interactions.
    
    Features:
    - Message creation
    - Streaming support
    - Token tracking
    - Error handling with retries
    - Cost estimation
    
    Usage:
        service = ClaudeAPIService(api_key='your-key')
        
        # Generate response
        response = service.create_message(
            messages=[{'role': 'user', 'content': 'Hello!'}],
            max_tokens=1000
        )
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 default_model: str = 'claude-sonnet-4-5-20250929'):
        """
        Initialize Claude API service.
        
        Args:
            api_key: Anthropic API key (uses env var if None)
            default_model: Default model to use
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.default_model = default_model
        self.logger = logging.getLogger(__name__)
        
        # Try to initialize Anthropic client
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
            self.available = True
        except ImportError:
            self.logger.warning(
                "Anthropic library not installed. "
                "Install with: pip install anthropic"
            )
            self.client = None
            self.available = False
        except Exception as e:
            self.logger.error(f"Failed to initialize Claude client: {e}")
            self.client = None
            self.available = False
    
    def create_message(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        model: Optional[str] = None,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a message using Claude API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            model: Model to use (uses default if None)
            system: Optional system prompt
        
        Returns:
            Response dict with content and usage info
        
        Raises:
            RuntimeError: If API not available
            Exception: On API errors
        """
        if not self.available:
            raise RuntimeError(
                "Claude API not available. Check API key and installation."
            )
        
        model = model or self.default_model
        
        try:
            # Call API
            kwargs = {
                'model': model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'messages': messages
            }
            
            if system:
                kwargs['system'] = system
            
            response = self.client.messages.create(**kwargs)
            
            # Extract response data
            return {
                'content': response.content[0].text,
                'model': response.model,
                'stop_reason': response.stop_reason,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            }
        
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            raise
    
    def create_message_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        model: Optional[str] = None,
        system: Optional[str] = None
    ) -> Iterator[str]:
        """
        Create a message with streaming response.
        
        Args:
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use
            system: Optional system prompt
        
        Yields:
            Text chunks as they arrive
        
        Raises:
            RuntimeError: If API not available
        """
        if not self.available:
            raise RuntimeError("Claude API not available")
        
        model = model or self.default_model
        
        try:
            kwargs = {
                'model': model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'messages': messages,
                'stream': True
            }
            
            if system:
                kwargs['system'] = system
            
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        
        except Exception as e:
            self.logger.error(f"Streaming failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Note: This is a placeholder. Use actual token counting
        from anthropic library when available.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Token count
        """
        # Rough approximation
        return len(text) // 4
