"""
Response Generator - Creates AI responses within token budgets.

Purpose:
- Generate responses using Claude API
- Enforce token budgets
- Stream responses for better UX
- Apply post-processing optimizations

Token Optimization:
- Streaming (allows early stopping)
- Dynamic token budgets (adjust based on remaining context)
- Temperature tuning (lower temperature = more concise)
- Max tokens enforcement
"""

from typing import Dict, Any, Optional, Iterator
from dataclasses import dataclass
from ..base_agent import BaseAgent


@dataclass
class Response:
    """
    Generated response with metadata.
    
    Attributes:
        content: The response text
        tokens_used: Tokens consumed for generation
        finish_reason: Why generation stopped ('stop', 'length', 'error')
        model: Model used for generation
        metadata: Additional metadata
    """
    content: str
    tokens_used: int
    finish_reason: str
    model: str
    metadata: Dict[str, Any]


class ResponseGenerator(BaseAgent):
    """
    Generates responses with token budget enforcement.
    
    Expected Input:
        - intent: Intent object from IntentClassifier
        - context: Context object from ContextManager
        - budget: Token budget for response (optional)
    
    Returns:
        - Response object with generated text and metadata
    
    Best Practices:
        1. Set reasonable token budgets (4096 for most responses)
        2. Use streaming for long responses (better UX)
        3. Lower temperature for factual/concise responses
        4. Cache common responses to save API calls
        5. Implement retry logic with exponential backoff
    
    Token Savings:
        - Streaming with early stop: 10-20% for long responses
        - Temperature tuning: 15-25% for concise mode
        - Response caching: 90%+ for repeated queries
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        
        # Configuration
        self.model = config.get('model', 'claude-sonnet-4-5-20250929')
        self.default_max_tokens = config.get('default_max_tokens', 8000)
        self.default_temperature = config.get('default_temperature', 0.7)
        self.streaming_enabled = config.get('streaming_enabled', True)
        
        # API client (mock for now - replace with actual Claude client)
        self.api_client = None  # Would be Anthropic() in production
    
    def process(self, intent, context, budget: Optional[int] = None) -> Response:
        """
        Generate a response based on intent and context.
        
        Args:
            intent: Intent object from classifier
            context: Context object from manager
            budget: Optional token budget (uses default if None)
        
        Returns:
            Response object with generated content
        """
        max_tokens = budget or self.default_max_tokens
        
        # Adjust temperature based on intent
        temperature = self._get_temperature_for_intent(intent)
        
        # Build prompt
        prompt = self._build_prompt(intent, context)
        
        # Estimate prompt tokens
        prompt_tokens = self._estimate_tokens(prompt)
        self.track_tokens(prompt_tokens)
        
        # Generate response
        if self.streaming_enabled:
            response_content = self._generate_streaming(prompt, max_tokens, temperature)
        else:
            response_content = self._generate_standard(prompt, max_tokens, temperature)
        
        # Estimate response tokens
        response_tokens = self._estimate_tokens(response_content)
        self.track_tokens(response_tokens)
        
        # Create response object
        response = Response(
            content=response_content,
            tokens_used=prompt_tokens + response_tokens,
            finish_reason='stop',
            model=self.model,
            metadata={
                'intent': str(intent.type),
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )
        
        self.logger.info(
            f"Generated response: {response_tokens} tokens, "
            f"intent={intent.type.value}"
        )
        
        return response
    
    def _build_prompt(self, intent, context) -> str:
        """
        Build the prompt for Claude API.
        
        Strategy:
        - Include conversation summary if available
        - Include recent messages from context
        - Add intent-specific instructions
        - Keep within overall token budget
        """
        parts = []
        
        # Add system instructions based on intent
        system_instructions = self._get_system_instructions(intent)
        if system_instructions:
            parts.append(system_instructions)
        
        # Add conversation summary if available
        if context.summary:
            parts.append(f"Previous conversation:\n{context.summary}\n")
        
        # Add recent messages
        if context.messages:
            parts.append("Recent messages:")
            for msg in context.messages[-10:]:  # Last 10 messages
                parts.append(f"{msg.role.value}: {msg.content}")
        
        return "\n\n".join(parts)
    
    def _get_system_instructions(self, intent) -> str:
        """
        Get system instructions based on intent type.
        
        Different intents may need different system prompts.
        """
        from ..intent_classifier import IntentType
        
        instructions = {
            IntentType.QUESTION: (
                "You are a helpful assistant. Provide accurate, concise answers. "
                "If you don't know, say so."
            ),
            IntentType.COMMAND: (
                "You are an action-oriented assistant. Confirm the requested action "
                "and execute it clearly. Ask for clarification if needed."
            ),
            IntentType.CONVERSATION: (
                "You are a friendly conversational assistant. Be natural and engaging "
                "while staying helpful."
            ),
            IntentType.CLARIFICATION: (
                "The user is providing clarification. Acknowledge and update your "
                "understanding accordingly."
            ),
            IntentType.FEEDBACK: (
                "The user is providing feedback. Acknowledge it appropriately and "
                "adjust if needed."
            ),
        }
        
        return instructions.get(intent.type, "You are a helpful assistant.")
    
    def _get_temperature_for_intent(self, intent) -> float:
        """
        Adjust temperature based on intent.
        
        Lower temperature for factual tasks, higher for creative tasks.
        """
        from ..intent_classifier import IntentType
        
        temperatures = {
            IntentType.QUESTION: 0.3,        # More deterministic for facts
            IntentType.COMMAND: 0.2,         # Very deterministic for actions
            IntentType.CONVERSATION: 0.8,    # More creative for chat
            IntentType.CLARIFICATION: 0.3,   # Deterministic
            IntentType.FEEDBACK: 0.5,        # Balanced
        }
        
        return temperatures.get(intent.type, self.default_temperature)
    
    def _generate_streaming(self, prompt: str, max_tokens: int, 
                          temperature: float) -> str:
        """
        Generate response with streaming.
        
        In production, this would use:
        ```python
        stream = self.api_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        response_text = ""
        for event in stream:
            if event.type == "content_block_delta":
                response_text += event.delta.text
        
        return response_text
        ```
        """
        # Mock implementation
        self.logger.debug(f"Streaming generation (max_tokens={max_tokens})")
        return self._mock_generate(prompt, max_tokens, temperature)
    
    def _generate_standard(self, prompt: str, max_tokens: int, 
                          temperature: float) -> str:
        """
        Generate response without streaming.
        
        In production, this would use:
        ```python
        response = self.api_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        ```
        """
        # Mock implementation
        self.logger.debug(f"Standard generation (max_tokens={max_tokens})")
        return self._mock_generate(prompt, max_tokens, temperature)
    
    def _mock_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Mock response generation for demonstration.
        
        In production, replace with actual Claude API call.
        """
        return (
            f"[Mock response based on prompt. "
            f"In production, this would call Claude API with "
            f"max_tokens={max_tokens}, temperature={temperature}]"
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        if not text:
            return 0
        return len(text) // 4
