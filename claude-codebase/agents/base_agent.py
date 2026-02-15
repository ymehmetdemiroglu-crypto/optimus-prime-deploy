"""
Base Agent - Abstract base class for all agents.

Provides common functionality for token tracking, logging, and configuration.
All specialized agents should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Purpose:
    - Enforce consistent interface across all agents
    - Provide shared token tracking functionality
    - Enable dependency injection for configuration
    
    Token Optimization Strategy:
    - Track token usage per operation
    - Log expensive operations
    - Provide hooks for token budget enforcement
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration dictionary with agent-specific settings
            logger: Optional logger instance (creates default if not provided)
        
        Expected config keys:
            - max_tokens: Maximum tokens for this agent's operations
            - cache_enabled: Whether to enable caching
            - log_level: Logging level for this agent
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.token_usage = 0
        self.operation_count = 0
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and return result.
        
        This is the main entry point for the agent. Each agent must implement
        this method with its specific logic.
        
        Args:
            input_data: The input to process (type varies by agent)
        
        Returns:
            Processed output (type varies by agent)
        
        Token Optimization:
            - Implement early stopping if approaching token limit
            - Use caching for repeated inputs
            - Log token usage for monitoring
        """
        pass
    
    def track_tokens(self, tokens_used: int) -> None:
        """
        Track token usage for this agent.
        
        Args:
            tokens_used: Number of tokens consumed in the operation
        """
        self.token_usage += tokens_used
        self.operation_count += 1
        self.logger.debug(
            f"Token usage: +{tokens_used} (total: {self.token_usage}, "
            f"avg: {self.token_usage / self.operation_count:.1f})"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for this agent.
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            'agent_type': self.__class__.__name__,
            'total_tokens': self.token_usage,
            'operations': self.operation_count,
            'avg_tokens_per_op': self.token_usage / max(self.operation_count, 1)
        }
    
    def reset_stats(self) -> None:
        """Reset token usage statistics."""
        self.token_usage = 0
        self.operation_count = 0
