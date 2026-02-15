"""
Context data structures.

Defines conversation context and related models.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .message import Message


@dataclass
class Context:
    """
    Conversation context snapshot.
    
    Attributes:
        messages: List of messages in context
        summary: Optional summary of older messages
        total_tokens: Total tokens in current context
        metadata: Additional context metadata
    """
    messages: List[Message] = field(default_factory=list)
    summary: Optional[str] = None
    total_tokens: int = 0
    metadata:Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        """Add a message to context."""
        self.messages.append(message)
        self.total_tokens += message.tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'messages': [msg.to_dict() for msg in self.messages],
            'summary': self.summary,
            'total_tokens': self.total_tokens,
            'metadata': self.metadata
        }


@dataclass
class ConversationContext(Context):
    """
    Extended context with conversation-specific features.
    
    Additional features over base Context:
    - Conversation ID tracking
    - User/session tracking
    - Compression history
    """
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    compression_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'conversation_id': self.conversation_id,
            'user_id': self.user_id,
            'compression_count': self.compression_count
        })
        return base_dict
