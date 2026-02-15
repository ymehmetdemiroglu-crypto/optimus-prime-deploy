"""
Message data structures.

Defines the core message types used throughout the application.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role: Who sent the message (user/assistant/system)
        content: Message text content
        timestamp: When the message was created
        metadata: Additional metadata (tokens, importance, etc.)
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tokens(self) -> int:
        """Get token count from metadata."""
        return self.metadata.get('tokens', 0)
    
    @property
    def importance(self) -> float:
        """Get importance score from metadata."""
        return self.metadata.get('importance', 0.5)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'role': self.role.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data['role']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            metadata=data.get('metadata', {})
        )
