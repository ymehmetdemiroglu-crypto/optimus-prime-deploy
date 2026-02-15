"""
Configuration data model.

Simple configuration wrapper.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    """
    Generic configuration container.
    
    Provides a simple way to pass configuration as a dataclass
    instead of raw dict.
    """
    data: Dict[str, Any]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.data[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self.data.update(updates)
