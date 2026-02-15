"""
Storage service for persisting data.

Handles saving/loading context snapshots, caches, and logs.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional
from datetime import datetime


class StorageService:
    """
    Handles data persistence to disk.
    
    Features:
    - Save/load context snapshots
    - Cache persistence
    - Log archival
    - JSON and pickle formats
    
    Usage:
        storage = StorageService(base_dir='data')
        
        # Save context
        storage.save_context(context, 'conversation_123')
        
        # Load context
        context = storage.load_context('conversation_123')
    """
    
    def __init__(self, base_dir: str = 'data'):
        """
        Initialize storage service.
        
        Args:
            base_dir: Base directory for all data storage
        """
        self.base_dir = Path(base_dir)
        self.context_dir = self.base_dir / 'context'
        self.cache_dir = self.base_dir / 'cache'
        self.logs_dir = self.base_dir / 'logs'
        
        # Create directories
        for directory in [self.context_dir, self.cache_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_context(self, context: Any, conversation_id: str) -> str:
        """
        Save conversation context.
        
        Args:
            context: Context object to save
            conversation_id: Unique conversation identifier
        
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{conversation_id}_{timestamp}.json"
        filepath = self.context_dir / filename
        
        # Convert to dict if has to_dict method
        if hasattr(context, 'to_dict'):
            data = context.to_dict()
        else:
            data = context
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def load_context(self, conversation_id: str, 
                    timestamp: Optional[str] = None) -> Optional[dict]:
        """
        Load conversation context.
        
        Args:
            conversation_id: Conversation identifier
            timestamp: Optional specific timestamp (loads latest if None)
        
        Returns:
            Context data or None if not found
        """
        # Find matching files
        pattern = f"{conversation_id}_*.json"
        matching_files = list(self.context_dir.glob(pattern))
        
        if not matching_files:
            return None
        
        # Get most recent if no timestamp specified
        if timestamp is None:
            filepath = max(matching_files, key=lambda p: p.stat().st_mtime)
        else:
            filename = f"{conversation_id}_{timestamp}.json"
            filepath = self.context_dir / filename
            if not filepath.exists():
                return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_cache(self, cache_data: dict, cache_name: str = 'default') -> str:
        """
        Persist cache to disk.
        
        Args:
            cache_data: Cache dictionary to save
            cache_name: Cache identifier
        
        Returns:
            Path to saved file
        """
        filepath = self.cache_dir / f"{cache_name}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)
        
        return str(filepath)
    
    def load_cache(self, cache_name: str = 'default') -> Optional[dict]:
        """
        Load cache from disk.
        
        Args:
            cache_name: Cache identifier
        
        Returns:
            Cache dictionary or None if not found
        """
        filepath = self.cache_dir / f"{cache_name}.pkl"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def list_contexts(self, conversation_id: Optional[str] = None) -> list[str]:
        """
        List saved context files.
        
        Args:
            conversation_id: Optional filter by conversation ID
        
        Returns:
            List of context file paths
        """
        if conversation_id:
            pattern = f"{conversation_id}_*.json"
        else:
            pattern = "*.json"
        
        files = self.context_dir.glob(pattern)
        return [str(f) for f in files]
