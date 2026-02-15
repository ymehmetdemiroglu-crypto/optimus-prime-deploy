import redis
import pickle
import json
import os
from typing import Any, Optional, Dict
from datetime import datetime

class PersistentSchedulerState:
    """Redis-backed persistent storage for scheduler state."""
    
    def __init__(self, redis_url: Optional[str] = None):
        if not redis_url:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.state_key_prefix = "scheduler_state:"
            self.enabled = True
        except Exception as e:
            print(f"Redis connection failed: {e}. Persistence disabled.")
            self.enabled = False
        
    def save_task_state(self, task_id: str, state: dict):
        """Persist task state to Redis."""
        if not self.enabled: return
        key = f"{self.state_key_prefix}{task_id}"
        state['last_updated'] = datetime.utcnow().isoformat()
        self.redis_client.set(key, pickle.dumps(state))
        self.redis_client.expire(key, 86400 * 7)  # 7-day TTL
        
    def load_task_state(self, task_id: str) -> Optional[dict]:
        """Restore task state from Redis."""
        if not self.enabled: return None
        key = f"{self.state_key_prefix}{task_id}"
        data = self.redis_client.get(key)
        return pickle.loads(data) if data else None
        
    def get_all_task_states(self) -> dict:
        """Restore all task states on startup."""
        if not self.enabled: return {}
        pattern = f"{self.state_key_prefix}*"
        states = {}
        # Using scan_iter for performance
        for key in self.redis_client.scan_iter(match=pattern):
            decoded_key = key.decode() if isinstance(key, bytes) else key
            task_id = decoded_key.replace(self.state_key_prefix, '')
            states[task_id] = self.load_task_state(task_id)
        return states
        
    def save_execution_history(self, task_id: str, execution: dict):
        """Append execution to history (circular buffer)."""
        if not self.enabled: return
        history_key = f"exec_history:{task_id}"
        self.redis_client.lpush(history_key, json.dumps(execution))
        self.redis_client.ltrim(history_key, 0, 99)  # Keep last 100
        self.redis_client.expire(history_key, 86400 * 30)  # 30-day TTL
