import psutil
import gc
import torch
import asyncio
from typing import List, Callable, Any
import logging

class ResourceManager:
    """Prevent OOM kills through intelligent resource management."""
    
    def __init__(self, max_memory_percent: float = 75.0):
        self.max_memory_percent = max_memory_percent
        self.semaphore = asyncio.Semaphore(2)  # Max 2 heavy tasks concurrently
        self.logger = logging.getLogger("resource_manager")
        
    def check_memory_available(self) -> bool:
        """Check if enough memory for heavy operation."""
        memory = psutil.virtual_memory()
        return memory.percent < self.max_memory_percent
        
    async def execute_with_memory_guard(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function only if memory available."""
        async with self.semaphore:
            if not self.check_memory_available():
                self.logger.warning("Memory usage high, waiting for resources...")
                await self._wait_for_memory()
                
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            finally:
                # Aggressive cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
    async def _wait_for_memory(self, max_wait: int = 300):
        """Wait for memory to free up."""
        waited = 0
        while not self.check_memory_available() and waited < max_wait:
            await asyncio.sleep(10)
            waited += 10
            gc.collect()
            
        if waited >= max_wait:
            self.logger.error("Memory threshold exceeded. Timeout waiting for resources.")
            raise MemoryError("Timeout waiting for available memory")

# Global instance
resource_manager = ResourceManager()
