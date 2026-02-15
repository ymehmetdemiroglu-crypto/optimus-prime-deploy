from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any
import asyncio
import logging
from .persistence import PersistentSchedulerState

class PersistentTaskScheduler:
    """Production-grade scheduler with Redis state persistence and crash recovery."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.state = PersistentSchedulerState(redis_url)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.logger = logging.getLogger("persistent_scheduler")
        self._restore_state()
        
    def _restore_state(self):
        """Restore scheduler state from Redis on startup."""
        if not self.state.enabled:
            self.logger.warning("Redis not available, starting with empty scheduler state.")
            return

        self.logger.info("Restoring scheduler state from Redis...")
        saved_states = self.state.get_all_task_states()
        
        for task_id, task_state in saved_states.items():
            if task_state:
                # Recalculate next_run based on last execution
                last_run_str = task_state.get('last_run')
                if last_run_str:
                    last_run = datetime.fromisoformat(last_run_str)
                else:
                    last_run = datetime.utcnow()
                    
                interval = task_state.get('interval_minutes', 60)
                next_run = last_run + timedelta(minutes=interval)
                
                # If next_run is in the past, schedule immediately (plus small offset)
                if next_run < datetime.utcnow():
                    next_run = datetime.utcnow() + timedelta(seconds=10)
                    
                self.tasks[task_id] = {
                    'func': None,  # Will be re-registered by application
                    'next_run': next_run,
                    'interval': interval,
                    'enabled': task_state.get('enabled', True)
                }
                self.logger.info(f"Restored task {task_id}, next run estimated: {next_run}")
                
    def schedule_task(self, task_id: str, func: Callable, interval_minutes: int):
        """Register or update a task with persistence."""
        # Check if we have a restored next_run
        if task_id in self.tasks and self.tasks[task_id]['func'] is None:
            next_run = self.tasks[task_id]['next_run']
        else:
            next_run = datetime.utcnow() + timedelta(minutes=interval_minutes)
        
        self.tasks[task_id] = {
            'func': func,
            'next_run': next_run,
            'interval': interval_minutes,
            'enabled': True
        }
        
        # Persist to Redis
        self.state.save_task_state(task_id, {
            'interval_minutes': interval_minutes,
            'last_run': (next_run - timedelta(minutes=interval_minutes)).isoformat(),
            'enabled': True
        })
        self.logger.info(f"Scheduled task {task_id}, next run: {next_run}")
        
    async def start(self):
        """Main scheduler loop with crash recovery."""
        self.running = True
        self.logger.info("Persistent scheduler started")
        
        while self.running:
            try:
                now = datetime.utcnow()
                
                for task_id, task_info in self.tasks.items():
                    if not task_info.get('enabled', True):
                        continue
                        
                    if now >= task_info['next_run'] and task_info.get('func'):
                        self.logger.info(f"Executing task: {task_id}")
                        
                        try:
                            # Execute task (handle both sync and async)
                            if asyncio.iscoroutinefunction(task_info['func']):
                                await task_info['func']()
                            else:
                                task_info['func']()
                            
                            # Update state for next run
                            task_info['next_run'] = now + timedelta(minutes=task_info['interval'])
                            self.state.save_task_state(task_id, {
                                'interval_minutes': task_info['interval'],
                                'last_run': now.isoformat(),
                                'enabled': True
                            })
                            
                            # Log execution history
                            self.state.save_execution_history(task_id, {
                                'timestamp': now.isoformat(),
                                'status': 'success'
                            })
                            
                        except Exception as e:
                            self.logger.error(f"Task {task_id} failed: {e}")
                            self.state.save_execution_history(task_id, {
                                'timestamp': now.isoformat(),
                                'status': 'error',
                                'error': str(e)
                            })
                            # Still schedule next run even on failure
                            task_info['next_run'] = now + timedelta(minutes=task_info['interval'])
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)

    def stop(self):
        """Stop the scheduler."""
        self.running = False
        self.logger.info("Persistent scheduler stopping...")

# Global instance
scheduler = PersistentTaskScheduler()
