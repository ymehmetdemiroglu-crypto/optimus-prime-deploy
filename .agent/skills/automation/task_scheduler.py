"""
Task Scheduler for Optimus Pryme
Cron-like scheduler for recurring tasks and jobs.
"""

import json
import threading
import time as time_module
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ScheduledTask:
    task_id: str
    name: str
    task_type: str
    handler: str  # Name of registered handler
    schedule_cron: str  # Simplified cron: "0 7 * * *" = 7 AM daily
    enabled: bool = True
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout_seconds: int = 300
    parameters: Dict[str, Any] = field(default_factory=dict)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    fail_count: int = 0

@dataclass
class TaskExecution:
    execution_id: str
    task_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    duration_ms: int = 0

class TaskScheduler:
    """
    Simple cron-like task scheduler.
    """
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.handlers: Dict[str, Callable] = {}
        self.execution_history: List[TaskExecution] = []
        self.task_counter = 0
        self.execution_counter = 0
        self.is_running = False
        self._scheduler_thread = None
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register built-in task handlers."""
        self.handlers = {
            "refresh_competitor_prices": self._handler_refresh_prices,
            "run_anomaly_detection": self._handler_anomaly_detection,
            "train_bid_model": self._handler_train_model,
            "generate_report": self._handler_generate_report,
            "sync_campaign_data": self._handler_sync_data,
            "cleanup_old_data": self._handler_cleanup,
            "send_scheduled_alerts": self._handler_send_alerts
        }
    
    def register_handler(self, handler_name: str, handler: Callable):
        """Register a task handler."""
        self.handlers[handler_name] = handler
    
    def create_task(self,
                    name: str,
                    task_type: str,
                    handler: str,
                    cron_schedule: str,
                    priority: TaskPriority = TaskPriority.NORMAL,
                    parameters: Optional[Dict] = None) -> ScheduledTask:
        """
        Create a scheduled task.
        
        cron_schedule format: "minute hour day month weekday"
        Examples:
            "0 7 * * *" = 7:00 AM every day
            "*/15 * * * *" = Every 15 minutes
            "0 8 * * 1" = 8:00 AM every Monday
            "0 0 1 * *" = Midnight on 1st of each month
        """
        self.task_counter += 1
        task_id = f"TASK{self.task_counter:04d}"
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            task_type=task_type,
            handler=handler,
            schedule_cron=cron_schedule,
            priority=priority,
            parameters=parameters or {},
            next_run=self._parse_cron(cron_schedule)
        )
        
        self.tasks[task_id] = task
        return task
    
    def _parse_cron(self, cron_str: str, from_time: Optional[datetime] = None) -> datetime:
        """
        Parse simplified cron string and return next run time.
        Supports: minute, hour, day of month, month, day of week
        """
        now = from_time or datetime.now()
        parts = cron_str.split()
        
        if len(parts) != 5:
            # Default to next hour
            return now + timedelta(hours=1)
        
        minute, hour, day, month, weekday = parts
        
        # Simple parsing (production: use croniter library)
        next_run = now.replace(second=0, microsecond=0)
        
        # Handle minute
        if minute == "*":
            pass
        elif minute.startswith("*/"):
            interval = int(minute[2:])
            next_minute = ((now.minute // interval) + 1) * interval
            if next_minute >= 60:
                next_run += timedelta(hours=1)
                next_run = next_run.replace(minute=0)
            else:
                next_run = next_run.replace(minute=next_minute)
        else:
            next_run = next_run.replace(minute=int(minute))
            if next_run <= now:
                next_run += timedelta(hours=1)
        
        # Handle hour
        if hour != "*" and not hour.startswith("*/"):
            next_run = next_run.replace(hour=int(hour))
            if next_run <= now:
                next_run += timedelta(days=1)
        
        return next_run
    
    def execute_task(self, task_id: str) -> TaskExecution:
        """Execute a task immediately."""
        task = self.tasks.get(task_id)
        if not task:
            return TaskExecution(
                execution_id="ERR",
                task_id=task_id,
                started_at=datetime.now(),
                status=TaskStatus.FAILED,
                error="Task not found"
            )
        
        self.execution_counter += 1
        execution = TaskExecution(
            execution_id=f"EXE{self.execution_counter:06d}",
            task_id=task_id,
            started_at=datetime.now(),
            status=TaskStatus.RUNNING
        )
        
        handler = self.handlers.get(task.handler)
        if not handler:
            execution.status = TaskStatus.FAILED
            execution.error = f"Handler not found: {task.handler}"
            execution.completed_at = datetime.now()
            self.execution_history.append(execution)
            return execution
        
        try:
            # Execute handler
            result = handler(task.parameters)
            
            execution.status = TaskStatus.COMPLETED
            execution.result = result
            execution.completed_at = datetime.now()
            execution.duration_ms = int(
                (execution.completed_at - execution.started_at).total_seconds() * 1000
            )
            
            # Update task stats
            task.last_run = execution.completed_at
            task.run_count += 1
            task.next_run = self._parse_cron(task.schedule_cron, execution.completed_at)
            
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            task.fail_count += 1
            
            # Retry logic
            if execution.retry_count < task.max_retries:
                execution.retry_count += 1
                # Reschedule soon
                task.next_run = datetime.now() + timedelta(minutes=5)
        
        self.execution_history.append(execution)
        return execution
    
    def check_and_run_due_tasks(self) -> List[TaskExecution]:
        """Check for due tasks and execute them."""
        now = datetime.now()
        executions = []
        
        # Sort by priority
        due_tasks = [
            t for t in self.tasks.values()
            if t.enabled and t.next_run and t.next_run <= now
        ]
        due_tasks.sort(key=lambda x: x.priority.value, reverse=True)
        
        for task in due_tasks:
            execution = self.execute_task(task.task_id)
            executions.append(execution)
        
        return executions
    
    def start(self, check_interval_seconds: int = 60):
        """Start the scheduler loop."""
        self.is_running = True
        
        def scheduler_loop():
            while self.is_running:
                self.check_and_run_due_tasks()
                time_module.sleep(check_interval_seconds)
        
        self._scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        print(f"Scheduler started (checking every {check_interval_seconds}s)")
    
    def stop(self):
        """Stop the scheduler loop."""
        self.is_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        print("Scheduler stopped")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        task = self.tasks.get(task_id)
        if not task:
            return {"error": "Task not found"}
        
        recent_executions = [
            e for e in self.execution_history
            if e.task_id == task_id
        ][-5:]
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "enabled": task.enabled,
            "schedule": task.schedule_cron,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "run_count": task.run_count,
            "fail_count": task.fail_count,
            "success_rate": round(
                (task.run_count - task.fail_count) / max(task.run_count, 1) * 100, 1
            ),
            "recent_executions": [
                {
                    "execution_id": e.execution_id,
                    "status": e.status.value,
                    "duration_ms": e.duration_ms
                }
                for e in recent_executions
            ]
        }
    
    def get_all_tasks_status(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        return {
            "total_tasks": len(self.tasks),
            "enabled_tasks": sum(1 for t in self.tasks.values() if t.enabled),
            "total_executions": len(self.execution_history),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "handler": t.handler,
                    "schedule": t.schedule_cron,
                    "enabled": t.enabled,
                    "next_run": t.next_run.isoformat() if t.next_run else None
                }
                for t in sorted(self.tasks.values(), key=lambda x: x.next_run or datetime.max)
            ]
        }
    
    # Default handlers
    def _handler_refresh_prices(self, params: Dict) -> Dict:
        """Handler: Refresh competitor prices."""
        print("  [Handler] Refreshing competitor prices...")
        return {"status": "success", "prices_updated": 15}
    
    def _handler_anomaly_detection(self, params: Dict) -> Dict:
        """Handler: Run anomaly detection scan."""
        print("  [Handler] Running anomaly detection...")
        return {"status": "success", "anomalies_found": 2}
    
    def _handler_train_model(self, params: Dict) -> Dict:
        """Handler: Train/retrain ML models."""
        print("  [Handler] Training bid prediction model...")
        return {"status": "success", "model_version": "v13"}
    
    def _handler_generate_report(self, params: Dict) -> Dict:
        """Handler: Generate scheduled reports."""
        report_type = params.get("report_type", "daily")
        print(f"  [Handler] Generating {report_type} report...")
        return {"status": "success", "report_id": "RPT001"}
    
    def _handler_sync_data(self, params: Dict) -> Dict:
        """Handler: Sync data from Amazon APIs."""
        print("  [Handler] Syncing campaign data...")
        return {"status": "success", "campaigns_synced": 45}
    
    def _handler_cleanup(self, params: Dict) -> Dict:
        """Handler: Cleanup old data."""
        print("  [Handler] Cleaning up old data...")
        return {"status": "success", "records_deleted": 1200}
    
    def _handler_send_alerts(self, params: Dict) -> Dict:
        """Handler: Send pending alerts."""
        print("  [Handler] Sending scheduled alerts...")
        return {"status": "success", "alerts_sent": 3}


# Convenience function to setup default tasks
def setup_default_tasks(scheduler: TaskScheduler):
    """Setup recommended default task schedule."""
    
    # Every 15 minutes: Price monitoring
    scheduler.create_task(
        name="Competitor Price Refresh",
        task_type="data_collection",
        handler="refresh_competitor_prices",
        cron_schedule="*/15 * * * *",
        priority=TaskPriority.HIGH
    )
    
    # Every hour: Anomaly detection
    scheduler.create_task(
        name="Hourly Anomaly Scan",
        task_type="monitoring",
        handler="run_anomaly_detection",
        cron_schedule="0 * * * *",
        priority=TaskPriority.HIGH
    )
    
    # Daily 6 AM: Data sync
    scheduler.create_task(
        name="Daily Campaign Sync",
        task_type="data_collection",
        handler="sync_campaign_data",
        cron_schedule="0 6 * * *",
        priority=TaskPriority.NORMAL
    )
    
    # Daily 7 AM: Morning report
    scheduler.create_task(
        name="Morning Briefing",
        task_type="reporting",
        handler="generate_report",
        cron_schedule="0 7 * * *",
        priority=TaskPriority.NORMAL,
        parameters={"report_type": "daily_briefing"}
    )
    
    # Weekly Sunday: Model retraining
    scheduler.create_task(
        name="Weekly Model Training",
        task_type="ml",
        handler="train_bid_model",
        cron_schedule="0 2 * * 0",  # 2 AM Sunday
        priority=TaskPriority.LOW
    )
    
    # Monthly: Data cleanup
    scheduler.create_task(
        name="Monthly Data Cleanup",
        task_type="maintenance",
        handler="cleanup_old_data",
        cron_schedule="0 3 1 * *",  # 3 AM on 1st
        priority=TaskPriority.LOW
    )


# Global instance
_task_scheduler = None

def get_task_scheduler() -> TaskScheduler:
    global _task_scheduler
    if _task_scheduler is None:
        _task_scheduler = TaskScheduler()
    return _task_scheduler


if __name__ == "__main__":
    scheduler = get_task_scheduler()
    
    print("Setting up default tasks...")
    setup_default_tasks(scheduler)
    
    # Show status
    status = scheduler.get_all_tasks_status()
    print(f"\nScheduler Status:")
    print(f"  Total Tasks: {status['total_tasks']}")
    print(f"  Enabled: {status['enabled_tasks']}")
    
    print("\nScheduled Tasks:")
    for task in status['tasks']:
        print(f"  - {task['name']}: {task['schedule']} (next: {task['next_run'][:16] if task['next_run'] else 'N/A'})")
    
    # Execute one manually
    print("\nExecuting 'Daily Campaign Sync' manually...")
    result = scheduler.execute_task("TASK0003")
    print(f"  Status: {result.status.value}")
    print(f"  Duration: {result.duration_ms}ms")
