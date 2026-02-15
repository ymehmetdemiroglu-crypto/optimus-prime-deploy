"""
Automation Package for Optimus Pryme
Central automation infrastructure for alerts, reports, and scheduled tasks.
"""

from .alert_manager import AlertManager, get_alert_manager, Alert, AlertConfig, AlertPriority, AlertChannel
from .scheduled_reporter import ReportScheduler, get_scheduler, ScheduledReport, ReportFrequency, ReportFormat
from .task_scheduler import TaskScheduler, get_task_scheduler, ScheduledTask, TaskPriority, setup_default_tasks

__all__ = [
    # Alert Manager
    "AlertManager",
    "get_alert_manager", 
    "Alert",
    "AlertConfig",
    "AlertPriority",
    "AlertChannel",
    
    # Report Scheduler
    "ReportScheduler",
    "get_scheduler",
    "ScheduledReport",
    "ReportFrequency",
    "ReportFormat",
    
    # Task Scheduler
    "TaskScheduler",
    "get_task_scheduler",
    "ScheduledTask",
    "TaskPriority",
    "setup_default_tasks"
]

# Version
__version__ = "1.0.0"


def initialize_automation():
    """
    Initialize all automation systems with default configurations.
    Call this at application startup.
    """
    # Initialize alert manager
    alert_manager = get_alert_manager()
    print(f"Alert Manager initialized with {len(alert_manager.alert_configs)} alert types")
    
    # Initialize report scheduler
    report_scheduler = get_scheduler()
    print(f"Report Scheduler initialized with {len(report_scheduler.report_generators)} generators")
    
    # Initialize task scheduler with defaults
    task_scheduler = get_task_scheduler()
    setup_default_tasks(task_scheduler)
    print(f"Task Scheduler initialized with {len(task_scheduler.tasks)} tasks")
    
    return {
        "alert_manager": alert_manager,
        "report_scheduler": report_scheduler,
        "task_scheduler": task_scheduler
    }


if __name__ == "__main__":
    print("Initializing Optimus Pryme Automation...")
    systems = initialize_automation()
    print("\nAll systems ready!")
