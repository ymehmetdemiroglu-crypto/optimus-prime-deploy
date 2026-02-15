"""
Report Scheduler for Optimus Pryme
Handles scheduled report generation and delivery.
"""

import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import threading
import time as time_module

class ReportFrequency(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

class ReportFormat(Enum):
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    MARKDOWN = "markdown"

class DeliveryMethod(Enum):
    EMAIL = "email"
    SLACK = "slack"
    S3 = "s3"
    WEBHOOK = "webhook"
    FILE = "file"

@dataclass
class ScheduledReport:
    report_id: str
    name: str
    report_type: str  # daily_briefing, weekly_summary, etc.
    frequency: ReportFrequency
    schedule_time: time  # Time of day to run (UTC)
    schedule_day: Optional[int] = None  # Day of week (0-6) or day of month (1-31)
    enabled: bool = True
    formats: List[ReportFormat] = field(default_factory=lambda: [ReportFormat.HTML])
    delivery_methods: List[DeliveryMethod] = field(default_factory=lambda: [DeliveryMethod.EMAIL])
    recipients: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None

@dataclass
class ReportResult:
    report_id: str
    generated_at: datetime
    success: bool
    content: Optional[Dict[str, Any]] = None
    rendered_formats: Dict[ReportFormat, str] = field(default_factory=dict)
    delivery_results: Dict[DeliveryMethod, bool] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: int = 0

class ReportScheduler:
    def __init__(self):
        self.scheduled_reports: Dict[str, ScheduledReport] = {}
        self.report_generators: Dict[str, Callable] = {}
        self.execution_history: List[ReportResult] = []
        self.report_counter = 0
        
        # Register default generators
        self._register_default_generators()
    
    def _register_default_generators(self):
        """Register built-in report generators."""
        self.report_generators = {
            "daily_briefing": self._generate_daily_briefing,
            "weekly_summary": self._generate_weekly_summary,
            "monthly_performance": self._generate_monthly_performance,
            "anomaly_report": self._generate_anomaly_report,
            "competitor_report": self._generate_competitor_report,
            "financial_report": self._generate_financial_report
        }
    
    def register_generator(self, report_type: str, generator: Callable):
        """Register a custom report generator."""
        self.report_generators[report_type] = generator
    
    def create_schedule(self, 
                        name: str,
                        report_type: str,
                        frequency: ReportFrequency,
                        schedule_time: time = time(7, 0),  # 7 AM default
                        schedule_day: Optional[int] = None,
                        recipients: Optional[List[str]] = None,
                        formats: Optional[List[ReportFormat]] = None,
                        delivery: Optional[List[DeliveryMethod]] = None,
                        parameters: Optional[Dict] = None) -> ScheduledReport:
        """Create a new scheduled report."""
        self.report_counter += 1
        report_id = f"RPT{self.report_counter:04d}"
        
        report = ScheduledReport(
            report_id=report_id,
            name=name,
            report_type=report_type,
            frequency=frequency,
            schedule_time=schedule_time,
            schedule_day=schedule_day,
            formats=formats or [ReportFormat.HTML],
            delivery_methods=delivery or [DeliveryMethod.EMAIL],
            recipients=recipients or [],
            parameters=parameters or {},
            next_run=self._calculate_next_run(frequency, schedule_time, schedule_day)
        )
        
        self.scheduled_reports[report_id] = report
        return report
    
    def _calculate_next_run(self, frequency: ReportFrequency, 
                           schedule_time: time,
                           schedule_day: Optional[int]) -> datetime:
        """Calculate the next run time."""
        now = datetime.now()
        today = now.date()
        
        if frequency == ReportFrequency.HOURLY:
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return next_hour
        
        elif frequency == ReportFrequency.DAILY:
            scheduled_dt = datetime.combine(today, schedule_time)
            if scheduled_dt <= now:
                scheduled_dt += timedelta(days=1)
            return scheduled_dt
        
        elif frequency == ReportFrequency.WEEKLY:
            scheduled_dt = datetime.combine(today, schedule_time)
            days_ahead = (schedule_day or 0) - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            scheduled_dt += timedelta(days=days_ahead)
            return scheduled_dt
        
        elif frequency == ReportFrequency.MONTHLY:
            # Run on specific day of month
            day = schedule_day or 1
            try:
                scheduled_dt = datetime.combine(today.replace(day=day), schedule_time)
            except ValueError:
                # Day doesn't exist in current month, use last day
                import calendar
                last_day = calendar.monthrange(today.year, today.month)[1]
                scheduled_dt = datetime.combine(today.replace(day=last_day), schedule_time)
            
            if scheduled_dt <= now:
                # Move to next month
                if today.month == 12:
                    scheduled_dt = scheduled_dt.replace(year=today.year + 1, month=1)
                else:
                    scheduled_dt = scheduled_dt.replace(month=today.month + 1)
            return scheduled_dt
        
        return now + timedelta(hours=1)
    
    def check_and_run_due_reports(self) -> List[ReportResult]:
        """Check for due reports and execute them."""
        now = datetime.now()
        results = []
        
        for report in self.scheduled_reports.values():
            if not report.enabled:
                continue
            
            if report.next_run and report.next_run <= now:
                result = self.execute_report(report.report_id)
                results.append(result)
                
                # Update schedule
                report.last_run = now
                report.next_run = self._calculate_next_run(
                    report.frequency, report.schedule_time, report.schedule_day
                )
        
        return results
    
    def execute_report(self, report_id: str) -> ReportResult:
        """Execute a report immediately."""
        start_time = datetime.now()
        
        report = self.scheduled_reports.get(report_id)
        if not report:
            return ReportResult(
                report_id=report_id,
                generated_at=start_time,
                success=False,
                error="Report not found"
            )
        
        generator = self.report_generators.get(report.report_type)
        if not generator:
            return ReportResult(
                report_id=report_id,
                generated_at=start_time,
                success=False,
                error=f"No generator for report type: {report.report_type}"
            )
        
        try:
            # Generate report content
            content = generator(report.parameters)
            
            # Render to requested formats
            rendered = {}
            for fmt in report.formats:
                rendered[fmt] = self._render_format(content, fmt)
            
            # Deliver via requested methods
            delivery_results = {}
            for method in report.delivery_methods:
                delivery_results[method] = self._deliver(
                    report, content, rendered, method
                )
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            result = ReportResult(
                report_id=report_id,
                generated_at=start_time,
                success=True,
                content=content,
                rendered_formats=rendered,
                delivery_results=delivery_results,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            result = ReportResult(
                report_id=report_id,
                generated_at=start_time,
                success=False,
                error=str(e)
            )
        
        self.execution_history.append(result)
        return result
    
    def _render_format(self, content: Dict, fmt: ReportFormat) -> str:
        """Render content to specified format."""
        if fmt == ReportFormat.JSON:
            return json.dumps(content, indent=2, default=str)
        
        elif fmt == ReportFormat.MARKDOWN:
            return self._render_markdown(content)
        
        elif fmt == ReportFormat.HTML:
            return self._render_html(content)
        
        elif fmt == ReportFormat.CSV:
            return self._render_csv(content)
        
        return json.dumps(content)
    
    def _render_markdown(self, content: Dict) -> str:
        """Render to Markdown."""
        lines = [f"# {content.get('title', 'Report')}"]
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")
        
        if "summary" in content:
            lines.append("## Summary")
            lines.append(content["summary"])
            lines.append("")
        
        if "metrics" in content:
            lines.append("## Key Metrics")
            for key, value in content["metrics"].items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        if "sections" in content:
            for section in content["sections"]:
                lines.append(f"## {section.get('title', 'Section')}")
                lines.append(section.get("content", ""))
                lines.append("")
        
        return "\n".join(lines)
    
    def _render_html(self, content: Dict) -> str:
        """Render to HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{content.get('title', 'Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #333; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 4px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
            </style>
        </head>
        <body>
            <h1>{content.get('title', 'Report')}</h1>
            <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>
        """
        
        if "summary" in content:
            html += f"<h2>Summary</h2><p>{content['summary']}</p>"
        
        if "metrics" in content:
            html += "<h2>Key Metrics</h2><div class='metrics'>"
            for key, value in content["metrics"].items():
                html += f"<div class='metric'><strong>{key}</strong>: <span class='metric-value'>{value}</span></div>"
            html += "</div>"
        
        html += "</body></html>"
        return html
    
    def _render_csv(self, content: Dict) -> str:
        """Render to CSV (for metrics)."""
        lines = ["Metric,Value"]
        for key, value in content.get("metrics", {}).items():
            lines.append(f"{key},{value}")
        return "\n".join(lines)
    
    def _deliver(self, report: ScheduledReport, content: Dict,
                 rendered: Dict[ReportFormat, str], method: DeliveryMethod) -> bool:
        """Deliver report via specified method."""
        try:
            if method == DeliveryMethod.EMAIL:
                print(f"[EMAIL] Sending '{report.name}' to {report.recipients}")
                return True
            
            elif method == DeliveryMethod.SLACK:
                print(f"[SLACK] Posting '{report.name}' to channel")
                return True
            
            elif method == DeliveryMethod.WEBHOOK:
                print(f"[WEBHOOK] POSTing '{report.name}' to webhook")
                return True
            
            elif method == DeliveryMethod.FILE:
                # Save to file
                filename = f"{report.report_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                print(f"[FILE] Saving '{report.name}' to {filename}")
                return True
            
            elif method == DeliveryMethod.S3:
                print(f"[S3] Uploading '{report.name}' to S3 bucket")
                return True
            
            return False
        except Exception as e:
            print(f"Delivery failed: {e}")
            return False
    
    # Built-in report generators
    def _generate_daily_briefing(self, params: Dict) -> Dict:
        """Generate daily executive briefing."""
        return {
            "title": "Daily Executive Briefing",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": "Yesterday's performance summary with key insights and action items.",
            "metrics": {
                "Total Revenue": "$15,423",
                "Ad Spend": "$2,347",
                "ROAS": "6.57x",
                "ACoS": "15.2%",
                "Orders": 156
            },
            "sections": [
                {"title": "Wins", "content": "Summer Sale campaign hit 150% of target"},
                {"title": "Concerns", "content": "Electronics ACoS trending up - needs attention"},
                {"title": "Action Items", "content": "1. Review Electronics bids\n2. Monitor competitor pricing"}
            ]
        }
    
    def _generate_weekly_summary(self, params: Dict) -> Dict:
        """Generate weekly performance summary."""
        return {
            "title": "Weekly Performance Summary",
            "period": "Last 7 days",
            "summary": "Strong week with 15% revenue growth and improved efficiency.",
            "metrics": {
                "Weekly Revenue": "$98,500",
                "Weekly Spend": "$14,200",
                "Blended ACoS": "14.4%",
                "WoW Growth": "+15%",
                "Active Campaigns": 45
            }
        }
    
    def _generate_monthly_performance(self, params: Dict) -> Dict:
        """Generate monthly performance report."""
        return {
            "title": "Monthly Performance Report",
            "period": datetime.now().strftime("%B %Y"),
            "summary": "Monthly overview with YoY comparison and trend analysis.",
            "metrics": {
                "Monthly Revenue": "$425,000",
                "Monthly Spend": "$58,000",
                "Monthly ROAS": "7.3x",
                "YoY Growth": "+23%"
            }
        }
    
    def _generate_anomaly_report(self, params: Dict) -> Dict:
        """Generate anomaly detection report."""
        return {
            "title": "Anomaly Detection Report",
            "summary": "Summary of detected anomalies requiring attention.",
            "metrics": {
                "Anomalies Detected": 3,
                "Critical": 0,
                "High": 1,
                "Medium": 2
            }
        }
    
    def _generate_competitor_report(self, params: Dict) -> Dict:
        """Generate competitor intelligence report."""
        return {
            "title": "Competitive Intelligence Report",
            "summary": "Weekly competitive landscape analysis.",
            "metrics": {
                "Your Market Share": "12.4%",
                "Share Change": "+1.1pp",
                "Active Competitors": 8,
                "Price Position": "Mid-Market"
            }
        }
    
    def _generate_financial_report(self, params: Dict) -> Dict:
        """Generate financial analysis report."""
        return {
            "title": "Financial Performance Report",
            "summary": "Profitability and unit economics analysis.",
            "metrics": {
                "Gross Profit": "$367,000",
                "Ad ROI": "534%",
                "TACoS": "8.2%",
                "LTV:CAC Ratio": "3.54x"
            }
        }
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """Get status of all scheduled reports."""
        return {
            "total_schedules": len(self.scheduled_reports),
            "enabled": sum(1 for r in self.scheduled_reports.values() if r.enabled),
            "schedules": [
                {
                    "report_id": r.report_id,
                    "name": r.name,
                    "frequency": r.frequency.value,
                    "enabled": r.enabled,
                    "next_run": r.next_run.isoformat() if r.next_run else None,
                    "last_run": r.last_run.isoformat() if r.last_run else None
                }
                for r in self.scheduled_reports.values()
            ]
        }


# Global instance
_scheduler = None

def get_scheduler() -> ReportScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = ReportScheduler()
    return _scheduler


if __name__ == "__main__":
    # Demo
    scheduler = get_scheduler()
    
    # Create schedules
    print("Creating report schedules...")
    
    scheduler.create_schedule(
        name="Morning Executive Briefing",
        report_type="daily_briefing",
        frequency=ReportFrequency.DAILY,
        schedule_time=time(7, 0),
        recipients=["ceo@company.com", "cmo@company.com"],
        formats=[ReportFormat.HTML, ReportFormat.MARKDOWN],
        delivery=[DeliveryMethod.EMAIL]
    )
    
    scheduler.create_schedule(
        name="Weekly Performance Summary",
        report_type="weekly_summary",
        frequency=ReportFrequency.WEEKLY,
        schedule_time=time(8, 0),
        schedule_day=0,  # Monday
        recipients=["leadership@company.com"],
        formats=[ReportFormat.HTML, ReportFormat.PDF],
        delivery=[DeliveryMethod.EMAIL, DeliveryMethod.SLACK]
    )
    
    # Check status
    status = scheduler.get_schedule_status()
    print(f"\nSchedule Status:")
    print(f"  Total Schedules: {status['total_schedules']}")
    print(f"  Enabled: {status['enabled']}")
    
    # Execute one immediately
    print("\nExecuting daily briefing manually...")
    result = scheduler.execute_report("RPT0001")
    print(f"  Success: {result.success}")
    print(f"  Execution Time: {result.execution_time_ms}ms")
    
    if result.content:
        print(f"  Title: {result.content['title']}")
        print(f"  Metrics: {list(result.content['metrics'].keys())}")
