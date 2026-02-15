"""
Central Alert Manager for Optimus Pryme
Handles alert creation, routing, and delivery across all skills.
"""

import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    ALL = "all"

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FAILED = "failed"

@dataclass
class AlertConfig:
    alert_type: str
    enabled: bool = True
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.IN_APP])
    priority: AlertPriority = AlertPriority.MEDIUM
    cooldown_minutes: int = 60  # Don't repeat same alert within this window
    recipients: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    alert_id: str
    alert_type: str
    title: str
    message: str
    priority: AlertPriority
    source_skill: str
    entity_id: Optional[str]
    entity_type: Optional[str]
    data: Dict[str, Any]
    created_at: datetime
    status: AlertStatus = AlertStatus.PENDING
    channels: List[AlertChannel] = field(default_factory=list)
    sent_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None

class AlertManager:
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_configs: Dict[str, AlertConfig] = {}
        self.alert_counter = 0
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Channel handlers
        self.channel_handlers: Dict[AlertChannel, Callable] = {
            AlertChannel.EMAIL: self._send_email,
            AlertChannel.SLACK: self._send_slack,
            AlertChannel.SMS: self._send_sms,
            AlertChannel.WEBHOOK: self._send_webhook,
            AlertChannel.IN_APP: self._send_in_app,
        }
        
        # Default configurations
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default alert configurations."""
        self.alert_configs = {
            "price_drop": AlertConfig(
                alert_type="price_drop",
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                priority=AlertPriority.HIGH,
                cooldown_minutes=30,
                conditions={"threshold_pct": 10}
            ),
            "anomaly_detected": AlertConfig(
                alert_type="anomaly_detected",
                channels=[AlertChannel.SLACK, AlertChannel.IN_APP],
                priority=AlertPriority.HIGH,
                cooldown_minutes=60
            ),
            "budget_warning": AlertConfig(
                alert_type="budget_warning",
                channels=[AlertChannel.EMAIL, AlertChannel.IN_APP],
                priority=AlertPriority.MEDIUM,
                cooldown_minutes=120,
                conditions={"threshold_pct": 80}
            ),
            "performance_threshold": AlertConfig(
                alert_type="performance_threshold",
                channels=[AlertChannel.IN_APP],
                priority=AlertPriority.LOW,
                cooldown_minutes=240
            ),
            "competitor_action": AlertConfig(
                alert_type="competitor_action",
                channels=[AlertChannel.SLACK],
                priority=AlertPriority.MEDIUM,
                cooldown_minutes=60
            ),
            "model_drift": AlertConfig(
                alert_type="model_drift",
                channels=[AlertChannel.EMAIL],
                priority=AlertPriority.MEDIUM,
                cooldown_minutes=1440  # Once per day
            ),
            "system_error": AlertConfig(
                alert_type="system_error",
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                priority=AlertPriority.CRITICAL,
                cooldown_minutes=5
            )
        }
    
    def configure_alert(self, alert_type: str, config: AlertConfig):
        """Configure an alert type."""
        self.alert_configs[alert_type] = config
    
    def create_alert(self, 
                     alert_type: str,
                     title: str,
                     message: str,
                     source_skill: str,
                     entity_id: Optional[str] = None,
                     entity_type: Optional[str] = None,
                     data: Optional[Dict] = None,
                     priority_override: Optional[AlertPriority] = None) -> Optional[Alert]:
        """
        Create and dispatch an alert.
        """
        # Check if alert type is configured
        config = self.alert_configs.get(alert_type)
        if not config or not config.enabled:
            return None
        
        # Check cooldown
        cooldown_key = f"{alert_type}_{entity_id or 'global'}"
        if cooldown_key in self.cooldown_tracker:
            last_alert = self.cooldown_tracker[cooldown_key]
            if datetime.now() - last_alert < timedelta(minutes=config.cooldown_minutes):
                return None  # Still in cooldown
        
        # Create alert
        self.alert_counter += 1
        alert = Alert(
            alert_id=f"ALT{self.alert_counter:06d}",
            alert_type=alert_type,
            title=title,
            message=message,
            priority=priority_override or config.priority,
            source_skill=source_skill,
            entity_id=entity_id,
            entity_type=entity_type,
            data=data or {},
            created_at=datetime.now(),
            channels=config.channels
        )
        
        self.alerts.append(alert)
        self.cooldown_tracker[cooldown_key] = datetime.now()
        
        # Dispatch to channels
        self._dispatch_alert(alert, config)
        
        return alert
    
    def _dispatch_alert(self, alert: Alert, config: AlertConfig):
        """Dispatch alert to configured channels."""
        success_count = 0
        
        for channel in config.channels:
            if channel == AlertChannel.ALL:
                for ch in AlertChannel:
                    if ch != AlertChannel.ALL:
                        if self._dispatch_to_channel(alert, ch, config.recipients):
                            success_count += 1
            else:
                if self._dispatch_to_channel(alert, channel, config.recipients):
                    success_count += 1
        
        if success_count > 0:
            alert.status = AlertStatus.SENT
            alert.sent_at = datetime.now()
        else:
            alert.status = AlertStatus.FAILED
    
    def _dispatch_to_channel(self, alert: Alert, channel: AlertChannel, 
                            recipients: List[str]) -> bool:
        """Dispatch to a specific channel."""
        handler = self.channel_handlers.get(channel)
        if handler:
            try:
                return handler(alert, recipients)
            except Exception as e:
                print(f"Failed to dispatch to {channel}: {e}")
                return False
        return False
    
    def _send_email(self, alert: Alert, recipients: List[str]) -> bool:
        """Send alert via email."""
        # In production: Use SendGrid, SES, or similar
        email_payload = {
            "to": recipients or ["alerts@company.com"],
            "subject": f"[{alert.priority.value.upper()}] {alert.title}",
            "body": self._format_email_body(alert)
        }
        print(f"[EMAIL] Would send: {email_payload['subject']} to {email_payload['to']}")
        return True
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format email body."""
        return f"""
        Alert: {alert.title}
        Priority: {alert.priority.value}
        Source: {alert.source_skill}
        
        {alert.message}
        
        Details:
        - Entity: {alert.entity_type} / {alert.entity_id}
        - Time: {alert.created_at.isoformat()}
        
        Data: {json.dumps(alert.data, indent=2)}
        """
    
    def _send_slack(self, alert: Alert, recipients: List[str]) -> bool:
        """Send alert via Slack."""
        # In production: Use Slack Webhook
        emoji = {
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.MEDIUM: "ℹ️",
            AlertPriority.LOW: "📝"
        }
        
        slack_message = {
            "channel": "#optimus-alerts",
            "text": f"{emoji.get(alert.priority, '📢')} *{alert.title}*\n{alert.message}",
            "attachments": [{
                "color": self._get_color(alert.priority),
                "fields": [
                    {"title": "Priority", "value": alert.priority.value, "short": True},
                    {"title": "Source", "value": alert.source_skill, "short": True}
                ]
            }]
        }
        print(f"[SLACK] Would send: {slack_message['text'][:100]}...")
        return True
    
    def _get_color(self, priority: AlertPriority) -> str:
        """Get Slack attachment color."""
        colors = {
            AlertPriority.CRITICAL: "#FF0000",
            AlertPriority.HIGH: "#FF9900",
            AlertPriority.MEDIUM: "#FFCC00",
            AlertPriority.LOW: "#00CC00"
        }
        return colors.get(priority, "#CCCCCC")
    
    def _send_sms(self, alert: Alert, recipients: List[str]) -> bool:
        """Send alert via SMS."""
        # In production: Use Twilio
        sms_message = f"[{alert.priority.value.upper()}] {alert.title}: {alert.message[:100]}"
        print(f"[SMS] Would send: {sms_message}")
        return True
    
    def _send_webhook(self, alert: Alert, recipients: List[str]) -> bool:
        """Send alert via webhook."""
        webhook_payload = asdict(alert)
        webhook_payload["created_at"] = alert.created_at.isoformat()
        webhook_payload["priority"] = alert.priority.value
        webhook_payload["status"] = alert.status.value
        print(f"[WEBHOOK] Would POST: {json.dumps(webhook_payload)[:200]}...")
        return True
    
    def _send_in_app(self, alert: Alert, recipients: List[str]) -> bool:
        """Store alert for in-app display."""
        # In production: Store in database/Redis for frontend to poll
        print(f"[IN_APP] Stored: {alert.alert_id} - {alert.title}")
        return True
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                return True
        return False
    
    def get_active_alerts(self, 
                          priority: Optional[AlertPriority] = None,
                          source_skill: Optional[str] = None) -> List[Alert]:
        """Get active (unresolved) alerts."""
        active = [a for a in self.alerts 
                 if a.status not in [AlertStatus.RESOLVED, AlertStatus.FAILED]]
        
        if priority:
            active = [a for a in active if a.priority == priority]
        if source_skill:
            active = [a for a in active if a.source_skill == source_skill]
        
        return sorted(active, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alerts in the given time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.alerts if a.created_at >= cutoff]
        
        by_priority = defaultdict(int)
        by_type = defaultdict(int)
        by_status = defaultdict(int)
        
        for alert in recent:
            by_priority[alert.priority.value] += 1
            by_type[alert.alert_type] += 1
            by_status[alert.status.value] += 1
        
        return {
            "period_hours": hours,
            "total_alerts": len(recent),
            "by_priority": dict(by_priority),
            "by_type": dict(by_type),
            "by_status": dict(by_status),
            "unresolved_count": by_status.get("pending", 0) + by_status.get("sent", 0),
            "critical_count": by_priority.get("critical", 0)
        }


# Global instance
_alert_manager = None

def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


if __name__ == "__main__":
    # Demo
    manager = get_alert_manager()
    
    # Create some test alerts
    print("Creating test alerts...")
    
    manager.create_alert(
        alert_type="price_drop",
        title="Competitor Price Drop Detected",
        message="RivalBrand dropped price on B0RIVAL001 by 20% (from $99.99 to $79.99)",
        source_skill="competitive-intelligence",
        entity_id="B0RIVAL001",
        entity_type="competitor_asin",
        data={"old_price": 99.99, "new_price": 79.99, "competitor": "RivalBrand"}
    )
    
    manager.create_alert(
        alert_type="anomaly_detected",
        title="CTR Anomaly on Campaign",
        message="CTR dropped to 0.18% (expected: 0.35-0.45%)",
        source_skill="data-scientist",
        entity_id="CAMP_001",
        entity_type="campaign",
        data={"metric": "CTR", "value": 0.18, "expected": 0.40}
    )
    
    # Get summary
    summary = manager.get_alert_summary(24)
    print(f"\nAlert Summary (24h):")
    print(f"  Total: {summary['total_alerts']}")
    print(f"  By Priority: {summary['by_priority']}")
    print(f"  Unresolved: {summary['unresolved_count']}")
