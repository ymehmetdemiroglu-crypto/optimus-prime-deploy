"""
Notification Dispatcher

Central service for dispatching alerts to users via multiple channels.
Supports logging (always on), with extensible hooks for Email, Slack, and WebSocket.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


async def dispatch_alert(
    user_id: str,
    title: str,
    message: str,
    severity: str = "info",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Dispatch an alert notification to a user.

    Currently logs the alert. In the future, this will fan-out to:
        - Email (SendGrid / AWS SES)
        - Slack webhook
        - WebSocket push to frontend
        - SMS via Twilio (enterprise tier)

    Args:
        user_id:  Target user/profile identifier.
        title:    Short alert title.
        message:  Full alert body.
        severity: One of 'critical', 'high', 'medium', 'low', 'info'.
        metadata: Optional dict of extra context (entity_id, metric values, etc.).

    Returns:
        True if the alert was dispatched successfully.
    """
    log_level = {
        "critical": logging.CRITICAL,
        "high": logging.ERROR,
        "medium": logging.WARNING,
        "low": logging.INFO,
        "info": logging.INFO,
    }.get(severity, logging.INFO)

    logger.log(
        log_level,
        "[ALERT] user=%s | severity=%s | %s — %s | meta=%s",
        user_id,
        severity,
        title,
        message,
        metadata or {},
    )

    # ── Future integration hooks ──────────────────────────────────
    # await _send_email(user_id, title, message, severity)
    # await _send_slack_webhook(title, message, severity)
    # await _push_websocket(user_id, {"title": title, "message": message})
    # ──────────────────────────────────────────────────────────────

    return True
