"""
Automation Agent — Trigger Handlers

Wraps the engine with trigger-type-specific methods for
API calls, scheduled jobs, events, and thresholds.
"""

import logging
from typing import Optional, Dict, Any

from .schemas import TriggerPayload, TriggerType, ExecutionResult
from .engine import engine

logger = logging.getLogger(__name__)


class TriggerHandler:
    """Convenience wrapper around the engine for different trigger sources."""

    async def handle_api_trigger(
        self,
        payload: TriggerPayload,
        user_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Handle a trigger from a REST API call."""
        payload.trigger_type = TriggerType.API
        logger.info(f"[Trigger] API trigger for {payload.script_id} by {user_id}")
        return await engine.trigger(payload, user_id)

    async def handle_schedule_trigger(
        self,
        script_id: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Handle a trigger from APScheduler / cron.
        Skips approval by default (scheduled = pre-approved).
        """
        payload = TriggerPayload(
            script_id=script_id,
            trigger_type=TriggerType.SCHEDULE,
            parameters=parameters or {},
            skip_approval=True,
        )
        logger.info(f"[Trigger] Schedule trigger for {script_id}")
        return await engine.trigger(payload, user_id="scheduler")

    async def handle_event_trigger(
        self,
        script_id: str,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Handle a trigger from an event (file change, DB update, etc.)."""
        payload = TriggerPayload(
            script_id=script_id,
            trigger_type=TriggerType.EVENT,
            parameters=event_data or {},
        )
        logger.info(f"[Trigger] Event trigger for {script_id}")
        return await engine.trigger(payload, user_id="event_system")

    async def handle_threshold_trigger(
        self,
        script_id: str,
        metric_name: str,
        current_value: float,
        threshold: float,
    ) -> ExecutionResult:
        """Handle a trigger when a metric exceeds a threshold."""
        payload = TriggerPayload(
            script_id=script_id,
            trigger_type=TriggerType.THRESHOLD,
            parameters={
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold": threshold,
            },
        )
        logger.info(
            f"[Trigger] Threshold trigger for {script_id}: "
            f"{metric_name}={current_value} (threshold={threshold})"
        )
        return await engine.trigger(payload, user_id="threshold_monitor")


# ── Singleton ──
trigger_handler = TriggerHandler()
