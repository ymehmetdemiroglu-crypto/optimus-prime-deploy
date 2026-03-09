"""
Automation Agent — Core Engine

Orchestrates script CRUD, execution with retries, validation,
and approval queue management. All state is in-memory.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

from .schemas import (
    ApprovalItem,
    AutomationDashboardResponse,
    AutomationLogEntry,
    ExecutionResult,
    ScriptConfig,
    ScriptConfigCreate,
    ScriptStatus,
    TriggerPayload,
    ApprovalDecision,
)
from .sandbox import ScriptExecutor
from .validator import ResultValidator
from .skills import skill_registry

logger = logging.getLogger(__name__)

MAX_LOG_ENTRIES = 200


class AutomationEngine:
    """
    Central orchestrator for the automation agent.

    Manages:
      - Script CRUD (in-memory dict)
      - Execution with exponential-backoff retries
      - JSON Schema output validation
      - Approval queue for sensitive scripts
      - Execution log with capped history
    """

    def __init__(self):
        self._scripts: Dict[str, ScriptConfig] = {}
        self._executions: Dict[str, ExecutionResult] = {}
        self._approval_queue: Dict[str, ApprovalItem] = {}
        self._log: deque[AutomationLogEntry] = deque(maxlen=MAX_LOG_ENTRIES)
        self._executor = ScriptExecutor()

    # ═════════════════════════════════════════════════════
    #  Script CRUD
    # ═════════════════════════════════════════════════════

    def register_script(self, create: ScriptConfigCreate) -> ScriptConfig:
        """Register a new script and return the full config with generated ID."""
        script = ScriptConfig(**create.model_dump())
        self._scripts[script.script_id] = script
        logger.info(f"[Engine] Registered script: {script.script_id} ({script.name})")
        return script

    def get_script(self, script_id: str) -> Optional[ScriptConfig]:
        return self._scripts.get(script_id)

    def list_scripts(self) -> List[ScriptConfig]:
        return list(self._scripts.values())

    def delete_script(self, script_id: str) -> bool:
        removed = self._scripts.pop(script_id, None)
        if removed:
            logger.info(f"[Engine] Deleted script: {script_id}")
        return removed is not None

    # ═════════════════════════════════════════════════════
    #  Trigger → Execute or Queue
    # ═════════════════════════════════════════════════════

    async def trigger(
        self, payload: TriggerPayload, user_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Main entry point: trigger script execution or queue for approval.
        """
        script = self._scripts.get(payload.script_id)
        if not script:
            result = ExecutionResult(script_id=payload.script_id)
            result.status = ScriptStatus.FAILED
            result.error_message = f"Script '{payload.script_id}' not found"
            return result

        # Create initial result
        result = ExecutionResult(script_id=script.script_id)
        result.triggered_by = user_id
        self._executions[result.execution_id] = result

        # Check if approval is required
        if script.requires_approval and not payload.skip_approval:
            result.status = ScriptStatus.AWAITING_APPROVAL
            approval = ApprovalItem(
                execution_id=result.execution_id,
                script_id=script.script_id,
                script_name=script.name,
                parameters=payload.parameters,
                requested_by=user_id,
            )
            self._approval_queue[result.execution_id] = approval
            self._add_log(result, script.name)
            logger.info(
                f"[Engine] Script {script.script_id} queued for approval "
                f"(execution_id={result.execution_id})"
            )
            return result

        # Execute immediately
        result = await self._execute_with_retries(
            script, payload.parameters, result, payload.force_executor
        )
        self._add_log(result, script.name)
        return result

    # ═════════════════════════════════════════════════════
    #  Execution with Retries
    # ═════════════════════════════════════════════════════

    async def _execute_with_retries(
        self,
        script: ScriptConfig,
        parameters: dict,
        result: ExecutionResult,
        force_executor=None,
    ) -> ExecutionResult:
        """Execute script with exponential backoff retries."""

        max_attempts = 1 + script.max_retries
        last_result = result

        for attempt in range(max_attempts):
            if attempt > 0:
                # Exponential backoff: 2^attempt seconds, capped at 30s
                delay = min(2 ** attempt, 30)
                logger.info(
                    f"[Engine] Retry {attempt}/{script.max_retries} for "
                    f"{script.script_id} in {delay}s"
                )
                await asyncio.sleep(delay)

            last_result.status = ScriptStatus.RUNNING
            last_result.retries_used = attempt

            exec_result = await self._executor.execute(
                script, parameters, force_executor
            )

            # Copy fields from executor result onto our tracked result
            last_result.status = exec_result.status
            last_result.output = exec_result.output
            last_result.raw_stdout = exec_result.raw_stdout
            last_result.raw_stderr = exec_result.raw_stderr
            last_result.exit_code = exec_result.exit_code
            last_result.duration_ms = exec_result.duration_ms
            last_result.executor_used = exec_result.executor_used
            last_result.error_message = exec_result.error_message
            last_result.started_at = exec_result.started_at
            last_result.completed_at = exec_result.completed_at
            last_result.retries_used = attempt

            if last_result.status == ScriptStatus.SUCCESS:
                # Validate output
                last_result = ResultValidator.validate(
                    last_result, script.validation_schema
                )
                break

            # If not retryable (timeout or last attempt), stop
            if attempt == max_attempts - 1:
                break

        self._executions[last_result.execution_id] = last_result
        return last_result

    # ═════════════════════════════════════════════════════
    #  Approval Queue
    # ═════════════════════════════════════════════════════

    async def approve(self, execution_id: str) -> Optional[ExecutionResult]:
        """Approve a pending execution and run it."""
        approval = self._approval_queue.pop(execution_id, None)
        if not approval:
            return None

        script = self._scripts.get(approval.script_id)
        if not script:
            return None

        result = self._executions.get(execution_id)
        if not result:
            return None

        logger.info(f"[Engine] Approved execution {execution_id}")
        result = await self._execute_with_retries(
            script, approval.parameters, result
        )
        self._add_log(result, script.name)
        return result

    async def reject(self, execution_id: str) -> Optional[ExecutionResult]:
        """Reject a pending execution."""
        approval = self._approval_queue.pop(execution_id, None)
        if not approval:
            return None

        result = self._executions.get(execution_id)
        if result:
            result.status = ScriptStatus.FAILED
            result.error_message = "Rejected by human reviewer"
            result.completed_at = datetime.utcnow()
            self._add_log(result, approval.script_name)
            logger.info(f"[Engine] Rejected execution {execution_id}")

        return result

    # ═════════════════════════════════════════════════════
    #  Queries
    # ═════════════════════════════════════════════════════

    def get_execution(self, execution_id: str) -> Optional[ExecutionResult]:
        return self._executions.get(execution_id)

    def get_recent_logs(self, limit: int = 20) -> List[AutomationLogEntry]:
        return list(self._log)[-limit:]

    def get_pending_approvals(self) -> List[ApprovalItem]:
        return list(self._approval_queue.values())

    def get_stats(self) -> dict:
        """Aggregate quick stats for the dashboard."""
        all_results = list(self._executions.values())
        total = len(all_results)
        success = sum(1 for r in all_results if r.status == ScriptStatus.SUCCESS)
        failed = sum(1 for r in all_results if r.status == ScriptStatus.FAILED)
        pending = len(self._approval_queue)

        return {
            "total_executions": total,
            "success_count": success,
            "failed_count": failed,
            "success_rate": round(success / total * 100, 1) if total > 0 else 0.0,
            "pending_approvals": pending,
            "registered_scripts": len(self._scripts),
            "registered_skills": len(skill_registry.list_skills()),
        }

    def get_dashboard(self) -> AutomationDashboardResponse:
        return AutomationDashboardResponse(
            recent_executions=self.get_recent_logs(20),
            pending_approvals=self.get_pending_approvals(),
            registered_skills=skill_registry.list_skills(),
            stats=self.get_stats(),
        )

    # ═════════════════════════════════════════════════════
    #  Internal Helpers
    # ═════════════════════════════════════════════════════

    def _add_log(self, result: ExecutionResult, script_name: str):
        entry = AutomationLogEntry(
            execution_id=result.execution_id,
            script_id=result.script_id,
            script_name=script_name,
            status=result.status,
            duration_ms=result.duration_ms,
            triggered_by=result.triggered_by,
        )
        self._log.append(entry)


# ── Singleton ──
engine = AutomationEngine()
