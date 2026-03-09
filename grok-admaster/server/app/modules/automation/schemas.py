"""
Automation Agent — Schemas

Pydantic v2 models for the deterministic automation engine.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


# ──────────────────────────────────────────────────────────
#  Enumerations
# ──────────────────────────────────────────────────────────

class ExecutorType(str, Enum):
    DOCKER = "docker"
    LOCAL = "local"


class ScriptStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    AWAITING_APPROVAL = "awaiting_approval"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class TriggerType(str, Enum):
    API = "api"
    SCHEDULE = "schedule"
    EVENT = "event"
    THRESHOLD = "threshold"


# ──────────────────────────────────────────────────────────
#  Script Configuration
# ──────────────────────────────────────────────────────────

class ScriptConfigCreate(BaseModel):
    """Input model for creating a new script."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    script_body: str = Field(..., min_length=1, description="Python or Bash script source code")
    executor: ExecutorType = ExecutorType.DOCKER
    timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_retries: int = Field(default=0, ge=0, le=10)
    requires_approval: bool = False
    validation_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON Schema to validate the script's JSON output",
    )
    tags: List[str] = Field(default_factory=list)


class ScriptConfig(ScriptConfigCreate):
    """Full script config with server-generated fields."""
    script_id: str = Field(default_factory=lambda: f"script_{uuid.uuid4().hex[:12]}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────
#  Execution Result
# ──────────────────────────────────────────────────────────

class ExecutionResult(BaseModel):
    """Structured result from a script execution."""
    script_id: str
    execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    status: ScriptStatus = ScriptStatus.PENDING
    output: Optional[Any] = None
    raw_stdout: str = ""
    raw_stderr: str = ""
    exit_code: Optional[int] = None
    duration_ms: Optional[float] = None
    retries_used: int = 0
    executor_used: Optional[ExecutorType] = None
    validation_passed: Optional[bool] = None
    validation_errors: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    triggered_by: Optional[str] = None


# ──────────────────────────────────────────────────────────
#  Triggers
# ──────────────────────────────────────────────────────────

class TriggerPayload(BaseModel):
    """Payload to trigger a script execution."""
    script_id: str
    trigger_type: TriggerType = TriggerType.API
    parameters: Dict[str, Any] = Field(default_factory=dict)
    force_executor: Optional[ExecutorType] = None
    skip_approval: bool = False


# ──────────────────────────────────────────────────────────
#  Approval Queue
# ──────────────────────────────────────────────────────────

class ApprovalItem(BaseModel):
    """An execution waiting for human approval."""
    execution_id: str
    script_id: str
    script_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    requested_by: Optional[str] = None


class ApprovalAction(BaseModel):
    """Human action on an approval item."""
    decision: ApprovalDecision


# ──────────────────────────────────────────────────────────
#  Skills
# ──────────────────────────────────────────────────────────

class SkillDefinition(BaseModel):
    """Definition of a registered agent skill."""
    skill_id: str
    name: str
    description: str = ""
    callable_name: str
    parameter_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────
#  Logs & Dashboard
# ──────────────────────────────────────────────────────────

class AutomationLogEntry(BaseModel):
    """Summary log entry for the activity feed."""
    execution_id: str
    script_id: str
    script_name: str
    status: ScriptStatus
    duration_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    triggered_by: Optional[str] = None


class AutomationDashboardResponse(BaseModel):
    """Full dashboard payload."""
    recent_executions: List[AutomationLogEntry] = Field(default_factory=list)
    pending_approvals: List[ApprovalItem] = Field(default_factory=list)
    registered_skills: List[SkillDefinition] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
