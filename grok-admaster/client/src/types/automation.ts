// Automation Agent — TypeScript Types
// Mirrors the backend schemas.py Pydantic models

export type ExecutorType = 'docker' | 'local'

export type ScriptStatus =
    | 'pending'
    | 'running'
    | 'success'
    | 'failed'
    | 'timeout'
    | 'awaiting_approval'

export type ApprovalDecision = 'approved' | 'rejected'

export type TriggerType = 'api' | 'schedule' | 'event' | 'threshold'

export interface ScriptConfig {
    script_id: string
    name: string
    description: string
    script_body: string
    executor: ExecutorType
    timeout_seconds: number
    max_retries: number
    requires_approval: boolean
    validation_schema: Record<string, unknown> | null
    tags: string[]
    created_at: string
    updated_at: string
}

export interface ScriptConfigCreate {
    name: string
    description?: string
    script_body: string
    executor?: ExecutorType
    timeout_seconds?: number
    max_retries?: number
    requires_approval?: boolean
    validation_schema?: Record<string, unknown>
    tags?: string[]
}

export interface ExecutionResult {
    script_id: string
    execution_id: string
    status: ScriptStatus
    output: unknown
    raw_stdout: string
    raw_stderr: string
    exit_code: number | null
    duration_ms: number | null
    retries_used: number
    executor_used: ExecutorType | null
    validation_passed: boolean | null
    validation_errors: string[]
    error_message: string | null
    started_at: string | null
    completed_at: string | null
    triggered_by: string | null
}

export interface TriggerPayload {
    script_id: string
    trigger_type?: TriggerType
    parameters?: Record<string, unknown>
    force_executor?: ExecutorType
    skip_approval?: boolean
}

export interface ApprovalItem {
    execution_id: string
    script_id: string
    script_name: string
    parameters: Record<string, unknown>
    requested_at: string
    requested_by: string | null
}

export interface ApprovalAction {
    decision: ApprovalDecision
}

export interface SkillDefinition {
    skill_id: string
    name: string
    description: string
    callable_name: string
    parameter_schema: Record<string, unknown> | null
    tags: string[]
}

export interface AutomationLogEntry {
    execution_id: string
    script_id: string
    script_name: string
    status: ScriptStatus
    duration_ms: number | null
    timestamp: string
    triggered_by: string | null
}

export interface AutomationDashboardData {
    recent_executions: AutomationLogEntry[]
    pending_approvals: ApprovalItem[]
    registered_skills: SkillDefinition[]
    stats: {
        total_executions: number
        success_count: number
        failed_count: number
        success_rate: number
        pending_approvals: number
        registered_scripts: number
        registered_skills: number
    }
}
