// Automation Agent — API Client
// Uses the shared axios apiClient with JWT interceptor

import { apiClient } from './client'
import type {
    AutomationDashboardData,
    ScriptConfig,
    ScriptConfigCreate,
    ExecutionResult,
    TriggerPayload,
    ApprovalItem,
    ApprovalAction,
    AutomationLogEntry,
    SkillDefinition,
} from '@/types/automation'

const BASE = '/api/v1/automation'

export const automationApi = {
    // Dashboard
    getDashboard: () =>
        apiClient.get<AutomationDashboardData>(`${BASE}/dashboard`).then((r) => r.data),

    // Scripts
    getScripts: () =>
        apiClient.get<ScriptConfig[]>(`${BASE}/scripts`).then((r) => r.data),

    createScript: (body: ScriptConfigCreate) =>
        apiClient.post<ScriptConfig>(`${BASE}/scripts`, body).then((r) => r.data),

    getScript: (scriptId: string) =>
        apiClient.get<ScriptConfig>(`${BASE}/scripts/${scriptId}`).then((r) => r.data),

    deleteScript: (scriptId: string) =>
        apiClient.delete(`${BASE}/scripts/${scriptId}`).then((r) => r.data),

    // Trigger
    triggerScript: (payload: TriggerPayload) =>
        apiClient.post<ExecutionResult>(`${BASE}/trigger`, payload).then((r) => r.data),

    // Executions
    getExecution: (executionId: string) =>
        apiClient.get<ExecutionResult>(`${BASE}/executions/${executionId}`).then((r) => r.data),

    // Approvals
    getApprovals: () =>
        apiClient.get<ApprovalItem[]>(`${BASE}/approvals`).then((r) => r.data),

    processApproval: (executionId: string, action: ApprovalAction) =>
        apiClient.post<ExecutionResult>(`${BASE}/approvals/${executionId}`, action).then((r) => r.data),

    // Logs
    getLogs: (limit = 20) =>
        apiClient.get<AutomationLogEntry[]>(`${BASE}/logs`, { params: { limit } }).then((r) => r.data),

    // Skills
    getSkills: () =>
        apiClient.get<SkillDefinition[]>(`${BASE}/skills`).then((r) => r.data),
}
