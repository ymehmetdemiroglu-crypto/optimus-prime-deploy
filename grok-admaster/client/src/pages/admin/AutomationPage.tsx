import { useState, useEffect, useCallback } from 'react'
import { automationApi } from '@/api/automation'
import type {
    AutomationDashboardData,
    ScriptConfig,
    AutomationLogEntry,
    ApprovalItem,
    SkillDefinition,
    ScriptConfigCreate,
} from '@/types/automation'

/* ───────── Status badge colours ───────── */
const statusColor: Record<string, string> = {
    success: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    failed: 'bg-red-500/20 text-red-400 border-red-500/30',
    timeout: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    running: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    pending: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
    awaiting_approval: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
}

/* ═══════════════════════════════════════════
   Main Page
   ═══════════════════════════════════════════ */

export default function AutomationPage() {
    const [dashboard, setDashboard] = useState<AutomationDashboardData | null>(null)
    const [scripts, setScripts] = useState<ScriptConfig[]>([])
    const [loading, setLoading] = useState(true)
    const [showAdvanced, setShowAdvanced] = useState(false)
    const [creating, setCreating] = useState(false)

    /* new-script form state */
    const [newName, setNewName] = useState('')
    const [newBody, setNewBody] = useState('result = {"hello": "world"}')
    const [newApproval, setNewApproval] = useState(false)

    const refresh = useCallback(async () => {
        try {
            const [dash, sc] = await Promise.all([
                automationApi.getDashboard(),
                automationApi.getScripts(),
            ])
            setDashboard(dash)
            setScripts(sc)
        } catch (e) {
            console.error('Failed to load automation data', e)
        } finally {
            setLoading(false)
        }
    }, [])

    useEffect(() => {
        refresh()
        const interval = setInterval(refresh, 10_000)
        return () => clearInterval(interval)
    }, [refresh])

    /* ─── Actions ─── */

    const handleApproval = async (executionId: string, decision: 'approved' | 'rejected') => {
        try {
            await automationApi.processApproval(executionId, { decision })
            await refresh()
        } catch (e) {
            console.error('Approval action failed', e)
        }
    }

    const handleCreateScript = async () => {
        if (!newName.trim() || !newBody.trim()) return
        setCreating(true)
        try {
            const payload: ScriptConfigCreate = {
                name: newName,
                script_body: newBody,
                requires_approval: newApproval,
            }
            await automationApi.createScript(payload)
            setNewName('')
            setNewBody('result = {"hello": "world"}')
            setNewApproval(false)
            await refresh()
        } catch (e) {
            console.error('Create script failed', e)
        } finally {
            setCreating(false)
        }
    }

    const handleDeleteScript = async (scriptId: string) => {
        try {
            await automationApi.deleteScript(scriptId)
            await refresh()
        } catch (e) {
            console.error('Delete failed', e)
        }
    }

    const handleTrigger = async (scriptId: string) => {
        try {
            await automationApi.triggerScript({ script_id: scriptId })
            await refresh()
        } catch (e) {
            console.error('Trigger failed', e)
        }
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500" />
            </div>
        )
    }

    const stats = dashboard?.stats

    return (
        <div className="space-y-6">
            {/* ══════ Header ══════ */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-slate-100">Automation Agent</h1>
                    <p className="text-sm text-slate-500 mt-1">
                        Deterministic script execution · Approval queue · Skill registry
                    </p>
                </div>
                <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="px-4 py-2 text-sm rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 transition-colors"
                >
                    {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
                </button>
            </div>

            {/* ══════ Stats Cards ══════ */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard label="Runs (Total)" value={stats?.total_executions ?? 0} icon="⚡" />
                <StatCard
                    label="Success Rate"
                    value={`${stats?.success_rate ?? 0}%`}
                    icon="✅"
                    accent={
                        (stats?.success_rate ?? 0) >= 80
                            ? 'text-emerald-400'
                            : 'text-amber-400'
                    }
                />
                <StatCard label="Scripts" value={stats?.registered_scripts ?? 0} icon="📜" />
                <StatCard
                    label="Pending Approvals"
                    value={stats?.pending_approvals ?? 0}
                    icon="🔔"
                    accent={
                        (stats?.pending_approvals ?? 0) > 0 ? 'text-purple-400' : undefined
                    }
                />
            </div>

            {/* ══════ Approval Queue ══════ */}
            <Section title="Approval Queue" badge={dashboard?.pending_approvals.length}>
                {dashboard?.pending_approvals.length === 0 ? (
                    <p className="text-slate-500 text-sm py-4 text-center">No pending approvals</p>
                ) : (
                    <div className="space-y-2">
                        {dashboard?.pending_approvals.map((item) => (
                            <ApprovalRow
                                key={item.execution_id}
                                item={item}
                                onApprove={(id) => handleApproval(id, 'approved')}
                                onReject={(id) => handleApproval(id, 'rejected')}
                            />
                        ))}
                    </div>
                )}
            </Section>

            {/* ══════ Execution Log ══════ */}
            <Section title="Execution Log" badge={dashboard?.recent_executions.length}>
                {dashboard?.recent_executions.length === 0 ? (
                    <p className="text-slate-500 text-sm py-4 text-center">No executions yet</p>
                ) : (
                    <div className="max-h-80 overflow-y-auto space-y-1">
                        {dashboard?.recent_executions
                            .slice()
                            .reverse()
                            .map((entry) => <LogRow key={entry.execution_id} entry={entry} />)}
                    </div>
                )}
            </Section>

            {/* ══════ Skills Panel ══════ */}
            <Section title="Agent Skills" badge={dashboard?.registered_skills.length}>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    {dashboard?.registered_skills.map((skill) => (
                        <SkillCard key={skill.skill_id} skill={skill} />
                    ))}
                </div>
            </Section>

            {/* ══════ Advanced Settings (Collapsible) ══════ */}
            {showAdvanced && (
                <Section title="Scripts Manager">
                    {/* Create form */}
                    <div className="mb-6 p-4 rounded-lg border border-slate-700 bg-slate-900/50 space-y-3">
                        <h4 className="text-sm font-semibold text-slate-300">Create Script</h4>
                        <input
                            placeholder="Script name"
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-sm focus:outline-none focus:border-blue-500"
                        />
                        <textarea
                            placeholder="Python script body..."
                            rows={4}
                            value={newBody}
                            onChange={(e) => setNewBody(e.target.value)}
                            className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-sm font-mono focus:outline-none focus:border-blue-500"
                        />
                        <label className="flex items-center gap-2 text-sm text-slate-400">
                            <input
                                type="checkbox"
                                checked={newApproval}
                                onChange={(e) => setNewApproval(e.target.checked)}
                                className="accent-purple-500"
                            />
                            Requires human approval before execution
                        </label>
                        <button
                            onClick={handleCreateScript}
                            disabled={creating || !newName.trim()}
                            className="px-4 py-2 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 transition-colors"
                        >
                            {creating ? 'Creating…' : 'Create Script'}
                        </button>
                    </div>

                    {/* Scripts list */}
                    {scripts.length === 0 ? (
                        <p className="text-slate-500 text-sm text-center py-4">No scripts registered</p>
                    ) : (
                        <div className="space-y-2">
                            {scripts.map((s) => (
                                <div
                                    key={s.script_id}
                                    className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700"
                                >
                                    <div>
                                        <p className="text-sm font-medium text-slate-200">{s.name}</p>
                                        <p className="text-xs text-slate-500">
                                            {s.script_id} · {s.executor} · timeout {s.timeout_seconds}s
                                            {s.requires_approval && ' · 🔒 approval required'}
                                        </p>
                                    </div>
                                    <div className="flex gap-2">
                                        <button
                                            onClick={() => handleTrigger(s.script_id)}
                                            className="px-3 py-1.5 text-xs rounded-md bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-600/40 transition-colors"
                                        >
                                            ▶ Run
                                        </button>
                                        <button
                                            onClick={() => handleDeleteScript(s.script_id)}
                                            className="px-3 py-1.5 text-xs rounded-md bg-red-600/20 text-red-400 border border-red-500/30 hover:bg-red-600/40 transition-colors"
                                        >
                                            ✕ Delete
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </Section>
            )}
        </div>
    )
}

/* ═══════════════════════════════════════════
   Sub-Components
   ═══════════════════════════════════════════ */

function StatCard({
    label,
    value,
    icon,
    accent,
}: {
    label: string
    value: string | number
    icon: string
    accent?: string
}) {
    return (
        <div className="p-4 rounded-xl bg-panel border border-slate-800">
            <div className="flex items-center justify-between mb-2">
                <span className="text-lg">{icon}</span>
            </div>
            <p className={`text-2xl font-bold ${accent ?? 'text-slate-100'}`}>{value}</p>
            <p className="text-xs text-slate-500 mt-1">{label}</p>
        </div>
    )
}

function Section({
    title,
    badge,
    children,
}: {
    title: string
    badge?: number
    children: React.ReactNode
}) {
    return (
        <div className="rounded-xl bg-panel border border-slate-800 overflow-hidden">
            <div className="px-5 py-3 border-b border-slate-800 flex items-center gap-2">
                <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">{title}</h3>
                {badge !== undefined && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-slate-700 text-slate-400">
                        {badge}
                    </span>
                )}
            </div>
            <div className="p-4">{children}</div>
        </div>
    )
}

function ApprovalRow({
    item,
    onApprove,
    onReject,
}: {
    item: ApprovalItem
    onApprove: (id: string) => void
    onReject: (id: string) => void
}) {
    return (
        <div className="flex items-center justify-between p-3 rounded-lg bg-purple-500/5 border border-purple-500/20">
            <div>
                <p className="text-sm font-medium text-slate-200">{item.script_name}</p>
                <p className="text-xs text-slate-500">
                    {item.execution_id} · requested by {item.requested_by ?? 'system'}
                </p>
            </div>
            <div className="flex gap-2">
                <button
                    onClick={() => onApprove(item.execution_id)}
                    className="px-3 py-1.5 text-xs rounded-md bg-emerald-600 text-white hover:bg-emerald-500 transition-colors"
                >
                    ✓ Approve
                </button>
                <button
                    onClick={() => onReject(item.execution_id)}
                    className="px-3 py-1.5 text-xs rounded-md bg-red-600/20 text-red-400 border border-red-500/30 hover:bg-red-600/40 transition-colors"
                >
                    ✕ Reject
                </button>
            </div>
        </div>
    )
}

function LogRow({ entry }: { entry: AutomationLogEntry }) {
    const color = statusColor[entry.status] ?? statusColor.pending
    return (
        <div className="flex items-center justify-between px-3 py-2 rounded-md hover:bg-slate-800/50 transition-colors">
            <div className="flex items-center gap-3 min-w-0">
                <span className={`text-xs px-2 py-0.5 rounded-full border ${color} whitespace-nowrap`}>
                    {entry.status}
                </span>
                <span className="text-sm text-slate-300 truncate">{entry.script_name}</span>
            </div>
            <div className="flex items-center gap-4 text-xs text-slate-500 shrink-0">
                {entry.duration_ms !== null && <span>{entry.duration_ms.toFixed(0)}ms</span>}
                <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
            </div>
        </div>
    )
}

function SkillCard({ skill }: { skill: SkillDefinition }) {
    return (
        <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700">
            <p className="text-sm font-medium text-slate-200">{skill.name}</p>
            <p className="text-xs text-slate-500 mt-1 line-clamp-2">{skill.description}</p>
            {skill.tags.length > 0 && (
                <div className="flex gap-1 flex-wrap mt-2">
                    {skill.tags.map((tag) => (
                        <span
                            key={tag}
                            className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-400"
                        >
                            {tag}
                        </span>
                    ))}
                </div>
            )}
        </div>
    )
}
