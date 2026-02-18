export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'
export const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'

export const AI_MODES = [
  { value: 'manual', label: 'Manual', color: 'text-slate-400' },
  { value: 'auto_pilot', label: 'Auto Pilot', color: 'text-blue-400' },
  { value: 'aggressive_growth', label: 'Aggressive Growth', color: 'text-orange-400' },
  { value: 'profit_guard', label: 'Profit Guard', color: 'text-emerald-400' },
  { value: 'advanced', label: 'Advanced', color: 'text-purple-400' },
  { value: 'autonomous', label: 'Autonomous', color: 'text-cyan-400' },
] as const

export const OPTIMIZATION_STRATEGIES = [
  { value: 'aggressive', label: 'Aggressive', description: 'Rapid growth (30% bid increase max)' },
  { value: 'balanced', label: 'Balanced', description: 'Steady growth (20% bid changes)' },
  { value: 'conservative', label: 'Conservative', description: 'Risk minimization (10% increase max)' },
  { value: 'profit', label: 'Profit', description: 'Maximize margins' },
  { value: 'volume', label: 'Volume', description: 'Maximize impressions' },
] as const
