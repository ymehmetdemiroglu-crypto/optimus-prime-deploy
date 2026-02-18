import { NavLink } from 'react-router-dom'
import { cn } from '@/utils/cn'
import {
  LayoutDashboard, Building2, Target, Zap,
  AlertTriangle, TrendingUp, MessageSquare, Settings,
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard',        href: '/admin/dashboard',    icon: LayoutDashboard },
  { name: 'Accounts',         href: '/admin/accounts',     icon: Building2 },
  { name: 'Campaigns',        href: '/admin/campaigns',    icon: Target },
  { name: 'Optimization',     href: '/admin/optimization', icon: Zap },
  { name: 'Anomalies',        href: '/admin/anomalies',    icon: AlertTriangle },
  { name: 'Competitive Intel',href: '/admin/competitive',  icon: TrendingUp },
  { name: 'AI Chat',          href: '/admin/chat',         icon: MessageSquare },
  { name: 'Settings',         href: '/admin/settings',     icon: Settings },
]

function Sidebar() {
  return (
    <aside className="w-60 bg-panel border-r border-[var(--border)] flex flex-col shrink-0">
      {/* Wordmark */}
      <div className="px-6 py-5 border-b border-[var(--border-subtle)]">
        <h1 className="text-xl font-serif font-semibold text-[var(--text-primary)] tracking-tight">
          Optimus Prime
        </h1>
        <p className="text-[10px] text-[var(--text-subtle)] mt-0.5 uppercase tracking-[0.12em] font-medium">
          The Data Sanctuary
        </p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2.5 py-3 space-y-0.5 overflow-y-auto">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3.5 py-2.5 rounded-institutional text-sm font-medium transition-all duration-150',
                isActive
                  ? 'bg-navy-600/12 text-slate-100 border-l-2 border-navy-500 pl-[calc(0.875rem_-_2px)]'
                  : 'text-[var(--text-subtle)] hover:bg-surface-high hover:text-[var(--text-muted)] border-l-2 border-transparent pl-[calc(0.875rem_-_2px)]',
              )
            }
          >
            <item.icon className="h-4 w-4 shrink-0" />
            {item.name}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-3.5 border-t border-[var(--border-subtle)]">
        <p className="text-[10px] text-[var(--text-subtle)] tracking-wider">v1.0.0</p>
      </div>
    </aside>
  )
}

export { Sidebar }
