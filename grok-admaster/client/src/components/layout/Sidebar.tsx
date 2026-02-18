import { NavLink } from 'react-router-dom'
import { cn } from '@/utils/cn'
import {
  LayoutDashboard, Building2, Target, Zap,
  AlertTriangle, TrendingUp, MessageSquare, Settings,
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/admin/dashboard', icon: LayoutDashboard },
  { name: 'Accounts', href: '/admin/accounts', icon: Building2 },
  { name: 'Campaigns', href: '/admin/campaigns', icon: Target },
  { name: 'Optimization', href: '/admin/optimization', icon: Zap },
  { name: 'Anomalies', href: '/admin/anomalies', icon: AlertTriangle },
  { name: 'Competitive Intel', href: '/admin/competitive', icon: TrendingUp },
  { name: 'AI Chat', href: '/admin/chat', icon: MessageSquare },
  { name: 'Settings', href: '/admin/settings', icon: Settings },
]

function Sidebar() {
  return (
    <aside className="w-64 bg-panel border-r border-slate-800 flex flex-col shrink-0">
      <div className="p-6 border-b border-slate-800">
        <h1 className="text-2xl font-serif font-bold text-slate-100">Optimus Prime</h1>
        <p className="text-xs text-slate-500 mt-1 uppercase tracking-wider">The Data Sanctuary</p>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-4 py-3 rounded-institutional text-sm font-medium transition-colors',
                isActive
                  ? 'bg-navy-600 text-white'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200',
              )
            }
          >
            <item.icon className="h-5 w-5 shrink-0" />
            {item.name}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-800">
        <p className="text-xs text-slate-600 text-center">v1.0.0</p>
      </div>
    </aside>
  )
}

export { Sidebar }
