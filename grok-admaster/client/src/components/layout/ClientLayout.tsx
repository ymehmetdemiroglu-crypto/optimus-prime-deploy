import { Outlet } from 'react-router-dom'
import { useAuth } from '@/hooks/useAuth'
import { Button } from '@/components/ui/Button'
import { LogOut, User, BarChart3 } from 'lucide-react'

function ClientLayout() {
  const { user, logout } = useAuth()

  return (
    <div className="min-h-screen bg-obsidian">
      <header className="h-16 bg-panel border-b border-slate-800 flex items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <BarChart3 className="h-8 w-8 text-electric" />
          <div>
            <h1 className="text-xl font-serif font-bold text-slate-100">Optimus Prime</h1>
            <p className="text-xs text-slate-500 uppercase tracking-wider">Client Portal</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3 text-sm">
            <User className="h-5 w-5 text-slate-400" />
            <div>
              <p className="text-slate-200 font-medium">{user?.full_name}</p>
              <p className="text-slate-500 text-xs">Read-only Access</p>
            </div>
          </div>

          <Button variant="ghost" size="sm" onClick={logout} icon={<LogOut className="h-4 w-4" />}>
            Logout
          </Button>
        </div>
      </header>

      <main className="p-6 max-w-7xl mx-auto">
        <Outlet />
      </main>
    </div>
  )
}

export { ClientLayout }
