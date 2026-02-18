import { useAuth } from '@/hooks/useAuth'
import { AccountSwitcher } from './AccountSwitcher'
import { Button } from '@/components/ui/Button'
import { LogOut, User } from 'lucide-react'

function Header() {
  const { user, logout } = useAuth()

  return (
    <header className="h-16 bg-panel border-b border-slate-800 flex items-center justify-between px-6 shrink-0">
      <AccountSwitcher />

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3 text-sm">
          <User className="h-5 w-5 text-slate-400" />
          <div>
            <p className="text-slate-200 font-medium">{user?.full_name}</p>
            <p className="text-slate-500 text-xs capitalize">{user?.role}</p>
          </div>
        </div>

        <Button variant="ghost" size="sm" onClick={logout} icon={<LogOut className="h-4 w-4" />}>
          Logout
        </Button>
      </div>
    </header>
  )
}

export { Header }
