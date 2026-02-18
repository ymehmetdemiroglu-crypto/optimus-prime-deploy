import { useAuth } from '@/hooks/useAuth'
import { AccountSwitcher } from './AccountSwitcher'
import { Button } from '@/components/ui/Button'
import { LogOut } from 'lucide-react'

function Header() {
  const { user, logout } = useAuth()

  return (
    <header className="h-14 bg-panel border-b border-[var(--border)] flex items-center justify-between px-6 shrink-0">
      <AccountSwitcher />

      <div className="flex items-center gap-3">
        <div className="text-right">
          <p className="text-sm font-medium text-[var(--text-primary)] leading-tight">{user?.full_name}</p>
          <p className="text-[10px] text-[var(--text-subtle)] uppercase tracking-wider capitalize">{user?.role}</p>
        </div>

        <div className="w-px h-6 bg-[var(--border)]" />

        <Button variant="ghost" size="sm" onClick={logout} icon={<LogOut className="h-3.5 w-3.5" />}>
          Sign out
        </Button>
      </div>
    </header>
  )
}

export { Header }
