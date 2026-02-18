import { useState } from 'react'
import { useActiveAccount } from '@/hooks/useActiveAccount'
import { Building2, ChevronDown } from 'lucide-react'

function AccountSwitcher() {
  const { activeAccount, accounts, setActiveAccount } = useActiveAccount()
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-3 px-4 py-2 bg-slate-800 rounded-institutional hover:bg-slate-700 transition-colors"
      >
        <Building2 className="h-5 w-5 text-electric" />
        <div className="text-left">
          <p className="text-sm font-medium text-slate-200">
            {activeAccount?.name || 'All Accounts'}
          </p>
          <p className="text-xs text-slate-500">
            {accounts.length} account{accounts.length !== 1 ? 's' : ''}
          </p>
        </div>
        <ChevronDown className="h-4 w-4 text-slate-400" />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute top-full left-0 mt-2 w-64 bg-panel border border-slate-700 rounded-institutional shadow-sanctuary z-50">
            <div className="p-2 border-b border-slate-800">
              <button
                onClick={() => { setActiveAccount(null); setOpen(false) }}
                className="w-full text-left px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 rounded-geometric"
              >
                All Accounts (Overview)
              </button>
            </div>
            <div className="p-2 max-h-64 overflow-y-auto">
              {accounts.map((account) => (
                <button
                  key={account.id}
                  onClick={() => { setActiveAccount(account); setOpen(false) }}
                  className="w-full text-left px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 rounded-geometric flex items-center justify-between"
                >
                  <span>{account.name}</span>
                  {account.region && (
                    <span className="text-xs text-slate-500">{account.region}</span>
                  )}
                </button>
              ))}
              {accounts.length === 0 && (
                <p className="px-3 py-2 text-sm text-slate-500">No accounts yet</p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export { AccountSwitcher }
