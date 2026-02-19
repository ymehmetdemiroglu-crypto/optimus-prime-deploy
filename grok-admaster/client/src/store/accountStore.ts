import { create } from 'zustand'
import type { AccountData } from '@/api/endpoints/accounts'

// AccountData from the API is the canonical account shape.
// Using it directly eliminates the parallel interface definition that was
// previously maintained separately and was prone to drift.
export type Account = AccountData

interface AccountState {
  activeAccount: Account | null
  accounts: Account[]
  setActiveAccount: (account: Account | null) => void
  setAccounts: (accounts: Account[]) => void
}

export const useAccountStore = create<AccountState>()((set) => ({
  activeAccount: null,
  accounts: [],
  setActiveAccount: (account) => set({ activeAccount: account }),
  setAccounts: (accounts) => set({ accounts }),
}))
