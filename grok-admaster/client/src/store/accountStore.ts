import { create } from 'zustand'

export interface Account {
  id: number
  name: string
  amazon_account_id?: string
  region?: string
  status?: string
  is_active?: boolean
  profiles?: Array<{
    profile_id: string
    country_code: string
    currency_code: string
    is_active: boolean
  }>
}

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
