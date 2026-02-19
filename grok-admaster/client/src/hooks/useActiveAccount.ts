import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useAccountStore } from '@/store/accountStore'
import { accountsApi } from '@/api/endpoints/accounts'

export function useActiveAccount() {
  const { activeAccount, accounts, setActiveAccount, setAccounts } = useAccountStore()

  const { data } = useQuery({
    queryKey: ['accounts'],
    queryFn: accountsApi.getAccounts,
    staleTime: 5 * 60 * 1000,
  })

  // Store type is AccountData, so data can be passed directly without reshaping.
  useEffect(() => {
    if (data) setAccounts(data)
  }, [data, setAccounts])

  return { activeAccount, accounts, setActiveAccount }
}
