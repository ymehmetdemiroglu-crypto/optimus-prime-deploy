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

  useEffect(() => {
    if (data) {
      setAccounts(
        data.map((a) => ({
          id: a.id,
          name: a.name,
          amazon_account_id: a.amazon_account_id,
          region: a.region,
          status: a.status,
          profiles: a.profiles?.map((p) => ({
            profile_id: p.profile_id,
            country_code: p.country_code,
            currency_code: p.currency_code,
            is_active: p.is_active,
          })),
        }))
      )
    }
  }, [data, setAccounts])

  return { activeAccount, accounts, setActiveAccount }
}
