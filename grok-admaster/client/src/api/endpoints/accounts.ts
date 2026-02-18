import { apiClient } from '../client'

export interface Profile {
  profile_id: string
  account_id: number
  country_code: string
  currency_code: string
  timezone: string
  account_info_id: string
  is_active: boolean
}

export interface AccountData {
  id: number
  name: string
  amazon_account_id: string
  region: string
  status: string
  created_at: string
  profiles: Profile[]
}

export interface CredentialPayload {
  account_id: number
  client_id: string
  client_secret: string
  refresh_token: string
}

export const accountsApi = {
  getAccounts: async (): Promise<AccountData[]> => {
    const response = await apiClient.get<AccountData[]>('/accounts')
    return response.data
  },

  createAccount: async (data: {
    company_name: string
    primary_contact_email?: string
  }): Promise<AccountData> => {
    const response = await apiClient.post<AccountData>('/accounts', data)
    return response.data
  },

  addCredentials: async (accountId: number, credentials: CredentialPayload): Promise<void> => {
    await apiClient.post(`/accounts/${accountId}/credentials`, credentials)
  },

  getProfiles: async (accountId: number): Promise<Profile[]> => {
    const response = await apiClient.get<Profile[]>(`/accounts/${accountId}/profiles`)
    return response.data
  },
}
