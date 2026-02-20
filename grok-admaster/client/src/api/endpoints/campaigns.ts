import { apiClient } from '../client'

export interface Campaign {
  id: string
  name: string
  campaign_type: string | null
  targeting_type: string | null
  status: string
  state: string
  daily_budget: number
  ai_mode: string
  target_acos: number | null
  target_roas: number | null
  spend: number
  sales: number
  clicks: number
  impressions: number
  orders: number
}

export const campaignsApi = {
  getCampaigns: async (accountId?: number): Promise<Campaign[]> => {
    const params = accountId ? { account_id: accountId } : {}
    const response = await apiClient.get<Campaign[]>('/campaigns', { params })
    return response.data
  },

  getCampaign: async (campaignId: string): Promise<Campaign> => {
    const response = await apiClient.get<Campaign>(`/campaigns/${campaignId}`)
    return response.data
  },

  updateStrategy: async (campaignId: string, aiMode: string): Promise<Campaign> => {
    const response = await apiClient.patch<Campaign>(`/campaigns/${campaignId}/strategy`, {
      ai_mode: aiMode,
    })
    return response.data
  },
}
