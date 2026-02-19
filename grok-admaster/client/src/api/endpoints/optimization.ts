import { apiClient } from '../client'
import { buildAccountParams } from '@/utils/api'

export interface OptimizationPlan {
  campaign_id: number
  changes: Array<{
    type: string
    description: string
    current_value: number
    proposed_value: number
  }>
  estimated_impact: {
    acos_change: number
    spend_change: number
  }
}

export interface OptimizationAlert {
  id: string
  campaign_id: number
  campaign_name: string
  alert_type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  timestamp: string
}

export const optimizationApi = {
  generatePlan: async (campaignId: number, strategy?: string): Promise<OptimizationPlan> => {
    const params = strategy ? { strategy } : {}
    const response = await apiClient.post<OptimizationPlan>(
      `/optimization/plan/${campaignId}`,
      {},
      { params }
    )
    return response.data
  },

  executePlan: async (campaignId: number): Promise<{ status: string; message: string }> => {
    const response = await apiClient.post(`/optimization/execute/${campaignId}`)
    return response.data
  },

  getAlerts: async (accountId?: number): Promise<OptimizationAlert[]> => {
    const response = await apiClient.get<OptimizationAlert[]>('/optimization/alerts', {
      params: buildAccountParams(accountId),
    })
    return response.data
  },
}
