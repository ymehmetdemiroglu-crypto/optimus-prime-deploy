import { apiClient } from '../client'

export interface DashboardSummary {
  total_sales: number
  ad_spend: number
  acos: number
  roas: number
  velocity_trend: 'up' | 'down' | 'flat'
}

export interface PerformanceMetric {
  timestamp: string
  organic_sales: number
  ad_sales: number
  spend: number
  impressions: number
  clicks: number
}

export interface AIAction {
  id: string
  action_type: string
  description: string
  timestamp: string
  campaign_name?: string
}

export interface ClientMatrixNode {
  id: string
  name: string
  status: 'healthy' | 'warning' | 'critical'
  sales: number
  spend: number
  acos: number
  acos_trend: number
}

export const dashboardApi = {
  getSummary: async (accountId?: number): Promise<DashboardSummary> => {
    const params = accountId ? { account_id: accountId } : {}
    const response = await apiClient.get<DashboardSummary>('/dashboard/summary', { params })
    return response.data
  },

  getChartData: async (accountId?: number, range = '7d'): Promise<PerformanceMetric[]> => {
    const params: Record<string, string | number> = { range }
    if (accountId) params.account_id = accountId
    const response = await apiClient.get<PerformanceMetric[]>('/dashboard/chart-data', { params })
    return response.data
  },

  getAIActions: async (accountId?: number): Promise<AIAction[]> => {
    const params = accountId ? { account_id: accountId } : {}
    const response = await apiClient.get<AIAction[]>('/dashboard/ai-actions', { params })
    return response.data
  },

  getClientMatrix: async (): Promise<ClientMatrixNode[]> => {
    const response = await apiClient.get<ClientMatrixNode[]>('/dashboard/matrix')
    return response.data
  },
}
