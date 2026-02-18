import { useQuery } from '@tanstack/react-query'
import { useAuthStore } from '@/store/authStore'
import { dashboardApi } from '@/api/endpoints/dashboard'
import { campaignsApi } from '@/api/endpoints/campaigns'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { MetricCard } from '@/components/charts/MetricCard'
import { PerformanceChart } from '@/components/charts/PerformanceChart'
import { CampaignTable } from '@/components/features/CampaignTable'
import { EmptyState } from '@/components/ui/EmptyState'
import { Lock, DollarSign, Target, TrendingDown, TrendingUp } from 'lucide-react'

function ClientDashboardPage() {
  const { user } = useAuthStore()
  const accountId = user?.client_account_id

  const { data: summary } = useQuery({
    queryKey: ['dashboard-summary', accountId],
    queryFn: () => dashboardApi.getSummary(accountId!),
    enabled: !!accountId,
  })

  const { data: chartData } = useQuery({
    queryKey: ['dashboard-chart', accountId],
    queryFn: () => dashboardApi.getChartData(accountId!, '30d'),
    enabled: !!accountId,
  })

  const { data: campaigns } = useQuery({
    queryKey: ['campaigns', accountId],
    queryFn: () => campaignsApi.getCampaigns(accountId!),
    enabled: !!accountId,
  })

  if (!accountId) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <EmptyState
          icon={<Lock className="h-12 w-12" />}
          title="No Account Assigned"
          description="Contact your administrator to link your account to this profile."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-serif font-bold text-slate-100 mb-1">Performance Dashboard</h1>
        <p className="text-slate-400 text-sm">Your Amazon PPC performance overview</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Sales"
          value={summary?.total_sales ?? 0}
          format="currency"
          icon={<DollarSign className="h-5 w-5" />}
          trend={summary?.velocity_trend}
        />
        <MetricCard
          title="Ad Spend"
          value={summary?.ad_spend ?? 0}
          format="currency"
          icon={<Target className="h-5 w-5" />}
        />
        <MetricCard
          title="ACoS"
          value={summary?.acos ?? 0}
          format="percentage"
          icon={<TrendingDown className="h-5 w-5" />}
        />
        <MetricCard
          title="ROAS"
          value={summary?.roas ?? 0}
          format="decimal"
          icon={<TrendingUp className="h-5 w-5" />}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>30-Day Performance Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <PerformanceChart data={chartData ?? []} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Campaign Performance</CardTitle>
        </CardHeader>
        <CardContent>
          {campaigns?.length ? (
            <CampaignTable campaigns={campaigns} readonly />
          ) : (
            <p className="text-slate-500 text-sm py-4">No campaign data available</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export { ClientDashboardPage }
