import { useQuery } from '@tanstack/react-query'
import { useActiveAccount } from '@/hooks/useActiveAccount'
import { dashboardApi } from '@/api/endpoints/dashboard'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { MetricCard } from '@/components/charts/MetricCard'
import { PerformanceChart } from '@/components/charts/PerformanceChart'
import { ClientMatrix } from '@/components/features/ClientMatrix'
import { AIActionFeed } from '@/components/features/AIActionFeed'
import { DollarSign, Target, TrendingDown, TrendingUp } from 'lucide-react'

function AdminDashboardPage() {
  const { activeAccount } = useActiveAccount()

  const { data: summary } = useQuery({
    queryKey: ['dashboard-summary', activeAccount?.id],
    queryFn: () => dashboardApi.getSummary(activeAccount?.id),
  })

  const { data: chartData } = useQuery({
    queryKey: ['dashboard-chart', activeAccount?.id],
    queryFn: () => dashboardApi.getChartData(activeAccount?.id, '7d'),
  })

  const { data: matrix } = useQuery({
    queryKey: ['client-matrix'],
    queryFn: () => dashboardApi.getClientMatrix(),
    enabled: !activeAccount,
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-serif font-bold text-slate-100 mb-1">
          {activeAccount ? activeAccount.name : 'Command Center'}
        </h1>
        <p className="text-slate-400 text-sm">
          {activeAccount ? 'Account performance overview' : 'Multi-account overview'}
        </p>
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
          <CardTitle>Performance Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <PerformanceChart data={chartData ?? []} />
        </CardContent>
      </Card>

      {!activeAccount && matrix && matrix.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Client Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <ClientMatrix clients={matrix} />
          </CardContent>
        </Card>
      )}

      <Card variant="glow">
        <CardHeader>
          <CardTitle>AI Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <AIActionFeed accountId={activeAccount?.id} />
        </CardContent>
      </Card>
    </div>
  )
}

export { AdminDashboardPage }
