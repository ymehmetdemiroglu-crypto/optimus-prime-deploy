import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/api/client'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { EmptyState } from '@/components/ui/EmptyState'
import { AlertTriangle } from 'lucide-react'
import { formatDate } from '@/utils/format'

interface AnomalyAlert {
  id: number
  profile_id: string
  metric_name: string
  severity: string
  message: string
  detected_at: string
  is_resolved: boolean
}

function AnomaliesPage() {
  const { data: alerts } = useQuery({
    queryKey: ['anomaly-alerts'],
    queryFn: async () => {
      const res = await apiClient.get<AnomalyAlert[]>('/anomaly-detection/alerts/all')
      return res.data
    },
  })

  const severityVariant = (s: string) => {
    if (s === 'critical') return 'danger' as const
    if (s === 'high') return 'danger' as const
    if (s === 'medium') return 'warning' as const
    return 'info' as const
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-serif font-bold text-slate-100 mb-1">Anomaly Detection</h1>
        <p className="text-slate-400 text-sm">AI-detected performance anomalies</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Active Alerts</CardTitle>
        </CardHeader>
        <CardContent>
          {!alerts?.length ? (
            <EmptyState icon={<AlertTriangle className="h-12 w-12" />} title="No anomalies detected" description="Your campaigns are performing normally" />
          ) : (
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div key={alert.id} className="flex items-start gap-4 p-4 bg-slate-800/50 rounded-institutional">
                  <AlertTriangle className={`h-5 w-5 shrink-0 mt-0.5 ${alert.severity === 'critical' || alert.severity === 'high' ? 'text-danger' : 'text-warning'}`} />
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant={severityVariant(alert.severity)}>{alert.severity}</Badge>
                      <span className="text-xs text-slate-500">{alert.metric_name}</span>
                    </div>
                    <p className="text-sm text-slate-200">{alert.message}</p>
                    <p className="text-xs text-slate-500 mt-1">{formatDate(alert.detected_at)}</p>
                  </div>
                  {alert.is_resolved && <Badge variant="success">Resolved</Badge>}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export { AnomaliesPage }
