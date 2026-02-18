import type { ClientMatrixNode } from '@/api/endpoints/dashboard'
import { Badge } from '@/components/ui/Badge'
import { formatCurrency, formatPercent } from '@/utils/format'
import { TrendIndicator } from '@/components/charts/TrendIndicator'

interface ClientMatrixProps {
  clients: ClientMatrixNode[]
}

function ClientMatrix({ clients }: ClientMatrixProps) {
  const statusVariant = (status: string) => {
    if (status === 'healthy') return 'success' as const
    if (status === 'warning') return 'warning' as const
    return 'danger' as const
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {clients.map((client) => (
        <div key={client.id} className="card-sanctuary p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-slate-200">{client.name}</h4>
            <Badge variant={statusVariant(client.status)}>{client.status}</Badge>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <p className="text-xs text-slate-500">Sales</p>
              <p className="text-sm font-data text-slate-200">{formatCurrency(client.sales)}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500">ACoS</p>
              <p className="text-sm font-data text-slate-200">{formatPercent(client.acos)}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500">Trend</p>
              <TrendIndicator value={client.acos_trend} />
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export { ClientMatrix }
