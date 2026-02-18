import type { ReactNode } from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { cn } from '@/utils/cn'
import { formatMetric } from '@/utils/format'

interface MetricCardProps {
  title: string
  value: number
  format: 'currency' | 'percentage' | 'number' | 'decimal'
  icon?: ReactNode
  trend?: 'up' | 'down' | 'flat'
  trendValue?: string
  className?: string
}

function MetricCard({ title, value, format, icon, trend, trendValue, className }: MetricCardProps) {
  return (
    <div className={cn('card-sanctuary p-6', className)}>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm font-medium text-slate-400">{title}</p>
        {icon && <span className="text-slate-500">{icon}</span>}
      </div>
      <p className="text-3xl font-bold text-slate-100 font-data">
        {formatMetric(value, format)}
      </p>
      {trend && trend !== 'flat' && (
        <div className={cn('flex items-center gap-1 mt-2 text-sm', trend === 'up' ? 'metric-positive' : 'metric-negative')}>
          {trend === 'up' ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
          {trendValue && <span>{trendValue}</span>}
        </div>
      )}
    </div>
  )
}

export { MetricCard }
export type { MetricCardProps }
