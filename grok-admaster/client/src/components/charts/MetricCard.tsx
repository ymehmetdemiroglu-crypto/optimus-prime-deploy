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
    <div className={cn('card-sanctuary p-5', className)}>
      <div className="flex items-start justify-between mb-4">
        <p className="text-xs font-medium uppercase tracking-wider text-[var(--text-subtle)]">{title}</p>
        {icon && <span className="text-[var(--text-subtle)] opacity-60">{icon}</span>}
      </div>

      <p className="text-2xl font-bold text-[var(--text-primary)] font-data leading-none">
        {formatMetric(value, format)}
      </p>

      {trend && trend !== 'flat' && (
        <div className={cn(
          'flex items-center gap-1.5 mt-3 text-xs font-medium',
          trend === 'up' ? 'metric-positive' : 'metric-negative',
        )}>
          {trend === 'up'
            ? <TrendingUp className="h-3.5 w-3.5" />
            : <TrendingDown className="h-3.5 w-3.5" />
          }
          {trendValue && <span>{trendValue}</span>}
        </div>
      )}
    </div>
  )
}

export { MetricCard }
export type { MetricCardProps }
