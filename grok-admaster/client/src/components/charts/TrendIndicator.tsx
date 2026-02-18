import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { cn } from '@/utils/cn'

interface TrendIndicatorProps {
  value: number
  suffix?: string
  className?: string
}

function TrendIndicator({ value, suffix = '%', className }: TrendIndicatorProps) {
  const isPositive = value > 0
  const isNeutral = value === 0

  return (
    <span className={cn('inline-flex items-center gap-1 text-sm font-medium', isNeutral ? 'text-slate-400' : isPositive ? 'metric-positive' : 'metric-negative', className)}>
      {isNeutral ? <Minus className="h-3 w-3" /> : isPositive ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
      {Math.abs(value).toFixed(1)}{suffix}
    </span>
  )
}

export { TrendIndicator }
