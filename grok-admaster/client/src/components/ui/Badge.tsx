import type { ReactNode } from 'react'
import { cn } from '@/utils/cn'

interface BadgeProps {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
  children: ReactNode
  className?: string
}

const variantStyles = {
  default: 'bg-slate-700 text-slate-300',
  success: 'bg-profit/20 text-profit',
  warning: 'bg-warning/20 text-warning',
  danger: 'bg-danger/20 text-danger',
  info: 'bg-electric/20 text-electric',
}

function Badge({ variant = 'default', children, className }: BadgeProps) {
  return (
    <span className={cn('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', variantStyles[variant], className)}>
      {children}
    </span>
  )
}

export { Badge }
export type { BadgeProps }
