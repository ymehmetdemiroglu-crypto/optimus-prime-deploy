import type { ReactNode } from 'react'
import { cn } from '@/utils/cn'

interface BadgeProps {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
  children: ReactNode
  className?: string
}

const variantStyles = {
  default: 'bg-surface-high text-[var(--text-muted)] border border-[var(--border)]',
  success: 'bg-profit/10 text-profit border border-profit/20',
  warning: 'bg-warning/10 text-warning border border-warning/20',
  danger:  'bg-danger/10 text-danger border border-danger/20',
  info:    'bg-electric/10 text-electric border border-electric/20',
}

function Badge({ variant = 'default', children, className }: BadgeProps) {
  return (
    <span className={cn(
      'inline-flex items-center px-2 py-0.5 rounded-geometric text-xs font-medium tracking-wide',
      variantStyles[variant],
      className,
    )}>
      {children}
    </span>
  )
}

export { Badge }
export type { BadgeProps }
