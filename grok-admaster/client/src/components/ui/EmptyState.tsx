import type { ReactNode } from 'react'
import { cn } from '@/utils/cn'

interface EmptyStateProps {
  icon: ReactNode
  title: string
  description?: string
  action?: ReactNode
  className?: string
}

function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center py-12 text-center', className)}>
      <div className="w-12 h-12 text-slate-600 mb-4">{icon}</div>
      <h3 className="text-lg font-serif font-bold text-slate-400">{title}</h3>
      {description && <p className="text-sm text-slate-500 mt-2 max-w-sm">{description}</p>}
      {action && <div className="mt-6">{action}</div>}
    </div>
  )
}

export { EmptyState }
export type { EmptyStateProps }
