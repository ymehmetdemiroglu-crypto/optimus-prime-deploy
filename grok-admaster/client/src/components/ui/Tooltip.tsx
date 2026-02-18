import type { ReactNode } from 'react'
import { cn } from '@/utils/cn'

interface TooltipProps {
  content: string
  children: ReactNode
  className?: string
}

function Tooltip({ content, children, className }: TooltipProps) {
  return (
    <div className={cn('group relative inline-flex', className)}>
      {children}
      <div className="invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-opacity absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-slate-700 text-slate-200 text-sm rounded-geometric whitespace-nowrap z-50">
        {content}
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-700" />
      </div>
    </div>
  )
}

export { Tooltip }
export type { TooltipProps }
