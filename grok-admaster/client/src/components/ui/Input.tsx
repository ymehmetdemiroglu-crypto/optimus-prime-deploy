import { forwardRef, useId } from 'react'
import type { InputHTMLAttributes, ReactNode } from 'react'
import { cn } from '@/utils/cn'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  icon?: ReactNode
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, icon, id: externalId, ...props }, ref) => {
    const generatedId = useId()
    const id = externalId ?? generatedId

    return (
      <div className="flex flex-col gap-1.5">
        {label && (
          <label htmlFor={id} className="text-xs font-medium tracking-wide uppercase text-[var(--text-muted)]">
            {label}
          </label>
        )}
        <div className="relative">
          {icon && (
            <span className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-subtle)]">
              {icon}
            </span>
          )}
          <input
            ref={ref}
            id={id}
            className={cn(
              'w-full px-3.5 py-2.5 bg-surface border border-[var(--border)] rounded-institutional',
              'text-[var(--text-primary)] placeholder-[var(--text-subtle)] text-sm',
              'transition-all duration-150 ease-out',
              'focus:outline-none focus:ring-2 focus:ring-navy-500/50 focus:border-navy-600/60',
              'hover:border-slate-600/80',
              'disabled:opacity-40 disabled:cursor-not-allowed',
              icon && 'pl-10',
              error && 'border-danger/60 focus:ring-danger/40 focus:border-danger/60',
              className,
            )}
            aria-invalid={error ? true : undefined}
            {...props}
          />
        </div>
        {error && <p className="text-xs text-danger/90">{error}</p>}
      </div>
    )
  },
)

Input.displayName = 'Input'

export { Input }
export type { InputProps }
