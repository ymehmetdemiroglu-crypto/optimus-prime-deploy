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
          <label htmlFor={id} className="text-sm font-medium text-slate-300">
            {label}
          </label>
        )}
        <div className="relative">
          {icon && (
            <span className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500">
              {icon}
            </span>
          )}
          <input
            ref={ref}
            id={id}
            className={cn(
              'w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-institutional',
              'text-slate-100 placeholder-slate-500',
              'focus:outline-none focus:ring-2 focus:ring-navy-500 focus:border-transparent',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              icon && 'pl-10',
              error && 'border-danger focus:ring-danger',
              className,
            )}
            aria-invalid={error ? true : undefined}
            {...props}
          />
        </div>
        {error && <p className="text-xs text-danger">{error}</p>}
      </div>
    )
  },
)

Input.displayName = 'Input'

export { Input }
export type { InputProps }
