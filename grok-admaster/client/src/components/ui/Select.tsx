import { forwardRef, useId } from 'react'
import type { SelectHTMLAttributes } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/utils/cn'

interface SelectOption {
  value: string
  label: string
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  options: SelectOption[]
  error?: string
  placeholder?: string
}

const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, label, options, error, placeholder, id: externalId, ...props }, ref) => {
    const generatedId = useId()
    const id = externalId ?? generatedId

    return (
      <div className="flex flex-col gap-1.5">
        {label && (
          <label htmlFor={id} className="text-sm font-medium text-slate-300">{label}</label>
        )}
        <div className="relative">
          <select
            ref={ref}
            id={id}
            className={cn(
              'w-full appearance-none px-4 py-2 pr-10 bg-slate-800 border border-slate-700 rounded-institutional',
              'text-slate-100 focus:outline-none focus:ring-2 focus:ring-navy-500 focus:border-transparent',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              error && 'border-danger focus:ring-danger',
              className,
            )}
            {...props}
          >
            {placeholder && <option value="" disabled>{placeholder}</option>}
            {options.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
        </div>
        {error && <p className="text-xs text-danger">{error}</p>}
      </div>
    )
  },
)

Select.displayName = 'Select'

export { Select }
export type { SelectProps, SelectOption }
