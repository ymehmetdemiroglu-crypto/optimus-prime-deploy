import { forwardRef } from 'react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/utils/cn'

const variantStyles = {
  primary:   'bg-navy-600 hover:bg-navy-700 text-white shadow-glow-navy tracking-wide',
  secondary: 'bg-surface-high hover:bg-slate-700/70 text-slate-200 border border-[var(--border)] hover:border-slate-600',
  outline:   'border border-[var(--border)] hover:border-navy-600/50 hover:bg-surface-high text-slate-400 hover:text-slate-100',
  ghost:     'hover:bg-surface-high text-slate-400 hover:text-slate-200',
  danger:    'bg-danger hover:bg-danger-dark text-white tracking-wide',
  success:   'bg-profit hover:bg-profit-dark text-white shadow-glow-emerald tracking-wide',
}

const sizeStyles = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-sm',
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: keyof typeof variantStyles
  size?: keyof typeof sizeStyles
  loading?: boolean
  icon?: ReactNode
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', loading, icon, children, disabled, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-institutional font-medium',
        'transition-all duration-150 ease-out',
        'focus:outline-none focus:ring-2 focus:ring-navy-500/70 focus:ring-offset-2 focus:ring-offset-obsidian',
        'disabled:opacity-40 disabled:cursor-not-allowed disabled:pointer-events-none',
        variantStyles[variant],
        sizeStyles[size],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin opacity-70" /> : icon}
      {children}
    </button>
  ),
)

Button.displayName = 'Button'

export { Button }
export type { ButtonProps }
