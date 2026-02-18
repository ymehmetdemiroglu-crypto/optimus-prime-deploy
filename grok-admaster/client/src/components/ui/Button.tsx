import { forwardRef } from 'react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/utils/cn'

const variantStyles = {
  primary: 'bg-navy-600 hover:bg-navy-700 text-white shadow-lg',
  secondary: 'bg-slate-700 hover:bg-slate-600 text-white',
  outline: 'border-2 border-slate-600 hover:border-slate-500 hover:bg-slate-800 text-slate-200',
  ghost: 'hover:bg-slate-800 text-slate-300',
  danger: 'bg-danger hover:bg-danger-dark text-white',
  success: 'bg-profit hover:bg-profit-dark text-white shadow-glow-emerald',
}

const sizeStyles = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-base',
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
        'inline-flex items-center justify-center gap-2 rounded-institutional font-medium transition-all',
        'focus:outline-none focus:ring-2 focus:ring-navy-500 focus:ring-offset-2 focus:ring-offset-obsidian',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        variantStyles[variant],
        sizeStyles[size],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : icon}
      {children}
    </button>
  ),
)

Button.displayName = 'Button'

export { Button }
export type { ButtonProps }
