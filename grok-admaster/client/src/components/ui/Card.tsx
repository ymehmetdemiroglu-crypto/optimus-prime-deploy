import { forwardRef } from 'react'
import type { HTMLAttributes } from 'react'
import { cn } from '@/utils/cn'

type CardVariant = 'default' | 'glow'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: CardVariant
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'card-sanctuary p-6 transition-shadow duration-200',
        variant === 'glow' && 'ai-glow',
        className,
      )}
      {...props}
    />
  ),
)

Card.displayName = 'Card'

const CardHeader = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('mb-5 flex items-center justify-between', className)}
      {...props}
    />
  ),
)

CardHeader.displayName = 'CardHeader'

const CardTitle = forwardRef<HTMLHeadingElement, HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn('font-serif font-semibold text-[var(--text-primary)] text-lg', className)}
      {...props}
    />
  ),
)

CardTitle.displayName = 'CardTitle'

const CardContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn(className)} {...props} />
  ),
)

CardContent.displayName = 'CardContent'

export { Card, CardHeader, CardTitle, CardContent }
export type { CardProps, CardVariant }
