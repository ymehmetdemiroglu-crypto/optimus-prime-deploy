import { forwardRef } from 'react'
import type { HTMLAttributes, TdHTMLAttributes, ThHTMLAttributes } from 'react'
import { cn } from '@/utils/cn'

const Table = forwardRef<HTMLTableElement, HTMLAttributes<HTMLTableElement>>(
  ({ className, ...props }, ref) => (
    <table ref={ref} className={cn('w-full text-left', className)} {...props} />
  ),
)
Table.displayName = 'Table'

const TableHeader = forwardRef<HTMLTableSectionElement, HTMLAttributes<HTMLTableSectionElement>>(
  ({ className, ...props }, ref) => <thead ref={ref} className={cn(className)} {...props} />,
)
TableHeader.displayName = 'TableHeader'

const TableBody = forwardRef<HTMLTableSectionElement, HTMLAttributes<HTMLTableSectionElement>>(
  ({ className, ...props }, ref) => (
    <tbody ref={ref} className={cn('divide-y divide-slate-800', className)} {...props} />
  ),
)
TableBody.displayName = 'TableBody'

const TableRow = forwardRef<HTMLTableRowElement, HTMLAttributes<HTMLTableRowElement>>(
  ({ className, ...props }, ref) => (
    <tr ref={ref} className={cn('hover:bg-slate-800/50 transition-colors', className)} {...props} />
  ),
)
TableRow.displayName = 'TableRow'

const TableHead = forwardRef<HTMLTableCellElement, ThHTMLAttributes<HTMLTableCellElement>>(
  ({ className, ...props }, ref) => (
    <th ref={ref} className={cn('px-4 py-3 text-xs font-medium text-slate-400 uppercase tracking-wider', className)} {...props} />
  ),
)
TableHead.displayName = 'TableHead'

const TableCell = forwardRef<HTMLTableCellElement, TdHTMLAttributes<HTMLTableCellElement>>(
  ({ className, ...props }, ref) => (
    <td ref={ref} className={cn('px-4 py-3 text-sm text-slate-300', className)} {...props} />
  ),
)
TableCell.displayName = 'TableCell'

export { Table, TableHeader, TableBody, TableRow, TableHead, TableCell }
