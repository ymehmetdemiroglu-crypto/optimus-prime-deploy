import type { ReactNode, MouseEvent } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/utils/cn'

interface ModalProps {
  title: string
  onClose: () => void
  children: ReactNode
  size?: 'sm' | 'md' | 'lg'
}

const sizeStyles = { sm: 'max-w-sm', md: 'max-w-lg', lg: 'max-w-2xl' }

function Modal({ title, onClose, children, size = 'md' }: ModalProps) {
  const handleBackdropClick = (e: MouseEvent) => {
    if (e.target === e.currentTarget) onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={handleBackdropClick}>
      <div className={cn('card-sanctuary p-0 w-full mx-4', sizeStyles[size])}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <h2 className="text-lg font-serif font-bold text-slate-100">{title}</h2>
          <button onClick={onClose} className="p-1 hover:bg-slate-800 rounded-geometric text-slate-400 hover:text-slate-200 transition-colors">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="px-6 py-4">{children}</div>
      </div>
    </div>
  )
}

export { Modal }
export type { ModalProps }
