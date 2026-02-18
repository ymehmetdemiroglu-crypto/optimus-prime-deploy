import { useState, useRef, useEffect } from 'react'
import type { FormEvent } from 'react'
import { useWebSocket } from '@/hooks/useWebSocket'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Send, Bot, User } from 'lucide-react'
import { cn } from '@/utils/cn'
import { formatTime } from '@/utils/format'

const QUICK_ACTIONS = [
  'Analyze campaign performance',
  'Suggest bid optimizations',
  'Generate weekly report',
  'Find wasted spend',
]

function ChatPage() {
  const [input, setInput] = useState('')
  const { messages, sendMessage, isConnected } = useWebSocket('admin-chat')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (input.trim() && isConnected) {
      sendMessage(input.trim())
      setInput('')
    }
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      <div className="mb-4">
        <h1 className="text-3xl font-serif font-bold text-slate-100 mb-1">AI Assistant</h1>
        <p className="text-slate-400 text-sm">Chat with Optimus for campaign insights and optimization</p>
      </div>

      <Card className="flex-1 flex flex-col p-0 overflow-hidden" variant="glow">
        <CardHeader className="px-6 py-4 border-b border-slate-800 mb-0">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-electric" />
            <CardTitle className="text-base">Optimus AI Agent</CardTitle>
            <Badge variant={isConnected ? 'success' : 'danger'} className="ml-auto">
              {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <Bot className="h-12 w-12 text-slate-600 mx-auto mb-4" />
              <p className="text-slate-400 mb-4">Ask Optimus anything about your campaigns</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {QUICK_ACTIONS.map((action) => (
                  <button
                    key={action}
                    onClick={() => sendMessage(action)}
                    className="px-3 py-1.5 text-sm bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-full transition-colors"
                    disabled={!isConnected}
                  >
                    {action}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={cn('flex gap-3', msg.sender === 'user' ? 'justify-end' : 'justify-start')}>
              {msg.sender === 'optimus' && (
                <div className="shrink-0 w-8 h-8 bg-electric/20 rounded-full flex items-center justify-center">
                  <Bot className="h-4 w-4 text-electric" />
                </div>
              )}
              <div className={cn('max-w-[70%] rounded-lg p-4', msg.sender === 'user' ? 'bg-navy-600 text-white' : 'bg-slate-800 text-slate-200')}>
                <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
                <p className="text-xs opacity-50 mt-2 font-data">{formatTime(msg.timestamp)}</p>
              </div>
              {msg.sender === 'user' && (
                <div className="shrink-0 w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center">
                  <User className="h-4 w-4 text-slate-300" />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </CardContent>

        <div className="border-t border-slate-800 px-6 py-4">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isConnected ? 'Ask Optimus about your campaigns...' : 'Connecting...'}
              className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-institutional text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-electric"
              disabled={!isConnected}
            />
            <Button type="submit" disabled={!isConnected || !input.trim()} icon={<Send className="h-4 w-4" />}>
              Send
            </Button>
          </form>
        </div>
      </Card>
    </div>
  )
}

export { ChatPage }
