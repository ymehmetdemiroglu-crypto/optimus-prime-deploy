import { useEffect, useRef, useState, useCallback } from 'react'
import { WS_BASE_URL } from '@/utils/constants'

export interface ChatMessage {
  id: string
  sender: 'user' | 'optimus'
  content: string
  timestamp: string
}

export function useWebSocket(clientId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  const connect = useCallback(() => {
    const ws = new WebSocket(`${WS_BASE_URL}/ws/${clientId}`)
    wsRef.current = ws

    ws.onopen = () => setIsConnected(true)

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string) as ChatMessage
        setMessages((prev) => [...prev, {
          ...data,
          id: data.id || `optimus-${Date.now()}`,
          sender: 'optimus',
          timestamp: data.timestamp || new Date().toISOString(),
        }])
      } catch {
        setMessages((prev) => [...prev, {
          id: `optimus-${Date.now()}`,
          sender: 'optimus',
          content: String(event.data),
          timestamp: new Date().toISOString(),
        }])
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => ws.close()
  }, [clientId])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  const sendMessage = useCallback((content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(content)
      setMessages((prev) => [...prev, {
        id: `user-${Date.now()}`,
        sender: 'user',
        content,
        timestamp: new Date().toISOString(),
      }])
    }
  }, [])

  return { messages, sendMessage, isConnected }
}
