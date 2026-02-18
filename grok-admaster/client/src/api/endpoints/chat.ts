import { WS_BASE_URL } from '@/utils/constants'

export interface ChatMessage {
  id: string
  sender: 'user' | 'optimus'
  content: string
  timestamp: string
}

export function createChatSocket(clientId: string): WebSocket {
  return new WebSocket(`${WS_BASE_URL}/ws/${clientId}`)
}
