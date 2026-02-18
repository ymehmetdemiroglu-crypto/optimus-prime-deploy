import { apiClient } from '../client'
import type { User } from '@/store/authStore'

export interface LoginResponse {
  access_token: string
  token_type: string
}

export const authApi = {
  login: async (email: string, password: string): Promise<LoginResponse> => {
    const formData = new URLSearchParams()
    formData.append('username', email)
    formData.append('password', password)

    const response = await apiClient.post<LoginResponse>('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })
    return response.data
  },

  getMe: async (): Promise<User> => {
    const response = await apiClient.get<User>('/auth/me')
    return response.data
  },

  createUser: async (data: {
    email: string
    password: string
    full_name: string
    role: 'admin' | 'client'
    client_account_id?: number
  }): Promise<User> => {
    const response = await apiClient.post<User>('/auth/users', data)
    return response.data
  },

  listUsers: async (): Promise<User[]> => {
    const response = await apiClient.get<User[]>('/auth/users')
    return response.data
  },

  updateUser: async (userId: number, data: {
    full_name?: string
    password?: string
    is_active?: boolean
    client_account_id?: number
  }): Promise<User> => {
    const response = await apiClient.patch<User>(`/auth/users/${userId}`, data)
    return response.data
  },

  deactivateUser: async (userId: number): Promise<void> => {
    await apiClient.delete(`/auth/users/${userId}`)
  },
}
