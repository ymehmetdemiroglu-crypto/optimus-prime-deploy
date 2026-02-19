import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { useAuthStore } from '@/store/authStore'
import { authApi } from '@/api/endpoints/auth'
import { apiClient } from '@/api/client'

export function useAuth() {
  const navigate = useNavigate()
  const { setAuth, logout: logoutStore, user, isAuthenticated } = useAuthStore()

  const loginMutation = useMutation({
    mutationFn: ({ email, password }: { email: string; password: string }) =>
      authApi.login(email, password),
    onSuccess: async (data) => {
      // Fetch user profile using the token directly (store not yet updated)
      try {
        const meResponse = await apiClient.get('/auth/me', {
          headers: { Authorization: `Bearer ${data.access_token}` },
        })
        setAuth(data.access_token, meResponse.data)
        navigate(meResponse.data.role === 'admin' ? '/admin/dashboard' : '/client/dashboard')
      } catch (err) {
        console.error('Failed to fetch user profile after login:', err)
        // Surface error through the mutation so callers can react
        throw err
      }
    },
  })

  const logout = () => {
    logoutStore()
    navigate('/login')
  }

  return {
    login: loginMutation.mutate,
    logout,
    isLoading: loginMutation.isPending,
    error: loginMutation.error,
    user,
    isAuthenticated,
  }
}
