import { createBrowserRouter, Navigate } from 'react-router-dom'
import { LoginPage } from '@/pages/auth/LoginPage'
import { AdminLayout } from '@/components/layout/AdminLayout'
import { ClientLayout } from '@/components/layout/ClientLayout'
import { ProtectedRoute, RootRedirect } from '@/components/auth/ProtectedRoute'
import { AdminDashboardPage } from '@/pages/admin/DashboardPage'
import { AccountsPage } from '@/pages/admin/AccountsPage'
import { CampaignsPage } from '@/pages/admin/CampaignsPage'
import { OptimizationPage } from '@/pages/admin/OptimizationPage'
import { AnomaliesPage } from '@/pages/admin/AnomaliesPage'
import { CompetitivePage } from '@/pages/admin/CompetitivePage'
import { ChatPage } from '@/pages/admin/ChatPage'
import { SettingsPage } from '@/pages/admin/SettingsPage'
import { ClientDashboardPage } from '@/pages/client/DashboardPage'

export const router = createBrowserRouter([
  { path: '/', element: <RootRedirect /> },
  { path: '/login', element: <LoginPage /> },
  {
    path: '/admin',
    element: (
      <ProtectedRoute allowedRoles={['admin']}>
        <AdminLayout />
      </ProtectedRoute>
    ),
    children: [
      { index: true, element: <Navigate to="/admin/dashboard" replace /> },
      { path: 'dashboard', element: <AdminDashboardPage /> },
      { path: 'accounts', element: <AccountsPage /> },
      { path: 'campaigns', element: <CampaignsPage /> },
      { path: 'optimization', element: <OptimizationPage /> },
      { path: 'anomalies', element: <AnomaliesPage /> },
      { path: 'competitive', element: <CompetitivePage /> },
      { path: 'chat', element: <ChatPage /> },
      { path: 'settings', element: <SettingsPage /> },
    ],
  },
  {
    path: '/client',
    element: (
      <ProtectedRoute allowedRoles={['client']}>
        <ClientLayout />
      </ProtectedRoute>
    ),
    children: [
      { index: true, element: <Navigate to="/client/dashboard" replace /> },
      { path: 'dashboard', element: <ClientDashboardPage /> },
    ],
  },
])
