import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { authApi } from '@/api/endpoints/auth'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Badge } from '@/components/ui/Badge'
import { Modal } from '@/components/ui/Modal'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/Table'
import { UserPlus, Shield, Eye } from 'lucide-react'
import { formatDate } from '@/utils/format'

function SettingsPage() {
  const [showCreateUser, setShowCreateUser] = useState(false)
  const queryClient = useQueryClient()

  const { data: users } = useQuery({
    queryKey: ['users'],
    queryFn: authApi.listUsers,
  })

  const toggleMutation = useMutation({
    mutationFn: ({ id, is_active }: { id: number; is_active: boolean }) =>
      authApi.updateUser(id, { is_active }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['users'] }),
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-serif font-bold text-slate-100 mb-1">Settings</h1>
        <p className="text-slate-400 text-sm">User management and system configuration</p>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between w-full">
            <CardTitle>User Management</CardTitle>
            <Button size="sm" onClick={() => setShowCreateUser(true)} icon={<UserPlus className="h-4 w-4" />}>
              Add User
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>User</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Last Login</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {users?.map((user) => (
                <TableRow key={user.id}>
                  <TableCell>
                    <div>
                      <p className="font-medium text-slate-200">{user.full_name}</p>
                      <p className="text-xs text-slate-500">{user.email}</p>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant={user.role === 'admin' ? 'info' : 'default'}>
                      {user.role === 'admin' ? <><Shield className="h-3 w-3 mr-1" /> Admin</> : <><Eye className="h-3 w-3 mr-1" /> Client</>}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={user.is_active ? 'success' : 'danger'}>
                      {user.is_active ? 'Active' : 'Inactive'}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs font-data">
                    {user.last_login ? formatDate(user.last_login) : 'Never'}
                  </TableCell>
                  <TableCell className="text-xs font-data">
                    {user.created_at ? formatDate(user.created_at) : '-'}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant={user.is_active ? 'danger' : 'success'}
                      size="sm"
                      onClick={() => toggleMutation.mutate({ id: user.id, is_active: !user.is_active })}
                    >
                      {user.is_active ? 'Deactivate' : 'Activate'}
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {showCreateUser && <CreateUserModal onClose={() => setShowCreateUser(false)} />}
    </div>
  )
}

function CreateUserModal({ onClose }: { onClose: () => void }) {
  const [email, setEmail] = useState('')
  const [fullName, setFullName] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState('client')
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: () => authApi.createUser({
      email, password, full_name: fullName, role: role as 'admin' | 'client',
    }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['users'] }); onClose() },
  })

  return (
    <Modal title="Create User" onClose={onClose}>
      <form onSubmit={(e) => { e.preventDefault(); mutation.mutate() }} className="space-y-4">
        <Input label="Full Name" value={fullName} onChange={(e) => setFullName(e.target.value)} required />
        <Input label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <Input label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-slate-300">Role</label>
          <div className="flex gap-3">
            <button type="button" onClick={() => setRole('client')} className={cn_role(role === 'client')}>
              <Eye className="h-4 w-4" /> Client (Read-only)
            </button>
            <button type="button" onClick={() => setRole('admin')} className={cn_role(role === 'admin')}>
              <Shield className="h-4 w-4" /> Admin (Full Access)
            </button>
          </div>
        </div>
        <div className="flex gap-3 pt-2">
          <Button type="button" variant="outline" onClick={onClose} className="flex-1">Cancel</Button>
          <Button type="submit" loading={mutation.isPending} className="flex-1">Create</Button>
        </div>
      </form>
    </Modal>
  )
}

function cn_role(active: boolean) {
  return `flex items-center gap-2 flex-1 px-4 py-3 rounded-institutional text-sm font-medium transition-colors ${
    active ? 'bg-navy-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
  }`
}

export { SettingsPage }
