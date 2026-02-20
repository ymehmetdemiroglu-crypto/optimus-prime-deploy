import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { accountsApi } from '@/api/endpoints/accounts'
import { authApi } from '@/api/endpoints/auth'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Badge } from '@/components/ui/Badge'
import { Modal } from '@/components/ui/Modal'
import { EmptyState } from '@/components/ui/EmptyState'
import { Plus, Key, UserPlus, Building2 } from 'lucide-react'

function AccountsPage() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [credentialAccountId, setCredentialAccountId] = useState<number | null>(null)
  const [clientUserAccountId, setClientUserAccountId] = useState<number | null>(null)

  const { data: accounts } = useQuery({
    queryKey: ['accounts'],
    queryFn: accountsApi.getAccounts,
  })

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-serif font-bold text-slate-100 mb-1">Client Accounts</h1>
          <p className="text-slate-400 text-sm">Manage Amazon seller accounts and credentials</p>
        </div>
        <Button onClick={() => setShowCreateModal(true)} icon={<Plus className="h-4 w-4" />}>
          Add Account
        </Button>
      </div>

      {!accounts?.length ? (
        <EmptyState
          icon={<Building2 className="h-12 w-12" />}
          title="No accounts yet"
          description="Add your first Amazon seller account to get started"
          action={
            <Button onClick={() => setShowCreateModal(true)} icon={<Plus className="h-4 w-4" />}>
              Add Account
            </Button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {accounts.map((account) => (
            <Card key={account.id}>
              <CardHeader>
                <CardTitle className="text-lg">{account.name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-sm space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Region:</span>
                    <span className="text-slate-200">{account.region}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Profiles:</span>
                    <span className="text-slate-200">{account.profiles?.length ?? 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Status:</span>
                    <Badge variant={account.status === 'active' ? 'success' : 'warning'}>
                      {account.status}
                    </Badge>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" className="flex-1" onClick={() => setCredentialAccountId(account.id)} icon={<Key className="h-4 w-4" />}>
                    Credentials
                  </Button>
                  <Button variant="outline" size="sm" className="flex-1" onClick={() => setClientUserAccountId(account.id)} icon={<UserPlus className="h-4 w-4" />}>
                    Client User
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {showCreateModal && <CreateAccountModal onClose={() => setShowCreateModal(false)} />}
      {credentialAccountId && <CredentialModal accountId={credentialAccountId} onClose={() => setCredentialAccountId(null)} />}
      {clientUserAccountId && <CreateClientUserModal accountId={clientUserAccountId} onClose={() => setClientUserAccountId(null)} />}
    </div>
  )
}

function CreateAccountModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: () => accountsApi.createAccount({ company_name: name, primary_contact_email: email || undefined }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['accounts'] }); onClose() },
  })

  return (
    <Modal title="Add Amazon Account" onClose={onClose}>
      <form onSubmit={(e) => { e.preventDefault(); mutation.mutate() }} className="space-y-4">
        <Input label="Account Name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Brand name" required />
        <Input label="Contact Email (optional)" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="contact@brand.com" />
        <div className="flex gap-3 pt-2">
          <Button type="button" variant="outline" onClick={onClose} className="flex-1">Cancel</Button>
          <Button type="submit" loading={mutation.isPending} className="flex-1">Create</Button>
        </div>
      </form>
    </Modal>
  )
}

function CredentialModal({ accountId, onClose }: { accountId: number; onClose: () => void }) {
  const [clientId, setClientId] = useState('')
  const [clientSecret, setClientSecret] = useState('')
  const [refreshToken, setRefreshToken] = useState('')
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: () => accountsApi.addCredentials(accountId, { account_id: accountId, client_id: clientId, client_secret: clientSecret, refresh_token: refreshToken }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['accounts'] }); onClose() },
  })

  return (
    <Modal title="Amazon API Credentials" onClose={onClose} size="lg">
      <div className="p-3 bg-warning/10 border border-warning/30 rounded-institutional text-warning text-sm mb-4">
        Credentials are encrypted at rest. Never share these values.
      </div>
      <form onSubmit={(e) => { e.preventDefault(); mutation.mutate() }} className="space-y-4">
        <Input label="Client ID" value={clientId} onChange={(e) => setClientId(e.target.value)} placeholder="amzn1.application-oa2-client.xxx" required />
        <Input label="Client Secret" type="password" value={clientSecret} onChange={(e) => setClientSecret(e.target.value)} required />
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-slate-300">Refresh Token</label>
          <textarea
            value={refreshToken}
            onChange={(e) => setRefreshToken(e.target.value)}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-institutional text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-navy-500 min-h-[80px] text-sm"
            required
          />
        </div>
        <div className="flex gap-3 pt-2">
          <Button type="button" variant="outline" onClick={onClose} className="flex-1">Cancel</Button>
          <Button type="submit" loading={mutation.isPending} className="flex-1">Save Credentials</Button>
        </div>
      </form>
    </Modal>
  )
}

function CreateClientUserModal({ accountId, onClose }: { accountId: number; onClose: () => void }) {
  const [email, setEmail] = useState('')
  const [fullName, setFullName] = useState('')
  const [password, setPassword] = useState('')

  const mutation = useMutation({
    mutationFn: () => authApi.createUser({ email, password, full_name: fullName, role: 'client', client_account_id: accountId }),
    onSuccess: onClose,
  })

  return (
    <Modal title="Create Client User" onClose={onClose}>
      <p className="text-sm text-slate-400 mb-4">This user will have read-only access to this account's performance data.</p>
      <form onSubmit={(e) => { e.preventDefault(); mutation.mutate() }} className="space-y-4">
        <Input label="Full Name" value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Client Name" required />
        <Input label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="client@brand.com" required />
        <Input label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Set a strong password" required />
        <div className="flex gap-3 pt-2">
          <Button type="button" variant="outline" onClick={onClose} className="flex-1">Cancel</Button>
          <Button type="submit" loading={mutation.isPending} className="flex-1">Create User</Button>
        </div>
      </form>
    </Modal>
  )
}

export { AccountsPage }
