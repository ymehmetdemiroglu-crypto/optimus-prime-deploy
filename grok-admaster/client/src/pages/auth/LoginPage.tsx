import { useState } from 'react'
import type { FormEvent } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useAuth } from '@/hooks/useAuth'
import { Lock, Mail } from 'lucide-react'

function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const { login, isLoading, error } = useAuth()

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    login({ email, password })
  }

  return (
    <div className="min-h-screen bg-obsidian flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-serif font-bold text-slate-100 mb-2">
            Optimus Prime
          </h1>
          <p className="text-slate-500 text-sm tracking-wide uppercase">The Data Sanctuary</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Sign In</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <Input
                label="Email Address"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="admin@company.com"
                icon={<Mail className="h-5 w-5" />}
                required
              />

              <Input
                label="Password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                icon={<Lock className="h-5 w-5" />}
                required
              />

              {error && (
                <div className="p-3 bg-danger/10 border border-danger/30 rounded-institutional text-danger text-sm">
                  Invalid email or password. Please try again.
                </div>
              )}

              <Button type="submit" className="w-full" size="lg" loading={isLoading}>
                Sign In
              </Button>
            </form>
          </CardContent>
        </Card>

        <p className="text-center text-slate-600 text-xs mt-6">
          Contact your administrator for account access
        </p>
      </div>
    </div>
  )
}

export { LoginPage }
