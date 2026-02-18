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
      <div className="w-full max-w-sm">

        {/* Wordmark */}
        <div className="text-center mb-10">
          <h1 className="text-3xl font-serif font-semibold text-[var(--text-primary)] tracking-tight mb-1">
            Optimus Prime
          </h1>
          <p className="text-[10px] text-[var(--text-subtle)] uppercase tracking-[0.16em] font-medium">
            The Data Sanctuary
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Sign in to your account</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <Input
                label="Email address"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="admin@company.com"
                icon={<Mail className="h-4 w-4" />}
                required
              />

              <Input
                label="Password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                icon={<Lock className="h-4 w-4" />}
                required
              />

              {error && (
                <div className="px-3.5 py-2.5 bg-danger/8 border border-danger/20 rounded-institutional text-danger/90 text-xs">
                  Invalid credentials. Please try again.
                </div>
              )}

              <Button type="submit" className="w-full mt-2" size="lg" loading={isLoading}>
                Sign In
              </Button>
            </form>
          </CardContent>
        </Card>

        <p className="text-center text-[var(--text-subtle)] text-xs mt-6 tracking-wide">
          Contact your administrator for access
        </p>
      </div>
    </div>
  )
}

export { LoginPage }
