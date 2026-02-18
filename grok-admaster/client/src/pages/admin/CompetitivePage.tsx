import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/api/client'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Search, TrendingUp } from 'lucide-react'

function CompetitivePage() {
  const [asin, setAsin] = useState('')

  const scanMutation = useMutation({
    mutationFn: async (targetAsin: string) => {
      const res = await apiClient.post(`/competitive/price-monitor/${targetAsin}/scan`)
      return res.data
    },
  })

  const forecastMutation = useMutation({
    mutationFn: async (targetAsin: string) => {
      const res = await apiClient.post(`/competitive/forecast/${targetAsin}`)
      return res.data
    },
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-serif font-bold text-slate-100 mb-1">Competitive Intelligence</h1>
        <p className="text-slate-400 text-sm">Price monitoring and competitor analysis</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>ASIN Monitor</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <Input
                placeholder="Enter ASIN (e.g. B0DWK3C1R7)"
                value={asin}
                onChange={(e) => setAsin(e.target.value)}
                icon={<Search className="h-5 w-5" />}
              />
            </div>
            <Button onClick={() => scanMutation.mutate(asin)} disabled={!asin} loading={scanMutation.isPending} icon={<Search className="h-4 w-4" />}>
              Scan Prices
            </Button>
            <Button variant="secondary" onClick={() => forecastMutation.mutate(asin)} disabled={!asin} loading={forecastMutation.isPending} icon={<TrendingUp className="h-4 w-4" />}>
              Forecast
            </Button>
          </div>
        </CardContent>
      </Card>

      {scanMutation.data && (
        <Card>
          <CardHeader>
            <CardTitle>Price Monitor Results</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-sm text-slate-300 font-data bg-slate-800/50 p-4 rounded-institutional overflow-x-auto">
              {JSON.stringify(scanMutation.data, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}

      {forecastMutation.data && (
        <Card variant="glow">
          <CardHeader>
            <CardTitle>Price Forecast</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-sm text-slate-300 font-data bg-slate-800/50 p-4 rounded-institutional overflow-x-auto">
              {JSON.stringify(forecastMutation.data, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export { CompetitivePage }
