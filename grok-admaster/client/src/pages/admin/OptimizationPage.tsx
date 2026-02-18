import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useActiveAccount } from '@/hooks/useActiveAccount'
import { campaignsApi } from '@/api/endpoints/campaigns'
import { optimizationApi } from '@/api/endpoints/optimization'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Select } from '@/components/ui/Select'
import { Badge } from '@/components/ui/Badge'
import { Zap, Play } from 'lucide-react'
import { OPTIMIZATION_STRATEGIES } from '@/utils/constants'

function OptimizationPage() {
  const { activeAccount } = useActiveAccount()
  const [selectedCampaign, setSelectedCampaign] = useState('')
  const [strategy, setStrategy] = useState('balanced')

  const { data: campaigns } = useQuery({
    queryKey: ['campaigns', activeAccount?.id],
    queryFn: () => campaignsApi.getCampaigns(activeAccount?.id),
  })

  const planMutation = useMutation({
    mutationFn: () => optimizationApi.generatePlan(Number(selectedCampaign), strategy),
  })

  const executeMutation = useMutation({
    mutationFn: () => optimizationApi.executePlan(Number(selectedCampaign)),
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-serif font-bold text-slate-100 mb-1">Optimization</h1>
        <p className="text-slate-400 text-sm">AI-powered bid and budget optimization</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Generate Optimization Plan</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Campaign"
              options={(campaigns ?? []).map((c) => ({ value: String(c.id), label: c.name }))}
              value={selectedCampaign}
              onChange={(e) => setSelectedCampaign(e.target.value)}
              placeholder="Select a campaign"
            />
            <Select
              label="Strategy"
              options={OPTIMIZATION_STRATEGIES.map((s) => ({ value: s.value, label: s.label }))}
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
            />
          </div>
          <Button
            onClick={() => planMutation.mutate()}
            disabled={!selectedCampaign}
            loading={planMutation.isPending}
            icon={<Zap className="h-4 w-4" />}
          >
            Generate Plan
          </Button>
        </CardContent>
      </Card>

      {planMutation.data && (
        <Card variant="glow">
          <CardHeader>
            <CardTitle>Optimization Plan</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {planMutation.data.changes.map((change, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-institutional">
                <div>
                  <p className="text-sm text-slate-200">{change.description}</p>
                  <p className="text-xs text-slate-500">{change.type}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-data text-slate-400">{change.current_value} &rarr;</p>
                  <p className="text-sm font-data text-profit">{change.proposed_value}</p>
                </div>
              </div>
            ))}
            <Button
              variant="success"
              onClick={() => executeMutation.mutate()}
              loading={executeMutation.isPending}
              icon={<Play className="h-4 w-4" />}
            >
              Execute Plan
            </Button>
            {executeMutation.data && (
              <Badge variant="success">{executeMutation.data.message}</Badge>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export { OptimizationPage }
