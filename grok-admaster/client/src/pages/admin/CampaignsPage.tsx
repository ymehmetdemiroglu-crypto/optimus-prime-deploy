import { useQuery } from '@tanstack/react-query'
import { useActiveAccount } from '@/hooks/useActiveAccount'
import { campaignsApi } from '@/api/endpoints/campaigns'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { CampaignTable } from '@/components/features/CampaignTable'
import { EmptyState } from '@/components/ui/EmptyState'
import { Target } from 'lucide-react'

function CampaignsPage() {
  const { activeAccount } = useActiveAccount()

  const { data: campaigns, isLoading } = useQuery({
    queryKey: ['campaigns', activeAccount?.id],
    queryFn: () => campaignsApi.getCampaigns(activeAccount?.id),
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-serif font-bold text-slate-100 mb-1">Campaigns</h1>
        <p className="text-slate-400 text-sm">
          {activeAccount ? `Campaigns for ${activeAccount.name}` : 'All campaigns across accounts'}
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Campaign Performance</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-slate-500 py-8 text-center">Loading campaigns...</p>
          ) : !campaigns?.length ? (
            <EmptyState icon={<Target className="h-12 w-12" />} title="No campaigns found" description="Sync your Amazon data to see campaigns here" />
          ) : (
            <CampaignTable campaigns={campaigns} />
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export { CampaignsPage }
