import type { Campaign } from '@/api/endpoints/campaigns'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/Table'
import { Badge } from '@/components/ui/Badge'
import { Select } from '@/components/ui/Select'
import { formatCurrency, formatPercent, formatNumber } from '@/utils/format'
import { AI_MODES } from '@/utils/constants'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { campaignsApi } from '@/api/endpoints/campaigns'

interface CampaignTableProps {
  campaigns: Campaign[]
  readonly?: boolean
}

function CampaignTable({ campaigns, readonly = false }: CampaignTableProps) {
  const queryClient = useQueryClient()

  const strategyMutation = useMutation({
    mutationFn: ({ id, mode }: { id: string; mode: string }) =>
      campaignsApi.updateStrategy(id, mode),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['campaigns'] }),
  })

  const stateVariant = (state: string) => {
    if (state === 'enabled') return 'success' as const
    if (state === 'paused') return 'warning' as const
    return 'default' as const
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Status</TableHead>
            <TableHead>Campaign</TableHead>
            <TableHead>Type</TableHead>
            {!readonly && <TableHead>AI Mode</TableHead>}
            <TableHead>Budget</TableHead>
            <TableHead>Spend</TableHead>
            <TableHead>Sales</TableHead>
            <TableHead>ACoS</TableHead>
            <TableHead>Impressions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {campaigns.map((c) => {
            const acos = c.sales > 0 ? (c.spend / c.sales) * 100 : 0
            return (
              <TableRow key={c.id}>
                <TableCell>
                  <Badge variant={stateVariant(c.state)}>{c.state}</Badge>
                </TableCell>
                <TableCell className="font-medium text-slate-100">{c.name}</TableCell>
                <TableCell className="text-xs">{c.campaign_type}</TableCell>
                {!readonly && (
                  <TableCell>
                    <Select
                      options={AI_MODES.map((m) => ({ value: m.value, label: m.label }))}
                      value={c.ai_mode}
                      onChange={(e) =>
                        strategyMutation.mutate({ id: String(c.id), mode: e.target.value })
                      }
                      className="w-40 py-1 text-xs"
                    />
                  </TableCell>
                )}
                <TableCell className="font-data">{formatCurrency(c.daily_budget, true)}</TableCell>
                <TableCell className="font-data">{formatCurrency(c.spend, true)}</TableCell>
                <TableCell className="font-data">{formatCurrency(c.sales, true)}</TableCell>
                <TableCell className={`font-data ${acos > 30 ? 'metric-negative' : 'metric-positive'}`}>
                  {formatPercent(acos)}
                </TableCell>
                <TableCell className="font-data">{formatNumber(c.impressions)}</TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}

export { CampaignTable }
