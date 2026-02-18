import { useQuery } from '@tanstack/react-query'
import { dashboardApi } from '@/api/endpoints/dashboard'
import { Zap } from 'lucide-react'
import { formatTime } from '@/utils/format'

interface AIActionFeedProps {
  accountId?: number
}

function AIActionFeed({ accountId }: AIActionFeedProps) {
  const { data: actions } = useQuery({
    queryKey: ['ai-actions', accountId],
    queryFn: () => dashboardApi.getAIActions(accountId),
    refetchInterval: 30000,
  })

  if (!actions?.length) {
    return <p className="text-sm text-slate-500">No recent AI actions</p>
  }

  return (
    <div className="space-y-3 max-h-64 overflow-y-auto">
      {actions.map((action) => (
        <div key={action.id} className="flex items-start gap-3 text-sm">
          <Zap className="h-4 w-4 text-electric mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-slate-200">{action.description}</p>
            {action.campaign_name && (
              <p className="text-xs text-slate-500 mt-0.5">{action.campaign_name}</p>
            )}
          </div>
          <span className="text-xs text-slate-500 shrink-0 font-data">
            {formatTime(action.timestamp)}
          </span>
        </div>
      ))}
    </div>
  )
}

export { AIActionFeed }
