import {
  ResponsiveContainer, ComposedChart, Bar, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts'
import type { PerformanceMetric } from '@/api/endpoints/dashboard'

interface PerformanceChartProps {
  data: PerformanceMetric[]
  height?: number
}

function PerformanceChart({ data, height = 350 }: PerformanceChartProps) {
  const formatted = data.map((d) => ({
    ...d,
    date: new Date(d.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    acos: d.spend && d.ad_sales ? ((d.spend / d.ad_sales) * 100) : 0,
  }))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={formatted} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 12 }} stroke="#334155" />
        <YAxis yAxisId="left" tick={{ fill: '#94a3b8', fontSize: 12 }} stroke="#334155" />
        <YAxis yAxisId="right" orientation="right" tick={{ fill: '#94a3b8', fontSize: 12 }} stroke="#334155" />
        <Tooltip
          contentStyle={{
            backgroundColor: '#141824',
            border: '1px solid #1e293b',
            borderRadius: '6px',
            color: '#f1f5f9',
          }}
        />
        <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
        <Bar yAxisId="left" dataKey="ad_sales" name="Ad Sales" fill="#4f46e5" radius={[4, 4, 0, 0]} />
        <Bar yAxisId="left" dataKey="spend" name="Spend" fill="#334155" radius={[4, 4, 0, 0]} />
        <Line yAxisId="right" dataKey="acos" name="ACoS %" stroke="#22d3ee" strokeWidth={2} dot={false} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export { PerformanceChart }
