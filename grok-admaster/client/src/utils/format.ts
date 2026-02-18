const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
})

const currencyFormatterPrecise = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const percentFormatter = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

const numberFormatter = new Intl.NumberFormat('en-US')

export function formatCurrency(value: number, precise = false): string {
  return precise ? currencyFormatterPrecise.format(value) : currencyFormatter.format(value)
}

export function formatPercent(value: number): string {
  return percentFormatter.format(value / 100)
}

export function formatNumber(value: number): string {
  return numberFormatter.format(value)
}

export function formatDecimal(value: number, digits = 2): string {
  return value.toFixed(digits)
}

export function formatMetric(value: number, format: 'currency' | 'percentage' | 'number' | 'decimal'): string {
  switch (format) {
    case 'currency':
      return formatCurrency(value)
    case 'percentage':
      return formatPercent(value)
    case 'number':
      return formatNumber(value)
    case 'decimal':
      return formatDecimal(value)
  }
}

export function formatDate(date: string | Date): string {
  return new Date(date).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function formatTime(date: string | Date): string {
  return new Date(date).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  })
}
