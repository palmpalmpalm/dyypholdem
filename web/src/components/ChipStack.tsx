// Presentational chip pile adapted from the MIT-licensed Elite-Poker project.
import { formatChips } from '../format'

interface ChipStackProps {
  amount: number
  label?: string
  compact?: boolean
}

export function ChipStack({ amount, label, compact = false }: ChipStackProps) {
  if (amount <= 0) return null

  const chipCount = compact ? 3 : Math.min(6, Math.max(3, Math.ceil(String(Math.round(amount)).length / 2)))

  return (
    <div className={`chip-stack${compact ? ' compact' : ''}`} aria-label={`${label ? `${label} ` : ''}${formatChips(amount)}`}>
      <span className="chip-pile" aria-hidden="true">
        {Array.from({ length: chipCount }, (_, index) => (
          <span className={`chip chip-${index % 3}`} key={index} />
        ))}
      </span>
      <span className="chip-amount">{formatChips(amount)}</span>
    </div>
  )
}
