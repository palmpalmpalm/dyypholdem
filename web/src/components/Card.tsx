import { cardPresentation } from '../format'

interface CardProps {
  card?: string
  hidden?: boolean
  compact?: boolean
}

export function Card({ card, hidden = false, compact = false }: CardProps) {
  if (hidden) {
    return (
      <span
        className={`playing-card card-back${compact ? ' compact' : ''}`}
        aria-label="Hidden card"
        data-testid="hidden-card"
      />
    )
  }
  if (!card) {
    return <span className={`playing-card card-slot${compact ? ' compact' : ''}`} aria-hidden="true" />
  }
  const presentation = cardPresentation(card)
  if (!presentation) {
    return <span className={`playing-card card-slot${compact ? ' compact' : ''}`} aria-hidden="true" />
  }
  return (
    <span
      className={`playing-card card-face${presentation.red ? ' red' : ''}${compact ? ' compact' : ''}`}
      aria-label={presentation.label}
      data-card={card}
    >
      <span className="card-rank">{presentation.rank}</span>
      <span className="card-suit" aria-hidden="true">
        {presentation.suit}
      </span>
    </span>
  )
}
