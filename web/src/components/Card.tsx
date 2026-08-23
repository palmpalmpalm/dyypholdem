import { cardPresentation } from '../format'

interface CardProps {
  card?: string
  hidden?: boolean
  compact?: boolean
  animate?: boolean
  animationIndex?: number
}

export function Card({
  card,
  hidden = false,
  compact = false,
  animate = false,
  animationIndex = 0,
}: CardProps) {
  const classNames = `${compact ? ' compact' : ''}${animate ? ' card-enter' : ''}`
  const animationStyle = animate ? { animationDelay: `${animationIndex * 70}ms` } : undefined

  if (hidden) {
    return (
      <span
        className={`playing-card card-back${classNames}`}
        aria-label="Hidden card"
        data-testid="hidden-card"
        style={animationStyle}
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
      className={`playing-card card-face${presentation.red ? ' red' : ''}${classNames}`}
      aria-label={presentation.label}
      data-card={card}
      style={animationStyle}
    >
      <span className="card-corner card-corner-top">
        <span className="card-rank">{presentation.rank}</span>
        <span className="card-suit" aria-hidden="true">
          {presentation.suit}
        </span>
      </span>
      <span className="card-center-suit" aria-hidden="true">
        {presentation.suit}
      </span>
      <span className="card-corner card-corner-bottom" aria-hidden="true">
        <span className="card-rank">{presentation.rank}</span>
        <span className="card-suit">{presentation.suit}</span>
      </span>
    </span>
  )
}
