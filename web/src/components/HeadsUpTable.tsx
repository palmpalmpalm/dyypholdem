import { formatChips } from '../format'
import type { PokerState } from '../types'
import { Card } from './Card'

interface HeadsUpTableProps {
  state: PokerState | null
}

function statusLabel(status: string | undefined): string {
  const labels: Record<string, string> = {
    connecting: 'Connecting to the table',
    waiting_for_dealer: 'Waiting for dealer',
    waiting_for_bot: 'Loading models and solving the root',
    your_turn: 'Your turn',
    bot_thinking: 'DyypHoldem is calculating',
    hand_complete: 'Hand complete',
    match_complete: 'Match complete',
    error: 'Session error',
  }
  return labels[status ?? 'connecting'] ?? (status || 'Connecting')
}

function Seat({
  name,
  stack,
  bet,
  cards,
  hidden,
  position,
  hero,
}: {
  name: string
  stack: number
  bet: number
  cards: string[]
  hidden?: boolean
  position?: string
  hero?: boolean
}) {
  return (
    <section className={`player-seat${hero ? ' hero-seat' : ' opponent-seat'}`} aria-label={`${name} seat`}>
      <div className="seat-cards">
        {hidden ? (
          <>
            <Card hidden />
            <Card hidden />
          </>
        ) : cards.length > 0 ? (
          cards.slice(0, 2).map((card, index) => <Card key={`${card}-${index}`} card={card} />)
        ) : (
          <>
            <Card />
            <Card />
          </>
        )}
      </div>
      <div className="seat-plaque">
        <div className="seat-heading">
          <strong>{name}</strong>
          {position ? <span className="position-pill">{position}</span> : null}
        </div>
        <span className="seat-stack">{formatChips(stack)} chips</span>
      </div>
      {bet > 0 ? <div className="committed-chip">{formatChips(bet)}</div> : null}
    </section>
  )
}

export function HeadsUpTable({ state }: HeadsUpTableProps) {
  const board = state?.board ?? []
  const boardSlots = Array.from({ length: 5 }, (_, index) => board[index])
  const opponentVisible = (state?.opponentHand.length ?? 0) > 0
  const startingStack = state?.tableStack ?? 20_000

  return (
    <section className="poker-table-shell" aria-label="Heads-up poker table">
      <div className="table-status" data-status={state?.status ?? 'connecting'} aria-live="polite">
        <span className="status-dot" aria-hidden="true" />
        {statusLabel(state?.status)}
      </div>
      <div className="felt-rail">
        <div className="felt">
          <Seat
            name="DyypHoldem"
            stack={state?.opponentStack ?? startingStack}
            bet={state?.opponentBet ?? 0}
            cards={state?.opponentHand ?? []}
            hidden={!opponentVisible}
          />

          <div className="table-center">
            <div className="street-label">{state?.street ?? 'waiting'}</div>
            <div className="community-cards" aria-label="Community cards">
              {boardSlots.map((card, index) => (
                <Card key={`${card ?? 'empty'}-${index}`} card={card} />
              ))}
            </div>
            <div className="pot-chip" aria-label={`Pot ${formatChips(state?.pot)}`}>
              <span>Pot</span>
              <strong>{formatChips(state?.pot)}</strong>
            </div>
          </div>

          <Seat
            name="You"
            stack={state?.heroStack ?? startingStack}
            bet={state?.heroBet ?? 0}
            cards={state?.heroHand ?? []}
            position={state?.heroPosition}
            hero
          />
        </div>
      </div>
      <div className="hand-caption">
        <span>{state?.handNumber !== undefined ? `Hand #${state.handNumber}` : 'Waiting for first hand'}</span>
        <span>
          Blinds {formatChips(state?.smallBlind)} / {formatChips(state?.bigBlind)}
        </span>
      </div>
    </section>
  )
}
