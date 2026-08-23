// Casino-table presentation adapted from the MIT-licensed Elite-Poker project.
// Game state, privacy, and action legality remain DyypHoldem/ACPC authoritative.
import { useEffect, useState } from 'react'
import { formatChips, formatSignedChips } from '../format'
import type { HandResult, PokerState } from '../types'
import { Card } from './Card'
import { ChipStack } from './ChipStack'

interface HeadsUpTableProps {
  state: PokerState | null
}

function statusLabel(status: string | undefined): string {
  const labels: Record<string, string> = {
    connecting: 'Connecting to the table',
    waiting_for_dealer: 'Waiting for the dealer',
    waiting_for_bot: 'DyypHoldem is loading',
    your_turn: 'Your turn',
    bot_thinking: 'DyypHoldem is thinking',
    hand_complete: 'Hand complete',
    match_complete: 'Match complete',
    error: 'Session error',
  }
  return labels[status ?? 'connecting'] ?? (status || 'Connecting')
}

function positionCode(position: string | undefined): 'SB' | 'BB' | undefined {
  const normalized = position?.trim().toUpperCase().replaceAll('-', '_').replaceAll(' ', '_')
  if (!normalized) return undefined
  if (
    normalized === 'SB' ||
    normalized === 'BUTTON' ||
    normalized === 'BTN' ||
    normalized === 'SMALL_BLIND'
  ) {
    return 'SB'
  }
  if (normalized === 'BB' || normalized === 'BIG_BLIND') return 'BB'
  return undefined
}

function Seat({
  name,
  stack,
  bet,
  cards,
  hidden,
  position,
  hero,
  active,
  winner,
}: {
  name: string
  stack: number
  bet: number
  cards: string[]
  hidden?: boolean
  position?: 'SB' | 'BB'
  hero?: boolean
  active?: boolean
  winner?: boolean
}) {
  const initials = hero ? 'YOU' : 'AI'
  const isDealer = position === 'SB'

  return (
    <section
      className={`casino-seat ${hero ? 'hero-seat' : 'opponent-seat'}${active ? ' is-active' : ''}${winner ? ' is-winner' : ''}`}
      aria-label={`${name} seat`}
      data-active={active || undefined}
    >
      {winner ? <div className="winner-ribbon">Winner</div> : null}
      <div className="seat-cards">
        {hidden ? (
          <>
            <Card hidden />
            <Card hidden />
          </>
        ) : cards.length > 0 ? (
          cards.slice(0, 2).map((card, index) => (
            <Card key={`${card}-${index}`} card={card} animate animationIndex={index} />
          ))
        ) : (
          <>
            <Card />
            <Card />
          </>
        )}
      </div>

      <div className="seat-profile">
        <div className="player-avatar" aria-hidden="true">
          <span>{initials}</span>
        </div>
        <div className="seat-plaque">
          <div className="seat-heading">
            <strong>{name}</strong>
            {active ? <span className="acting-label">Acting</span> : null}
          </div>
          <span className="seat-stack">{formatChips(stack)}</span>
        </div>
        <div
          className="seat-markers"
          aria-label={position ? `${position}${isDealer ? ', dealer' : ''}` : undefined}
        >
          {isDealer ? <span className="dealer-button">D</span> : null}
          {position ? <span className={`blind-button ${position.toLowerCase()}`}>{position}</span> : null}
        </div>
      </div>

      <div className="seat-bet">
        <ChipStack amount={bet} label={`${name} committed`} compact />
      </div>
    </section>
  )
}

export function HeadsUpTable({ state }: HeadsUpTableProps) {
  const [recentResult, setRecentResult] = useState<HandResult | null>(null)
  const board = state?.board ?? []
  const boardSlots = Array.from({ length: 5 }, (_, index) => board[index])
  const opponentVisible =
    (state?.status === 'hand_complete' || state?.status === 'match_complete') &&
    (state?.opponentHand.length ?? 0) > 0
  const startingStack = state?.tableStack ?? 20_000
  const heroPosition = positionCode(state?.heroPosition)
  const opponentPosition = heroPosition === 'SB' ? 'BB' : heroPosition === 'BB' ? 'SB' : undefined
  const heroActive = state?.status === 'your_turn'
  const opponentActive = state?.status === 'bot_thinking'
  const completedResult =
    (state?.status === 'hand_complete' || state?.status === 'match_complete') &&
    state.lastResult?.handNumber === state.handNumber
      ? state.lastResult
      : null
  const heroWinner = (completedResult?.winnings ?? 0) > 0
  const opponentWinner = (completedResult?.winnings ?? 0) < 0

  useEffect(() => {
    const result = state?.lastResult
    if (!result) {
      setRecentResult(null)
      return
    }
    setRecentResult(result)
    const timer = window.setTimeout(() => {
      setRecentResult((current) =>
        current?.handNumber === result.handNumber ? null : current,
      )
    }, 4_000)
    return () => window.clearTimeout(timer)
  }, [state?.lastResult?.handNumber])

  return (
    <section className="poker-table-shell" aria-label="Heads-up poker table">
      <div className="table-status" data-status={state?.status ?? 'connecting'} aria-live="polite">
        <span className="status-dot" aria-hidden="true" />
        {statusLabel(state?.status)}
      </div>

      <div className="table-meta table-meta-left">
        <span>{state?.handNumber !== undefined ? `Hand #${state.handNumber}` : 'Waiting'}</span>
        <strong>{state?.street ?? 'preflop'}</strong>
      </div>
      <div className="table-meta table-meta-right">
        <span>Blinds</span>
        <strong>
          {formatChips(state?.smallBlind)} / {formatChips(state?.bigBlind)}
        </strong>
      </div>

      <div className="table-rail">
        <div className="casino-felt">
          <div className="felt-watermark" aria-hidden="true">
            <span>D</span>
            <small>Heads-up</small>
          </div>

          <Seat
            name="DyypHoldem"
            stack={state?.opponentStack ?? startingStack}
            bet={state?.opponentBet ?? 0}
            cards={state?.opponentHand ?? []}
            hidden={!opponentVisible}
            position={opponentPosition}
            active={opponentActive}
            winner={opponentWinner}
          />

          <div className="table-center">
            <div className="community-cards" aria-label="Community cards">
              {boardSlots.map((card, index) => (
                <Card
                  key={`${state?.handNumber ?? 'waiting'}-${card ?? 'empty'}-${index}`}
                  card={card}
                  animate={Boolean(card)}
                  animationIndex={index}
                />
              ))}
            </div>
            <div className="pot-display">
              <ChipStack amount={state?.pot ?? 0} label="Pot" />
              <span>Pot</span>
            </div>
          </div>

          <Seat
            name="You"
            stack={state?.heroStack ?? startingStack}
            bet={state?.heroBet ?? 0}
            cards={state?.heroHand ?? []}
            position={heroPosition}
            hero
            active={heroActive}
            winner={heroWinner}
          />

          {completedResult ? (
            <div
              className={`hand-result-overlay${completedResult.winnings >= 0 ? ' won' : ' lost'}`}
              role="status"
            >
              <strong>
                {completedResult.winnings > 0
                  ? 'You win'
                  : completedResult.winnings < 0
                    ? 'DyypHoldem wins'
                    : 'Split pot'}
              </strong>
              <span>{formatSignedChips(completedResult.winnings)} chips</span>
            </div>
          ) : null}
        </div>
      </div>

      {recentResult && !completedResult ? (
        <div
          className={`previous-result-toast${recentResult.winnings >= 0 ? ' won' : ' lost'}`}
          role="status"
        >
          <strong>Previous hand</strong>
          <span>
            {recentResult.winnings > 0
              ? `You won ${formatSignedChips(recentResult.winnings)}`
              : recentResult.winnings < 0
                ? `DyypHoldem won ${formatChips(Math.abs(recentResult.winnings))}`
                : 'Split pot'}
          </span>
        </div>
      ) : null}
    </section>
  )
}
