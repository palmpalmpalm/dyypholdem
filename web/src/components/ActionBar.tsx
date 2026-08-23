// Interaction pattern inspired by Pip Web's MIT-licensed ActionBar.
// See ../../THIRD_PARTY_NOTICES.md for attribution and license text.
import { useCallback, useEffect, useMemo, useState } from 'react'
import { buildRaisePlan, foldAction, passiveAction } from '../actionSizing'
import { formatChips } from '../format'
import type { ActionSubmission, PokerState } from '../types'
import { BetSizer } from './BetSizer'

interface ActionBarProps {
  state: PokerState
  busy: boolean
  onSubmit(submission: ActionSubmission): Promise<boolean>
  onStale(): void
}

export function ActionBar({ state, busy, onSubmit, onStale }: ActionBarProps) {
  const [sizerOpen, setSizerOpen] = useState(false)
  const [openedNonce, setOpenedNonce] = useState(state.stateNonce)
  const facingBet = state.opponentBet > state.heroBet
  const fold = foldAction(state.availableActions)
  const passive = passiveAction(state.availableActions, facingBet)
  const raisePlan = useMemo(() => buildRaisePlan(state), [state])
  const canAct = state.status === 'your_turn' && state.availableActions.length > 0 && !busy
  const verb: 'Bet' | 'Raise' = facingBet ? 'Raise' : 'Bet'

  useEffect(() => {
    if (sizerOpen && state.stateNonce !== openedNonce) {
      setSizerOpen(false)
      onStale()
    }
  }, [openedNonce, onStale, sizerOpen, state.stateNonce])

  const closeSizer = useCallback(() => setSizerOpen(false), [])

  const directAction = async (actionId: string) => {
    await onSubmit({ actionId, stateNonce: state.stateNonce })
  }

  if (!canAct && !sizerOpen) {
    return (
      <div className="action-dock waiting-dock" aria-live="polite">
        <span className={state.status.includes('thinking') || state.status.includes('waiting') ? 'spinner' : ''} />
        <span>{state.status === 'your_turn' && busy ? 'Sending action…' : 'Waiting for action'}</span>
      </div>
    )
  }

  const passiveLabel = passive?.label || (facingBet ? `Call ${formatChips(state.opponentBet - state.heroBet)}` : 'Check')

  return (
    <>
      <nav className="action-dock" aria-label="Poker actions">
        {fold ? (
            <button className="action-button fold-button" type="button" disabled={!canAct} onClick={() => directAction(fold.id)}>
            Fold
          </button>
        ) : null}
        {passive ? (
          <button
            className="action-button passive-button"
            type="button"
            disabled={!canAct}
            onClick={() => directAction(passive.id)}
          >
            {passiveLabel}
          </button>
        ) : null}
        {raisePlan ? (
          <button
            className="action-button raise-button"
            type="button"
            aria-label={`${verb}, choose size`}
            disabled={!canAct}
            onClick={() => {
              setOpenedNonce(state.stateNonce)
              setSizerOpen(true)
            }}
          >
            {verb}
            <small>Choose size</small>
          </button>
        ) : null}
      </nav>
      {sizerOpen && raisePlan ? (
        <BetSizer
          state={state}
          plan={raisePlan}
          verb={verb}
          openedNonce={openedNonce}
          busy={busy}
          onClose={closeSizer}
          onSubmit={onSubmit}
        />
      ) : null}
    </>
  )
}
