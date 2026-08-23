// Visual grouping adapted from the MIT-licensed Elite-Poker action dock.
// Legal actions and all raise totals remain server-authoritative.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  buildRaisePlan,
  foldAction,
  passiveAction,
  resolveRaiseAction,
} from '../actionSizing'
import { formatChips } from '../format'
import type { ActionSubmission, PokerState } from '../types'
import { BetSizer } from './BetSizer'

interface ActionBarProps {
  state: PokerState
  busy: boolean
  onSubmit(submission: ActionSubmission): Promise<boolean>
  onStale(): void
}

function waitingMessage(status: string): string {
  const labels: Record<string, string> = {
    connecting: 'Connecting to the dealer…',
    waiting_for_dealer: 'Waiting for the dealer…',
    waiting_for_bot: 'DyypHoldem is loading…',
    bot_thinking: 'DyypHoldem is thinking…',
    hand_complete: 'Dealing the next hand…',
    match_complete: 'Session complete',
    error: 'The table needs attention',
  }
  return labels[status] ?? 'Waiting for your turn…'
}

export function ActionBar({ state, busy, onSubmit, onStale }: ActionBarProps) {
  const [sizerOpen, setSizerOpen] = useState(false)
  const [openedNonce, setOpenedNonce] = useState<number | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const submittingRef = useRef(false)
  const raiseButtonRef = useRef<HTMLButtonElement>(null)
  const fold = useMemo(() => foldAction(state.availableActions), [state.availableActions])
  const facingBet = state.opponentBet > state.heroBet
  const passive = useMemo(
    () => passiveAction(state.availableActions, facingBet),
    [facingBet, state.availableActions],
  )
  const plan = useMemo(() => buildRaisePlan(state), [state])
  const advertisedAllIn = useMemo(() => {
    if (!plan) return undefined
    const candidates = [...plan.advertisedPresets, ...plan.discreteTargets]
    const target = candidates.find((candidate) =>
      /all[\s_-]?in/i.test(`${candidate.actionId} ${candidate.label}`),
    )
    return target ? resolveRaiseAction(plan, target.raiseTo) : undefined
  }, [plan])
  const canChooseRaiseSize = Boolean(
    plan &&
      (plan.genericActionId !== null
        ? plan.maximum > plan.minimum
        : plan.discreteTargets.length > 1),
  )
  const canAct =
    state.status === 'your_turn' && state.availableActions.length > 0 && !busy && !submitting
  const raiseVerb: 'Bet' | 'Raise' = state.street === 'preflop' || facingBet ? 'Raise' : 'Bet'

  const closeSizer = useCallback((restoreFocus = true) => {
    setSizerOpen(false)
    setOpenedNonce(null)
    if (restoreFocus) window.requestAnimationFrame(() => raiseButtonRef.current?.focus())
  }, [])

  useEffect(() => {
    if (!sizerOpen || openedNonce === null || openedNonce === state.stateNonce) return
    if (submitting) return
    closeSizer(false)
    onStale()
  }, [closeSizer, onStale, openedNonce, sizerOpen, state.stateNonce, submitting])

  useEffect(() => {
    if (state.status !== 'your_turn' && sizerOpen) closeSizer(false)
  }, [closeSizer, sizerOpen, state.status])

  const runSubmission = useCallback(
    async (submission: ActionSubmission) => {
      if (busy || submittingRef.current || state.status !== 'your_turn') return false
      submittingRef.current = true
      setSubmitting(true)
      try {
        return await onSubmit(submission)
      } finally {
        submittingRef.current = false
        setSubmitting(false)
      }
    },
    [busy, onSubmit, state.status],
  )

  const submitAdvertised = (actionId: string) =>
    void runSubmission({ actionId, stateNonce: state.stateNonce })

  const submitAllIn = () => {
    if (!plan || !advertisedAllIn) return
    void runSubmission({
      actionId: advertisedAllIn.actionId,
      stateNonce: state.stateNonce,
      raiseTo: advertisedAllIn.raiseTo,
    })
  }

  const openSizer = () => {
    if (!canAct || !plan || !canChooseRaiseSize) return
    setOpenedNonce(state.stateNonce)
    setSizerOpen(true)
  }

  return (
    <section className={`action-zone${sizerOpen ? ' sizer-open' : ''}`} aria-label="Poker actions">
      {sizerOpen && plan && openedNonce !== null ? (
        <BetSizer
          state={state}
          plan={plan}
          verb={raiseVerb}
          openedNonce={openedNonce}
          busy={busy || submitting}
          onClose={closeSizer}
          onSubmit={runSubmission}
        />
      ) : null}

      <nav
        className="action-dock"
        aria-label="Available actions"
        aria-hidden={sizerOpen || undefined}
        inert={sizerOpen || undefined}
      >
        {state.status === 'your_turn' && state.availableActions.length > 0 ? (
          <>
            {fold ? (
              <button
                className="poker-action fold-action"
                type="button"
                disabled={!canAct}
                aria-label={fold.label}
                onClick={() => submitAdvertised(fold.id)}
              >
                <span>Fold</span>
                <small>Give up hand</small>
              </button>
            ) : null}

            {passive ? (
              <button
                className="poker-action passive-action"
                type="button"
                disabled={!canAct}
                aria-label={passive.label}
                onClick={() => submitAdvertised(passive.id)}
              >
                <span>{passive.label}</span>
                <small>
                  {facingBet && passive.amount ? `${formatChips(passive.amount)} chips` : 'Continue'}
                </small>
              </button>
            ) : null}

            {plan && canChooseRaiseSize ? (
              <button
                ref={raiseButtonRef}
                className="poker-action raise-action"
                type="button"
                disabled={!canAct}
                aria-expanded={sizerOpen}
                aria-haspopup="dialog"
                aria-label={`${raiseVerb}, choose size`}
                onClick={openSizer}
              >
                <span>{raiseVerb}</span>
                <small>
                  {formatChips(plan.minimum)}–{formatChips(plan.maximum)}
                </small>
              </button>
            ) : null}

            {advertisedAllIn ? (
              <button
                className="poker-action all-in-action"
                type="button"
                disabled={!canAct}
                aria-label={`All-in to ${formatChips(advertisedAllIn.raiseTo)}`}
                onClick={submitAllIn}
              >
                <span>All-in</span>
                <small>{formatChips(advertisedAllIn.raiseTo)}</small>
              </button>
            ) : null}
          </>
        ) : (
          <div className="action-waiting" aria-live="polite">
            {state.status !== 'match_complete' && state.status !== 'error' ? (
              <span className="spinner" aria-hidden="true" />
            ) : null}
            <strong>{waitingMessage(state.status)}</strong>
          </div>
        )}
      </nav>
    </section>
  )
}
