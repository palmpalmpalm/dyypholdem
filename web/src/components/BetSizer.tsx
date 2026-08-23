import { useEffect, useMemo, useRef, useState } from 'react'
import {
  isLegalRaiseTo,
  raisePresets,
  resolveRaiseAction,
  type RaisePlan,
} from '../actionSizing'
import { formatChips } from '../format'
import type { ActionSubmission, PokerState } from '../types'

interface BetSizerProps {
  state: PokerState
  plan: RaisePlan
  verb: 'Bet' | 'Raise'
  openedNonce: number
  busy: boolean
  onClose(): void
  onSubmit(submission: ActionSubmission): Promise<boolean>
}

function initialTarget(state: PokerState, plan: RaisePlan): number {
  const potPreset = raisePresets(state, plan).find((preset) => preset.key === 'pot' && preset.enabled)
  return potPreset?.raiseTo ?? plan.discreteTargets[0]?.raiseTo ?? plan.minimum
}

export function BetSizer({
  state,
  plan,
  verb,
  openedNonce,
  busy,
  onClose,
  onSubmit,
}: BetSizerProps) {
  const [raiseTo, setRaiseTo] = useState(() => initialTarget(state, plan))
  const [inputValue, setInputValue] = useState(() => String(initialTarget(state, plan)))
  const inputRef = useRef<HTMLInputElement>(null)
  const stale = state.stateNonce !== openedNonce
  const legal = isLegalRaiseTo(plan, raiseTo) && !stale
  const presets = useMemo(() => raisePresets(state, plan), [plan, state])
  const discrete = plan.genericActionId === null

  useEffect(() => {
    inputRef.current?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const updateTarget = (value: number) => {
    if (!Number.isFinite(value)) return
    const rounded = Math.round(value)
    setRaiseTo(rounded)
    setInputValue(String(rounded))
  }

  const sliderTargets = plan.discreteTargets
  const discreteIndex = Math.max(
    0,
    sliderTargets.findIndex((target) => target.raiseTo === raiseTo),
  )
  const sliderMin = discrete ? 0 : plan.minimum
  const sliderMax = discrete ? Math.max(0, sliderTargets.length - 1) : plan.maximum
  const sliderValue = discrete ? discreteIndex : Math.max(plan.minimum, Math.min(plan.maximum, raiseTo))

  const submit = async () => {
    const resolved = resolveRaiseAction(plan, raiseTo)
    if (!resolved || stale) return
    const accepted = await onSubmit({
      actionId: resolved.actionId,
      stateNonce: openedNonce,
      raiseTo: resolved.raiseTo,
    })
    if (accepted) onClose()
  }

  return (
    <section
      className="bet-sizer"
      role="dialog"
      aria-modal="false"
      aria-labelledby="bet-sizer-title"
      aria-describedby="bet-sizer-help"
    >
        <header className="sizer-header">
          <div>
            <span className="eyebrow">Choose a total</span>
            <h2 id="bet-sizer-title">{verb} size</h2>
          </div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close bet sizer">
            ×
          </button>
        </header>

        <div className="raise-total" aria-live="polite">
          {raiseTo === plan.maximum ? 'All-in' : formatChips(raiseTo)}
          {raiseTo !== plan.maximum ? <small> chips total</small> : null}
        </div>

        <label className="field-label" htmlFor="raise-to-input">
          Raise to
        </label>
        <div className="number-field">
          <button
            type="button"
            aria-label="Decrease raise"
            onClick={() =>
              updateTarget(
                discrete
                  ? sliderTargets[Math.max(0, discreteIndex - 1)]?.raiseTo ?? raiseTo
                  : Math.max(plan.minimum, raiseTo - state.bigBlind),
              )
            }
          >
            −
          </button>
          <input
            ref={inputRef}
            id="raise-to-input"
            aria-invalid={!legal}
            inputMode="numeric"
            type="number"
            min={plan.minimum}
            max={plan.maximum}
            step={1}
            value={inputValue}
            onChange={(event) => {
              setInputValue(event.target.value)
              const parsed = Number(event.target.value)
              if (Number.isFinite(parsed)) setRaiseTo(Math.round(parsed))
            }}
          />
          <button
            type="button"
            aria-label="Increase raise"
            onClick={() =>
              updateTarget(
                discrete
                  ? sliderTargets[Math.min(sliderTargets.length - 1, discreteIndex + 1)]?.raiseTo ??
                      raiseTo
                  : Math.min(plan.maximum, raiseTo + state.bigBlind),
              )
            }
          >
            +
          </button>
        </div>

        <label className="visually-hidden" htmlFor="raise-slider">
          {verb} size slider
        </label>
        <input
          id="raise-slider"
          className="raise-slider"
          type="range"
          min={sliderMin}
          max={sliderMax}
          step={1}
          value={sliderValue}
          aria-valuetext={`${formatChips(raiseTo)} chips total`}
          onChange={(event) => {
            const value = Number(event.target.value)
            updateTarget(discrete ? sliderTargets[value]?.raiseTo ?? raiseTo : value)
          }}
        />
        <div className="range-labels" aria-hidden="true">
          <span>{formatChips(plan.minimum)}</span>
          <span>{formatChips(plan.maximum)}</span>
        </div>

        <div className="preset-grid" aria-label="Common bet sizes">
          {presets.map((preset) => (
            <button
              key={preset.key}
              type="button"
              disabled={!preset.enabled || busy || stale}
              aria-label={`${preset.key === 'three-quarter' ? 'Three-quarter pot' : preset.key === 'half' ? 'Half pot' : preset.label}, raise to ${formatChips(preset.raiseTo)}`}
              title={preset.enabled ? undefined : 'This size is not available in the current abstraction'}
              data-selected={raiseTo === preset.raiseTo}
              onClick={() => updateTarget(preset.raiseTo)}
            >
              <span>{preset.label}</span>
              <small>{formatChips(preset.raiseTo)}</small>
            </button>
          ))}
        </div>

        <p id="bet-sizer-help" className={`sizer-help${!legal ? ' warning' : ''}`} role="status">
          {stale
            ? 'The table changed while this panel was open. Close it and choose again.'
            : !isLegalRaiseTo(plan, raiseTo)
              ? `That total is unavailable. Choose ${plan.discreteTargets.map((item) => formatChips(item.raiseTo)).join(' or ')}.`
              : discrete
                ? 'This model currently exposes discrete raise sizes only.'
                : `Legal range: ${formatChips(plan.minimum)} to ${formatChips(plan.maximum)} chips total.`}
        </p>

        <button className="confirm-raise" type="button" disabled={!legal || busy} onClick={submit}>
          {busy ? 'Sending…' : `${verb} to ${formatChips(raiseTo)}`}
        </button>
    </section>
  )
}
