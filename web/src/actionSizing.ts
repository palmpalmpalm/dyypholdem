import type { LegalAction, PokerState } from './types'

export interface RaiseTarget {
  actionId: string
  raiseTo: number
  label: string
}

export interface RaisePlan {
  minimum: number
  maximum: number
  genericActionId: string | null
  discreteTargets: RaiseTarget[]
  advertisedPresets: RaiseTarget[]
  potRaiseTo: number | null
}

export interface RaisePreset {
  key: 'min' | 'half' | 'three-quarter' | 'pot' | 'all-in'
  label: string
  raiseTo: number
  enabled: boolean
}

function canonical(value: string): string {
  return value.trim().toLowerCase().replace(/[\s-]+/g, '_')
}

export function foldAction(actions: LegalAction[]): LegalAction | undefined {
  return actions.find((action) => canonical(action.id).includes('fold'))
}

export function passiveAction(actions: LegalAction[], facingBet: boolean): LegalAction | undefined {
  const candidates = actions.filter((action) => {
    const id = canonical(action.id)
    return id.includes('call') || id.includes('check') || id === 'passive'
  })
  return (
    candidates.find((action) => canonical(action.id).includes(facingBet ? 'call' : 'check')) ??
    candidates[0]
  )
}

function isGenericRaiseAction(action: LegalAction): boolean {
  return ['raise', 'bet', 'bet_raise', 'raise_to', 'wager'].includes(canonical(action.id))
}

function isRaiseAction(action: LegalAction): boolean {
  const id = canonical(action.id)
  return (
    isGenericRaiseAction(action) ||
    action.raiseTo !== undefined ||
    id.includes('all_in') ||
    id.includes('allin') ||
    id.includes('pot')
  )
}

export function buildRaisePlan(state: PokerState): RaisePlan | null {
  const raiseActions = state.availableActions.filter(isRaiseAction)
  if (raiseActions.length === 0) return null

  const discreteTargets = raiseActions
    .filter((action): action is LegalAction & { raiseTo: number } => action.raiseTo !== undefined)
    .map((action) => ({ actionId: action.id, raiseTo: Math.round(action.raiseTo), label: action.label }))
    .filter((target) => target.raiseTo > state.heroBet)
    .sort((left, right) => left.raiseTo - right.raiseTo)
    .filter((target, index, all) => index === 0 || target.raiseTo !== all[index - 1].raiseTo)

  const generic = raiseActions.find(isGenericRaiseAction)
  const advertisedPresets = generic
    ? generic.presets
        .map((preset) => ({
          actionId: preset.id,
          raiseTo: Math.round(preset.raiseTo),
          label: preset.label,
        }))
        .filter((preset) => preset.raiseTo > state.heroBet)
    : []
  const advertisedMinimum =
    state.raiseBounds.minRaiseTo ?? generic?.minRaiseTo ?? raiseActions.find((a) => a.minRaiseTo)?.minRaiseTo
  const advertisedMaximum =
    state.raiseBounds.maxRaiseTo ?? generic?.maxRaiseTo ?? raiseActions.find((a) => a.maxRaiseTo)?.maxRaiseTo
  // Bounds must come from the server. Legacy discrete targets are themselves
  // an explicit legal set, but stack/pot guesses must never create legality.
  const minimumSource = advertisedMinimum ?? discreteTargets[0]?.raiseTo
  const maximumSource = advertisedMaximum ?? discreteTargets.at(-1)?.raiseTo
  if (minimumSource === undefined || maximumSource === undefined) return null
  const minimum = Math.round(minimumSource)
  const maximum = Math.round(maximumSource)
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || maximum < minimum) return null
  return {
    minimum,
    maximum,
    genericActionId: generic?.id ?? null,
    discreteTargets,
    advertisedPresets,
    potRaiseTo: generic?.potRaiseTo !== undefined ? Math.round(generic.potRaiseTo) : null,
  }
}

export function isLegalRaiseTo(plan: RaisePlan, raiseTo: number): boolean {
  const rounded = Math.round(raiseTo)
  if (plan.genericActionId !== null) return rounded >= plan.minimum && rounded <= plan.maximum
  return plan.discreteTargets.some((target) => target.raiseTo === rounded)
}

export function resolveRaiseAction(plan: RaisePlan, raiseTo: number): RaiseTarget | null {
  const rounded = Math.round(raiseTo)
  if (plan.genericActionId !== null && isLegalRaiseTo(plan, rounded)) {
    return { actionId: plan.genericActionId, raiseTo: rounded, label: 'Raise' }
  }
  const exact = plan.discreteTargets.find((target) => target.raiseTo === rounded)
  if (exact) return exact
  return null
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, Math.round(value)))
}

export function potFractionRaiseTo(state: PokerState, plan: RaisePlan, fraction: number): number {
  const toCall = Math.max(0, state.opponentBet - state.heroBet)
  const matchedBet = Math.max(state.heroBet, state.opponentBet)
  const potAfterCall = state.pot + toCall
  return clamp(matchedBet + potAfterCall * fraction, plan.minimum, plan.maximum)
}

export function raisePresets(state: PokerState, plan: RaisePlan): RaisePreset[] {
  const advertised = (key: RaisePreset['key'], ...matches: string[]): number | undefined => {
    const normalizedMatches = matches.map(canonical)
    const exact = plan.advertisedPresets.find((preset) => {
      const id = canonical(preset.label)
      const action = canonical(preset.actionId)
      return normalizedMatches.some((match) => id === match || action === match)
    })
    if (exact) return exact.raiseTo
    return plan.advertisedPresets.find((preset) => {
      const text = canonical(`${preset.label} ${preset.actionId}`)
      return normalizedMatches.some((match) => text.includes(match))
    })?.raiseTo
  }
  const fractionTarget = (fraction: number) => {
    if (plan.genericActionId !== null) return potFractionRaiseTo(state, plan, fraction)
    const toCall = Math.max(0, state.opponentBet - state.heroBet)
    return Math.round(Math.max(state.heroBet, state.opponentBet) + (state.pot + toCall) * fraction)
  }
  const definitions: Array<[RaisePreset['key'], string, number]> = [
    ['min', 'Min', advertised('min', 'min') ?? plan.minimum],
    ['half', '½ pot', advertised('half', 'half', 'half_pot', '1_2', '50') ?? fractionTarget(0.5)],
    [
      'three-quarter',
      '¾ pot',
      advertised('three-quarter', 'three_quarter', '3_4', '75') ??
        fractionTarget(0.75),
    ],
    [
      'pot',
      'Pot',
      advertised('pot', 'pot') ?? plan.potRaiseTo ?? fractionTarget(1),
    ],
    ['all-in', 'All-in', advertised('all-in', 'all_in', 'allin', 'max') ?? plan.maximum],
  ]
  return definitions.map(([key, label, raiseTo]) => ({
    key,
    label,
    raiseTo,
    enabled: isLegalRaiseTo(plan, raiseTo),
  }))
}
