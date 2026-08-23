import type {
  HandResult,
  LegalAction,
  PokerState,
  RecentDecision,
  SolverReport,
  Street,
  StreetTiming,
  TimingStats,
} from './types'

const STREETS: Street[] = ['preflop', 'flop', 'turn', 'river']
const CARD_PATTERN = /^[2-9TJQKA][cdhs]$/

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function finiteNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function optionalFiniteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function shortText(value: unknown, fallback = '', maximum = 120): string {
  return typeof value === 'string' ? value.slice(0, maximum) : fallback
}

function nullableShortText(value: unknown, maximum = 120): string | null {
  return typeof value === 'string' ? value.slice(0, maximum) : null
}

function cards(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.filter((card): card is string => typeof card === 'string' && CARD_PATTERN.test(card))
}

function firstNumber(record: Record<string, unknown>, keys: string[]): number | undefined {
  for (const key of keys) {
    const candidate = optionalFiniteNumber(record[key])
    if (candidate !== undefined) return candidate
  }
  return undefined
}

function normalizeLegalAction(value: unknown): LegalAction | null {
  if (!isRecord(value)) return null
  const identifier = ['id', 'action_id', 'kind', 'type']
    .map((key) => shortText(value[key], '', 40))
    .find(Boolean)
  if (!identifier) return null
  const label = shortText(value.label, identifier.replaceAll('_', ' '), 80)
  const amount = firstNumber(value, ['amount', 'call_amount', 'to_call'])
  const raiseTo = firstNumber(value, ['raise_to', 'raiseTo', 'target'])
  const minRaiseTo = firstNumber(value, ['min_raise_to', 'minRaiseTo', 'minimum'])
  const maxRaiseTo = firstNumber(value, ['max_raise_to', 'maxRaiseTo', 'maximum'])
  const potRaiseTo = firstNumber(value, ['pot_raise_to', 'potRaiseTo'])
  const presets = Array.isArray(value.presets)
    ? value.presets
        .map((preset) => {
          if (!isRecord(preset)) return null
          const presetId = ['id', 'action_id', 'kind']
            .map((key) => shortText(preset[key], '', 40))
            .find(Boolean)
          const presetRaiseTo = firstNumber(preset, ['raise_to', 'raiseTo', 'target'])
          if (!presetId || presetRaiseTo === undefined) return null
          return {
            id: presetId,
            label: shortText(preset.label, presetId.replaceAll('_', ' '), 60),
            raiseTo: presetRaiseTo,
          }
        })
        .filter(
          (preset): preset is { id: string; label: string; raiseTo: number } => preset !== null,
        )
    : []
  return {
    id: identifier,
    label,
    presets,
    ...(amount !== undefined ? { amount } : {}),
    ...(raiseTo !== undefined ? { raiseTo } : {}),
    ...(minRaiseTo !== undefined ? { minRaiseTo } : {}),
    ...(maxRaiseTo !== undefined ? { maxRaiseTo } : {}),
    ...(potRaiseTo !== undefined ? { potRaiseTo } : {}),
  }
}

function normalizeHandResult(value: unknown): HandResult | null {
  if (!isRecord(value)) return null
  const handNumber = firstNumber(value, ['hand_number', 'handNumber'])
  if (handNumber === undefined) return null
  const cumulativeWinnings = firstNumber(value, ['cumulative_winnings', 'cumulativeWinnings'])
  return {
    handNumber,
    winnings: finiteNumber(value.winnings),
    ...(cumulativeWinnings !== undefined ? { cumulativeWinnings } : {}),
    board: cards(value.board),
    heroHand: cards(value.hero_hand ?? value.heroHand),
    // Opponent cards are accepted only when the server explicitly supplies them.
    opponentHand: cards(value.opponent_hand ?? value.opponentHand),
  }
}

function normalizeStreet(value: unknown): Street | undefined {
  return STREETS.includes(value as Street) ? (value as Street) : undefined
}

export function normalizePokerState(value: unknown): PokerState {
  const source = isRecord(value) ? value : {}
  const rawActions = source.available_actions ?? source.legal_actions
  const actions = Array.isArray(rawActions)
    ? rawActions.map(normalizeLegalAction).filter((item): item is LegalAction => item !== null)
    : []
  const rawHistory = Array.isArray(source.hand_history) ? source.hand_history : []
  const history = rawHistory
    .map(normalizeHandResult)
    .filter((item): item is HandResult => item !== null)
    .slice(0, 50)
  const rawBounds = isRecord(source.raise_bounds) ? source.raise_bounds : {}
  const minRaiseTo =
    firstNumber(source, ['min_raise_to', 'minRaiseTo']) ??
    firstNumber(rawBounds, ['min_raise_to', 'minRaiseTo', 'min'])
  const maxRaiseTo =
    firstNumber(source, ['max_raise_to', 'maxRaiseTo']) ??
    firstNumber(rawBounds, ['max_raise_to', 'maxRaiseTo', 'max'])
  const lastResult = normalizeHandResult(source.last_result ?? source.lastResult)
  const street = normalizeStreet(source.street)
  return {
    status: shortText(source.status, 'connecting', 40),
    error: nullableShortText(source.error, 240),
    stateNonce: Math.trunc(firstNumber(source, ['state_nonce', 'stateNonce']) ?? 0),
    ...(firstNumber(source, ['hand_number', 'handNumber']) !== undefined
      ? { handNumber: firstNumber(source, ['hand_number', 'handNumber']) }
      : {}),
    ...(street ? { street } : {}),
    board: cards(source.board),
    heroHand: cards(source.hero_hand ?? source.heroHand),
    // Never infer or reconstruct the opponent's private cards.
    opponentHand: cards(source.opponent_hand ?? source.opponentHand),
    ...(source.hero_position || source.heroPosition
      ? { heroPosition: shortText(source.hero_position ?? source.heroPosition, '', 12) }
      : {}),
    heroBet: firstNumber(source, ['hero_bet', 'heroBet']) ?? 0,
    opponentBet: firstNumber(source, ['opponent_bet', 'opponentBet']) ?? 0,
    heroStack: firstNumber(source, ['hero_stack', 'heroStack']) ?? finiteNumber(source.stack, 20_000),
    opponentStack:
      firstNumber(source, ['opponent_stack', 'opponentStack']) ?? finiteNumber(source.stack, 20_000),
    pot: finiteNumber(source.pot),
    tableStack: finiteNumber(source.stack, 20_000),
    smallBlind: firstNumber(source, ['small_blind', 'smallBlind']) ?? 50,
    bigBlind: firstNumber(source, ['big_blind', 'bigBlind']) ?? 100,
    cumulativeWinnings:
      firstNumber(source, ['cumulative_winnings', 'cumulativeWinnings']) ?? 0,
    handsCompleted: firstNumber(source, ['hands_completed', 'handsCompleted']) ?? 0,
    lastResult,
    handHistory: history,
    availableActions: actions,
    raiseBounds: {
      ...(minRaiseTo !== undefined ? { minRaiseTo } : {}),
      ...(maxRaiseTo !== undefined ? { maxRaiseTo } : {}),
    },
  }
}

const EMPTY_STATS: TimingStats = {
  count: 0,
  total: 0,
  mean: 0,
  p50: 0,
  p95: 0,
  max: 0,
}

function normalizeStats(value: unknown): TimingStats {
  if (!isRecord(value)) return { ...EMPTY_STATS }
  return {
    count: Math.max(0, Math.trunc(finiteNumber(value.count))),
    total: Math.max(0, finiteNumber(value.total)),
    mean: Math.max(0, finiteNumber(value.mean)),
    p50: Math.max(0, finiteNumber(value.p50)),
    p95: Math.max(0, finiteNumber(value.p95)),
    max: Math.max(0, finiteNumber(value.max)),
  }
}

function normalizeStreetTiming(value: unknown): StreetTiming {
  const source = isRecord(value) ? value : {}
  const timings = isRecord(source.timing_seconds) ? source.timing_seconds : {}
  return {
    decisions: Math.max(0, Math.trunc(finiteNumber(source.decisions))),
    latestAction: nullableShortText(source.latest_action, 40),
    totalResponse: normalizeStats(timings.total_response),
    cfr: normalizeStats(timings.cfr),
    resolveTotal: normalizeStats(timings.resolve_total),
    sampling: normalizeStats(timings.sampling),
  }
}

function normalizeRecentDecision(value: unknown): RecentDecision | null {
  if (!isRecord(value)) return null
  return {
    timestamp: nullableShortText(value.timestamp, 48),
    handNumber: optionalFiniteNumber(value.hand_number) ?? null,
    decisionNumber: optionalFiniteNumber(value.decision_number) ?? null,
    street: shortText(value.street, 'unknown', 20),
    pot: optionalFiniteNumber(value.pot) ?? null,
    chosenAction: shortText(value.chosen_action, 'unknown', 40),
    cfrIterations: optionalFiniteNumber(value.cfr_iterations) ?? null,
    totalResponseSeconds: Math.max(0, finiteNumber(value.total_response_seconds)),
    cfrSeconds: Math.max(0, finiteNumber(value.cfr_seconds)),
  }
}

export function normalizeSolverReport(value: unknown): SolverReport {
  const source = isRecord(value) ? value : {}
  const metadata = isRecord(source.metadata) ? source.metadata : {}
  const initialization = isRecord(source.initialization) ? source.initialization : {}
  const byStreetSource = isRecord(source.by_street) ? source.by_street : {}
  const byStreet = Object.fromEntries(
    STREETS.map((street) => [street, normalizeStreetTiming(byStreetSource[street])]),
  ) as Record<Street, StreetTiming>
  const recent = Array.isArray(source.recent_decisions)
    ? source.recent_decisions
        .map(normalizeRecentDecision)
        .filter((item): item is RecentDecision => item !== null)
        .slice(-12)
    : []
  return {
    updatedAt: nullableShortText(source.updated_at, 48),
    decisionCount: Math.max(0, Math.trunc(finiteNumber(source.decision_count))),
    metadata: {
      gpuName: nullableShortText(metadata.gpu_name, 100),
      cfrIterations: optionalFiniteNumber(metadata.cfr_iterations) ?? null,
      cfrSkipIterations: optionalFiniteNumber(metadata.cfr_skip_iterations) ?? null,
    },
    initializationSeconds: optionalFiniteNumber(initialization.seconds) ?? null,
    byStreet,
    recentDecisions: recent,
  }
}
