import type { PokerState, SolverReport } from '../types'

export function pokerState(overrides: Partial<PokerState> = {}): PokerState {
  return {
    status: 'your_turn',
    error: null,
    stateNonce: 42,
    handNumber: 19,
    street: 'flop',
    board: ['2c', '7d', 'Jh'],
    heroHand: ['As', 'Kd'],
    opponentHand: [],
    heroPosition: 'BB',
    heroBet: 300,
    opponentBet: 500,
    heroStack: 19_700,
    opponentStack: 19_500,
    pot: 800,
    tableStack: 20_000,
    smallBlind: 50,
    bigBlind: 100,
    cumulativeWinnings: 1250,
    handsCompleted: 18,
    lastResult: null,
    handHistory: [],
    availableActions: [
      { id: 'fold', label: 'Fold', presets: [] },
      { id: 'call', label: 'Call 200', amount: 200, presets: [] },
      {
        id: 'raise',
        label: 'Raise',
        minRaiseTo: 700,
        maxRaiseTo: 20_000,
        potRaiseTo: 1500,
        presets: [
          { id: 'min', label: 'Min', raiseTo: 700 },
          { id: 'half_pot', label: '½ pot', raiseTo: 1000 },
          { id: 'three_quarter_pot', label: '¾ pot', raiseTo: 1250 },
          { id: 'pot', label: 'Pot', raiseTo: 1500 },
          { id: 'all_in', label: 'All-in', raiseTo: 20_000 },
        ],
      },
    ],
    raiseBounds: {},
    ...overrides,
  }
}

export function emptyReport(): SolverReport {
  const stats = { count: 0, total: 0, mean: 0, p50: 0, p95: 0, max: 0 }
  const street = {
    decisions: 0,
    latestAction: null,
    totalResponse: stats,
    cfr: stats,
    resolveTotal: stats,
    sampling: stats,
  }
  return {
    updatedAt: null,
    decisionCount: 0,
    metadata: { gpuName: null, cfrIterations: null, cfrSkipIterations: null },
    initializationSeconds: null,
    byStreet: { preflop: street, flop: street, turn: street, river: street },
    recentDecisions: [],
  }
}
