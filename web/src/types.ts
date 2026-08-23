export type Street = 'preflop' | 'flop' | 'turn' | 'river'

export interface LegalAction {
  id: string
  label: string
  amount?: number
  raiseTo?: number
  minRaiseTo?: number
  maxRaiseTo?: number
  potRaiseTo?: number
  presets: LegalRaisePreset[]
}

export interface LegalRaisePreset {
  id: string
  label: string
  raiseTo: number
}

export interface HandResult {
  handNumber: number
  winnings: number
  cumulativeWinnings?: number
  board: string[]
  heroHand: string[]
  opponentHand: string[]
}

export interface RaiseBounds {
  minRaiseTo?: number
  maxRaiseTo?: number
}

export interface PokerState {
  status: string
  error: string | null
  stateNonce: number
  handNumber?: number
  street?: Street
  board: string[]
  heroHand: string[]
  opponentHand: string[]
  heroPosition?: string
  heroBet: number
  opponentBet: number
  heroStack: number
  opponentStack: number
  pot: number
  tableStack: number
  smallBlind: number
  bigBlind: number
  cumulativeWinnings: number
  handsCompleted: number
  lastResult: HandResult | null
  handHistory: HandResult[]
  availableActions: LegalAction[]
  raiseBounds: RaiseBounds
}

export interface TimingStats {
  count: number
  total: number
  mean: number
  p50: number
  p95: number
  max: number
}

export interface StreetTiming {
  decisions: number
  latestAction: string | null
  totalResponse: TimingStats
  cfr: TimingStats
  resolveTotal: TimingStats
  sampling: TimingStats
}

export interface RecentDecision {
  timestamp: string | null
  handNumber: number | null
  decisionNumber: number | null
  street: string
  pot: number | null
  chosenAction: string
  cfrIterations: number | null
  totalResponseSeconds: number
  cfrSeconds: number
}

export interface SolverReport {
  updatedAt: string | null
  decisionCount: number
  metadata: {
    gpuName: string | null
    cfrIterations: number | null
    cfrSkipIterations: number | null
  }
  initializationSeconds: number | null
  byStreet: Record<Street, StreetTiming>
  recentDecisions: RecentDecision[]
}

export interface ActionSubmission {
  actionId: string
  stateNonce: number
  raiseTo?: number
}
