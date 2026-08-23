import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin } from 'vite'

const report = {
  schema_version: 1,
  updated_at: '2026-08-23T12:00:00Z',
  decision_count: 9,
  metadata: {
    gpu_name: 'NVIDIA GeForce RTX 4090 (mock)',
    cfr_iterations: 1000,
    cfr_skip_iterations: 500,
  },
  initialization: { seconds: 10.239 },
  by_street: {
    preflop: streetTiming(3, 0.084, 0.102, 0.111, 0),
    flop: streetTiming(3, 5.632, 6.104, 6.221, 4.941),
    turn: streetTiming(2, 2.841, 3.027, 3.048, 2.392),
    river: streetTiming(1, 0.714, 0.714, 0.714, 0.488),
  },
  recent_decisions: [
    {
      timestamp: '2026-08-23T11:58:02Z',
      hand_number: 18,
      decision_number: 8,
      street: 'turn',
      pot: 3500,
      chosen_action: 'call',
      cfr_iterations: 1000,
      total_response_seconds: 2.841,
      cfr_seconds: 2.392,
    },
    {
      timestamp: '2026-08-23T11:59:20Z',
      hand_number: 19,
      decision_number: 9,
      street: 'flop',
      pot: 800,
      chosen_action: 'raise',
      cfr_iterations: 1000,
      total_response_seconds: 5.632,
      cfr_seconds: 4.941,
    },
  ],
}

function stats(count: number, p50: number, p95: number, maximum: number) {
  return { count, total: p50 * count, mean: p50, p50, p95, max: maximum }
}

function streetTiming(count: number, p50: number, p95: number, maximum: number, cfr: number) {
  return {
    decisions: count,
    latest_action: count ? 'raise' : null,
    timing_seconds: {
      total_response: stats(count, p50, p95, maximum),
      cfr: stats(count, cfr, cfr, cfr),
      resolve_total: stats(count, cfr + 0.3, cfr + 0.4, cfr + 0.5),
      sampling: stats(count, 0.04, 0.06, 0.07),
    },
  }
}

let nonce = 42
let status = 'your_turn'

function state() {
  return {
    status,
    error: null,
    started_at: '2026-08-23T10:45:29Z',
    state_nonce: nonce,
    hand_number: 19,
    street: 'flop',
    stack: 20_000,
    small_blind: 50,
    big_blind: 100,
    board: ['2c', '7d', 'Jh'],
    hero_hand: ['As', 'Kd'],
    opponent_hand: [],
    hero_position: 'BB',
    hero_bet: 300,
    opponent_bet: 500,
    hero_stack: 19_700,
    opponent_stack: 19_500,
    pot: 800,
    cumulative_winnings: 1250,
    hands_completed: 18,
    available_actions: [
      { id: 'fold', label: 'Fold' },
      { id: 'call', label: 'Call 200', amount: 200 },
      {
        id: 'raise',
        label: 'Raise',
        min_raise_to: 700,
        max_raise_to: 20_000,
        pot_raise_to: 1500,
        presets: [
          { id: 'half_pot', label: '½ pot', raise_to: 1000 },
          { id: 'three_quarter_pot', label: '¾ pot', raise_to: 1250 },
          { id: 'pot', label: 'Pot', raise_to: 1500 },
          { id: 'all_in', label: 'All-in', raise_to: 20_000 },
        ],
      },
    ],
    last_result: {
      hand_number: 18,
      winnings: 750,
      cumulative_winnings: 1250,
      board: ['Ah', '7s', '7c', 'Td', '2s'],
      hero_hand: ['Ac', 'Qd'],
      opponent_hand: ['Ks', 'Kh'],
    },
    hand_history: [
      {
        hand_number: 18,
        winnings: 750,
        cumulative_winnings: 1250,
        board: ['Ah', '7s', '7c', 'Td', '2s'],
        hero_hand: ['Ac', 'Qd'],
        opponent_hand: ['Ks', 'Kh'],
      },
      {
        hand_number: 17,
        winnings: -300,
        cumulative_winnings: 500,
        board: ['Ts', '8s', '5s'],
        hero_hand: ['Qh', 'Jd'],
        opponent_hand: [],
      },
    ],
  }
}

function sendJson(response: ServerResponse, payload: unknown, statusCode = 200) {
  response.statusCode = statusCode
  response.setHeader('Content-Type', 'application/json; charset=utf-8')
  response.setHeader('Cache-Control', 'no-store')
  response.end(JSON.stringify(payload))
}

async function requestBody(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = []
  for await (const chunk of request) chunks.push(Buffer.from(chunk))
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'))
    return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

export function mockApiPlugin(): Plugin {
  return {
    name: 'dyypholdem-dev-mock-api',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const path = new URL(request.url ?? '/', 'http://localhost').pathname
        if (request.method === 'GET' && path === '/api/state') return sendJson(response, state())
        if (request.method === 'GET' && path === '/api/report') return sendJson(response, report)
        if (request.method !== 'POST' || path !== '/api/action') return next()

        const body = await requestBody(request)
        if (Number(body.state_nonce) !== nonce || status !== 'your_turn') {
          return sendJson(response, { error: 'stale state; refresh before acting' }, 409)
        }
        const action = String(body.action ?? body.action_id ?? '')
        const raiseTo = Number(body.raise_to)
        const legal =
          action === 'fold' ||
          action === 'call' ||
          (action === 'raise' && Number.isFinite(raiseTo) && raiseTo >= 700 && raiseTo <= 20_000)
        if (!legal) return sendJson(response, { error: 'action is not legal in this state' }, 400)

        status = 'bot_thinking'
        nonce += 1
        sendJson(response, { accepted: true })
        setTimeout(() => {
          status = 'your_turn'
          nonce += 1
        }, 900)
      })
    },
  }
}
