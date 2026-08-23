import { normalizePokerState, normalizeSolverReport } from './normalizers'
import type { ActionSubmission, PokerState, SolverReport } from './types'

const REQUEST_TIMEOUT_MS = 10_000

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export class StaleActionError extends ApiError {
  constructor(message = 'The table changed before that action was sent.') {
    super(message, 409)
    this.name = 'StaleActionError'
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as unknown
    if (typeof payload === 'object' && payload !== null && 'error' in payload) {
      const message = (payload as { error?: unknown }).error
      if (typeof message === 'string') return message.slice(0, 240)
    }
  } catch {
    // The fallback below intentionally avoids echoing an arbitrary HTML response.
  }
  return `Request failed (${response.status})`
}

export function removeLegacyTokenFromAddressBar(): void {
  const url = new URL(window.location.href)
  if (!url.searchParams.has('token')) return
  url.searchParams.delete('token')
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`)
}

export interface PokerApi {
  getState(signal?: AbortSignal): Promise<PokerState>
  getReport(signal?: AbortSignal): Promise<SolverReport>
  submitAction(submission: ActionSubmission, signal?: AbortSignal): Promise<void>
}

export function createPokerApi(fetcher: typeof fetch = window.fetch.bind(window)): PokerApi {
  async function request(
    path: string,
    options: RequestInit,
    outerSignal?: AbortSignal,
  ): Promise<Response> {
    const controller = new AbortController()
    const abort = () => controller.abort()
    if (outerSignal?.aborted) controller.abort()
    else outerSignal?.addEventListener('abort', abort, { once: true })
    const timeout = window.setTimeout(abort, REQUEST_TIMEOUT_MS)
    try {
      return await fetcher(path, { ...options, signal: controller.signal })
    } finally {
      window.clearTimeout(timeout)
      outerSignal?.removeEventListener('abort', abort)
    }
  }

  async function get(path: string, signal?: AbortSignal): Promise<unknown> {
    const response = await request(path, {
      method: 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { Accept: 'application/json' },
    }, signal)
    if (!response.ok) throw new ApiError(await errorMessage(response), response.status)
    return response.json() as Promise<unknown>
  }

  return {
    async getState(signal) {
      return normalizePokerState(await get('/api/state', signal))
    },
    async getReport(signal) {
      return normalizeSolverReport(await get('/api/report', signal))
    },
    async submitAction(submission, signal) {
      const payload: Record<string, number | string> = {
        // `action` keeps compatibility with the current bridge; `action_id` is
        // the explicit form accepted by the next server adapter.
        action: submission.actionId,
        action_id: submission.actionId,
        state_nonce: submission.stateNonce,
      }
      if (submission.raiseTo !== undefined) payload.raise_to = submission.raiseTo
      const response = await request('/api/action', {
        method: 'POST',
        credentials: 'same-origin',
        cache: 'no-store',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      }, signal)
      if (response.status === 409) throw new StaleActionError(await errorMessage(response))
      if (!response.ok) throw new ApiError(await errorMessage(response), response.status)
    },
  }
}
