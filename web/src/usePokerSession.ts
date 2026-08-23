import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiError,
  createPokerApi,
  removeLegacyTokenFromAddressBar,
  StaleActionError,
  type PokerApi,
} from './api'
import type { ActionSubmission, PokerState, SolverReport } from './types'

export interface SessionNotice {
  tone: 'error' | 'info'
  message: string
}

export function usePokerSession(injectedApi?: PokerApi) {
  const api = useMemo(() => injectedApi ?? createPokerApi(), [injectedApi])
  const [state, setState] = useState<PokerState | null>(null)
  const [report, setReport] = useState<SolverReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<SessionNotice | null>(null)
  const stateRef = useRef<PokerState | null>(null)
  const latestRefreshRef = useRef(0)
  const reportInFlightRef = useRef(false)
  const lastReportStartedAtRef = useRef(0)

  const refresh = useCallback(
    async (signal?: AbortSignal, includeReport = true) => {
      const refreshId = ++latestRefreshRef.current
      const now = Date.now()
      if (
        includeReport &&
        !reportInFlightRef.current &&
        now - lastReportStartedAtRef.current >= 4_000
      ) {
        reportInFlightRef.current = true
        lastReportStartedAtRef.current = now
        void api
          .getReport(signal)
          .then((nextReport) => {
            if (!signal?.aborted) setReport(nextReport)
          })
          .catch(() => {
            // Solver diagnostics never block or downgrade the live table state.
          })
          .finally(() => {
            reportInFlightRef.current = false
          })
      }

      const stateResult = await api.getState(signal).then(
        (value) => ({ status: 'fulfilled' as const, value }),
        (reason: unknown) => ({ status: 'rejected' as const, reason }),
      )
      // A polling request and a post-action refresh can overlap. Only the
      // newest request may update the table or restore an older state nonce.
      if (refreshId !== latestRefreshRef.current) return
      if (stateResult.status === 'fulfilled') {
        stateRef.current = stateResult.value
        setState(stateResult.value)
        setConnected(true)
      } else if (!(stateResult.reason instanceof Error && stateResult.reason.name === 'AbortError')) {
        setConnected(false)
        const reason = stateResult.reason
        const message =
          reason instanceof ApiError && reason.status === 401
            ? 'This play session is no longer authorized.'
            : 'Connection interrupted. Retrying…'
        setNotice((current) => current?.tone === 'error' ? current : { tone: 'error', message })
      }
      setLoading(false)
    },
    [api],
  )

  useEffect(() => {
    // Authentication belongs to the HttpOnly same-origin cookie. This only
    // removes a legacy query parameter from browser history without reading it.
    removeLegacyTokenFromAddressBar()
    let active = true
    let timer: number | undefined
    let controller = new AbortController()

    const poll = async () => {
      await refresh(controller.signal)
      if (active) timer = window.setTimeout(poll, 1200)
    }
    void poll()
    return () => {
      active = false
      latestRefreshRef.current += 1
      if (timer !== undefined) window.clearTimeout(timer)
      controller.abort()
      controller = new AbortController()
    }
  }, [refresh])

  const submitAction = useCallback(
    async (submission: ActionSubmission): Promise<boolean> => {
      if (!stateRef.current || stateRef.current.stateNonce !== submission.stateNonce) {
        setNotice({ tone: 'info', message: 'The table changed. Refreshed before sending your action.' })
        await refresh(undefined, false)
        return false
      }
      setBusy(true)
      try {
        await api.submitAction(submission)
        await refresh(undefined, false)
        return true
      } catch (error) {
        if (error instanceof StaleActionError) {
          setNotice({ tone: 'info', message: 'That action was stale. The latest table state is now shown.' })
        } else if (error instanceof ApiError) {
          setNotice({ tone: 'error', message: error.message })
        } else {
          setNotice({ tone: 'error', message: 'The action could not be sent. Please try again.' })
        }
        await refresh(undefined, false)
        return false
      } finally {
        setBusy(false)
      }
    },
    [api, refresh],
  )

  const notifyStale = useCallback(() => {
    setNotice({ tone: 'info', message: 'The table changed while sizing. Choose the action again.' })
  }, [])

  return {
    state,
    report,
    loading,
    connected,
    busy,
    notice,
    submitAction,
    notifyStale,
    dismissNotice: () => setNotice(null),
  }
}
