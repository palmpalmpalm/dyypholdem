import { useCallback, useEffect, useRef, useState } from 'react'
import { ActionBar } from './components/ActionBar'
import { HandHistory } from './components/HandHistory'
import { HeadsUpTable } from './components/HeadsUpTable'
import { SolverTimingDrawer } from './components/SolverTimingDrawer'
import { formatSeconds, formatSignedChips } from './format'
import { usePokerSession } from './usePokerSession'

export default function App() {
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false)
  const diagnosticsTriggerRef = useRef<HTMLButtonElement>(null)
  const diagnosticsPanelRef = useRef<HTMLElement>(null)
  const {
    state,
    report,
    loading,
    connected,
    busy,
    notice,
    submitAction,
    notifyStale,
    dismissNotice,
  } = usePokerSession()

  const sessionStatus = connected ? 'Connected' : loading ? 'Connecting' : 'Reconnecting'

  const closeDiagnostics = useCallback(() => setDiagnosticsOpen(false), [])

  useEffect(() => {
    if (!diagnosticsOpen) return
    const panel = diagnosticsPanelRef.current
    if (!panel) return

    const focusableElements = () =>
      Array.from(
        panel.querySelectorAll<HTMLElement>(
          'button:not([disabled]), a[href], summary, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => {
        const closedDetails = element.closest('details:not([open])')
        return !closedDetails || element.tagName === 'SUMMARY'
      })

    focusableElements()[0]?.focus()
    const containFocus = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeDiagnostics()
        return
      }
      if (event.key !== 'Tab') return
      const focusable = focusableElements()
      if (focusable.length === 0) {
        event.preventDefault()
        panel.focus()
        return
      }
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && (document.activeElement === first || !panel.contains(document.activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', containFocus)
    return () => {
      document.removeEventListener('keydown', containFocus)
      diagnosticsTriggerRef.current?.focus()
    }
  }, [closeDiagnostics, diagnosticsOpen])

  return (
    <div className="app-shell">
      <div
        className="table-layer"
        aria-hidden={diagnosticsOpen || undefined}
        inert={diagnosticsOpen || undefined}
      >
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            D
          </div>
          <div>
            <div className="eyebrow">Heads-up no-limit hold’em</div>
            <h1>DyypHoldem Table</h1>
          </div>
        </div>
        <div className="header-controls">
          <span className="header-badge connection-badge" data-connected={connected}>
            <span aria-hidden="true" />
            {sessionStatus}
          </span>
          <button
            ref={diagnosticsTriggerRef}
            className="diagnostics-toggle"
            type="button"
            aria-controls="diagnostics-panel"
            aria-expanded={diagnosticsOpen}
            onClick={() => setDiagnosticsOpen(true)}
          >
            <span aria-hidden="true">☰</span>
            Session
          </button>
        </div>
      </header>

      {notice ? (
        <div className={`notice-banner ${notice.tone}`} role="alert">
          <span>{notice.message}</span>
          <button type="button" onClick={dismissNotice} aria-label="Dismiss message">
            ×
          </button>
        </div>
      ) : null}

      <main className="table-room">
        <div className="play-column">
          <HeadsUpTable state={state} />
          {state ? (
            <ActionBar state={state} busy={busy} onSubmit={submitAction} onStale={notifyStale} />
          ) : (
            <section className="action-zone" aria-label="Poker actions">
              <div className="action-dock">
                <div className="action-waiting" aria-live="polite">
                  <span className="spinner" aria-hidden="true" />
                  <strong>Connecting to the dealer…</strong>
                </div>
              </div>
            </section>
          )}
          <p className="action-note">
            Bet amounts are total chips committed. The dealer supplies every legal action and limit.
          </p>
        </div>
      </main>
      </div>

      {diagnosticsOpen ? (
        <>
          <div
            className="diagnostics-backdrop"
            aria-hidden="true"
            onClick={closeDiagnostics}
          />
          <aside
            ref={diagnosticsPanelRef}
            id="diagnostics-panel"
            className="side-rail"
            role="dialog"
            aria-modal="true"
            aria-labelledby="diagnostics-title"
            tabIndex={-1}
          >
            <header className="drawer-header">
              <div>
                <span className="eyebrow">Private test table</span>
                <h2 id="diagnostics-title">Session</h2>
              </div>
              <button
                className="icon-button"
                type="button"
                onClick={closeDiagnostics}
                aria-label="Close session panel"
              >
                ×
              </button>
            </header>

            <section className="side-section session-card">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">Table status</span>
                  <h3>{state?.street ? `${state.street} play` : 'Starting table'}</h3>
                </div>
                <span className={`live-pill${state?.status === 'your_turn' ? ' your-turn' : ''}`}>
                  {state?.status === 'your_turn' ? 'Act now' : 'Live'}
                </span>
              </div>
              <div className="metric-grid">
                <div>
                  <span>Hands</span>
                  <strong>{state?.handsCompleted ?? 0}</strong>
                </div>
                <div>
                  <span>Net chips</span>
                  <strong className={(state?.cumulativeWinnings ?? 0) >= 0 ? 'positive' : 'negative'}>
                    {formatSignedChips(state?.cumulativeWinnings ?? 0)}
                  </strong>
                </div>
                <div>
                  <span>Root solve</span>
                  <strong>{formatSeconds(report?.initializationSeconds, 2)}</strong>
                </div>
                <div>
                  <span>Decisions</span>
                  <strong>{report?.decisionCount ?? 0}</strong>
                </div>
              </div>
              {state?.lastResult ? (
                <div className="last-result">
                  <span>Last hand</span>
                  <strong className={state.lastResult.winnings >= 0 ? 'positive' : 'negative'}>
                    {formatSignedChips(state.lastResult.winnings)} chips
                  </strong>
                </div>
              ) : null}
              {state?.error ? <p className="server-error">{state.error}</p> : null}
            </section>

            <HandHistory hands={state?.handHistory ?? []} />
            <SolverTimingDrawer report={report} />

            <details className="side-section notices-section">
              <summary>Open-source notices</summary>
              <p className="privacy-note">
                Table presentation is adapted from Elite-Poker; sizing interaction is inspired by
                Pip Web. Both are MIT-licensed. DyypHoldem’s game and solver code remain separate.
              </p>
              <a href="/THIRD_PARTY_NOTICES.txt" target="_blank" rel="noreferrer">
                Read full notices
              </a>
            </details>
          </aside>
        </>
      ) : null}
    </div>
  )
}
