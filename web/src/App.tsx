import { ActionBar } from './components/ActionBar'
import { HandHistory } from './components/HandHistory'
import { HeadsUpTable } from './components/HeadsUpTable'
import { SolverTimingDrawer } from './components/SolverTimingDrawer'
import { formatSeconds, formatSignedChips } from './format'
import { usePokerSession } from './usePokerSession'

export default function App() {
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
  const gpuName = report?.metadata.gpuName?.replace('NVIDIA GeForce ', '') ?? 'GPU pending'

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            D
          </div>
          <div>
            <div className="eyebrow">Live model test</div>
            <h1>DyypHoldem</h1>
          </div>
        </div>
        <div className="header-badges">
          <span className="header-badge connection-badge" data-connected={connected}>
            <span aria-hidden="true" />
            {sessionStatus}
          </span>
          <span className="header-badge gpu-badge">{gpuName}</span>
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

      <main className="workspace">
        <div className="play-column">
          <HeadsUpTable state={state} />
          {state ? (
            <ActionBar state={state} busy={busy} onSubmit={submitAction} onStale={notifyStale} />
          ) : (
            <div className="action-dock waiting-dock" aria-live="polite">
              <span className="spinner" />
              <span>Connecting to the dealer</span>
            </div>
          )}
          <p className="action-note">
            Legal actions and raise bounds come from the server. Raise values are the total amount
            committed, not the additional chips.
          </p>
        </div>

        <aside className="side-rail" aria-label="Session information">
          <section className="side-section session-card">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Session</span>
                <h2>{state?.street ? `${state.street} play` : 'Starting table'}</h2>
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
        </aside>
      </main>

      <footer className="app-footer">
        <span>Play-money research interface · ACPC dealer remains authoritative</span>
        <span>
          Sizing interaction inspired by{' '}
          <a href="https://github.com/playpip/pip-web" target="_blank" rel="noreferrer">
            Pip Web
          </a>{' '}
          (MIT)
        </span>
      </footer>
    </div>
  )
}
