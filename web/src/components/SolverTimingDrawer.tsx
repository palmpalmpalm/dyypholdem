import { formatSeconds } from '../format'
import type { SolverReport, Street } from '../types'

const STREETS: Street[] = ['preflop', 'flop', 'turn', 'river']

export function SolverTimingDrawer({ report }: { report: SolverReport | null }) {
  return (
    <details className="side-section timing-drawer">
      <summary>
        <span>Solver timing</span>
        <span className="summary-count">{report?.decisionCount ?? 0}</span>
      </summary>
      <p className="privacy-note">Sanitized timings only. No private cards, ranges, or strategy probabilities.</p>
      <dl className="solver-meta">
        <div>
          <dt>GPU</dt>
          <dd>{report?.metadata.gpuName ?? '—'}</dd>
        </div>
        <div>
          <dt>Root solve</dt>
          <dd>{formatSeconds(report?.initializationSeconds, 2)}</dd>
        </div>
        <div>
          <dt>CFR iterations</dt>
          <dd>{report?.metadata.cfrIterations?.toLocaleString('en-US') ?? '—'}</dd>
        </div>
      </dl>
      <div className="timing-table-wrap">
        <table className="timing-table">
          <caption className="visually-hidden">Response time by street</caption>
          <thead>
            <tr>
              <th scope="col">Street</th>
              <th scope="col">n</th>
              <th scope="col">p50</th>
              <th scope="col">p95</th>
              <th scope="col">max</th>
            </tr>
          </thead>
          <tbody>
            {STREETS.map((street) => {
              const timing = report?.byStreet[street]
              return (
                <tr key={street}>
                  <th scope="row">{street}</th>
                  <td>{timing?.decisions ?? 0}</td>
                  <td>{formatSeconds(timing?.totalResponse.p50, 2)}</td>
                  <td>{formatSeconds(timing?.totalResponse.p95, 2)}</td>
                  <td>{formatSeconds(timing?.totalResponse.max, 2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <h3 className="recent-title">Recent decisions</h3>
      {report?.recentDecisions.length ? (
        <ol className="decision-list">
          {[...report.recentDecisions].reverse().map((decision, index) => (
            <li key={`${decision.decisionNumber ?? index}-${decision.timestamp ?? ''}`}>
              <div>
                <strong>{decision.street}</strong>
                <span>{decision.chosenAction}</span>
              </div>
              <span>{formatSeconds(decision.totalResponseSeconds)}</span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="empty-copy">No bot decisions recorded yet.</p>
      )}
    </details>
  )
}
