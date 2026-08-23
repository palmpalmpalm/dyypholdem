import { formatSignedChips } from '../format'
import type { HandResult } from '../types'
import { Card } from './Card'

export function HandHistory({ hands }: { hands: HandResult[] }) {
  return (
    <details className="side-section history-section">
      <summary>
        <span>Hand history</span>
        <span className="summary-count">{hands.length}</span>
      </summary>
      {hands.length === 0 ? (
        <p className="empty-copy">Completed hands will appear here.</p>
      ) : (
        <ol className="hand-list">
          {hands.map((hand) => (
            <li key={hand.handNumber}>
              <div className="history-heading">
                <strong>Hand #{hand.handNumber}</strong>
                <span className={hand.winnings >= 0 ? 'positive' : 'negative'}>
                  {formatSignedChips(hand.winnings)}
                </span>
              </div>
              <div className="history-row">
                <span>You</span>
                <span className="mini-cards">
                  {hand.heroHand.map((card, index) => (
                    <Card key={`${card}-${index}`} card={card} compact />
                  ))}
                </span>
              </div>
              {hand.opponentHand.length > 0 ? (
                <div className="history-row">
                  <span>Bot</span>
                  <span className="mini-cards">
                    {hand.opponentHand.map((card, index) => (
                      <Card key={`${card}-${index}`} card={card} compact />
                    ))}
                  </span>
                </div>
              ) : null}
              <div className="history-board" aria-label="Board">
                {hand.board.map((card, index) => (
                  <Card key={`${card}-${index}`} card={card} compact />
                ))}
              </div>
            </li>
          ))}
        </ol>
      )}
    </details>
  )
}
