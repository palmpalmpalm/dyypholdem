import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { emptyReport, pokerState } from './test/fixtures'

vi.mock('./usePokerSession', () => ({
  usePokerSession: () => ({
    state: pokerState(),
    report: emptyReport(),
    loading: false,
    connected: true,
    busy: false,
    notice: null,
    submitAction: vi.fn().mockResolvedValue(true),
    notifyStale: vi.fn(),
    dismissNotice: vi.fn(),
  }),
}))

import App from './App'

describe('table chrome', () => {
  it('keeps diagnostics out of the play surface until requested', async () => {
    const user = userEvent.setup()
    render(<App />)

    const trigger = screen.getByRole('button', { name: 'Session' })
    expect(screen.queryByRole('dialog', { name: 'Session' })).not.toBeInTheDocument()
    await user.click(trigger)
    expect(screen.getByRole('dialog', { name: 'Session' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Close session panel' })).toHaveFocus()
    expect(document.querySelector('.table-layer')).toHaveAttribute('inert')
    expect(screen.getByText('Open-source notices')).toBeInTheDocument()

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog', { name: 'Session' })).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })
})
