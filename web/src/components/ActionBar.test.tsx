import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { pokerState } from '../test/fixtures'
import { ActionBar } from './ActionBar'
import { HeadsUpTable } from './HeadsUpTable'

describe('table actions', () => {
  it('offers a full raise sizer and submits an arbitrary canonical raise', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn().mockResolvedValue(true)
    render(<ActionBar state={pokerState()} busy={false} onSubmit={onSubmit} onStale={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Fold' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Call 200' })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: 'Raise, choose size' }))
    expect(screen.getByRole('dialog', { name: 'Raise size' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /three-quarter pot, raise to 1,250/i })).toBeEnabled()

    const input = screen.getByLabelText('Raise to')
    await user.clear(input)
    await user.type(input, '1234')
    await user.click(screen.getByRole('button', { name: 'Raise to 1,234' }))
    expect(onSubmit).toHaveBeenCalledWith({ actionId: 'raise', stateNonce: 42, raiseTo: 1234 })
  })

  it('closes the sizer and reports when its nonce becomes stale', async () => {
    const user = userEvent.setup()
    const onStale = vi.fn()
    const props = { busy: false, onSubmit: vi.fn().mockResolvedValue(true), onStale }
    const view = render(<ActionBar state={pokerState()} {...props} />)
    await user.click(screen.getByRole('button', { name: 'Raise, choose size' }))
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    view.rerender(<ActionBar state={pokerState({ stateNonce: 43 })} {...props} />)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(onStale).toHaveBeenCalledOnce()
  })

  it('submits the passive action identifier advertised by the server', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn().mockResolvedValue(true)
    const state = pokerState({
      heroBet: 300,
      opponentBet: 300,
      availableActions: [{ id: 'check', label: 'Check', presets: [] }],
    })
    render(<ActionBar state={state} busy={false} onSubmit={onSubmit} onStale={vi.fn()} />)
    await user.click(screen.getByRole('button', { name: 'Check' }))
    expect(onSubmit).toHaveBeenCalledWith({ actionId: 'check', stateNonce: 42 })
  })

  it('keeps opponent cards hidden unless the snapshot supplies them', () => {
    const view = render(<HeadsUpTable state={pokerState()} />)
    expect(screen.getAllByLabelText('Hidden card')).toHaveLength(2)
    expect(screen.queryByLabelText('Ace of hearts')).not.toBeInTheDocument()

    view.rerender(<HeadsUpTable state={pokerState({ opponentHand: ['Ah', 'Ad'] })} />)
    expect(screen.queryAllByLabelText('Hidden card')).toHaveLength(0)
    expect(screen.getByLabelText('Ace of hearts')).toBeInTheDocument()
    expect(screen.getByLabelText('Ace of diamonds')).toBeInTheDocument()
  })
})
