import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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

  it('does not report a successful in-flight raise as stale when its refresh advances the nonce', async () => {
    const user = userEvent.setup()
    let finish: ((accepted: boolean) => void) | undefined
    const onSubmit = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finish = resolve
        }),
    )
    const onStale = vi.fn()
    const props = { busy: false, onSubmit, onStale }
    const view = render(<ActionBar state={pokerState()} {...props} />)

    await user.click(screen.getByRole('button', { name: 'Raise, choose size' }))
    await user.click(screen.getByRole('button', { name: 'Raise to 1,500' }))
    view.rerender(<ActionBar state={pokerState({ stateNonce: 43 })} {...props} />)
    expect(onStale).not.toHaveBeenCalled()
    expect(screen.getByRole('dialog', { name: 'Raise size' })).toBeInTheDocument()

    await act(async () => finish?.(true))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    expect(onStale).not.toHaveBeenCalled()
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

  it('labels an unopened postflop wager as a bet despite cumulative commitments', () => {
    const state = pokerState({
      heroBet: 300,
      opponentBet: 300,
      availableActions: pokerState().availableActions.filter((action) => action.id !== 'call'),
    })
    render(<ActionBar state={state} busy={false} onSubmit={vi.fn()} onStale={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Bet, choose size' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Raise, choose size' })).not.toBeInTheDocument()
  })

  it('submits all-in through the exact generic raise advertised by the server', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn().mockResolvedValue(true)
    render(<ActionBar state={pokerState()} busy={false} onSubmit={onSubmit} onStale={vi.fn()} />)

    await user.click(screen.getByRole('button', { name: 'All-in to 20,000' }))
    expect(onSubmit).toHaveBeenCalledWith({ actionId: 'raise', stateNonce: 42, raiseTo: 20_000 })
  })

  it('shows one all-in control instead of a duplicate sizer for a short all-in', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn().mockResolvedValue(true)
    const state = pokerState({
      availableActions: [
        { id: 'fold', label: 'Fold', presets: [] },
        { id: 'call', label: 'Call 200', amount: 200, presets: [] },
        {
          id: 'raise',
          label: 'Raise',
          minRaiseTo: 19_700,
          maxRaiseTo: 19_700,
          presets: [{ id: 'all_in', label: 'All-in', raiseTo: 19_700 }],
        },
      ],
    })
    render(<ActionBar state={state} busy={false} onSubmit={onSubmit} onStale={vi.fn()} />)

    expect(screen.queryByRole('button', { name: 'Raise, choose size' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'All-in to 19,700' }))
    expect(onSubmit).toHaveBeenCalledWith({ actionId: 'raise', stateNonce: 42, raiseTo: 19_700 })
  })

  it('suppresses duplicate action submissions while the first request is pending', () => {
    const onSubmit = vi.fn().mockReturnValue(new Promise<boolean>(() => undefined))
    render(<ActionBar state={pokerState()} busy={false} onSubmit={onSubmit} onStale={vi.fn()} />)

    const fold = screen.getByRole('button', { name: 'Fold' })
    fireEvent.click(fold)
    fireEvent.click(fold)
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('keeps the dock stable without inventing actions outside the human turn', () => {
    render(
      <ActionBar
        state={pokerState({ status: 'bot_thinking', availableActions: [] })}
        busy={false}
        onSubmit={vi.fn()}
        onStale={vi.fn()}
      />,
    )

    expect(screen.getByText('DyypHoldem is thinking…')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Fold' })).not.toBeInTheDocument()
  })

  it('keeps opponent cards hidden unless the snapshot supplies them', () => {
    const view = render(<HeadsUpTable state={pokerState()} />)
    expect(screen.getAllByLabelText('Hidden card')).toHaveLength(2)
    expect(screen.queryByLabelText('Ace of hearts')).not.toBeInTheDocument()

    view.rerender(<HeadsUpTable state={pokerState({ opponentHand: ['Ah', 'Ad'] })} />)
    expect(screen.getAllByLabelText('Hidden card')).toHaveLength(2)
    expect(screen.queryByLabelText('Ace of hearts')).not.toBeInTheDocument()

    view.rerender(
      <HeadsUpTable
        state={pokerState({ status: 'hand_complete', opponentHand: ['Ah', 'Ad'] })}
      />,
    )
    expect(screen.queryAllByLabelText('Hidden card')).toHaveLength(0)
    expect(screen.getByLabelText('Ace of hearts')).toBeInTheDocument()
    expect(screen.getByLabelText('Ace of diamonds')).toBeInTheDocument()
  })

  it('derives heads-up position markers and highlights only the current actor', () => {
    render(<HeadsUpTable state={pokerState({ heroPosition: 'SB', status: 'bot_thinking' })} />)

    const opponent = screen.getByRole('region', { name: 'DyypHoldem seat' })
    const hero = screen.getByRole('region', { name: 'You seat' })
    expect(opponent).toHaveAttribute('data-active', 'true')
    expect(hero).not.toHaveAttribute('data-active')
    expect(within(opponent).getByLabelText('BB')).toBeInTheDocument()
    expect(within(hero).getByLabelText('SB, dealer')).toBeInTheDocument()
  })

  it('shows a winner only for the matching completed hand', () => {
    const result = {
      handNumber: 19,
      winnings: 1_250,
      board: ['2c', '7d', 'Jh', 'Ts', '3c'],
      heroHand: ['As', 'Kd'],
      opponentHand: ['Qh', 'Qc'],
    }
    const view = render(<HeadsUpTable state={pokerState({ lastResult: result })} />)
    expect(screen.queryByText('You win')).not.toBeInTheDocument()

    view.rerender(
      <HeadsUpTable state={pokerState({ status: 'hand_complete', handNumber: 20, lastResult: result })} />,
    )
    expect(screen.queryByText('You win')).not.toBeInTheDocument()

    view.rerender(<HeadsUpTable state={pokerState({ status: 'hand_complete', lastResult: result })} />)
    expect(screen.getByText('You win')).toBeInTheDocument()
    expect(screen.getByText('Winner')).toBeInTheDocument()
  })

  it('shows a prior-hand result separately when the dealer already advanced', () => {
    const result = {
      handNumber: 19,
      winnings: 1_250,
      board: ['2c', '7d', 'Jh', 'Ts', '3c'],
      heroHand: ['As', 'Kd'],
      opponentHand: ['Qh', 'Qc'],
    }
    render(
      <HeadsUpTable
        state={pokerState({ handNumber: 20, status: 'your_turn', lastResult: result })}
      />,
    )

    expect(screen.getByText('Previous hand')).toBeInTheDocument()
    expect(screen.getByText('You won +1,250')).toBeInTheDocument()
    expect(screen.queryByText('Winner')).not.toBeInTheDocument()
  })
})
