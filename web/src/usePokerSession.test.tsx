import { act, renderHook, waitFor } from '@testing-library/react'
import type { PokerApi } from './api'
import { emptyReport, pokerState } from './test/fixtures'
import { usePokerSession } from './usePokerSession'
import type { PokerState } from './types'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((accept) => {
    resolve = accept
  })
  return { promise, resolve }
}

describe('poker session refresh ordering', () => {
  it('does not let an older overlapping refresh restore stale state', async () => {
    const older = deferred<PokerState>()
    const newer = deferred<PokerState>()
    const getState = vi
      .fn()
      .mockResolvedValueOnce(pokerState({ stateNonce: 42 }))
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise)
    const api: PokerApi = {
      getState,
      getReport: vi.fn().mockResolvedValue(emptyReport()),
      submitAction: vi.fn().mockResolvedValue(undefined),
    }
    const view = renderHook(() => usePokerSession(api))
    await waitFor(() => expect(view.result.current.state?.stateNonce).toBe(42))

    let first!: Promise<boolean>
    act(() => {
      first = view.result.current.submitAction({ actionId: 'call', stateNonce: 42 })
    })
    await waitFor(() => expect(getState).toHaveBeenCalledTimes(2))

    let second!: Promise<boolean>
    act(() => {
      second = view.result.current.submitAction({ actionId: 'call', stateNonce: 42 })
    })
    await waitFor(() => expect(getState).toHaveBeenCalledTimes(3))

    await act(async () => {
      newer.resolve(pokerState({ stateNonce: 43, status: 'bot_thinking' }))
      await second
    })
    await act(async () => {
      older.resolve(pokerState({ stateNonce: 42, status: 'your_turn' }))
      await first
    })

    expect(view.result.current.state?.stateNonce).toBe(43)
    expect(view.result.current.state?.status).toBe('bot_thinking')
    view.unmount()
  })
})
