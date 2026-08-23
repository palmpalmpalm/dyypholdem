import { createPokerApi, removeLegacyTokenFromAddressBar, StaleActionError } from './api'

describe('poker API', () => {
  it('uses same-origin cookie credentials and canonical action payload', async () => {
    const fetcher = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ accepted: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    const api = createPokerApi(fetcher as typeof fetch)
    await api.submitAction({ actionId: 'raise', stateNonce: 7, raiseTo: 1250 })
    expect(fetcher).toHaveBeenCalledOnce()
    const [, options] = fetcher.mock.calls[0]
    expect(options.credentials).toBe('same-origin')
    expect(JSON.parse(options.body)).toEqual({
      action: 'raise',
      action_id: 'raise',
      state_nonce: 7,
      raise_to: 1250,
    })
    expect(options.headers).not.toHaveProperty('X-Session-Token')
  })

  it('turns a conflict into a stale-action error', async () => {
    const fetcher = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: 'stale state; refresh before acting' }), {
        status: 409,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    const api = createPokerApi(fetcher as typeof fetch)
    await expect(api.submitAction({ actionId: 'call', stateNonce: 4 })).rejects.toBeInstanceOf(
      StaleActionError,
    )
  })

  it('aborts a half-open request so polling can continue', async () => {
    vi.useFakeTimers()
    try {
      const fetcher = vi.fn((_path: RequestInfo | URL, options?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          options?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Request timed out', 'AbortError'))
          })
        }),
      )
      const api = createPokerApi(fetcher as typeof fetch)
      const rejection = expect(api.getState()).rejects.toMatchObject({ name: 'AbortError' })
      await vi.advanceTimersByTimeAsync(10_000)
      await rejection
    } finally {
      vi.useRealTimers()
    }
  })

  it('removes a legacy token from the address bar without storage access', () => {
    const getItem = vi.spyOn(Storage.prototype, 'getItem')
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    window.history.pushState({}, '', '/table?token=do-not-read&view=compact#hand')
    removeLegacyTokenFromAddressBar()
    expect(window.location.pathname + window.location.search + window.location.hash).toBe(
      '/table?view=compact#hand',
    )
    expect(getItem).not.toHaveBeenCalled()
    expect(setItem).not.toHaveBeenCalled()
  })
})
