import { buildRaisePlan, raisePresets, resolveRaiseAction } from './actionSizing'
import { pokerState } from './test/fixtures'

describe('raise sizing', () => {
  it('uses canonical server bounds and advertised presets for arbitrary raises', () => {
    const state = pokerState()
    const plan = buildRaisePlan(state)
    expect(plan).not.toBeNull()
    expect(plan).toMatchObject({ minimum: 700, maximum: 20_000, genericActionId: 'raise' })
    expect(raisePresets(state, plan!).map((preset) => [preset.key, preset.raiseTo, preset.enabled])).toEqual([
      ['min', 700, true],
      ['half', 1000, true],
      ['three-quarter', 1250, true],
      ['pot', 1500, true],
      ['all-in', 20_000, true],
    ])
    expect(resolveRaiseAction(plan!, 1234)).toEqual({ actionId: 'raise', raiseTo: 1234, label: 'Raise' })
  })

  it('supports legacy discrete pot/all-in actions without inventing legality', () => {
    const state = pokerState({
      availableActions: [
        { id: 'fold', label: 'Fold', presets: [] },
        { id: 'call', label: 'Call 50', amount: 50, presets: [] },
        { id: 'pot', label: 'Pot', raiseTo: 300, presets: [] },
        { id: 'all_in', label: 'All-in', raiseTo: 20_000, presets: [] },
      ],
      heroBet: 50,
      opponentBet: 100,
      pot: 150,
    })
    const plan = buildRaisePlan(state)!
    expect(plan).toMatchObject({ minimum: 300, maximum: 20_000, genericActionId: null })
    expect(resolveRaiseAction(plan, 300)?.actionId).toBe('pot')
    expect(resolveRaiseAction(plan, 700)).toBeNull()
    const presets = raisePresets(state, plan)
    expect(presets.find((preset) => preset.key === 'half')?.enabled).toBe(false)
    expect(presets.find((preset) => preset.key === 'pot')?.enabled).toBe(true)
  })

  it('does not expose a raise plan when a generic action has no legal bounds', () => {
    const state = pokerState({
      availableActions: [{ id: 'raise', label: 'Raise', presets: [] }],
      raiseBounds: {},
    })
    expect(buildRaisePlan(state)).toBeNull()
  })
})
