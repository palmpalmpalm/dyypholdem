import { normalizePokerState, normalizeSolverReport } from './normalizers'

describe('API normalization and privacy', () => {
  it('parses canonical raise bounds/presets and only valid supplied cards', () => {
    const state = normalizePokerState({
      status: 'your_turn',
      state_nonce: 9,
      hero_hand: ['As', 'Kd'],
      opponent_hand: ['not-a-card', '<script>'],
      available_actions: [
        {
          id: 'raise',
          label: 'Raise',
          min_raise_to: 300,
          max_raise_to: 20_000,
          pot_raise_to: 900,
          presets: [{ id: 'pot', label: 'Pot', raise_to: 900 }],
        },
      ],
    })
    expect(state.heroHand).toEqual(['As', 'Kd'])
    expect(state.opponentHand).toEqual([])
    expect(state.availableActions[0]).toMatchObject({
      id: 'raise',
      minRaiseTo: 300,
      maxRaiseTo: 20_000,
      potRaiseTo: 900,
      presets: [{ id: 'pot', label: 'Pot', raiseTo: 900 }],
    })
  })

  it('allowlists the safe solver report instead of retaining private fields', () => {
    const report = normalizeSolverReport({
      metadata: { gpu_name: 'RTX 4090', model_paths: ['/private/model.pt'] },
      initialization: { seconds: 10, private_cards: ['As', 'Ah'] },
      recent_decisions: [
        {
          street: 'flop',
          chosen_action: 'call',
          total_response_seconds: 1.25,
          private_strategy: [0.1, 0.9],
        },
      ],
    })
    const serialized = JSON.stringify(report)
    expect(serialized).not.toContain('model.pt')
    expect(serialized).not.toContain('private_strategy')
    expect(serialized).not.toContain('As')
    expect(report.metadata.gpuName).toBe('RTX 4090')
  })
})
