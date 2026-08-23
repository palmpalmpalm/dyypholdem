# DyypHoldem versus seeded random — 100-hand benchmark

This benchmark uses the same guarded Secure RunPod RTX 4090 path as live play,
but a local automated client occupies the browser seat. The bundled ACPC dealer
remains authoritative for cards, turns, legal transitions, and payouts. The
web bridge validates every action with PokerKit 0.7.5 before forwarding it.

## Random policy

The opponent is deterministic for a fixed opponent seed and public state
sequence. The launch also seeds DyypHoldem's Torch/Python action sampler with
the dealer seed, so all action-sampling sources are explicitly seeded:

- facing a wager: uniformly choose fold, call, or raise;
- when checking is free: uniformly choose check or raise (no dominated free
  folds);
- after choosing raise: uniformly choose one of the deduplicated server
presets — minimum, half pot, three-quarter pot, pot, or all-in.

The client polls at 10 Hz and backs off/retries the same deterministic action
if the UI's defensive HTTP action limiter responds with 429.

The policy never invents a raise amount. The bridge supplies the allowed
presets and validates the nonce and cumulative raise-to amount; the dealer
performs the final protocol-level legality check.

## Commands

Preflight without renting:

```bash
make random-benchmark-dry-run
```

Launch the guarded 100-hand match:

```bash
DYYPHOLDEM_UI_GUARD_SECONDS=3600 \
DYYPHOLDEM_GPU_CLOUD_TYPE=SECURE \
DYYPHOLDEM_GPU_TYPE='NVIDIA GeForce RTX 4090' \
DYYPHOLDEM_UI_SEED=20260824 \
DYYPHOLDEM_UI_OPPONENT_SEED=20260824 \
make random-benchmark
```

The controller is detached. It copies artifacts back periodically, stops as
soon as the match completes, and permanently deletes the pod. Independent
local and authenticated remote deadline guards cover controller failure. The
one-hour limit includes pod startup, asset download, CUDA model validation,
and root solving; if it expires before hand 100, the run is marked incomplete
instead of being reported as a benchmark result.

## Outputs

Artifacts are copied to `runs/play-ui/<run-id>/`:

- `timing_report.json` and `timing_report.txt`: root initialization, bot hand
  count/winnings, per-street decision count and latency statistics, and a
  cached-root versus fresh-resolve preflop split;
- `decisions.jsonl`: private full solver telemetry for every bot decision;
- `random-summary.json` and `random-events.jsonl`: opponent policy, seed,
  action frequencies, completion status, and safe action events;
- `<run-id>.log`: authoritative dealer hand log;
- `bot.log`, `autoplay.log`, and dealer/UI diagnostic logs.

Final success requires both bot telemetry and the random opponent to record
exactly 100 hands. Timing percentiles are per bot decision, not per hand.
`total_response_seconds` is CUDA-synchronized solver latency and excludes
telemetry file I/O, socket transmission, and post-action garbage collection.
Root precomputation is reported separately because it is performed once and
reused on eligible first actions.

This match measures runtime and end-to-end correctness. One hundred hands
against a random policy is far too small and is not an exploitability estimate.
