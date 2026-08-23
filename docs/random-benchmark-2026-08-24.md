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

## Verified RTX 4090 result

The completed run `dyypholdem-ui-20260823T205907Z` used source commit
`5bbc2dbd18ec554d124d3cd86453f34c15279ee3`, dealer/opponent/bot seed
`20260824`, and the recovered compact value networks. Exact completion
validation passed with 100 bot results, 100 opponent results, hand IDs 0–99,
230 timed bot decisions, and 208 opponent actions. Bot and opponent winnings
were exactly zero-sum at +20,350 and -20,350 chips. All stale-state conflict,
rate-limit backoff, request-retry, and error counters were zero.

The full match took 1,085.603 seconds (18m05.603s), excluding pod setup and
shutdown. Timings below are CUDA-synchronized total bot decision latency:

| Street | Decisions | Mean | p50 | p95 | Max | Resolve mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preflop | 100 | 1.940s | 0.461s | 4.586s | 4.619s | 1.904s |
| Flop | 61 | 9.505s | 10.565s | 11.476s | 15.853s | 5.363s |
| Turn | 39 | 5.037s | 4.934s | 6.089s | 11.054s | 4.898s |
| River | 30 | 2.766s | 2.876s | 3.878s | 3.920s | 2.699s |

Preflop is intentionally bimodal: 50 cached-root decisions averaged 0.0189s,
while 50 fresh resolves averaged 3.860s. The one-time root precompute took
11.221s, including 6.102s to build the lookahead and 4.921s for CFR.

Two deterministic edge cases were found and fixed before accepting the final
result. A preflop limp followed by an opponent all-in previously mislabeled a
call as a nonterminal check and crashed the lookahead; a called all-in before
the river previously skipped the bot's showdown result after the dealer ran
out the board. Regression coverage now includes both states plus ordinary
street transitions, uncalled shoves, and normal river completion. The final
backend suite passed 61 tests with five optional-dependency skips.

The successful pod's conservative acquisition-to-verified-deletion lifetime
was 1,410.579 seconds and cost about $0.290 at $0.74/hour. Including the two
diagnostic runs that exposed the edge cases, total authorized RTX 4090 time was
2,972.258 seconds (49m32.258s), about $0.611. Final copyback succeeded and the
provider API verified that the benchmark pod was deleted.
