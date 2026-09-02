# DyypHoldem versus Slumbot — guarded 1,000-hand benchmark

Date: 2026-09-02 (Asia/Bangkok)

## Why

Every DyypHoldem result before this run measured runtime, determinism, or
end-to-end correctness. None measured playing strength. Slumbot is a public,
always-available heads-up no-limit bot with an HTTP API that plays the same
game DyypHoldem is configured for (20,000-chip stacks, 50/100 blinds), so it
is the cheapest external strength signal available. This run validates the
client end to end on real hardware and produces the first chip result with an
honest confidence interval.

## Protocol adapter

`src/server/slumbot_game.py` keeps Slumbot's reference action parser and
derives both conversion directions from it instead of tracking chip
commitments locally:

- Slumbot's street-local `b<amount>` levels become cumulative ACPC `r<total>`
  amounts by adding the matched commitment of earlier streets; `k` becomes
  ACPC `c`.
- A DyypHoldem raise to cumulative `R` becomes `b(R - previous_streets)` where
  `previous_streets = total_last_bet_to - street_last_bet_to` from the parsed
  Slumbot state. Check versus call uses `last_bet_size`, so a big-blind check
  behind a limp is sent as `k`, not `c` (the original client sent `c`, which
  Slumbot's parser rejects).
- Bets below Slumbot's minimum raise are lifted to it and bets above the stack
  are capped to all-in. Both are counted and logged as corrections; the tree's
  own bet sizing already respects the same rules, so the count is expected to
  stay at zero.
- Slumbot `client_pos` 1 is the small blind and acts first preflop, matching
  ACPC player 1 in the bundled reverse-blinds game definition, so the position
  is copied verbatim into `MATCHSTATE:<pos>:<hand>:<actions>:<cards>`.
- Every reconstructed state must have DyypHoldem to act, and the Slumbot
  parser's next-to-act position must equal the client seat; anything else
  raises a protocol error rather than solving from the wrong seat.
- Transport errors and HTTP 5xx retry with exponential backoff; server
  rejections (`error_msg`, HTTP 4xx) never retry.

`src/player/slumbot_match.py` runs the fixed-length match. A failed hand is
recorded, the next hand starts, and the match aborts after three consecutive
failures or more than ten in total. It writes the same private
`decisions.jsonl` and safe `timing_report.{json,txt}` as the ACPC player plus
`slumbot-events.jsonl` (no hole cards) and `slumbot-summary.json` with chip
totals, milli-big-blind per hand, standard error, and a 95% interval.

## Guarded launch

The Slumbot mode reuses the live-UI pod controller so it inherits the spend
cap, absolute deadline, remote self-delete guard, local watchdog, periodic
copyback, and verified exact-name deletion. Differences: no dealer, no web
bundle, no public port; readiness is the first hand attempt in the copied-back
summary; completion is the summary reaching `complete` plus the remote
validator. The guard ceiling was raised from four to six hours because a
1,000-hand match at roughly 12-15 seconds per hand needs about four hours plus
setup.

```shell
make test
make slumbot-benchmark-dry-run
make slumbot-benchmark            # 1,000 hands, 6 h guard, $5.00 cap
make play-ui-status
make play-ui-logs
make play-ui-stop                 # only if you need to end it early
```

`SLUMBOT_HANDS`, `SLUMBOT_SEED`, `SLUMBOT_GUARD_SECONDS`, and
`SLUMBOT_MAX_TOTAL_COST_USD` override the defaults.

Local smoke test without renting (CPU, production iteration counts):

```shell
cd src
DYYPHOLDEM_DEVICE=cpu \
DYYPHOLDEM_COMPACT_MODEL_PATH="$PWD/../runs/model-recovery/compact" \
python3 player/dyypholdem_slumbot_player.py 2 --seed 20260902 \
  --telemetry /tmp/slumbot-smoke/decisions.jsonl \
  --events /tmp/slumbot-smoke/slumbot-events.jsonl \
  --summary /tmp/slumbot-smoke/slumbot-summary.json
```

`DYYPHOLDEM_DEVICE` defaults to `cuda`; the only other accepted value is
`cpu`.

## Reading the result

Slumbot reports winnings from the client's perspective in chips. With a
100-chip big blind, milli-big-blinds per hand is chips per hand times ten.
One thousand hands is enough to prove the client and get a rough sign, not to
rank the bot. The run below measured a per-hand standard deviation of 15.9 big
blinds, so the 95% interval on 1,000 hands is roughly plus or minus 1,000
mbb/hand. Treat anything inside that band as "not distinguishable from
break-even" and plan a much longer, parallel run before drawing conclusions.

## Results

Run `dyypholdem-slumbot-20260901T202522Z` (ignored under `runs/play-ui/`),
Secure RTX 4090 at $0.74/hour, PyTorch 2.8 with CUDA 12.8, the four recovered
compact networks, 1,000 CFR iterations with 500 skipped, bot seed `20260902`,
working tree on top of commit `d975f8b` plus the uncommitted Slumbot changes
described above. The pod-side asset checksum, CUDA model validation, and the
strict preflop/chance regression capture (three bit-identical repeats) all
passed before the first hand.

| Metric | Value |
|---|---:|
| Hands completed / requested | 1,000 / 1,000 |
| Bot decisions (per hand) | 2,898 (2.90) |
| Net chips | -18,350 |
| Result | -183.5 mbb/hand, standard error 503.5, 95% CI ±986.9 |
| Hands won / lost / tied | 520 / 471 / 9 |
| Small blind hands (chips) | 500 (-5,450) |
| Big blind hands (chips) | 500 (-12,900) |
| Per-hand standard deviation | 1,592 chips (15.9 big blinds) |
| Actions | 990 checks, 845 calls, 686 raises, 368 folds, 9 all-ins |
| Hand errors / request retries / bet-size corrections | 0 / 0 / 0 |
| Match wall time | 11,910 s (3h18m, 11.9 s/hand) |

Per-street CUDA-synchronized decision latency:

| Street | Decisions | Response mean | p95 | Max | CFR mean |
|---|---:|---:|---:|---:|---:|
| preflop | 1,087 | 2.197 s | 4.305 s | 4.522 s | 2.138 s |
| flop | 829 | 5.524 s | 6.603 s | 11.606 s | 4.099 s |
| turn | 547 | 4.767 s | 5.695 s | 11.316 s | 3.924 s |
| river | 435 | 2.619 s | 3.420 s | 3.603 s | 2.525 s |

Preflop splits into 500 cached-root decisions at 0.032 s and 587 fresh
resolves at 4.04 s. All 636 flop arrivals used captured trajectories (no
legacy replay). The postflop transform cache hit 280 of 1,375 eligible
decisions (20.4%). `scripts/slumbot_run_report.py --run-dir <run>` reproduces
the tables from the artifacts.

### Interpretation

The harness is validated: 2,898 decisions across all four streets and both
seats were accepted by Slumbot with no protocol errors, no transport retries,
and no bet-size corrections, and the chip totals reconcile between the
private telemetry and the safe summary.

The chip result is not a strength verdict. The measured per-hand standard
deviation of 15.9 big blinds is far larger than the "several big blinds"
assumed before the run, so the 95% interval on 1,000 hands is roughly ±1,000
mbb/hand and comfortably contains zero. DyypHoldem won more hands than it lost
but lost chips, and lost more from the big blind than the small blind, which is
consistent with giving up too much in large pots but is not established by
this sample. Reaching ±100 mbb/hand at 95% against Slumbot's API needs on the
order of 100,000 hands, because the API offers no duplicate or variance-reduced
matches. At the measured 11.9 s/hand that is about 330 GPU-hours, roughly $245
at the observed rate, before any solver speedup. That cost is the strongest
argument yet for the CFR-loop optimizations and for a local best-response
harness that can run far more hands per dollar.

### Cost and lifecycle

Pod lifetime from acquisition to verified deletion was about 4.95 hours
(about $3.67 at $0.74/hour; provider billing is authoritative), of which 96
minutes were setup. Almost all of that setup was the controller's rsync
uploading roughly 700 MB of tracked bucket tables and hand ranks from the
local checkout over the home uplink; the pod-side downloader then found every
asset already present and only checksummed it. The code sync now excludes
`*.pt`, `*.pkl`, and `*.sqlite`, so the pod downloads those assets itself as
it did in earlier runs, and a launcher test asserts that every play asset in
`scripts/materialize_assets.py` matches an excluded pattern. One periodic
telemetry copyback timed out mid-run and succeeded on the next 12-second
cycle. The final validator, copyback, exact-name stop/delete, and six absence
checks all passed, and an independent pod listing afterwards showed no
DyypHoldem pods.
