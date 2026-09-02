# Local Best Response Harness

Date: 2026-09-02 (Asia/Bangkok)

## Why

Slumbot gives an external number, but at 15.9 big blinds of per-hand standard
deviation it needs about 100,000 hands for ±100 mbb/hand and offers no
variance reduction. A local best response (LBR, Lisý and Bowling 2017) is the
cheapest lower bound on exploitability that runs entirely on our own hardware
and against our own dealer, so it can be repeated on every solver change.

## What LBR knows and assumes

DyypHoldem publishes, at every decision, its full root strategy over all
1,326 private hands together with the action menu it chose from
(`--strategy-channel`, written before the action reaches the dealer). The LBR
opponent multiplies its belief over the bot's hand by the probability of the
action the bot actually took, so the belief is exact given the bot's own
strategy, and masks hands that collide with the board or LBR's cards.

At its own decisions LBR values each candidate under the call-down
assumption: after this action both players check or call to showdown.

| Action | Value in future chips |
|---|---|
| fold | 0 |
| check / call | (T / 2)(1 + eq) − C, with T the pot after the call and C the amount to call |
| raise to R | R·eq + own commitment, the bot assumed to call |

`eq` is the range-conditional win-minus-loss expectation of LBR's hand
against the bot's belief on the current board, taken from DyypHoldem's own
terminal-equity matrices (runouts averaged on the flop and turn, checked
against known hands: KK on a K-high river is +1.00, 4-5 offsuit is −0.99).
LBR plays the argmax and folds only when nothing beats zero; a free check is
never turned into a fold. The raise menu defaults to pot and all-in
(`DYYPHOLDEM_LBR_RAISE_MENU`, from `half_pot,pot,double_pot,all_in`, empty
for fold/call only).

Call-down is a heuristic: it never credits a bet with fold equity and never
charges it for the bot folding weak hands and continuing with strong ones. The
result is therefore a lower bound on how exploitable the bot is, not a best
response, and it is most informative when it is large.

## Running it

```shell
make test
make lbr-benchmark-dry-run
make lbr-benchmark            # 1,000 hands, 6 h guard, $5.00 cap
make play-ui-status
make play-ui-logs
```

The guarded controller treats LBR like a headless single session: the pod runs
the ACPC dealer with LBR in the `LBR` seat on port 18901 and DyypHoldem on
18902, both artifacts under `runs/play-ui/<run>/session-0/`
(`decisions.jsonl`, `timing_report.*`, `bot-strategy.jsonl`, `lbr-events.jsonl`,
`lbr-summary.json`). Completion requires the bot's telemetry and the LBR summary
to agree on the hand count and to be exactly zero-sum
(`scripts/validate_lbr_benchmark.py`). The strategy channel grows by roughly
50 KB per bot decision. The report:

```shell
python3 scripts/slumbot_run_report.py --run-dir runs/play-ui/<run> \
  --summary-name lbr-summary.json --events-name lbr-events.jsonl
```

## Results

Two guarded attempts on 2026-09-02 validated the harness end to end but did
not complete a measurement.

- `dyypholdem-lbr-20260902T105614Z` played 35 hands before both ACPC clients
  died on a pre-existing bug: a called flop all-in makes the dealer deal the
  turn and river with empty action lists, and the debug repr of the parsed
  state indexed the previous street's empty list. Slumbot hands never reach
  that state, an LBR that shoves finds it within 35 hands. Fixed with a
  regression test on the exact message.
- `dyypholdem-lbr-20260902T111054Z` played 13 clean hands with the fix and
  was then stopped by RunPod because the account balance reached zero; the
  provider also stopped the other lane's pod. The next launch after a top-up
  is `DYYPHOLDEM_UI_GRAPH_GATE=1 make lbr-benchmark`.

### `dyypholdem-lbr-20260902T135954Z`: 1,000 hands, pot and all-in menu

The first complete match. All 1,000 hands validated, zero-sum against the
bot's telemetry, no channel waits, 7.8 s/hand.

| Metric | Value |
|---|---:|
| LBR result | −326 mbb/hand, 95% CI ±2,292 |
| Per-hand standard deviation | 3,698 chips (37 big blinds) |
| Hands won / lost / tied by LBR | 708 / 278 / 14 |
| LBR as small blind (chips) | 500 hands, +78,600 |
| LBR as big blind (chips) | 500 hands, −111,150 |

**The measurement is inconclusive, and the reason is the raise formula.** LBR
lost, but a negative LBR result never means the bot is unexploitable; it means
this LBR variant is a poor exploiter. Splitting the hands shows where the
result comes from:

| Hands | Count | Net chips |
|---|---:|---:|
| LBR shoved at some point | 668 | −32,500 |
| LBR never shoved | 332 | −50 |

Every chip LBR lost, it lost in hands where it shoved, and it shoved with a
median equity of +0.18 with 54% of shoves below +0.20. That follows directly
from the call-down value of a raise, `R·eq + own commitment`: with `R` the
20,000 stack, any positive equity produces a huge number, so all-in dominates
calling on thin edges. The formula credits no fold equity and, more
importantly, charges nothing for the fact that the bot folds its worst hands
and calls a shove with a range far stronger than the +0.18 average LBR
measured against its whole range.

The proper LBR raise value (Lisý and Bowling) weights the opponent's actual
fold probability at that node and recomputes equity against the range that
continues. Both quantities require querying the agent at a counterfactual
node, which the strategy channel cannot supply: it only carries decisions the
bot actually faced. Supporting that would mean running a second resolver as an
oracle, one full resolve per candidate action.

The cheap and standard alternative is to restrict the action set. With
fold and call only, the bound is weaker in principle but the variance
collapses, since no hand builds a 20,000-chip pot. The 332 non-shoving hands
above already hint at the answer, being within 50 chips of break-even. The
next run uses `DYYPHOLDEM_LBR_RAISE_MENU=` (empty).

### `dyypholdem-lbr-20260902T161854Z`: fold or call only, stopped at 206 hands

Run with `DYYPHOLDEM_LBR_RAISE_MENU=none` to test the prediction that removing
raises would collapse the variance. It did not, and the run was stopped early
rather than pay for a foregone result.

| Metric | Pot and all-in menu | Fold or call only |
|---|---:|---:|
| Per-hand standard deviation | 3,698 chips | 2,939 chips |
| Projected 95% CI at 1,000 hands | ±2,292 | ±1,822 |
| LBR fold rate | 13% | 8% |

Only 1.3x lower, because the variance was never LBR's raises. Five hands of
206 (2%) hold −31,900 chips while the other 201 total −3,200. In those five
LBR called bets of up to 19,700 chips at a median equity of +0.315. The
call-down value of a call is correct in isolation, but the assumption behind
it, that the hand checks down after this decision, is exactly what an
aggressive opponent violates: the bot keeps betting on later streets and LBR,
folding only 8% of the time, calls all the way with a marginal holding.

**Conclusion: this LBR cannot bound the bot's exploitability in either
configuration.** With raises it shoves on thin equity; without them it is a
calling station. Both failures come from the same root, the call-down
assumption, which prices an action as if the hand ended there. A negative
result from a broken exploiter says nothing about the bot.

### What a working LBR needs

The published method values a raise using the opponent's actual probability of
folding at that node, then recomputes equity against the range that continues.
Both numbers require asking the agent what it would do at a node it did not
reach. The strategy channel cannot answer that; it only carries decisions the
bot actually faced. The fix is an oracle process: a second continual resolver
that LBR queries per candidate action, which costs one full resolve per
question. With graph replay a river resolve is 0.25 s and a flop resolve
1.3 s, so a three-action probe costs a few seconds per LBR decision. That is
affordable now and was not before, but it is a real build rather than a
parameter change.

Until then the honest instruments are the deterministic solver-regression gate
for solution quality, which is free and sensitive, and Slumbot for external
strength, which needs about 100,000 hands (roughly $35 of GPU at the current
1.7 s/hand) to resolve differences smaller than a few hundred mbb/hand.

Across the earlier 48 validation hands the channel bookkeeping never waited or mismatched, the
bot's telemetry and the LBR summary were exactly zero-sum, and LBR chose
all-in in 36 of 104 decisions, call in 55, fold in 13. LBR was ahead by
3,200 chips, which at that sample size says nothing beyond "the bot folds the
blinds to most shoves", as call-down LBR is designed to probe. Every bot
decision in those hands replayed CUDA Graphs after the on-pod gate passed on
the default tree (its third bitwise pass of the day).
