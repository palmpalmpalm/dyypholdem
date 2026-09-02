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

Pending: no LBR match has been run yet.
