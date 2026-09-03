# Cheaper strength measurement: what Slumbot's baseline is, and is not

Date: 2026-09-03 (Asia/Bangkok)

## Why this matters

Strength against Slumbot is measured in chips, and chips are noisy: the bot's
per-hand standard deviation is about 1,421 chips (14.2 big blinds), so a 95%
interval of ±100 mbb/hand needs roughly 78,000 hands. At 2.3 s/hand that is
about 50 GPU-hours. Anything that lowers the variance of the score lowers that
bill proportionally, which is why variance reduction is worth studying before
buying hands.

## What the API actually returns

Probing `https://slumbot.com/api` showed the hand-end response carries more
than the client ever read:

```json
{"action": "b200c/kb200c/kb400c/kk", "winnings": 800, "won_pot": 1600,
 "bot_hole_cards": ["7d", "6h"], "baseline_winnings": 1000,
 "session_total": 800, "session_baseline_total": 1000, "session_num_hands": 1}
```

Both `baseline_winnings` and `bot_hole_cards` are present on **every** hand,
including hands that end in a fold with no showdown. Every run before
2026-09-03 discarded both.

`bot_hole_cards` is the more important of the two in the long run: knowing the
opponent's actual hand is what makes a full AIVAT estimator implementable.

## The baseline is not a variance-reduced score

The obvious hope was that `baseline_winnings` is a lower-variance estimate of
the same quantity as `winnings`, so it could simply replace it. It is not.
`scripts/slumbot_baseline_probe.py` played 1,000 hands with a fixed call-down
policy and no solver, so the measurement cost nothing:

| Quantity | Value |
|---|---:|
| Hands | 1,000 |
| Raw result | −1,017 mbb/hand, 95% CI ±733 (sd 1,182 chips) |
| Baseline result | −644 mbb/hand, 95% CI ±1,130 (sd 1,824 chips) |
| Raw minus baseline | −373 mbb/hand, 95% CI ±1,079 |
| Correlation | 0.393 |

Two conclusions:

* **As a replacement score the baseline is 2.4x worse.** Its standard deviation
  is 1,824 chips against the raw 1,182, so using it would multiply the hands
  needed by 2.4, not divide them.
* **As a control variate it is also worse, for now.** The useful form is
  `raw − baseline`, whose variance is
  `var_raw + var_baseline − 2ρ·sd_raw·sd_baseline`. That only beats the raw
  score when `ρ > sd_baseline / (2·sd_raw)`, here 0.771. The measured
  correlation is 0.393, giving `raw − baseline` a standard deviation of 1,741
  chips: 2.2x worse again.

The paired difference of −373 ±1,079 mbb/hand does contain zero, which is
consistent with the baseline being an unbiased estimate of the same quantity.
Nothing here suggests the baseline is wrong; it is simply noisier.

## What this does not settle

The probe played a call-down policy, which is nothing like the reference
strategy the baseline is presumably built from, and low correlation is exactly
what a mismatched strategy would produce. The threshold is not far away: a
correlation above 0.771 would flip the answer and make `raw − baseline` a
genuine saving. A real bot playing a real strategy could plausibly clear it.

That question is now answered for free. `src/player/slumbot_match.py` records
the baseline for every hand and the run summary reports `baseline_statistics`
and `baseline_comparison`, including the correlation and both variance ratios,
so the next GPU run measures this for the actual solver at no extra cost. Until
one has, the raw chip count stays the score.

## If the baseline stays useless

A full AIVAT estimator remains available and is now unblocked by
`bot_hole_cards`. The parts that need only our own strategy and the public
chance events -- corrections at the deal of each street, and imaginary
observations over our own private cards -- are zero-mean by construction
regardless of how good the value estimator is, so they cannot bias the result;
the value network's quality only decides how much variance they remove. The
part that needs the opponent's strategy is unavailable and would simply be
omitted. The blocking work is telemetry: the client currently logs the strategy
only for the hand actually held, not the full range, so an AIVAT pass needs
wider logging before it has anything to consume.
