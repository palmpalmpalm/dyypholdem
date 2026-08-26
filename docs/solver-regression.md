# Deterministic Solver Optimization Gate

Date: 2026-08-27 (Asia/Bangkok)

## Purpose

`scripts/solver_regression.py` is the A/B gate for solver engineering changes
and CFR-iteration experiments. It runs fixed repository public nodes and
compares complete solver outputs. CPU is the portable default; the identical
capture can run on CUDA for a short rental-GPU validation. It does not use
random-match winnings as a quality signal.

For every selected spot the capture includes:

- the exact legal action vector;
- the full root strategy tensor;
- root, both-player, achieved-opponent, and per-action child CFVs;
- range-weighted strategy, action, CFV, and root-EV comparison metrics;
- wall, public-tree, lookahead-build, CFR, and result-extraction timings;
- on preflop, full chance-action CFVs for every resolved root action on two
  fixed flop textures, including one synchronized timing per call;
- on CUDA, device/runtime metadata and peak allocated/reserved memory for
  setup, every warmup/repeat, and every preflop chance-action call;
- exact asset, compact-weight, fixture, source-tree, and tensor fingerprints.

The strict comparison policy permits only tiny floating-point drift. Legal
actions and the undefined-child-CFV mask must remain identical. A separate
timing gate can require a candidate runtime ratio, but only captures with at
least three measured repeats are accepted for that gate.

## Fast River Gate

The existing `src/tests/test_river.py` fixture is an exact terminal-street solve
and therefore needs no neural-network inference. Its three large equity assets
are still checksum-verified before imports begin.

```shell
make solver-regression-preflight
make solver-regression-river \
  SOLVER_REGRESSION_OUTPUT=runs/solver-regression/baseline-river.json
```

If the Git LFS files are only pointer text, stage the public assets under an
ignored root instead of dirtying the tracked pointer files:

```shell
python scripts/solver_regression.py stage-assets \
  --asset-root runs/solver-regression/assets \
  --spot river-7d7c8s5sQd

make solver-regression-river \
  SOLVER_REGRESSION_ASSET_ROOT=runs/solver-regression/assets
```

Missing, truncated, wrong-checksum, or LFS-pointer assets cause a nonzero exit
before the solver is imported. Non-river spots additionally require the exact
recovered compact checkpoints under `runs/model-recovery/compact`; run
`make recover-models compact-models` if those are absent.

## Source A/B Workflow

Use separate source worktrees with one shared verified asset/model root. This
prevents uncommitted candidate code from contaminating the baseline capture.

```shell
python scripts/solver_regression.py capture \
  --source-root /path/to/baseline-worktree \
  --asset-root "$PWD" \
  --model-root "$PWD/runs/model-recovery/compact" \
  --spot river-7d7c8s5sQd \
  --iterations 1000 --warmups 1 --repeats 3 --threads 1 \
  --output runs/solver-regression/baseline.json

python scripts/solver_regression.py capture \
  --source-root "$PWD" \
  --asset-root "$PWD" \
  --model-root "$PWD/runs/model-recovery/compact" \
  --spot river-7d7c8s5sQd \
  --iterations 1000 --warmups 1 --repeats 3 --threads 1 \
  --output runs/solver-regression/candidate.json

python scripts/solver_regression.py compare \
  --baseline runs/solver-regression/baseline.json \
  --candidate runs/solver-regression/candidate.json \
  --max-runtime-ratio 1.10 \
  --output runs/solver-regression/comparison.json
```

`make solver-regression-compare` provides the same strict comparison when
`SOLVER_REGRESSION_BASELINE` and `SOLVER_REGRESSION_CANDIDATE` are supplied.

## Full Public-Node Suite

Omitting `--spot` selects the repository's existing preflop, flop, turn, and
river test nodes. The harness uses their tracked range files and exact public
state metadata. It verifies only the street-specific tables and models each
spot actually requires.

```shell
python scripts/solver_regression.py preflight
python scripts/solver_regression.py capture \
  --iterations 1000 --warmups 1 --repeats 3 --threads 1 \
  --output runs/solver-regression/full-current.json
```

This remains a CPU path, but preflop/flop/turn captures are materially heavier
than the river check. Run them when an optimization touches transition boxes,
bucketing, or neural value evaluation; the river-only gate is sufficient for a
change isolated to terminal-street CFR buffers.

The preflop row also exercises the API continual resolving consumes. After the
final measured root resolve, the harness calls `get_chance_action_cfv` once for
every entry in `resolver.lookahead.action_to_index` on `2s3d4h` and `Ah7d2c`.
Those calls happen once per capture, not inside every timing repeat. Comparison
requires identical board/action/index keys, tensor shapes, and NaN masks, then
applies the normal strict CFV max-delta and range-weighted-RMSE thresholds to
the full tensors. This works with both the untouched replay implementation and
the captured-trajectory implementation.

## CUDA / RTX 4090 Validation

Use the same device for baseline and candidate. The comparator rejects a CPU
capture versus a CUDA capture unless its configuration checks are explicitly
changed, because that would mix implementation and hardware effects.

```shell
python scripts/solver_regression.py preflight \
  --device cuda --spot preflop-root

python scripts/solver_regression.py capture \
  --device cuda --spot preflop-root \
  --iterations 1000 --warmups 1 --repeats 3 --threads 1 \
  --output runs/solver-regression/candidate-4090.json

make solver-regression-river \
  SOLVER_REGRESSION_DEVICE=cuda \
  SOLVER_REGRESSION_OUTPUT=runs/solver-regression/river-4090.json
```

CUDA wall timings are synchronized before the timer starts and after the work
finishes. Each capture records the PyTorch/CUDA/cuDNN versions, GPU name,
compute capability, total VRAM, `CUBLAS_WORKSPACE_CONFIG`, and peak allocated
and reserved bytes. The harness sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` when it
is absent and CUDA has not been initialized. An incompatible existing value or
an already-initialized CUDA context without deterministic cuBLAS configuration
fails preflight and requires a fresh process.

Preflight enables strict deterministic algorithms and runs CUDA `scatter_add`,
indexing, and matrix-multiplication probes before loading the solver. Captures
keep strict mode enabled with `warn_only=False`, disable TF32, and require
bit-identical full tensors across repeats. If the installed PyTorch/CUDA build
cannot provide deterministic behavior, the command exits with an actionable
error; it never silently weakens the quality gate. A CPU-only machine can run
the unit tests and receives a clear `cuda-unavailable` preflight result.

## Iteration Sweeps

Iteration changes are rejected unless explicitly acknowledged. Capture each
iteration count, then compare it with the 1,000-iteration reference:

```shell
python scripts/solver_regression.py compare \
  --baseline runs/solver-regression/river-1000.json \
  --candidate runs/solver-regression/river-500.json \
  --allow-iteration-change \
  --output runs/solver-regression/river-500-comparison.json
```

The default quality limits remain strict even with
`--allow-iteration-change`. Any relaxed `--max-strategy-*`,
`--max-action-disagreement-*`, `--max-cfv-*`, or `--max-root-ev-delta` value
must be an explicit project decision rather than an accidental consequence of
asking for more speed.

## Verified Initial Result

The first real CPU gate compared commit `0f13fd4` with the current buffer-reuse
candidate using the same 1,000 iterations, one warmup, and three measured
repeats:

| Check | Result |
|---|---:|
| Full strategy max delta | `0` |
| Full CFV max delta | `0` |
| Range-weighted argmax action disagreements | `0 / 160` |
| Baseline median | `0.5970 s` |
| Candidate median | `0.5801 s` |
| Observed runtime ratio | `0.9718` |

The timing difference is too small for a causal speedup claim, but the
candidate passed the 1.10 no-regression timing ceiling and exact quality gate.

A 500-iteration probe took `0.2882 s` (2.07x faster) but correctly failed the
strict quality gate: two of 160 supported hands changed argmax action, the
range-weighted action disagreement was `0.0117`, strategy weighted L1 was
`0.0828`, and root EV moved by about `0.756` chips on this spot. Reducing the
iteration count therefore needs a separately justified quality budget or a
better convergence/warm-start mechanism.

The first preflop chance-boundary comparison exposed an old reset inconsistency:
fresh lookaheads initialized deeper reach ranges to zero and deeper regrets to
`regret_epsilon`, while the legacy replay reset used uniform ranges and zero
regrets at every depth. That changed replay trajectories by roughly `2e-10`
before bucketing and amplified to a maximum `4.8828e-4` chip CFV difference.
The default gate caught it; its threshold was not relaxed.

After making reset reproduce constructor state, the captured path and corrected
replay are bit-identical on all three mapped actions for both `2s3d4h` and
`Ah7d2c`: maximum absolute delta `0`, weighted RMSE `0`, and identical tensor
hashes. The six direct calls took `0.05364 s` versus `0.22578 s` for corrected
replay in the interleaved 10-iteration CPU diagnostic, a `4.21x` speedup. The
capture occupies `159,120` bytes at 10/5 iterations and exactly `15,912,000`
bytes at the production 1000/500 settings. The earlier 7.31x comparison against
the inconsistent replay is retained under ignored `runs/solver-regression/`
as diagnostic history, not as the accepted quality result.

A production-size CPU capture then ran the preflop fixture twice at 1000/500
iterations with zero repeat delta. Root solve wall samples were `3.3457 s`
(including first network/bucket initialization) and `2.1762 s`; the six direct
chance calls averaged `0.4147 s` each and all reported
`captured_flop=true`, `replayed_flop=false`. The ignored evidence file is
`runs/solver-regression/final-cpu-preflop-1000.json`. RTX 4090 parity, timing,
and peak VRAM remain the required final hardware gate.

These comparisons are deterministic regression evidence, not an exploitability
measurement. Exploitability, LBR, or another best-response evaluation remains
necessary before claiming that a quality-changing iteration policy preserves
playing strength.
