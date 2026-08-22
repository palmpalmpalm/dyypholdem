# DyypHoldem GPU River Baseline

Date: 2026-08-23 (Asia/Bangkok)

## Outcome

DyypHoldem runs successfully on a modern PyTorch/CUDA stack. The first
evidence-driven optimization changes bucket lookup tables from eager global
loading to lazy, street-specific loading.

The optimized river run:

- completed two 1,000-iteration resolves on a secure RTX 4090;
- measured 0.597 and 0.517 seconds per resolve on that host;
- allocated about 298-299 MB of peak CUDA memory during solving;
- produced bit-identical repeats;
- matched the original root-CFV and strategy SHA-256 hashes exactly.

The solve timings are a useful optimized-host baseline, not a causal speedup
claim. The legacy and optimized runs landed on different RTX 4090 hosts. The
causal result from this patch is the removal of unrelated table loading.

## Matched output contract

Both implementations produced:

- root CFVs:
  `820d911cad2f15416b19b02e0e61de4b5740d8d6eeffaf15c1746d58361c5ca6`
- strategy:
  `16529973d061d3c594a067fa251af7a44569377958b5f8255c748ab889610346`

The optimized benchmark enforces these hashes and fails if either changes.

## Startup finding

The original module import eagerly loaded every postflop bucket table even for
a river-only solve:

| Component | Legacy time |
|---|---:|
| Flop categories | 0.268 s |
| Turn categories | 5.438 s |
| River categories | 6.925 s |
| All bucket initialization | 12.632 s |
| Total imports | 14.942 s |

After lazy loading, total imports measured 1.028 seconds on the optimized
host. That is a 93.1% reduction (14.5x), with the caveat that host-level I/O
also contributes to absolute timing.

The river deployment asset requirement fell from seven files / 589,692,151
bytes to three files / 273,971,073 bytes, a 53.5% reduction. River play no
longer requires the 10 MB flop table, 117 MB turn table, or 189 MB river
category table merely to import the solver.

## Reproduction

```shell
make test
make gpu-baseline-dry-run
DYYPHOLDEM_GPU_CLOUD_TYPE=SECURE \
DYYPHOLDEM_GPU_TYPE='NVIDIA GeForce RTX 4090' \
make gpu-baseline
```

The successful ignored artifact is under:

`runs/gpu-baseline/dyypholdem-river-20260822T171215Z/`

Its environment was Python 3.11.11, PyTorch 2.8.0 development build with CUDA
12.8, and an RTX 4090 with 24 GB VRAM.

## Rental safety record

- Two community RTX 3090 attempts were rejected: one never exposed SSH and
  one exposed the GPU but could not initialize CUDA in PyTorch.
- Both were stopped and terminated immediately after diagnosis.
- The secure RTX 4090 runs used a one-hour remote self-stop guard plus a local
  cleanup trap.
- Every DyypHoldem-owned pod was terminated and verified absent after its run.
- The separate Supremus distribution-screen pod was not touched.

Provider billing is authoritative. Observed rates were $0.22/hour for the
community 3090 attempts and $0.74/hour for the secure 4090 runs.
