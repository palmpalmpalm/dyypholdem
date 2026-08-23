# Pretrained Model Recovery and Compact Conversion

Date: 2026-08-23 (Asia/Bangkok)

## Outcome

The pretrained networks can still be used without retraining, but not through
this repository's GitHub LFS storage:

- all eight CPU/GPU LFS objects return HTTP 410;
- a scan of the mounted local home directory and volumes found no materialized
  LFS payloads;
- all four original Torch7 `.model` files and their `.info` metadata remain
  public at the links in DeepHoldem issue #28;
- the files were downloaded, size/SHA-256 verified, parsed with DyypHoldem's
  Torch7 reader, and converted into native device-neutral PyTorch checkpoints.

The original downloads and generated outputs are ignored under
`runs/model-recovery/`.

## Original model evidence

| Network | Epoch | Validation loss | Model bytes | Model SHA-256 |
|---|---:|---:|---:|---|
| Preflop auxiliary | 67 | 0.001736354 | 65,805,286 | `a761dfc985cbe93d1cc2fb470fa70b90354aecf402b8f25fe36d651b41c39817` |
| Flop | 50 | 0.067476701 | 192,130,574 | `4164d7026f86efe7fc63a2f1c7e6f8eeb9de52381f40f84525923a2ed288766d` |
| Turn | 50 | 0.072930857 | 192,130,574 | `213b4ea517b57ef3a8b389c0f49f7e3222822db0ae3c9fc19561386c19d9134a` |
| River | 95 | 0.057868460 | 116,730,638 | `b8c619349de35f7427aa3dc768d80d32a7f96edd5106baf606e28fa87668e493` |

The four original model files total 566,797,072 bytes. Including their four
86-byte metadata files, the verified recovery set totals 566,797,416 bytes.

## Compact checkpoints

| Network | Parameters | Compact bytes | Reduction | Max output delta |
|---|---:|---:|---:|---:|
| Preflop auxiliary | 843,341 | 3,394,549 | 19.39x | `6.56e-7` |
| Flop | 2,507,003 | 10,049,269 | 19.12x | `1.31e-6` |
| Turn | 2,507,003 | 10,049,269 | 19.12x | `1.07e-6` |
| River | 1,514,011 | 6,077,301 | 19.21x | `1.55e-6` |

The compact set totals 29,570,388 bytes, a 19.17x reduction from the original
Torch7 models. It stores only architecture metadata, provenance, and a native
PyTorch state dictionary. It excludes legacy CUDA runtime buffers and training
workspace tensors.

Every conversion gate checks:

- exact original file size and SHA-256;
- an ordered FP32 parameter fingerprint;
- deterministic legacy versus native forward outputs;
- checkpoint reload identity;
- range-weighted zero-sum residual (measured maximum below `3e-8`).

Native operators are not bit-identical to the old Torch7-style graph because
their matrix/batch-normalization kernels accumulate in a slightly different
order. The measured maximum error is around one part in a million and is the
explicit compatibility tolerance.

## CUDA validation

The guarded model-only validation passed on source commit
`5bd2009fb71b00bdc13067bbec8a8d7f35fbee01` using a secure RTX 4090, Python
3.11.11, and PyTorch `2.8.0.dev20250319+cu128`:

- all four compact checkpoints loaded through `ValueNn.load_for_street()`;
- maximum CPU-versus-CUDA output difference: `7.7486038e-7`;
- maximum range-weighted zero-sum residual: `2.9802322e-8`;
- median batch-of-two inference latency: `0.343-0.358 ms` after warmup;
- peak CUDA allocation per loaded network: `12,973,568-19,666,944` bytes;
- total uploaded checkpoint payload: `29,570,388` bytes.

The successful ignored artifact is:

`runs/gpu-model-validation/dyypholdem-models-20260823T090842Z/summary.json`

The first guarded attempt exposed a repository-relative path bug and wrote no
summary. It was terminated automatically; the fix gained a regression test
before the successful rerun. The successful runner terminated its throwaway pod
and both its own exact-name check and an independent account audit found zero
remaining DyypHoldem pods. Provider billing remains authoritative; the two
short attempts imply roughly five cents at the observed `$0.74/hour` rate.

This proves checkpoint recovery, native conversion, runtime mapping, and CUDA
inference parity. It does not yet prove full flop/turn/preflop resolving parity
or playing strength; those require the next street-resolve benchmark suite.

## Commands

```shell
make model-recovery-progress
make recover-models
make compact-models
make compact-model-progress
make test
make gpu-model-validation-dry-run
make gpu-model-validation
```

To select compact models at runtime:

```shell
export DYYPHOLDEM_COMPACT_MODEL_PATH="$PWD/runs/model-recovery/compact"
```

`ValueNn` maps that root to `preflop-aux`, `flop`, `turn`, and `river`, loads
the restricted weights-only payload when supported by PyTorch, moves the model
to the configured runtime device, and executes inference with autograd disabled.
