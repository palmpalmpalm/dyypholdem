#!/usr/bin/env python3
"""Validate every recovered compact value net on CUDA against a CPU reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


PROCESS_START = time.perf_counter()
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
os.chdir(SOURCE_DIR)
sys.path.insert(0, str(SOURCE_DIR))

import torch  # noqa: E402

from nn.compact_value_net import load_compact_checkpoint  # noqa: E402


MODELS = (
    ("preflop-aux", 1, True),
    ("flop", 1, False),
    ("turn", 2, False),
    ("river", 3, False),
)


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def deterministic_inputs(input_size: int, output_size: int) -> torch.Tensor:
    buckets = output_size // 2
    inputs = torch.zeros(2, input_size, dtype=torch.float32)
    inputs[0, :buckets] = 1.0 / buckets
    inputs[0, buckets:output_size] = 1.0 / buckets
    inputs[0, -1] = 0.5
    ramp = torch.arange(1, buckets + 1, dtype=torch.float32)
    ramp /= ramp.sum()
    inputs[1, :buckets] = ramp
    inputs[1, buckets:output_size] = torch.flip(ramp, dims=(0,))
    inputs[1, -1] = 0.9
    return inputs


def validate_model(
    model_root: Path,
    folder: str,
    current_street: int,
    auxiliary: bool,
    repeats: int,
) -> dict:
    checkpoint = model_root / folder / "final_compact.pt"
    cpu_load_started = time.perf_counter()
    cpu_model = load_compact_checkpoint(_safe_torch_load(checkpoint))
    cpu_load_seconds = time.perf_counter() - cpu_load_started
    inputs = deterministic_inputs(cpu_model.input_size, cpu_model.output_size)
    with torch.inference_mode():
        cpu_output = cpu_model(inputs).detach().cpu().contiguous()

    import settings.arguments as arguments
    from nn.value_nn import ValueNn

    if arguments.device != torch.device("cuda"):
        raise RuntimeError(f"DyypHoldem configured unexpected device {arguments.device}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    gpu_load_started = time.perf_counter()
    value_nn = ValueNn().load_for_street(current_street, auxiliary)
    torch.cuda.synchronize()
    gpu_load_seconds = time.perf_counter() - gpu_load_started

    gpu_inputs = inputs.cuda()
    gpu_output = torch.empty(2, cpu_model.output_size, device="cuda")
    torch.cuda.synchronize()
    first_started = time.perf_counter()
    value_nn.get_value(gpu_inputs, gpu_output)
    torch.cuda.synchronize()
    first_inference_seconds = time.perf_counter() - first_started

    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        value_nn.get_value(gpu_inputs, gpu_output)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)

    gpu_output_cpu = gpu_output.detach().cpu().contiguous()
    torch.testing.assert_close(gpu_output_cpu, cpu_output, rtol=1e-4, atol=2e-5)
    difference = torch.abs(gpu_output_cpu - cpu_output)
    zero_sum = torch.sum(
        gpu_output_cpu * inputs[:, : cpu_model.output_size], dim=1
    ).abs()
    if not bool(torch.isfinite(gpu_output_cpu).all()):
        raise RuntimeError(f"{folder} produced non-finite CUDA output")

    return {
        "folder": folder,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "input_size": cpu_model.input_size,
        "output_size": cpu_model.output_size,
        "parameters": sum(parameter.numel() for parameter in cpu_model.parameters()),
        "cpu_load_seconds": cpu_load_seconds,
        "gpu_load_seconds": gpu_load_seconds,
        "first_inference_seconds": first_inference_seconds,
        "median_inference_seconds": statistics.median(timings),
        "min_inference_seconds": min(timings),
        "max_inference_seconds": max(timings),
        "repeats": repeats,
        "cpu_output_sha256": tensor_sha256(cpu_output),
        "gpu_output_sha256": tensor_sha256(gpu_output_cpu),
        "max_abs_cpu_gpu_difference": float(difference.max()),
        "mean_abs_cpu_gpu_difference": float(difference.mean()),
        "max_zero_sum_residual": float(zero_sum.max()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("repeats must be at least 2")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for model validation")

    model_root = args.model_root.resolve()
    os.environ["DYYPHOLDEM_COMPACT_MODEL_PATH"] = str(model_root)
    rows = [
        validate_model(model_root, folder, street, auxiliary, args.repeats)
        for folder, street, auxiliary in MODELS
    ]

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    summary = {
        "schema_version": 1,
        "benchmark": "dyypholdem-compact-model-cuda-validation",
        "source_commit": args.source_commit,
        "model_root": str(model_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": properties.total_memory,
        },
        "startup_seconds": time.perf_counter() - PROCESS_START,
        "models": rows,
        "verified": True,
        "total_checkpoint_bytes": sum(int(row["checkpoint_bytes"]) for row in rows),
        "max_abs_cpu_gpu_difference": max(
            float(row["max_abs_cpu_gpu_difference"]) for row in rows
        ),
        "max_zero_sum_residual": max(float(row["max_zero_sum_residual"]) for row in rows),
    }

    output = args.output
    if not output.is_absolute():
        output = PROJECT_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
