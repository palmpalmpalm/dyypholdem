#!/usr/bin/env python3
"""Convert recovered Torch7 value nets into compact device-neutral checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
TORCH7_DIR = SOURCE_DIR / "torch7"
DEFAULT_SOURCE_ROOT = PROJECT_DIR / "runs" / "model-recovery" / "original"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "runs" / "model-recovery" / "compact"

sys.path[:0] = [str(SOURCE_DIR), str(TORCH7_DIR), str(PROJECT_DIR / "scripts")]

import settings.arguments as arguments  # noqa: E402

# The Torch7 reader chooses its allocation device through this legacy module.
# Parsing on CPU is both portable and sufficient because the compact payload is
# device-neutral and the runtime loader moves it to the configured device.
arguments.use_gpu = False
arguments.Tensor = torch.FloatTensor
arguments.LongTensor = torch.LongTensor
arguments.device = torch.device("cpu")

import torch7_file  # noqa: E402
from nn.compact_value_net import (  # noqa: E402
    checkpoint_payload,
    from_legacy_model,
    load_compact_checkpoint,
)
from recover_models import MODEL_ASSETS, SOURCE_ISSUE, file_sha256, is_verified  # noqa: E402


STREETS = {
    "preflop": {"street_id": 1, "folder": "preflop-aux"},
    "flop": {"street_id": 2, "folder": "flop"},
    "turn": {"street_id": 3, "folder": "turn"},
    "river": {"street_id": 4, "folder": "river"},
}

EXPECTED_PARAMETER_SHA256 = {
    "preflop": "09f78c1e71eac5ca45e0aaad3e61c78c99ba750aeb8fe356583b264b7b1d03a0",
    "flop": "e0b77b64c6b726afb8650cf7e64a738207999b81511fbbef59da786ffba36328",
    "turn": "565a97bebd2548b13216175394b347752fe380c601bacc541c67a401e75f8e92",
    "river": "36cbc6d5fb81c4c6b3c0d360c698ffeaf4a188beb3218c37568f26b8eaa9093c",
}


def _source_asset(street: str, kind: str):
    return next(
        asset for asset in MODEL_ASSETS if asset.street == street and asset.kind == kind
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def parameter_sha256(model) -> str:
    digest = hashlib.sha256()
    parameters = model.parameters()
    if parameters is None:
        raise ValueError("legacy model has no parameters")
    for tensor in parameters[0]:
        value = tensor.detach().cpu().contiguous()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def convert_street(street: str, source_root: Path, output_root: Path) -> dict:
    started = time.perf_counter()
    model_asset = _source_asset(street, "model")
    info_asset = _source_asset(street, "info")
    model_path = source_root / model_asset.filename
    info_path = source_root / info_asset.filename
    if not is_verified(model_path, model_asset) or not is_verified(info_path, info_asset):
        raise RuntimeError(
            f"unverified {street} source; run scripts/recover_models.py first"
        )

    info = torch7_file.read_model_from_torch7_file(str(info_path), "rb")
    legacy = torch7_file.read_model_from_torch7_file(str(model_path), "rb")
    legacy.evaluate()
    fingerprint = parameter_sha256(legacy)
    if fingerprint != EXPECTED_PARAMETER_SHA256[street]:
        raise RuntimeError(
            f"unexpected {street} parameter SHA-256: {fingerprint}"
        )

    compact = from_legacy_model(legacy)
    inputs = deterministic_inputs(compact.input_size, compact.output_size)
    with torch.inference_mode():
        legacy_output = legacy.forward(inputs).detach().cpu().contiguous()
        compact_output = compact(inputs).detach().cpu().contiguous()

    difference = torch.abs(legacy_output - compact_output)
    max_abs_difference = float(difference.max())
    mean_abs_difference = float(difference.mean())
    torch.testing.assert_close(
        compact_output,
        legacy_output,
        rtol=1e-5,
        atol=2e-6,
    )

    config = STREETS[street]
    destination = output_root / str(config["folder"]) / "final_compact.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_info = {
        "street": int(config["street_id"]),
        "epoch": int(info["epoch"]),
        "valid_loss": float(info["valid_loss"]),
        "device": "cpu",
        "datatype": "float32",
    }
    source = {
        "issue": SOURCE_ISSUE,
        "model_filename": model_asset.filename,
        "model_size": model_asset.size,
        "model_sha256": model_asset.sha256,
        "info_filename": info_asset.filename,
        "info_size": info_asset.size,
        "info_sha256": info_asset.sha256,
        "parameter_sha256": fingerprint,
    }
    payload = checkpoint_payload(compact, model_info, source)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)

    reloaded = load_compact_checkpoint(_safe_torch_load(destination))
    with torch.inference_mode():
        reloaded_output = reloaded(inputs).detach().cpu().contiguous()
    if not torch.equal(compact_output, reloaded_output):
        raise RuntimeError(f"{street} compact checkpoint changed after reload")

    ranges = inputs[:, : compact.output_size]
    zero_sum_residual = torch.sum(reloaded_output * ranges, dim=1).abs()
    return {
        "street": street,
        "source_model": str(model_path),
        "source_model_bytes": model_asset.size,
        "source_model_sha256": model_asset.sha256,
        "parameter_sha256": fingerprint,
        "checkpoint": str(destination),
        "checkpoint_bytes": destination.stat().st_size,
        "checkpoint_sha256": file_sha256(destination),
        "size_reduction_ratio": round(model_asset.size / destination.stat().st_size, 3),
        "input_size": compact.input_size,
        "output_size": compact.output_size,
        "parameters": sum(parameter.numel() for parameter in compact.parameters()),
        "legacy_output_sha256": tensor_sha256(legacy_output),
        "compact_output_sha256": tensor_sha256(compact_output),
        "max_abs_output_difference": max_abs_difference,
        "mean_abs_output_difference": mean_abs_difference,
        "max_zero_sum_residual": float(zero_sum_residual.max()),
        "epoch": model_info["epoch"],
        "valid_loss": model_info["valid_loss"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def progress_report(output_root: Path, streets) -> dict:
    rows = []
    for street in streets:
        destination = output_root / str(STREETS[street]["folder"]) / "final_compact.pt"
        row = {
            "street": street,
            "checkpoint": str(destination),
            "present": destination.is_file(),
            "bytes": destination.stat().st_size if destination.is_file() else 0,
        }
        if destination.is_file():
            try:
                model = load_compact_checkpoint(_safe_torch_load(destination))
                row.update(
                    {
                        "verified": True,
                        "input_size": model.input_size,
                        "output_size": model.output_size,
                    }
                )
            except Exception as exc:
                row.update({"verified": False, "error": str(exc)})
        else:
            row["verified"] = False
        rows.append(row)
    return {
        "output_root": str(output_root),
        "checkpoints": rows,
        "verified": bool(rows) and all(row["verified"] for row in rows),
        "checkpoint_bytes": sum(int(row["bytes"]) for row in rows),
    }


def _write_manifest(output_root: Path, report: Mapping[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        **dict(report),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = output_root / ".manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_root / "manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--street",
        action="append",
        choices=tuple(STREETS),
        default=None,
    )
    parser.add_argument("--progress-report", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(1)
    streets = tuple(args.street or STREETS)
    if args.progress_report:
        print(json.dumps(progress_report(args.output_root, streets), indent=2, sort_keys=True))
        return

    rows = [convert_street(street, args.source_root, args.output_root) for street in streets]
    report = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "models": rows,
        "verified": True,
        "source_model_bytes": sum(int(row["source_model_bytes"]) for row in rows),
        "checkpoint_bytes": sum(int(row["checkpoint_bytes"]) for row in rows),
    }
    report["overall_size_reduction_ratio"] = round(
        report["source_model_bytes"] / report["checkpoint_bytes"], 3
    )
    _write_manifest(args.output_root, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
