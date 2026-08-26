#!/usr/bin/env python3
"""Deterministic CPU/CUDA A/B quality gate for DyypHoldem solver changes.

The harness captures complete root strategies and CFV tensors for the tracked
public-node fixtures, then compares a candidate capture with a baseline.  It is
deliberately independent of match winnings. CPU is the portable default; CUDA
adds synchronized timings and memory telemetry when a compatible GPU exists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
BENCHMARK_NAME = "dyypholdem-solver-regression"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
ACCEPTED_CUBLAS_WORKSPACE_CONFIGS = (":4096:8", ":16:8")


# These are the same public, checksum-pinned assets used by
# ``scripts/materialize_assets.py``.  Keeping the requirements explicit lets a
# street-specific CPU run avoid downloading unrelated tables.
ASSET_SPECS = {
    "src/game/evaluation/hand_ranks.pt": {
        "drive_id": "1aDIOsaDROQBaMtpXetThSmduGY46FwNT",
        "size": 259_903_403,
        "sha256": "f896304f2dde706945978fed38069dfc9a9a06d3f2970afb702f1514f9587a68",
    },
    "src/terminal_equity/block_matrix.pt": {
        "drive_id": "1VixteYtYtdsorWc039Pyl7ZWn6Uq8lTN",
        "size": 7_033_835,
        "sha256": "d28b9561b182e43dc86901f713d60c7e94cd4a69f76bf5a27c825d5b3333e80d",
    },
    "src/terminal_equity/preflop_equity.pt": {
        "drive_id": "1oePwh3S27UM-URi8bZUqTp_lAYal4RZS",
        "size": 7_033_835,
        "sha256": "ad47a518612c5a0c92d44fbef570fb2c005cd96b76536e7ca4d420663cfba7c8",
    },
    "src/nn/bucketing/ihr_pair_to_bucket.pkl": {
        "drive_id": "19VUnYVzRzHmicGA-P1tQkoNtGdNHvULA",
        "size": 3_041,
        "sha256": "8f6df2a556c25f6e5f59417cc7a99558d4300520278844411523894698c24857",
    },
    "src/nn/bucketing/turn_dist_cats.pkl": {
        "drive_id": "1gK82FqtSIghEPnkfvPmyoQxaE5O30rzZ",
        "size": 116_507_098,
        "sha256": "4697af9bfc5e17e243557d74092326b669148fb35fd10d151cf130f7493037f7",
    },
    "src/nn/bucketing/river_ihr.pkl": {
        "drive_id": "1X6PbbT2m7Dhr--IesIDy3kyPs0mDVuT-",
        "size": 188_711_781,
        "sha256": "cbe82220f1ea5082e9f3f6daa525c2ac4df89e6043f8928f2eb134859ac50d33",
    },
    "src/nn/bucketing/preflop_buckets.pt": {
        "drive_id": "1VQnqGBDwY39oDdgJjsrAk0RuVNfShs6y",
        "size": 117_219_115,
        "sha256": "131814be7cec451cd4cdc894007db16b5c0eb83a9afc6ff7132e361ee2f4a1bc",
    },
}

COMMON_ASSETS = (
    "src/game/evaluation/hand_ranks.pt",
    "src/terminal_equity/block_matrix.pt",
    "src/terminal_equity/preflop_equity.pt",
)

STREET_ASSETS = {
    1: COMMON_ASSETS + ("src/nn/bucketing/preflop_buckets.pt",),
    2: COMMON_ASSETS + ("src/nn/bucketing/turn_dist_cats.pkl",),
    3: COMMON_ASSETS
    + (
        "src/nn/bucketing/ihr_pair_to_bucket.pkl",
        "src/nn/bucketing/river_ihr.pkl",
    ),
    4: COMMON_ASSETS,
}

# Native checkpoints recovered from the public Torch7 files.  Whole-file
# fingerprints pin the exact repository weights, not merely a compatible
# architecture.
MODEL_SPECS = {
    "preflop-aux": {
        "size": 3_394_549,
        "sha256": "2e51d058c8158e49ad43dd8c8e05b325857acb0b69f1bf448c1348d53c23485e",
    },
    "flop": {
        "size": 10_049_269,
        "sha256": "92d0192c2e87554b7e7a1bedc3fb19df9edbc2db824a3c5a5f94c7e0fc3b1094",
    },
    "turn": {
        "size": 10_049_269,
        "sha256": "034d2098333b8f1bbe8d7c44ba429579fe4d3e2aa14d7baa1d0f6a136d559293",
    },
    "river": {
        "size": 6_077_301,
        "sha256": "60b089f2cec2cdb32778a035aed7916cd446e34aac4f968d9acb491c1fa91eb9",
    },
}

STREET_MODELS = {
    1: ("preflop-aux", "flop"),
    2: ("turn",),
    3: ("river",),
    4: (),
}


# The four fixtures below are the repository's existing public-node tests.  The
# ranges are tracked inputs, so captures from different worktrees exercise the
# same poker spots without depending on a random match trajectory.
SPOTS = {
    "preflop-root": {
        "street": 1,
        "board": "",
        "current_player": "P1",
        "bets": [50, 100],
        "num_bets": 1,
        "player_range": "uniform",
        "opponent_range": "uniform",
        "source_fixture": "src/tests/test_preflop.py",
    },
    "flop-3cAdKc": {
        "street": 2,
        "board": "3cAdKc",
        "current_player": "P2",
        "bets": [600, 600],
        "num_bets": 0,
        "player_range": "src/tests/ranges/flop-situation3-p2.txt",
        "opponent_range": "src/tests/ranges/flop-situation3-p1.txt",
        "source_fixture": "src/tests/test_flop.py",
    },
    "turn-3c5h4h3h": {
        "street": 3,
        "board": "3c5h4h3h",
        "current_player": "P2",
        "bets": [600, 600],
        "num_bets": 0,
        "player_range": "src/tests/ranges/situation3-p2.txt",
        "opponent_range": "src/tests/ranges/situation3-p1.txt",
        "source_fixture": "src/tests/test_turn.py",
    },
    "river-7d7c8s5sQd": {
        "street": 4,
        "board": "7d7c8s5sQd",
        "current_player": "P2",
        "bets": [8000, 8000],
        "num_bets": 0,
        "player_range": "src/tests/ranges/situation-p2.txt",
        "opponent_range": "src/tests/ranges/situation-p1.txt",
        "source_fixture": "src/tests/test_river.py",
    },
}

# These texture probes exercise the preflop-to-flop CFV boundary that continual
# resolving consumes.  They are deterministic and intentionally independent of
# the public-node flop fixture above.
PREFLOP_CHANCE_BOARDS = (
    {"name": "low-connected-rainbow", "board": "2s3d4h"},
    {"name": "ace-high-dry", "board": "Ah7d2c"},
)

TENSOR_FIELDS = (
    "player_range",
    "opponent_range",
    "strategy",
    "root_cfvs",
    "root_cfvs_both_players",
    "achieved_cfvs",
    "children_cfvs",
)

CFV_FIELDS = (
    "root_cfvs",
    "root_cfvs_both_players",
    "achieved_cfvs",
    "children_cfvs",
)

TIMING_FIELDS = (
    "wall_seconds",
    "public_tree_seconds",
    "lookahead_tensor_seconds",
    "lookahead_build_seconds",
    "cfr_seconds",
    "results_seconds",
    "resolve_total_seconds",
)


class RegressionError(RuntimeError):
    """Raised when a capture or comparison cannot be trusted."""


@dataclass(frozen=True)
class Thresholds:
    max_strategy_abs_delta: float = 1e-6
    max_strategy_weighted_l1: float = 1e-6
    max_action_disagreement_weight: float = 0.0
    max_action_disagreement_fraction: float = 0.0
    max_cfv_abs_delta: float = 1e-4
    max_weighted_cfv_rmse: float = 1e-4
    max_root_ev_delta: float = 1e-4
    max_runtime_ratio: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_strategy_abs_delta",
            "max_strategy_weighted_l1",
            "max_action_disagreement_weight",
            "max_action_disagreement_fraction",
            "max_cfv_abs_delta",
            "max_weighted_cfv_rmse",
            "max_root_ev_delta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise RegressionError(
                    f"{name} must be finite and nonnegative"
                )
        if self.max_runtime_ratio is not None:
            runtime_ratio = float(self.max_runtime_ratio)
            if not math.isfinite(runtime_ratio) or runtime_ratio <= 0:
                raise RegressionError(
                    "max_runtime_ratio must be finite and positive"
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(80).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
    except OSError:
        return False


def inspect_artifact(
    path: Path, expected_size: int, expected_sha256: str
) -> dict[str, object]:
    row: dict[str, object] = {
        "path": str(path),
        "expected_size": int(expected_size),
        "expected_sha256": expected_sha256,
        "verified": False,
    }
    if not path.is_file():
        row["status"] = "missing"
        return row
    actual_size = path.stat().st_size
    row["actual_size"] = actual_size
    if _looks_like_lfs_pointer(path):
        row["status"] = "git-lfs-pointer"
        return row
    if actual_size != expected_size:
        row["status"] = "size-mismatch"
        return row
    actual_sha256 = _sha256_file(path)
    row["actual_sha256"] = actual_sha256
    if actual_sha256 != expected_sha256:
        row["status"] = "sha256-mismatch"
        return row
    row.update({"status": "verified", "verified": True})
    return row


def selected_spot_names(names: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(names or SPOTS)
    if not selected:
        raise RegressionError("at least one spot is required")
    unknown = sorted(set(selected) - set(SPOTS))
    if unknown:
        raise RegressionError(f"unknown spots: {', '.join(unknown)}")
    if len(set(selected)) != len(selected):
        raise RegressionError("spot names must be unique")
    return selected


def required_asset_paths(spot_names: Sequence[str]) -> tuple[str, ...]:
    paths: set[str] = set()
    for name in spot_names:
        paths.update(STREET_ASSETS[int(SPOTS[name]["street"])])
    return tuple(sorted(paths))


def required_model_names(spot_names: Sequence[str]) -> tuple[str, ...]:
    names: set[str] = set()
    for spot_name in spot_names:
        names.update(STREET_MODELS[int(SPOTS[spot_name]["street"])])
    return tuple(sorted(names))


def configure_cuda_determinism_environment(torch_module=None) -> dict[str, object]:
    """Set cuBLAS deterministic workspace config before a CUDA context exists."""

    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
    initialized_reader = getattr(cuda, "is_initialized", None)
    initialized = bool(initialized_reader()) if initialized_reader else False
    if configured in ACCEPTED_CUBLAS_WORKSPACE_CONFIGS:
        return {
            "verified": True,
            "status": "preconfigured",
            "value": configured,
            "message": f"CUBLAS_WORKSPACE_CONFIG={configured}",
        }
    if initialized:
        return {
            "verified": False,
            "status": "cuda-context-already-initialized",
            "value": configured,
            "message": (
                "CUDA was initialized before a deterministic cuBLAS workspace "
                "was configured. Start a fresh process with "
                f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG}; the harness "
                "will not run a warn-only quality gate."
            ),
        }
    if configured is not None:
        return {
            "verified": False,
            "status": "invalid-cublas-workspace-config",
            "value": configured,
            "message": (
                f"unsupported CUBLAS_WORKSPACE_CONFIG={configured!r}; start a "
                "fresh process with :4096:8 (preferred) or :16:8 for strict "
                "deterministic CUDA matrix multiplication"
            ),
        }
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    return {
        "verified": True,
        "status": "configured",
        "value": CUBLAS_WORKSPACE_CONFIG,
        "message": (
            f"set CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG} before CUDA use"
        ),
    }


def probe_cuda_determinism(torch_module) -> dict[str, object]:
    """Exercise CUDA ops used by preflop bucketing and neural inference."""

    previous_enabled = bool(torch_module.are_deterministic_algorithms_enabled())
    warn_only_reader = getattr(
        torch_module, "is_deterministic_algorithms_warn_only_enabled", None
    )
    previous_warn_only = bool(warn_only_reader()) if warn_only_reader else False
    try:
        torch_module.use_deterministic_algorithms(True, warn_only=False)
        device = torch_module.device("cuda")
        target = torch_module.zeros((2, 2, 4), device=device)
        indexes = torch_module.tensor(
            [[[0, 0, 1, 1], [2, 2, 3, 3]]],
            dtype=torch_module.long,
            device=device,
        ).expand(2, 2, 4)
        source = torch_module.arange(
            16, dtype=torch_module.float32, device=device
        ).view(2, 2, 4)
        target.scatter_add_(2, indexes, source)

        selected = torch_module.empty_like(target)
        torch_module.index_select(
            target,
            0,
            torch_module.tensor([1, 0], dtype=torch_module.long, device=device),
            out=selected,
        )
        destination = torch_module.zeros((3, 2, 4), device=device)
        destination.index_copy_(
            0,
            torch_module.tensor([0, 2], dtype=torch_module.long, device=device),
            selected,
        )
        left = torch_module.arange(
            256, dtype=torch_module.float32, device=device
        ).view(16, 16)
        right = torch_module.arange(
            256, 512, dtype=torch_module.float32, device=device
        ).view(16, 16)
        first_product = torch_module.mm(left, right)
        second_product = torch_module.mm(left, right)
        torch_module.cuda.synchronize()
        if not bool(torch_module.equal(first_product, second_product)):
            raise RuntimeError("CUDA matrix multiplication was not bit-identical")
    except RuntimeError as exc:
        return {
            "verified": False,
            "status": "cuda-determinism-unsupported",
            "message": (
                "CUDA is available, but strict deterministic scatter/index/mm "
                f"operations failed: {exc}. Use a compatible PyTorch/CUDA "
                "build or run the CPU gate; this harness will not fall back "
                "to warn-only determinism."
            ),
        }
    finally:
        torch_module.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )
    return {
        "verified": True,
        "status": "available",
        "message": "strict deterministic CUDA scatter/index/mm probe passed",
    }


def inspect_runtime_device(
    device: str, *, torch_module=None, determinism_probe=None
) -> dict[str, object]:
    """Return a fail-closed runtime report without requiring CUDA in tests."""

    if device not in ("cpu", "cuda"):
        return {
            "requested": device,
            "verified": False,
            "status": "unsupported-device",
            "message": f"unsupported device {device!r}; choose cpu or cuda",
        }
    if device == "cpu":
        return {
            "requested": "cpu",
            "verified": True,
            "status": "available",
            "message": "CPU execution selected",
        }

    loaded_torch = torch_module or sys.modules.get("torch")
    workspace = configure_cuda_determinism_environment(loaded_torch)
    if not bool(workspace["verified"]):
        return {
            "requested": "cuda",
            "verified": False,
            "status": workspace["status"],
            "message": workspace["message"],
            "cublas_workspace_config": workspace,
        }

    try:
        if torch_module is None:
            import torch as torch_module
    except ImportError as exc:
        return {
            "requested": "cuda",
            "verified": False,
            "status": "torch-unavailable",
            "message": f"CUDA requested but PyTorch could not be imported: {exc}",
        }

    report: dict[str, object] = {
        "requested": "cuda",
        "torch": str(torch_module.__version__),
        "cuda_runtime": torch_module.version.cuda,
        "cublas_workspace_config": workspace,
    }
    if not bool(torch_module.cuda.is_available()):
        report.update(
            {
                "verified": False,
                "status": "cuda-unavailable",
                "message": (
                    "CUDA requested but torch.cuda.is_available() is false "
                    f"(torch={torch_module.__version__}, "
                    f"cuda_runtime={torch_module.version.cuda!r})"
                ),
            }
        )
        return report

    device_index = int(torch_module.cuda.current_device())
    properties = torch_module.cuda.get_device_properties(device_index)
    capability = torch_module.cuda.get_device_capability(device_index)
    probe = (determinism_probe or probe_cuda_determinism)(torch_module)
    report.update(
        {
            "verified": bool(probe["verified"]),
            "status": (
                "available" if bool(probe["verified"]) else str(probe["status"])
            ),
            "message": (
                "CUDA execution available"
                if bool(probe["verified"])
                else str(probe["message"])
            ),
            "device_index": device_index,
            "device_name": str(torch_module.cuda.get_device_name(device_index)),
            "device_total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(capability[0]), int(capability[1])],
            "deterministic_probe": probe,
        }
    )
    return report


def _fixture_rows(source_root: Path, spot_names: Sequence[str]) -> list[dict]:
    rows = []
    fixture_paths: set[str] = set()
    for name in spot_names:
        spot = SPOTS[name]
        fixture_paths.add(str(spot["source_fixture"]))
        for key in ("player_range", "opponent_range"):
            value = str(spot[key])
            if value != "uniform":
                fixture_paths.add(value)
    for relative in sorted(fixture_paths):
        path = source_root / relative
        row = {"relative_path": relative, "path": str(path), "verified": False}
        if not path.is_file():
            row["status"] = "missing"
        elif _looks_like_lfs_pointer(path):
            row["status"] = "git-lfs-pointer"
        else:
            row.update(
                {
                    "status": "verified",
                    "verified": True,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        rows.append(row)
    return rows


def preflight(
    source_root: Path,
    asset_root: Path,
    model_root: Path,
    spot_names: Sequence[str],
    device: str = "cpu",
) -> dict[str, object]:
    source_root = source_root.resolve()
    asset_root = asset_root.resolve()
    model_root = model_root.resolve()
    assets = []
    for relative in required_asset_paths(spot_names):
        spec = ASSET_SPECS[relative]
        row = inspect_artifact(
            asset_root / relative, int(spec["size"]), str(spec["sha256"])
        )
        row["relative_path"] = relative
        assets.append(row)

    models = []
    for name in required_model_names(spot_names):
        spec = MODEL_SPECS[name]
        row = inspect_artifact(
            model_root / name / "final_compact.pt",
            int(spec["size"]),
            str(spec["sha256"]),
        )
        row["name"] = name
        models.append(row)

    fixtures = _fixture_rows(source_root, spot_names)
    source_dir = source_root / "src"
    device_report = inspect_runtime_device(device)
    verified = (
        source_dir.is_dir()
        and bool(device_report["verified"])
        and all(bool(row["verified"]) for row in assets)
        and all(bool(row["verified"]) for row in models)
        and all(bool(row["verified"]) for row in fixtures)
    )
    fingerprint_rows = [
        ("asset", row["relative_path"], row.get("actual_sha256")) for row in assets
    ] + [("model", row["name"], row.get("actual_sha256")) for row in models]
    return {
        "verified": verified,
        "source_root": str(source_root),
        "asset_root": str(asset_root),
        "model_root": str(model_root),
        "spots": list(spot_names),
        "source_directory_present": source_dir.is_dir(),
        "assets": assets,
        "models": models,
        "fixtures": fixtures,
        "device": device_report,
        "artifact_fingerprint": _json_sha256(fingerprint_rows),
    }


def _preflight_failures(report: Mapping[str, object]) -> list[str]:
    failures = []
    device = report.get("device", {})
    if isinstance(device, Mapping) and not bool(device.get("verified")):
        failures.append(str(device.get("message", "requested device is unavailable")))
    if not bool(report.get("source_directory_present")):
        failures.append(f"missing source directory under {report.get('source_root')}")
    for group in ("assets", "models", "fixtures"):
        for row in report.get(group, []):
            if not bool(row.get("verified")):
                label = row.get("relative_path") or row.get("name") or row.get("path")
                failures.append(f"{label}: {row.get('status', 'unverified')}")
    return failures


def stage_assets(asset_root: Path, spot_names: Sequence[str]) -> dict[str, object]:
    """Download needed large assets into an explicit, preferably ignored root."""

    asset_root = asset_root.resolve()
    if asset_root == Path(asset_root.anchor):
        raise RegressionError("refusing to stage assets at a filesystem root")
    rows = []
    for relative in required_asset_paths(spot_names):
        spec = ASSET_SPECS[relative]
        destination = asset_root / relative
        current = inspect_artifact(
            destination, int(spec["size"]), str(spec["sha256"])
        )
        if bool(current["verified"]):
            current.update({"relative_path": relative, "stage_status": "reused"})
            rows.append(current)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.download-{os.getpid()}")
        temporary.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gdown",
                    str(spec["drive_id"]),
                    "-O",
                    str(temporary),
                ],
                check=True,
            )
            downloaded = inspect_artifact(
                temporary, int(spec["size"]), str(spec["sha256"])
            )
            if not bool(downloaded["verified"]):
                raise RegressionError(
                    f"download verification failed for {relative}: "
                    f"{downloaded.get('status')}"
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        verified = inspect_artifact(
            destination, int(spec["size"]), str(spec["sha256"])
        )
        verified.update({"relative_path": relative, "stage_status": "downloaded"})
        rows.append(verified)
    return {
        "verified": all(bool(row["verified"]) for row in rows),
        "asset_root": str(asset_root),
        "spots": list(spot_names),
        "assets": rows,
        "total_bytes": sum(int(row["expected_size"]) for row in rows),
    }


def _source_fingerprint(source_root: Path) -> dict[str, object]:
    paths = sorted((source_root / "src").rglob("*.py"))
    paths.extend(sorted((source_root / "src/tests/ranges").glob("*.txt")))
    digest = hashlib.sha256()
    file_rows = []
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        file_digest = _sha256_file(path)
        digest.update(
            relative.encode("utf-8")
            + b"\0"
            + file_digest.encode("ascii")
            + b"\n"
        )
        file_rows.append((relative, file_digest))
    try:
        commit = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "commit": commit,
        "solver_tree_sha256": digest.hexdigest(),
        "files": len(file_rows),
    }


def _tensor_payload(tensor, *, allow_nan: bool = False) -> dict[str, object]:
    import torch

    if tensor is None:
        raise RegressionError("solver returned a required tensor as None")
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if bool(torch.isinf(value).any().item()):
        raise RegressionError("solver returned an infinite tensor value")
    nan_indices = torch.where(torch.isnan(value.reshape(-1)))[0].tolist()
    if nan_indices and not allow_nan:
        raise RegressionError("solver returned a NaN in a required finite tensor")
    flat = value.reshape(-1)
    raw = value.numpy().tobytes()
    finite = flat[torch.isfinite(flat)]
    serialized = torch.nan_to_num(flat, nan=0.0)
    payload = {
        "dtype": "float32",
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "min": float(finite.min().item()) if finite.numel() else None,
        "max": float(finite.max().item()) if finite.numel() else None,
        # ``children_cfvs`` uses NaN for structurally undefined action/hand
        # pairs (zero average action reach).  Persist an explicit mask and
        # canonical zeros so JSON remains standards-compliant and comparisons
        # cannot silently ignore a changed undefined region.
        "nan_indices": [int(index) for index in nan_indices],
        "values": [float(item) for item in serialized.tolist()],
    }
    payload["content_sha256"] = _json_sha256(
        {
            "dtype": payload["dtype"],
            "shape": payload["shape"],
            "nan_indices": payload["nan_indices"],
            "values": payload["values"],
        }
    )
    return payload


def _tensor_max_delta(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    if left.get("shape") != right.get("shape"):
        raise RegressionError(
            f"tensor shape mismatch: {left.get('shape')} != {right.get('shape')}"
        )
    if left.get("nan_indices", []) != right.get("nan_indices", []):
        raise RegressionError("tensor NaN masks changed")
    left_values = left.get("values")
    right_values = right.get("values")
    if not isinstance(left_values, list) or not isinstance(right_values, list):
        raise RegressionError("tensor payload is missing values")
    if len(left_values) != len(right_values):
        raise RegressionError("tensor value length mismatch")
    return max(
        (abs(float(a) - float(b)) for a, b in zip(left_values, right_values)),
        default=0.0,
    )


def _prepare_spot(spec: Mapping[str, object], source_root: Path):
    import settings.arguments as arguments
    import settings.constants as constants
    import game.card_tools as card_tools
    import game.card_to_string_conversion as card_to_string
    from tree.tree_node import TreeNode

    node = TreeNode()
    node.board = card_to_string.string_to_board(str(spec["board"]))
    node.street = int(spec["street"])
    node.current_player = getattr(constants.Players, str(spec["current_player"]))
    node.bets = arguments.Tensor([float(value) for value in spec["bets"]])
    node.num_bets = int(spec["num_bets"])

    ranges = []
    for key in ("player_range", "opponent_range"):
        source = str(spec[key])
        if source == "uniform":
            value = card_tools.get_uniform_range(node.board)
        else:
            value = card_tools.get_file_range(str(source_root / source))
        ranges.append(value.view(1, -1).clone())
    return node, ranges[0], ranges[1]


def _synchronize_runtime(torch_module, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()


def _cuda_memory_state(torch_module) -> dict[str, int]:
    device = torch_module.device("cuda")
    return {
        "allocated_bytes": int(torch_module.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch_module.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(device)),
    }


def _cuda_memory_sample(
    torch_module, before: Mapping[str, int]
) -> dict[str, int]:
    after = _cuda_memory_state(torch_module)
    return {
        **after,
        "allocated_before_bytes": int(before["allocated_bytes"]),
        "reserved_before_bytes": int(before["reserved_bytes"]),
        "incremental_peak_allocated_bytes": max(
            0, int(after["peak_allocated_bytes"]) - int(before["allocated_bytes"])
        ),
        "incremental_peak_reserved_bytes": max(
            0, int(after["peak_reserved_bytes"]) - int(before["reserved_bytes"])
        ),
    }


def _capture_preflop_chance_action_cfvs(resolver, device: str) -> dict[str, object]:
    import torch
    import game.card_to_string_conversion as card_to_string

    action_to_index = getattr(resolver.lookahead, "action_to_index", None)
    if not isinstance(action_to_index, Mapping) or not action_to_index:
        raise RegressionError("preflop lookahead has no chance-action index mapping")
    action_pairs = sorted(
        (
            float(action),
            int(index.item() if hasattr(index, "item") else index),
        )
        for action, index in action_to_index.items()
    )
    if len({action for action, _index in action_pairs}) != len(action_to_index):
        raise RegressionError("preflop chance-action keys are not uniquely numeric")

    board_rows = []
    for board_spec in PREFLOP_CHANCE_BOARDS:
        board = card_to_string.string_to_board(str(board_spec["board"]))
        action_rows = []
        for action, lookahead_index in action_pairs:
            if device == "cuda":
                _synchronize_runtime(torch, device)
                torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
                memory_before = _cuda_memory_state(torch)
            _synchronize_runtime(torch, device)
            started = time.perf_counter()
            values = resolver.get_chance_action_cfv(action, board)
            _synchronize_runtime(torch, device)
            wall_seconds = time.perf_counter() - started
            solver_timing = {}
            for key, value in dict(
                getattr(resolver, "last_chance_timing", {})
            ).items():
                if isinstance(value, bool):
                    solver_timing[key] = value
                elif isinstance(value, (int, float)):
                    solver_timing[key] = float(value)
            action_row = {
                "action": action,
                "lookahead_index": lookahead_index,
                "tensor": _tensor_payload(values, allow_nan=True),
                "timing": {
                    "wall_seconds": wall_seconds,
                    "solver": solver_timing,
                },
            }
            if device == "cuda":
                action_row["cuda_memory"] = _cuda_memory_sample(
                    torch, memory_before
                )
            action_rows.append(action_row)
        board_rows.append(
            {
                "name": str(board_spec["name"]),
                "board": str(board_spec["board"]),
                "actions": action_rows,
            }
        )
    return {
        "suite_sha256": _json_sha256(PREFLOP_CHANCE_BOARDS),
        "boards": board_rows,
    }


def _spot_capture(
    name: str,
    spec: Mapping[str, object],
    source_root: Path,
    warmups: int,
    repeats: int,
    device: str = "cpu",
) -> dict[str, object]:
    import torch
    from lookahead.resolving import Resolving
    from terminal_equity.terminal_equity import TerminalEquity

    node, player_range, opponent_range = _prepare_spot(spec, source_root)
    terminal_memory = None
    if device == "cuda":
        _synchronize_runtime(torch, device)
        torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
        terminal_memory_before = _cuda_memory_state(torch)
    _synchronize_runtime(torch, device)
    terminal_started = time.perf_counter()
    terminal_equity = TerminalEquity()
    terminal_equity.set_board(node.board)
    _synchronize_runtime(torch, device)
    terminal_seconds = time.perf_counter() - terminal_started
    if device == "cuda":
        terminal_memory = _cuda_memory_sample(torch, terminal_memory_before)

    warmup_seconds = []
    warmup_memory = []
    timing_samples: list[dict[str, float]] = []
    memory_samples: list[dict[str, int]] = []
    reference = None
    repeat_deltas = []
    resolver = None
    results = None
    for index in range(warmups + repeats):
        if device == "cuda":
            _synchronize_runtime(torch, device)
            torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
            memory_before = _cuda_memory_state(torch)
        resolver = Resolving(terminal_equity)
        _synchronize_runtime(torch, device)
        started = time.perf_counter()
        results = resolver.resolve_first_node(node, player_range, opponent_range)
        _synchronize_runtime(torch, device)
        wall_seconds = time.perf_counter() - started
        memory_sample = (
            _cuda_memory_sample(torch, memory_before) if device == "cuda" else None
        )
        output = {
            "strategy": _tensor_payload(results.strategy),
            "root_cfvs": _tensor_payload(results.root_cfvs),
            "root_cfvs_both_players": _tensor_payload(results.root_cfvs_both_players),
            "achieved_cfvs": _tensor_payload(results.achieved_cfvs),
            "children_cfvs": _tensor_payload(results.children_cfvs, allow_nan=True),
        }
        if index < warmups:
            warmup_seconds.append(wall_seconds)
            if memory_sample is not None:
                warmup_memory.append(memory_sample)
            continue
        if reference is None:
            reference = output
            repeat_deltas.append(0.0)
        else:
            repeat_deltas.append(
                max(
                    _tensor_max_delta(reference[field], output[field])
                    for field in output
                )
            )
        phase = dict(getattr(resolver, "last_timing", {}))
        phase["wall_seconds"] = wall_seconds
        timing_samples.append({key: float(phase[key]) for key in TIMING_FIELDS})
        if memory_sample is not None:
            memory_samples.append(memory_sample)

    if reference is None or resolver is None or results is None:
        raise RegressionError(f"spot {name} produced no measured result")
    if max(repeat_deltas, default=0.0) != 0.0:
        raise RegressionError(
            f"spot {name} was not bit-identical across repeats; "
            f"max delta={max(repeat_deltas):.9g}"
        )

    chance_action_cfvs = None
    if int(spec["street"]) == 1:
        # Capture this boundary API once from the final measured resolve.  It is
        # intentionally outside the repeated root timing loop because the
        # untouched implementation replays preflop CFR for each query.
        chance_action_cfvs = _capture_preflop_chance_action_cfvs(resolver, device)

    phase_summary = {}
    for field in TIMING_FIELDS:
        values = [sample[field] for sample in timing_samples]
        phase_summary[field] = {
            "samples": values,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    actions = [float(value) for value in resolver.get_possible_actions().tolist()]
    strategy = reference["strategy"]
    hand_count = int(strategy["shape"][-1])
    action_count = int(strategy["shape"][0])
    strategy_values = strategy["values"]
    argmax_indices = []
    for hand in range(hand_count):
        probabilities = [
            float(strategy_values[action * hand_count + hand])
            for action in range(action_count)
        ]
        argmax_indices.append(max(range(action_count), key=probabilities.__getitem__))

    row = {
        "name": name,
        "spec": dict(spec),
        "spec_sha256": _json_sha256(spec),
        "actions": actions,
        "argmax_action_indices_sha256": _json_sha256(argmax_indices),
        "tensors": {
            "player_range": _tensor_payload(player_range),
            "opponent_range": _tensor_payload(opponent_range),
            **reference,
        },
        "timing": {
            "terminal_equity_setup_seconds": terminal_seconds,
            "warmup_seconds": warmup_seconds,
            "measured_repeats": repeats,
            "bit_identical_repeats": True,
            "max_repeat_tensor_delta": max(repeat_deltas, default=0.0),
            "phases": phase_summary,
            "median_wall_seconds": phase_summary["wall_seconds"]["median"],
        },
    }
    if chance_action_cfvs is not None:
        row["chance_action_cfvs"] = chance_action_cfvs
    if device == "cuda":
        chance_memory_samples = [
            action["cuda_memory"]
            for board in (chance_action_cfvs or {}).get("boards", [])
            for action in board["actions"]
        ]
        all_memory_samples = [
            terminal_memory,
            *warmup_memory,
            *memory_samples,
            *chance_memory_samples,
        ]
        row["cuda_memory"] = {
            "terminal_equity": terminal_memory,
            "warmups": warmup_memory,
            "measured_repeats": memory_samples,
            "chance_action_calls": chance_memory_samples,
            "peak_allocated_bytes": max(
                int(sample["peak_allocated_bytes"]) for sample in all_memory_samples
            ),
            "peak_reserved_bytes": max(
                int(sample["peak_reserved_bytes"]) for sample in all_memory_samples
            ),
            "max_incremental_peak_allocated_bytes": max(
                int(sample["incremental_peak_allocated_bytes"])
                for sample in all_memory_samples
            ),
            "max_incremental_peak_reserved_bytes": max(
                int(sample["incremental_peak_reserved_bytes"])
                for sample in all_memory_samples
            ),
        }
    return row


def capture_snapshot(
    source_root: Path,
    asset_root: Path,
    model_root: Path,
    spot_names: Sequence[str],
    iterations: int,
    skip_iterations: int,
    warmups: int,
    repeats: int,
    seed: int,
    threads: int,
    device: str = "cpu",
) -> dict[str, object]:
    if iterations < 2:
        raise RegressionError("iterations must be at least 2")
    if skip_iterations < 0 or skip_iterations >= iterations:
        raise RegressionError("skip iterations must satisfy 0 <= skip < iterations")
    if warmups < 0 or repeats < 1 or threads < 1:
        raise RegressionError(
            "warmups must be nonnegative; repeats and threads positive"
        )
    if device not in ("cpu", "cuda"):
        raise RegressionError("device must be cpu or cuda")

    source_root = source_root.resolve()
    asset_root = asset_root.resolve()
    model_root = model_root.resolve()
    report = preflight(source_root, asset_root, model_root, spot_names, device)
    if not bool(report["verified"]):
        raise RegressionError(
            "preflight failed: " + "; ".join(_preflight_failures(report))
        )

    source_dir = source_root / "src"
    runtime_dir = asset_root / "src"
    os.chdir(runtime_dir)
    sys.path.insert(0, str(source_dir))
    os.environ["DYYPHOLDEM_COMPACT_MODEL_PATH"] = str(model_root)

    import torch
    import settings.arguments as arguments

    arguments.use_gpu = device == "cuda"
    arguments.Tensor = (
        torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
    )
    arguments.LongTensor = (
        torch.cuda.LongTensor if device == "cuda" else torch.LongTensor
    )
    arguments.device = torch.device(device)
    arguments.value_net_name = "final_gpu" if device == "cuda" else "final_cpu"
    arguments.cfr_iters = iterations
    arguments.cfr_skip_iters = skip_iterations
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_grad_enabled(False)

    started = time.perf_counter()
    try:
        spots = [
            _spot_capture(name, SPOTS[name], source_root, warmups, repeats, device)
            for name in spot_names
        ]
    except RuntimeError as exc:
        lowered = str(exc).lower()
        if device == "cuda" and (
            "determin" in lowered
            or "scatter" in lowered
            or "index_add" in lowered
            or "cublas_workspace_config" in lowered
        ):
            raise RegressionError(
                "CUDA capture reached an operation that this PyTorch/CUDA build "
                f"cannot run under strict deterministic algorithms: {exc}. Use "
                "a compatible build or run --device cpu; the quality gate will "
                "not fall back to warn-only determinism."
            ) from exc
        raise
    source = _source_fingerprint(source_root)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            getattr(
                torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
            )()
        ),
    }
    if device == "cuda":
        device_report = report["device"]
        environment.update(
            {
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": (
                    torch.backends.cudnn.version()
                    if hasattr(torch.backends, "cudnn")
                    else None
                ),
                "gpu_device_index": device_report["device_index"],
                "gpu_name": device_report["device_name"],
                "gpu_total_memory_bytes": device_report[
                    "device_total_memory_bytes"
                ],
                "gpu_compute_capability": device_report["compute_capability"],
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
            }
        )
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "configuration": {
            "device": device,
            "dtype": "float32",
            "iterations": iterations,
            "skip_iterations": skip_iterations,
            "warmups": warmups,
            "repeats": repeats,
            "seed": seed,
            "threads": threads,
            "spots": list(spot_names),
            "suite_sha256": _json_sha256([(name, SPOTS[name]) for name in spot_names]),
        },
        "environment": environment,
        "preflight": report,
        "spots": spots,
        "total_seconds": time.perf_counter() - started,
    }
    if device == "cuda":
        snapshot["cuda_memory"] = {
            "peak_allocated_bytes": max(
                int(row["cuda_memory"]["peak_allocated_bytes"]) for row in spots
            ),
            "peak_reserved_bytes": max(
                int(row["cuda_memory"]["peak_reserved_bytes"]) for row in spots
            ),
            "max_incremental_peak_allocated_bytes": max(
                int(row["cuda_memory"]["max_incremental_peak_allocated_bytes"])
                for row in spots
            ),
            "max_incremental_peak_reserved_bytes": max(
                int(row["cuda_memory"]["max_incremental_peak_reserved_bytes"])
                for row in spots
            ),
        }
    return snapshot


def _validate_tensor_payload(
    payload: Mapping[str, object], label: str, *, allow_nan_mask: bool = False
) -> None:
    shape = payload.get("shape")
    values = payload.get("values")
    if payload.get("dtype") != "float32" or not isinstance(shape, list):
        raise RegressionError(f"{label} has an invalid dtype or shape")
    if not isinstance(values, list):
        raise RegressionError(f"{label} is missing tensor values")
    expected = math.prod(int(value) for value in shape)
    if expected != len(values):
        raise RegressionError(
            f"{label} contains {len(values)} values but shape requires {expected}"
        )
    if not all(math.isfinite(float(value)) for value in values):
        raise RegressionError(f"{label} contains non-finite values")
    nan_indices = payload.get("nan_indices", [])
    if not isinstance(nan_indices, list):
        raise RegressionError(f"{label} has an invalid NaN mask")
    normalized = [int(index) for index in nan_indices]
    if normalized != sorted(set(normalized)) or any(
        index < 0 or index >= expected for index in normalized
    ):
        raise RegressionError(f"{label} has an invalid NaN mask")
    if normalized and not allow_nan_mask:
        raise RegressionError(f"{label} unexpectedly contains masked NaNs")
    expected_content_sha256 = _json_sha256(
        {
            "dtype": payload["dtype"],
            "shape": shape,
            "nan_indices": normalized,
            "values": values,
        }
    )
    if payload.get("content_sha256") != expected_content_sha256:
        raise RegressionError(f"{label} content fingerprint does not match")


def _validate_cuda_memory_sample(sample: object, label: str) -> None:
    if not isinstance(sample, Mapping):
        raise RegressionError(f"{label} is missing CUDA memory data")
    keys = (
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "allocated_before_bytes",
        "reserved_before_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
    )
    for key in keys:
        value = sample.get(key)
        if not isinstance(value, int) or value < 0:
            raise RegressionError(f"{label}.{key} is not a nonnegative integer")
    if int(sample["peak_allocated_bytes"]) < int(sample["allocated_before_bytes"]):
        raise RegressionError(f"{label} peak allocation is below its starting value")
    if int(sample["peak_reserved_bytes"]) < int(sample["reserved_before_bytes"]):
        raise RegressionError(f"{label} peak reservation is below its starting value")


def _validate_cuda_memory_summary(summary: object, label: str) -> None:
    if not isinstance(summary, Mapping):
        raise RegressionError(f"{label} is missing CUDA memory summary")
    for key in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "max_incremental_peak_allocated_bytes",
        "max_incremental_peak_reserved_bytes",
    ):
        value = summary.get(key)
        if not isinstance(value, int) or value < 0:
            raise RegressionError(f"{label}.{key} is not a nonnegative integer")


def _validate_chance_action_cfvs(row: Mapping[str, object], device: str) -> None:
    chance = row.get("chance_action_cfvs")
    if not isinstance(chance, Mapping):
        raise RegressionError(
            f"spot {row.get('name')} is missing preflop chance-action CFVs"
        )
    if chance.get("suite_sha256") != _json_sha256(PREFLOP_CHANCE_BOARDS):
        raise RegressionError(
            f"spot {row.get('name')} has the wrong chance-board suite"
        )
    boards = chance.get("boards")
    if not isinstance(boards, list) or len(boards) < 2:
        raise RegressionError(
            f"spot {row.get('name')} needs at least two chance boards"
        )
    seen_boards = set()
    for board in boards:
        if not isinstance(board, Mapping):
            raise RegressionError("chance-board row is invalid")
        key = (board.get("name"), board.get("board"))
        if key in seen_boards:
            raise RegressionError(f"duplicate chance board {key!r}")
        seen_boards.add(key)
        actions = board.get("actions")
        if not isinstance(actions, list) or not actions:
            raise RegressionError(f"chance board {key!r} has no actions")
        seen_actions = set()
        for action in actions:
            if not isinstance(action, Mapping):
                raise RegressionError(f"chance board {key!r} has an invalid action")
            action_key = float(action.get("action"))
            if action_key in seen_actions:
                raise RegressionError(
                    f"chance board {key!r} repeats action {action_key}"
                )
            seen_actions.add(action_key)
            tensor = action.get("tensor")
            if not isinstance(tensor, Mapping):
                raise RegressionError(
                    f"chance board {key!r} action {action_key} has no tensor"
                )
            _validate_tensor_payload(
                tensor,
                f"{row.get('name')}.chance[{key!r}][{action_key}]",
                allow_nan_mask=True,
            )
            timing = action.get("timing")
            if (
                not isinstance(timing, Mapping)
                or not math.isfinite(float(timing.get("wall_seconds", -1)))
                or float(timing.get("wall_seconds", -1)) < 0
            ):
                raise RegressionError(
                    f"chance board {key!r} action {action_key} has invalid timing"
                )
            if device == "cuda":
                _validate_cuda_memory_sample(
                    action.get("cuda_memory"),
                    f"{row.get('name')}.chance[{key!r}][{action_key}].cuda_memory",
                )


def validate_snapshot(snapshot: Mapping[str, object]) -> None:
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise RegressionError("unsupported solver-regression schema")
    if snapshot.get("benchmark") != BENCHMARK_NAME:
        raise RegressionError("not a DyypHoldem solver-regression capture")
    if not bool(snapshot.get("preflight", {}).get("verified")):
        raise RegressionError("snapshot records an unverified preflight")
    configured = snapshot.get("configuration", {}).get("spots")
    rows = snapshot.get("spots")
    if not isinstance(configured, list) or not isinstance(rows, list):
        raise RegressionError("snapshot is missing configured spots")
    if configured != [row.get("name") for row in rows]:
        raise RegressionError("configured spots do not match captured rows")
    device = str(snapshot.get("configuration", {}).get("device", "cpu"))
    if device not in ("cpu", "cuda"):
        raise RegressionError(f"snapshot has unsupported device {device!r}")
    if device == "cuda":
        environment = snapshot.get("environment")
        if not isinstance(environment, Mapping):
            raise RegressionError("CUDA snapshot is missing environment metadata")
        for key in (
            "cuda_runtime",
            "cublas_workspace_config",
            "gpu_device_index",
            "gpu_name",
            "gpu_total_memory_bytes",
            "gpu_compute_capability",
        ):
            if environment.get(key) is None:
                raise RegressionError(f"CUDA snapshot is missing environment.{key}")
        if environment.get("cublas_workspace_config") not in (
            ACCEPTED_CUBLAS_WORKSPACE_CONFIGS
        ):
            raise RegressionError(
                "CUDA snapshot used an invalid cuBLAS workspace config"
            )
        if environment.get("deterministic_algorithms") is not True:
            raise RegressionError("CUDA snapshot did not use deterministic algorithms")
        if environment.get("deterministic_warn_only") is not False:
            raise RegressionError("CUDA snapshot used warn-only determinism")
        device_report = snapshot.get("preflight", {}).get("device")
        deterministic_probe = (
            device_report.get("deterministic_probe")
            if isinstance(device_report, Mapping)
            else None
        )
        if not isinstance(deterministic_probe, Mapping) or not bool(
            deterministic_probe.get("verified")
        ):
            raise RegressionError(
                "CUDA snapshot lacks a verified strict-determinism preflight probe"
            )
        _validate_cuda_memory_summary(
            snapshot.get("cuda_memory"), "snapshot.cuda_memory"
        )
    for row in rows:
        tensors = row.get("tensors")
        if not isinstance(tensors, Mapping):
            raise RegressionError(f"spot {row.get('name')} has no tensors")
        for field in TENSOR_FIELDS:
            payload = tensors.get(field)
            if not isinstance(payload, Mapping):
                raise RegressionError(f"spot {row.get('name')} missing {field}")
            _validate_tensor_payload(
                payload,
                f"{row.get('name')}.{field}",
                allow_nan_mask=field == "children_cfvs",
            )
        if not bool(row.get("timing", {}).get("bit_identical_repeats")):
            raise RegressionError(f"spot {row.get('name')} is nondeterministic")
        if int(row.get("spec", {}).get("street", 0)) == 1:
            _validate_chance_action_cfvs(row, device)
        if device == "cuda":
            cuda_memory = row.get("cuda_memory")
            if not isinstance(cuda_memory, Mapping):
                raise RegressionError(
                    f"spot {row.get('name')} is missing CUDA memory summary"
                )
            _validate_cuda_memory_summary(
                cuda_memory, f"{row.get('name')}.cuda_memory"
            )
            _validate_cuda_memory_sample(
                cuda_memory.get("terminal_equity"),
                f"{row.get('name')}.terminal_equity.cuda_memory",
            )
            measured = cuda_memory.get("measured_repeats")
            if not isinstance(measured, list) or len(measured) != int(
                row.get("timing", {}).get("measured_repeats", -1)
            ):
                raise RegressionError(
                    f"spot {row.get('name')} CUDA samples do not match repeats"
                )
            for index, sample in enumerate(measured):
                _validate_cuda_memory_sample(
                    sample, f"{row.get('name')}.repeat[{index}].cuda_memory"
                )


def _weighted_rmse(differences: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(float(value) for value in weights)
    if total <= 0:
        raise RegressionError("range weights do not have positive mass")
    return math.sqrt(
        sum(
            float(weight) * float(delta) ** 2
            for delta, weight in zip(differences, weights)
        )
        / total
    )


def _flatten_chance_action_cfvs(
    chance: Mapping[str, object], label: str
) -> dict[tuple[str, str, float, int], Mapping[str, object]]:
    rows = {}
    for board in chance.get("boards", []):
        for action in board.get("actions", []):
            key = (
                str(board.get("name")),
                str(board.get("board")),
                float(action.get("action")),
                int(action.get("lookahead_index")),
            )
            if key in rows:
                raise RegressionError(f"{label} repeats chance-action key {key!r}")
            rows[key] = action
    return rows


def _spot_metrics(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    if baseline.get("spec_sha256") != candidate.get("spec_sha256"):
        raise RegressionError(f"spot definition changed for {baseline.get('name')}")
    if baseline.get("actions") != candidate.get("actions"):
        raise RegressionError(f"legal actions changed for {baseline.get('name')}")
    bt = baseline["tensors"]
    ct = candidate["tensors"]
    for range_field in ("player_range", "opponent_range"):
        if bt[range_field].get("sha256") != ct[range_field].get("sha256"):
            raise RegressionError(
                f"input {range_field} changed for {baseline.get('name')}"
            )

    strategy_shape = bt["strategy"]["shape"]
    if strategy_shape != ct["strategy"]["shape"] or len(strategy_shape) != 3:
        raise RegressionError(f"strategy shape changed for {baseline.get('name')}")
    action_count, batch_size, hand_count = (int(value) for value in strategy_shape)
    if batch_size != 1 or action_count != len(baseline["actions"]):
        raise RegressionError(f"unexpected strategy layout for {baseline.get('name')}")
    base_strategy = [float(value) for value in bt["strategy"]["values"]]
    cand_strategy = [float(value) for value in ct["strategy"]["values"]]
    player_weights = [float(value) for value in bt["player_range"]["values"]]
    opponent_weights = [float(value) for value in bt["opponent_range"]["values"]]
    if len(player_weights) != hand_count or len(opponent_weights) != hand_count:
        raise RegressionError(f"range shape mismatch for {baseline.get('name')}")

    strategy_differences = [
        abs(left - right) for left, right in zip(base_strategy, cand_strategy)
    ]
    weighted_l1 = 0.0
    disagreement_weight = 0.0
    disagreements = 0
    support = 0
    for hand in range(hand_count):
        weight = player_weights[hand]
        weighted_l1 += weight * sum(
            strategy_differences[action * hand_count + hand]
            for action in range(action_count)
        )
        if weight <= 0:
            continue
        support += 1
        base_probabilities = [
            base_strategy[action * hand_count + hand] for action in range(action_count)
        ]
        cand_probabilities = [
            cand_strategy[action * hand_count + hand] for action in range(action_count)
        ]
        base_action = max(range(action_count), key=base_probabilities.__getitem__)
        cand_action = max(range(action_count), key=cand_probabilities.__getitem__)
        if base_action != cand_action:
            disagreements += 1
            disagreement_weight += weight

    cfv_maxima = {}
    for field in CFV_FIELDS:
        cfv_maxima[field] = _tensor_max_delta(bt[field], ct[field])

    chance_metrics = None
    baseline_chance = baseline.get("chance_action_cfvs")
    candidate_chance = candidate.get("chance_action_cfvs")
    if (baseline_chance is None) != (candidate_chance is None):
        raise RegressionError(
            f"preflop chance-action capture presence changed for {baseline.get('name')}"
        )
    if baseline_chance is not None:
        if not isinstance(baseline_chance, Mapping) or not isinstance(
            candidate_chance, Mapping
        ):
            raise RegressionError("invalid preflop chance-action capture")
        if baseline_chance.get("suite_sha256") != candidate_chance.get(
            "suite_sha256"
        ):
            raise RegressionError("preflop chance-board suite changed")
        baseline_actions = _flatten_chance_action_cfvs(
            baseline_chance, "baseline"
        )
        candidate_actions = _flatten_chance_action_cfvs(
            candidate_chance, "candidate"
        )
        if baseline_actions.keys() != candidate_actions.keys():
            raise RegressionError("preflop chance-action keys changed")
        action_metrics = []
        chance_max_abs = 0.0
        chance_max_rmse = 0.0
        for key, baseline_action in baseline_actions.items():
            candidate_action = candidate_actions[key]
            tensor_delta = _tensor_max_delta(
                baseline_action["tensor"], candidate_action["tensor"]
            )
            differences = [
                float(right) - float(left)
                for left, right in zip(
                    baseline_action["tensor"]["values"],
                    candidate_action["tensor"]["values"],
                )
            ]
            if len(differences) != len(opponent_weights):
                raise RegressionError(
                    f"chance-action CFV shape does not match range for {key!r}"
                )
            weighted_rmse = _weighted_rmse(differences, opponent_weights)
            baseline_call_seconds = float(
                baseline_action["timing"]["wall_seconds"]
            )
            candidate_call_seconds = float(
                candidate_action["timing"]["wall_seconds"]
            )
            chance_max_abs = max(chance_max_abs, tensor_delta)
            chance_max_rmse = max(chance_max_rmse, weighted_rmse)
            action_metrics.append(
                {
                    "board_name": key[0],
                    "board": key[1],
                    "action": key[2],
                    "lookahead_index": key[3],
                    "max_abs_delta": tensor_delta,
                    "range_weighted_rmse": weighted_rmse,
                    "baseline_seconds": baseline_call_seconds,
                    "candidate_seconds": candidate_call_seconds,
                    "runtime_ratio": (
                        candidate_call_seconds / baseline_call_seconds
                        if baseline_call_seconds > 0
                        else None
                    ),
                    "speedup": (
                        baseline_call_seconds / candidate_call_seconds
                        if candidate_call_seconds > 0
                        else None
                    ),
                }
            )
        cfv_maxima["chance_action_cfvs"] = chance_max_abs
        chance_metrics = {
            "max_abs_delta": chance_max_abs,
            "max_range_weighted_rmse": chance_max_rmse,
            "calls": action_metrics,
            "baseline_total_seconds": sum(
                float(row["timing"]["wall_seconds"])
                for row in baseline_actions.values()
            ),
            "candidate_total_seconds": sum(
                float(row["timing"]["wall_seconds"])
                for row in candidate_actions.values()
            ),
        }

    root_delta = [
        float(right) - float(left)
        for left, right in zip(bt["root_cfvs"]["values"], ct["root_cfvs"]["values"])
    ]
    achieved_delta = [
        float(right) - float(left)
        for left, right in zip(
            bt["achieved_cfvs"]["values"], ct["achieved_cfvs"]["values"]
        )
    ]
    root_rmse = _weighted_rmse(root_delta, player_weights)
    achieved_rmse = _weighted_rmse(achieved_delta, opponent_weights)
    root_ev_delta = abs(sum(d * w for d, w in zip(root_delta, player_weights)))
    achieved_ev_delta = abs(
        sum(d * w for d, w in zip(achieved_delta, opponent_weights))
    )

    baseline_seconds = float(baseline["timing"]["median_wall_seconds"])
    candidate_seconds = float(candidate["timing"]["median_wall_seconds"])
    if baseline_seconds <= 0 or candidate_seconds <= 0:
        raise RegressionError(f"invalid timing for {baseline.get('name')}")
    phase_ratios = {}
    for field in TIMING_FIELDS:
        base_phase = float(baseline["timing"]["phases"][field]["median"])
        cand_phase = float(candidate["timing"]["phases"][field]["median"])
        phase_ratios[field] = cand_phase / base_phase if base_phase > 0 else None

    result = {
        "name": baseline["name"],
        "actions_equal": True,
        "strategy": {
            "max_abs_delta": max(strategy_differences, default=0.0),
            "range_weighted_l1": weighted_l1,
        },
        "argmax_actions": {
            "support_hands": support,
            "disagreements": disagreements,
            "disagreement_fraction": disagreements / support if support else 0.0,
            "disagreement_weight": disagreement_weight,
        },
        "cfvs": {
            "max_abs_delta": max(cfv_maxima.values(), default=0.0),
            "max_abs_delta_by_tensor": cfv_maxima,
            "root_range_weighted_rmse": root_rmse,
            "achieved_range_weighted_rmse": achieved_rmse,
            "max_range_weighted_rmse": max(
                root_rmse,
                achieved_rmse,
                (
                    float(chance_metrics["max_range_weighted_rmse"])
                    if chance_metrics is not None
                    else 0.0
                ),
            ),
            "root_ev_delta": root_ev_delta,
            "achieved_ev_delta": achieved_ev_delta,
        },
        "timing": {
            "baseline_median_seconds": baseline_seconds,
            "candidate_median_seconds": candidate_seconds,
            "runtime_ratio": candidate_seconds / baseline_seconds,
            "speedup": baseline_seconds / candidate_seconds,
            "phase_runtime_ratios": phase_ratios,
        },
    }
    if chance_metrics is not None:
        baseline_chance_seconds = float(chance_metrics["baseline_total_seconds"])
        candidate_chance_seconds = float(chance_metrics["candidate_total_seconds"])
        chance_metrics["runtime_ratio"] = (
            candidate_chance_seconds / baseline_chance_seconds
            if baseline_chance_seconds > 0
            else None
        )
        chance_metrics["speedup"] = (
            baseline_chance_seconds / candidate_chance_seconds
            if candidate_chance_seconds > 0
            else None
        )
        result["chance_action_cfvs"] = chance_metrics
    return result


def compare_snapshots(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    thresholds: Thresholds = Thresholds(),
    *,
    allow_iteration_change: bool = False,
    allow_environment_change: bool = False,
) -> dict[str, object]:
    validate_snapshot(baseline)
    validate_snapshot(candidate)
    baseline_config = baseline["configuration"]
    candidate_config = candidate["configuration"]
    failures = []

    for key in ("device", "dtype", "seed", "threads", "spots", "suite_sha256"):
        if baseline_config.get(key) != candidate_config.get(key):
            failures.append(
                f"configuration {key} changed: {baseline_config.get(key)!r} != "
                f"{candidate_config.get(key)!r}"
            )
    for key in ("iterations", "skip_iterations"):
        if (
            baseline_config.get(key) != candidate_config.get(key)
            and not allow_iteration_change
        ):
            failures.append(
                f"configuration {key} changed without --allow-iteration-change"
            )
    if (
        baseline["preflight"].get("artifact_fingerprint")
        != candidate["preflight"].get("artifact_fingerprint")
    ):
        failures.append("asset or model fingerprints changed")
    if (
        baseline.get("environment") != candidate.get("environment")
        and not allow_environment_change
    ):
        failures.append("capture environments changed")

    baseline_rows = {row["name"]: row for row in baseline["spots"]}
    candidate_rows = {row["name"]: row for row in candidate["spots"]}
    if baseline_rows.keys() != candidate_rows.keys():
        raise RegressionError("baseline and candidate spot sets differ")

    spot_metrics = []
    for name in baseline_config["spots"]:
        metrics = _spot_metrics(baseline_rows[name], candidate_rows[name])
        spot_metrics.append(metrics)
        checks = (
            (
                metrics["strategy"]["max_abs_delta"],
                thresholds.max_strategy_abs_delta,
                "strategy max absolute delta",
            ),
            (
                metrics["strategy"]["range_weighted_l1"],
                thresholds.max_strategy_weighted_l1,
                "strategy range-weighted L1",
            ),
            (
                metrics["argmax_actions"]["disagreement_weight"],
                thresholds.max_action_disagreement_weight,
                "argmax action disagreement weight",
            ),
            (
                metrics["argmax_actions"]["disagreement_fraction"],
                thresholds.max_action_disagreement_fraction,
                "argmax action disagreement fraction",
            ),
            (
                metrics["cfvs"]["max_abs_delta"],
                thresholds.max_cfv_abs_delta,
                "CFV max absolute delta",
            ),
            (
                metrics["cfvs"]["max_range_weighted_rmse"],
                thresholds.max_weighted_cfv_rmse,
                "CFV range-weighted RMSE",
            ),
            (
                metrics["cfvs"]["root_ev_delta"],
                thresholds.max_root_ev_delta,
                "root EV delta",
            ),
        )
        for actual, limit, label in checks:
            if float(actual) > float(limit):
                failures.append(
                    f"{name}: {label} {actual:.9g} exceeds {limit:.9g}"
                )
        if (
            thresholds.max_runtime_ratio is not None
            and metrics["timing"]["runtime_ratio"] > thresholds.max_runtime_ratio
        ):
            failures.append(
                f"{name}: runtime ratio {metrics['timing']['runtime_ratio']:.6f} "
                f"exceeds {thresholds.max_runtime_ratio:.6f}"
            )

    if thresholds.max_runtime_ratio is not None:
        for label, snapshot in (("baseline", baseline), ("candidate", candidate)):
            if int(snapshot["configuration"].get("repeats", 0)) < 3:
                failures.append(
                    f"{label}: timing gate requires at least 3 measured repeats"
                )

    baseline_total = sum(
        float(row["timing"]["median_wall_seconds"]) for row in baseline_rows.values()
    )
    candidate_total = sum(
        float(row["timing"]["median_wall_seconds"]) for row in candidate_rows.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": f"{BENCHMARK_NAME}-comparison",
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        "baseline_source": baseline.get("source"),
        "candidate_source": candidate.get("source"),
        "baseline_iterations": baseline_config.get("iterations"),
        "candidate_iterations": candidate_config.get("iterations"),
        "thresholds": thresholds.__dict__,
        "spots": spot_metrics,
        "aggregate_timing": {
            "baseline_sum_of_medians_seconds": baseline_total,
            "candidate_sum_of_medians_seconds": candidate_total,
            "runtime_ratio": candidate_total / baseline_total,
            "speedup": baseline_total / candidate_total,
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path = path if path.is_absolute() else PROJECT_DIR / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> Mapping[str, object]:
    path = path if path.is_absolute() else PROJECT_DIR / path
    path = path.resolve()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RegressionError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RegressionError(f"{path} does not contain a JSON object")
    return payload


def _add_location_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--asset-root", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_DIR / "runs/model-recovery/compact",
    )
    parser.add_argument(
        "--spot",
        action="append",
        choices=tuple(SPOTS),
        help="repeat to select spots; default is all tracked public nodes",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="solver execution device (default: cpu)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", help="verify exact assets, weights, and fixture files"
    )
    _add_location_arguments(preflight_parser)

    stage_parser = subparsers.add_parser(
        "stage-assets", help="download checksum-pinned assets to an explicit root"
    )
    stage_parser.add_argument("--asset-root", type=Path, required=True)
    stage_parser.add_argument("--spot", action="append", choices=tuple(SPOTS))

    capture_parser = subparsers.add_parser(
        "capture", help="capture deterministic tensors and synchronized timings"
    )
    _add_location_arguments(capture_parser)
    capture_parser.add_argument("--iterations", type=int, default=1000)
    capture_parser.add_argument("--skip-iterations", type=int)
    capture_parser.add_argument("--warmups", type=int, default=1)
    capture_parser.add_argument("--repeats", type=int, default=3)
    capture_parser.add_argument("--seed", type=int, default=0)
    capture_parser.add_argument("--threads", type=int, default=1)
    capture_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser(
        "compare", help="apply quality and optional timing gates to two captures"
    )
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.add_argument("--allow-iteration-change", action="store_true")
    compare_parser.add_argument("--allow-environment-change", action="store_true")
    compare_parser.add_argument("--max-strategy-abs-delta", type=float, default=1e-6)
    compare_parser.add_argument(
        "--max-strategy-weighted-l1", type=float, default=1e-6
    )
    compare_parser.add_argument(
        "--max-action-disagreement-weight", type=float, default=0.0
    )
    compare_parser.add_argument(
        "--max-action-disagreement-fraction", type=float, default=0.0
    )
    compare_parser.add_argument("--max-cfv-abs-delta", type=float, default=1e-4)
    compare_parser.add_argument(
        "--max-weighted-cfv-rmse", type=float, default=1e-4
    )
    compare_parser.add_argument("--max-root-ev-delta", type=float, default=1e-4)
    compare_parser.add_argument("--max-runtime-ratio", type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "stage-assets":
            names = selected_spot_names(args.spot)
            payload = stage_assets(args.asset_root, names)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if args.command == "preflight":
            names = selected_spot_names(args.spot)
            payload = preflight(
                args.source_root,
                args.asset_root,
                args.model_root,
                names,
                args.device,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["verified"] else 2

        if args.command == "capture":
            names = selected_spot_names(args.spot)
            skip = args.skip_iterations
            if skip is None:
                skip = args.iterations // 2
            payload = capture_snapshot(
                args.source_root,
                args.asset_root,
                args.model_root,
                names,
                args.iterations,
                skip,
                args.warmups,
                args.repeats,
                args.seed,
                args.threads,
                args.device,
            )
            _write_json(args.output, payload)
            print(
                json.dumps(
                    {
                        "output": str(
                            (
                                args.output
                                if args.output.is_absolute()
                                else PROJECT_DIR / args.output
                            ).resolve()
                        ),
                        "source": payload["source"],
                        "configuration": payload["configuration"],
                        "spots": [
                            {
                                "name": row["name"],
                                "median_wall_seconds": row["timing"][
                                    "median_wall_seconds"
                                ],
                                "strategy_sha256": row["tensors"]["strategy"][
                                    "sha256"
                                ],
                                "root_cfvs_sha256": row["tensors"]["root_cfvs"][
                                    "sha256"
                                ],
                                **(
                                    {"cuda_memory": row["cuda_memory"]}
                                    if args.device == "cuda"
                                    else {}
                                ),
                                **(
                                    {
                                        "chance_action_calls": sum(
                                            len(board["actions"])
                                            for board in row[
                                                "chance_action_cfvs"
                                            ]["boards"]
                                        )
                                    }
                                    if "chance_action_cfvs" in row
                                    else {}
                                ),
                            }
                            for row in payload["spots"]
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        thresholds = Thresholds(
            max_strategy_abs_delta=args.max_strategy_abs_delta,
            max_strategy_weighted_l1=args.max_strategy_weighted_l1,
            max_action_disagreement_weight=args.max_action_disagreement_weight,
            max_action_disagreement_fraction=args.max_action_disagreement_fraction,
            max_cfv_abs_delta=args.max_cfv_abs_delta,
            max_weighted_cfv_rmse=args.max_weighted_cfv_rmse,
            max_root_ev_delta=args.max_root_ev_delta,
            max_runtime_ratio=args.max_runtime_ratio,
        )
        payload = compare_snapshots(
            _load_json(args.baseline),
            _load_json(args.candidate),
            thresholds,
            allow_iteration_change=args.allow_iteration_change,
            allow_environment_change=args.allow_environment_change,
        )
        if args.output:
            _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 1
    except RegressionError as exc:
        print(f"solver regression error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
