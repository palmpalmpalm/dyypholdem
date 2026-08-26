#!/usr/bin/env python3

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from solver_regression import (  # noqa: E402
    BENCHMARK_NAME,
    RegressionError,
    SCHEMA_VERSION,
    TIMING_FIELDS,
    Thresholds,
    _preflight_failures,
    _tensor_payload,
    build_parser,
    compare_snapshots,
    configure_cuda_determinism_environment,
    inspect_artifact,
    inspect_runtime_device,
    validate_snapshot,
)


def _timing(seconds: float) -> dict:
    phases = {}
    for field in TIMING_FIELDS:
        value = seconds if field == "wall_seconds" else seconds / 2
        phases[field] = {
            "samples": [value, value, value],
            "median": value,
            "min": value,
            "max": value,
        }
    return {
        "terminal_equity_setup_seconds": 0.1,
        "warmup_seconds": [seconds],
        "measured_repeats": 3,
        "bit_identical_repeats": True,
        "max_repeat_tensor_delta": 0.0,
        "phases": phases,
        "median_wall_seconds": seconds,
    }


def _snapshot(seconds: float = 2.0, iterations: int = 1000) -> dict:
    strategy = torch.tensor(
        [[[0.8, 0.4, 0.1]], [[0.2, 0.6, 0.9]]], dtype=torch.float32
    )
    player_range = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    opponent_range = torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float32)
    root = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float32)
    achieved = torch.tensor([[-0.25, 0.5, -1.0]], dtype=torch.float32)
    both = torch.stack((root, achieved), dim=1)
    children = torch.tensor(
        [[-1.0, 0.0, 1.0], [0.5, -0.5, 0.25]], dtype=torch.float32
    )
    spot = {
        "name": "synthetic-river",
        "spec": {"street": 4, "board": "test"},
        "spec_sha256": "spot-fingerprint",
        "actions": [-1.0, 100.0],
        "argmax_action_indices_sha256": "action-fingerprint",
        "tensors": {
            "player_range": _tensor_payload(player_range),
            "opponent_range": _tensor_payload(opponent_range),
            "strategy": _tensor_payload(strategy),
            "root_cfvs": _tensor_payload(root),
            "root_cfvs_both_players": _tensor_payload(both),
            "achieved_cfvs": _tensor_payload(achieved),
            "children_cfvs": _tensor_payload(children, allow_nan=True),
        },
        "timing": _timing(seconds),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "captured_at": "2026-08-27T00:00:00+00:00",
        "source": {"commit": "baseline", "solver_tree_sha256": "source"},
        "configuration": {
            "device": "cpu",
            "dtype": "float32",
            "iterations": iterations,
            "skip_iterations": iterations // 2,
            "warmups": 1,
            "repeats": 3,
            "seed": 0,
            "threads": 1,
            "spots": ["synthetic-river"],
            "suite_sha256": "suite",
        },
        "environment": {
            "python": "test",
            "torch": "test",
            "platform": "test",
            "machine": "test",
            "processor": "test",
        },
        "preflight": {"verified": True, "artifact_fingerprint": "assets"},
        "spots": [spot],
        "total_seconds": seconds,
    }


def _replace_tensor(snapshot: dict, field: str, tensor: torch.Tensor) -> None:
    snapshot["spots"][0]["tensors"][field] = _tensor_payload(
        tensor, allow_nan=field == "children_cfvs"
    )


def _memory_sample(value: int = 100) -> dict:
    return {
        "allocated_bytes": value,
        "reserved_bytes": value + 10,
        "peak_allocated_bytes": value + 20,
        "peak_reserved_bytes": value + 30,
        "allocated_before_bytes": value - 10,
        "reserved_before_bytes": value,
        "incremental_peak_allocated_bytes": 30,
        "incremental_peak_reserved_bytes": 30,
    }


def _add_cuda_metadata(snapshot: dict) -> None:
    snapshot["configuration"]["device"] = "cuda"
    snapshot["environment"].update(
        {
            "cuda_runtime": "12.8",
            "cublas_workspace_config": ":4096:8",
            "cudnn_version": 9100,
            "gpu_device_index": 0,
            "gpu_name": "Test GPU",
            "gpu_total_memory_bytes": 24_000_000_000,
            "gpu_compute_capability": [8, 9],
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
        }
    )
    snapshot["preflight"]["device"] = {
        "requested": "cuda",
        "verified": True,
        "status": "available",
        "deterministic_probe": {
            "verified": True,
            "status": "available",
        },
    }
    sample = _memory_sample()
    snapshot["spots"][0]["cuda_memory"] = {
        "terminal_equity": copy.deepcopy(sample),
        "warmups": [copy.deepcopy(sample)],
        "measured_repeats": [copy.deepcopy(sample) for _ in range(3)],
        "chance_action_calls": [],
        "peak_allocated_bytes": sample["peak_allocated_bytes"],
        "peak_reserved_bytes": sample["peak_reserved_bytes"],
        "max_incremental_peak_allocated_bytes": sample[
            "incremental_peak_allocated_bytes"
        ],
        "max_incremental_peak_reserved_bytes": sample[
            "incremental_peak_reserved_bytes"
        ],
    }
    snapshot["cuda_memory"] = {
        "peak_allocated_bytes": sample["peak_allocated_bytes"],
        "peak_reserved_bytes": sample["peak_reserved_bytes"],
        "max_incremental_peak_allocated_bytes": sample[
            "incremental_peak_allocated_bytes"
        ],
        "max_incremental_peak_reserved_bytes": sample[
            "incremental_peak_reserved_bytes"
        ],
    }


def _add_preflop_chance_capture(snapshot: dict, delta: float = 0.0) -> None:
    snapshot["spots"][0]["spec"]["street"] = 1
    boards = []
    for board_index, (name, board) in enumerate(
        (("low-connected-rainbow", "2s3d4h"), ("ace-high-dry", "Ah7d2c"))
    ):
        tensor = torch.tensor(
            [1.0 + delta + board_index, -0.5, 0.25], dtype=torch.float32
        )
        boards.append(
            {
                "name": name,
                "board": board,
                "actions": [
                    {
                        "action": -1.0,
                        "lookahead_index": 0,
                        "tensor": _tensor_payload(tensor),
                        "timing": {
                            "wall_seconds": 0.5,
                            "solver": {"replayed_flop": True, "seconds": 0.49},
                        },
                    }
                ],
            }
        )
    from solver_regression import PREFLOP_CHANCE_BOARDS, _json_sha256

    snapshot["spots"][0]["chance_action_cfvs"] = {
        "suite_sha256": _json_sha256(PREFLOP_CHANCE_BOARDS),
        "boards": boards,
    }


class SolverRegressionTest(unittest.TestCase):
    def test_thresholds_reject_nonfinite_and_invalid_values(self):
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    RegressionError, "finite and nonnegative"
                ):
                    Thresholds(max_cfv_abs_delta=value)

        for value in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(runtime_ratio=value):
                with self.assertRaisesRegex(
                    RegressionError, "finite and positive"
                ):
                    Thresholds(max_runtime_ratio=value)

    def test_device_cli_defaults_to_cpu_and_accepts_cuda(self):
        cpu = build_parser().parse_args(
            ["capture", "--output", "capture.json"]
        )
        cuda = build_parser().parse_args(
            ["preflight", "--device", "cuda"]
        )

        self.assertEqual(cpu.device, "cpu")
        self.assertEqual(cuda.device, "cuda")

    def test_cuda_unavailable_preflight_is_clear(self):
        class FakeVersion:
            cuda = None

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            __version__ = "test-cpu-build"
            version = FakeVersion()
            cuda = FakeCuda()

        report = inspect_runtime_device("cuda", torch_module=FakeTorch())
        failures = _preflight_failures(
            {"device": report, "source_directory_present": True}
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "cuda-unavailable")
        self.assertIn("torch.cuda.is_available() is false", report["message"])
        self.assertEqual(failures, [report["message"]])

    def test_cuda_determinism_probe_failure_is_actionable(self):
        class FakeVersion:
            cuda = "12.8"

        class FakeProperties:
            total_memory = 24_000_000_000

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def get_device_properties(_index):
                return FakeProperties()

            @staticmethod
            def get_device_capability(_index):
                return (8, 9)

            @staticmethod
            def get_device_name(_index):
                return "Test GPU"

        class FakeTorch:
            __version__ = "test-cuda-build"
            version = FakeVersion()
            cuda = FakeCuda()

        report = inspect_runtime_device(
            "cuda",
            torch_module=FakeTorch(),
            determinism_probe=lambda _torch: {
                "verified": False,
                "status": "cuda-determinism-unsupported",
                "message": "strict deterministic scatter failed; no warn-only fallback",
            },
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "cuda-determinism-unsupported")
        self.assertIn("no warn-only fallback", report["message"])

    def test_initialized_cuda_without_cublas_config_fails_closed(self):
        class FakeCuda:
            @staticmethod
            def is_initialized():
                return True

        class FakeTorch:
            cuda = FakeCuda()

        with mock.patch.dict(os.environ, {}, clear=True):
            report = configure_cuda_determinism_environment(FakeTorch())

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "cuda-context-already-initialized")
        self.assertIn("fresh process", report["message"])

    def test_cuda_snapshot_requires_and_accepts_memory_metadata(self):
        snapshot = _snapshot()
        _add_cuda_metadata(snapshot)
        del snapshot["spots"][0]["cuda_memory"]
        del snapshot["cuda_memory"]

        with self.assertRaisesRegex(RegressionError, "CUDA memory summary"):
            validate_snapshot(snapshot)

        _add_cuda_metadata(snapshot)
        validate_snapshot(snapshot)

    def test_preflop_chance_action_cfvs_are_quality_gated(self):
        baseline = _snapshot()
        candidate = _snapshot()
        _add_preflop_chance_capture(baseline)
        _add_preflop_chance_capture(candidate, delta=1.0)

        report = compare_snapshots(baseline, candidate)

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["spots"][0]["chance_action_cfvs"]["max_abs_delta"], 1.0
        )
        self.assertTrue(
            any("CFV max absolute delta" in item for item in report["failures"])
        )

    def test_exact_outputs_pass_and_timing_is_reported(self):
        baseline = _snapshot(seconds=2.0)
        candidate = _snapshot(seconds=1.0)

        report = compare_snapshots(
            baseline,
            candidate,
            Thresholds(max_runtime_ratio=0.75),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["aggregate_timing"]["speedup"], 2.0)
        self.assertEqual(
            report["spots"][0]["argmax_actions"]["disagreements"], 0
        )

    def test_strategy_and_argmax_action_regression_fails(self):
        baseline = _snapshot()
        candidate = copy.deepcopy(baseline)
        changed = torch.tensor(
            [[[0.1, 0.4, 0.1]], [[0.9, 0.6, 0.9]]], dtype=torch.float32
        )
        _replace_tensor(candidate, "strategy", changed)

        report = compare_snapshots(baseline, candidate)

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["spots"][0]["argmax_actions"]["disagreements"], 1
        )
        self.assertTrue(
            any("strategy max absolute delta" in item for item in report["failures"])
        )
        self.assertTrue(
            any("argmax action disagreement" in item for item in report["failures"])
        )

    def test_cfv_and_ev_regression_fails(self):
        baseline = _snapshot()
        candidate = copy.deepcopy(baseline)
        _replace_tensor(
            candidate,
            "root_cfvs",
            torch.tensor([[2.0, -0.5, 0.25]], dtype=torch.float32),
        )

        report = compare_snapshots(baseline, candidate)

        self.assertFalse(report["passed"])
        self.assertGreater(report["spots"][0]["cfvs"]["root_ev_delta"], 0.19)
        self.assertTrue(any("root EV delta" in item for item in report["failures"]))

    def test_iteration_change_requires_explicit_opt_in(self):
        baseline = _snapshot(iterations=1000)
        candidate = _snapshot(iterations=500)

        strict = compare_snapshots(baseline, candidate)
        allowed = compare_snapshots(
            baseline, candidate, allow_iteration_change=True
        )

        self.assertFalse(strict["passed"])
        self.assertTrue(
            any("iterations changed" in item for item in strict["failures"])
        )
        self.assertTrue(allowed["passed"])

    def test_legal_action_change_fails_closed(self):
        baseline = _snapshot()
        candidate = copy.deepcopy(baseline)
        candidate["spots"][0]["actions"] = [-1.0, 200.0]

        with self.assertRaisesRegex(RegressionError, "legal actions changed"):
            compare_snapshots(baseline, candidate)

    def test_tampered_tensor_content_is_rejected(self):
        snapshot = _snapshot()
        snapshot["spots"][0]["tensors"]["strategy"]["values"][0] = 0.7

        with self.assertRaisesRegex(RegressionError, "fingerprint does not match"):
            validate_snapshot(snapshot)

    def test_lfs_pointer_is_not_accepted_as_an_asset(self):
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:deadbeef\nsize 123\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.pt"
            path.write_text(pointer)
            row = inspect_artifact(path, len(pointer), "unused")

        self.assertFalse(row["verified"])
        self.assertEqual(row["status"], "git-lfs-pointer")

    def test_children_cfv_nan_mask_must_remain_stable(self):
        baseline = _snapshot()
        candidate = copy.deepcopy(baseline)
        baseline_children = torch.tensor(
            [[float("nan"), 0.0, 1.0], [0.5, -0.5, 0.25]], dtype=torch.float32
        )
        candidate_children = torch.tensor(
            [[0.0, 0.0, 1.0], [0.5, -0.5, 0.25]], dtype=torch.float32
        )
        _replace_tensor(baseline, "children_cfvs", baseline_children)
        _replace_tensor(candidate, "children_cfvs", candidate_children)

        with self.assertRaisesRegex(RegressionError, "NaN masks changed"):
            compare_snapshots(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
