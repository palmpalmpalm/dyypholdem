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
    _output_hash_mismatches,
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


def _add_cuda_graph_metadata(
    snapshot: dict,
    mode: str,
    *,
    used: bool,
    reason: str,
    eager: int,
    captures: int,
    replays: int,
) -> None:
    snapshot["configuration"]["cuda_graph_mode"] = mode
    snapshot["configuration"]["cuda_graph_eager_warmups"] = 3
    sample = {
        "cuda_graph_mode": mode,
        "cuda_graph_requested": mode != "off",
        "cuda_graph_used": used,
        "cuda_graph_reason": reason,
        "cuda_graph_eager_iterations": eager,
        "cuda_graph_captures": captures,
        "cuda_graph_replays": replays,
    }
    snapshot["spots"][0]["cuda_graph"] = {
        "warmups": [copy.deepcopy(sample)],
        "measured_repeats": [copy.deepcopy(sample) for _ in range(3)],
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
        graphed = build_parser().parse_args(
            [
                "capture",
                "--device",
                "cuda",
                "--cuda-graphs",
                "required",
                "--output",
                "capture.json",
            ]
        )
        cuda = build_parser().parse_args(
            ["preflight", "--device", "cuda"]
        )

        self.assertEqual(cpu.device, "cpu")
        self.assertEqual(cpu.cuda_graphs, "off")
        self.assertEqual(graphed.cuda_graphs, "required")
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

    def test_required_cuda_graph_capture_validates_every_solve(self):
        snapshot = _snapshot()
        _add_cuda_metadata(snapshot)
        _add_cuda_graph_metadata(
            snapshot,
            "required",
            used=True,
            reason="enabled",
            eager=6,
            captures=2,
            replays=994,
        )

        validate_snapshot(snapshot)

        snapshot["spots"][0]["cuda_graph"]["measured_repeats"][1][
            "cuda_graph_replays"
        ] = 993
        with self.assertRaisesRegex(RegressionError, "executed 999 CFR iterations"):
            validate_snapshot(snapshot)

    def test_required_cuda_graph_fallback_and_unreported_auto_fail_closed(self):
        required = _snapshot()
        _add_cuda_metadata(required)
        _add_cuda_graph_metadata(
            required,
            "required",
            used=False,
            reason="river-only",
            eager=1000,
            captures=0,
            replays=0,
        )
        with self.assertRaisesRegex(RegressionError, "required CUDA Graph"):
            validate_snapshot(required)

        automatic = _snapshot()
        automatic["configuration"]["cuda_graph_mode"] = "auto"
        with self.assertRaisesRegex(RegressionError, "lacks per-solve"):
            validate_snapshot(automatic)

    def test_partial_nested_off_graph_telemetry_is_not_treated_as_legacy(self):
        snapshot = _snapshot()
        snapshot["configuration"]["cuda_graph_mode"] = "off"
        snapshot["spots"][0]["cuda_graph"] = {
            "warmups": [],
            "summary": {"configured_mode": "off"},
        }

        with self.assertRaisesRegex(RegressionError, "incomplete per-solve"):
            validate_snapshot(snapshot)

    def test_comparison_surfaces_graph_modes_and_counts(self):
        baseline = _snapshot()
        candidate = copy.deepcopy(baseline)
        _add_cuda_metadata(baseline)
        _add_cuda_metadata(candidate)
        _add_cuda_graph_metadata(
            candidate,
            "required",
            used=True,
            reason="enabled",
            eager=6,
            captures=2,
            replays=994,
        )

        report = compare_snapshots(baseline, candidate, require_bitwise=True)

        self.assertTrue(report["passed"])
        self.assertEqual(report["baseline_cuda_graph_mode"], "off")
        self.assertEqual(report["candidate_cuda_graph_mode"], "required")
        candidate_graph = report["spots"][0]["cuda_graph"]["candidate"]
        self.assertTrue(candidate_graph["all_measured_used"])
        self.assertEqual(candidate_graph["eager_iterations"], [6])
        self.assertEqual(candidate_graph["captures"], [2])
        self.assertEqual(candidate_graph["replays"], [994])

    def test_repeat_hash_check_catches_signed_zero(self):
        reference = {
            "strategy": _tensor_payload(torch.tensor([0.0], dtype=torch.float32))
        }
        changed = {
            "strategy": _tensor_payload(torch.tensor([-0.0], dtype=torch.float32))
        }

        self.assertEqual(
            _output_hash_mismatches(reference, changed), ["strategy"]
        )

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
        self.assertTrue(
            report["spots"][0]["bitwise"]["all_output_hashes_equal"]
        )

    def test_bitwise_gate_catches_signed_zero_with_zero_numeric_delta(self):
        baseline = _snapshot()
        candidate = copy.deepcopy(baseline)
        changed = torch.tensor(
            [[-1.0, -0.0, 1.0], [0.5, -0.5, 0.25]],
            dtype=torch.float32,
        )
        _replace_tensor(candidate, "children_cfvs", changed)

        numeric = compare_snapshots(
            baseline,
            candidate,
            Thresholds(
                max_strategy_abs_delta=0,
                max_strategy_weighted_l1=0,
                max_action_disagreement_weight=0,
                max_action_disagreement_fraction=0,
                max_cfv_abs_delta=0,
                max_weighted_cfv_rmse=0,
                max_root_ev_delta=0,
            ),
        )
        bitwise = compare_snapshots(
            baseline,
            candidate,
            Thresholds(
                max_strategy_abs_delta=0,
                max_strategy_weighted_l1=0,
                max_action_disagreement_weight=0,
                max_action_disagreement_fraction=0,
                max_cfv_abs_delta=0,
                max_weighted_cfv_rmse=0,
                max_root_ev_delta=0,
            ),
            require_bitwise=True,
        )

        self.assertTrue(numeric["passed"])
        self.assertFalse(bitwise["passed"])
        self.assertEqual(
            bitwise["spots"][0]["bitwise"]["mismatches"],
            ["children_cfvs"],
        )
        self.assertTrue(
            any(
                "bitwise output hashes changed" in failure
                for failure in bitwise["failures"]
            )
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

    def test_stale_or_missing_raw_tensor_hash_is_rejected(self):
        snapshot = _snapshot()
        original_hash = snapshot["spots"][0]["tensors"]["children_cfvs"][
            "sha256"
        ]
        changed = torch.tensor(
            [[-1.0, -0.0, 1.0], [0.5, -0.5, 0.25]],
            dtype=torch.float32,
        )
        _replace_tensor(snapshot, "children_cfvs", changed)
        snapshot["spots"][0]["tensors"]["children_cfvs"][
            "sha256"
        ] = original_hash

        with self.assertRaisesRegex(RegressionError, "raw tensor fingerprint"):
            validate_snapshot(snapshot)

        del snapshot["spots"][0]["tensors"]["children_cfvs"]["sha256"]
        with self.assertRaisesRegex(RegressionError, "raw tensor fingerprint"):
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
