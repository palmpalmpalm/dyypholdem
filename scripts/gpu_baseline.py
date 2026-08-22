#!/usr/bin/env python3
"""Run a deterministic DyypHoldem resolve and record CUDA timing and parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


PROCESS_START = time.perf_counter()
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
os.chdir(SOURCE_DIR)
sys.path.insert(0, str(SOURCE_DIR))

import torch  # noqa: E402


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def finite_stats(tensor: torch.Tensor) -> dict[str, float | int]:
    value = tensor.detach()
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("benchmark produced non-finite tensor values")
    return {
        "elements": value.numel(),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=("river",), default="river")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--source-diff-sha256", default=None)
    parser.add_argument("--expected-root-sha256", default=None)
    parser.add_argument("--expected-strategy-sha256", default=None)
    args = parser.parse_args()

    if args.iterations < 1 or args.repeats < 2:
        parser.error("iterations must be positive and repeats must be at least 2")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path

    import settings.arguments as arguments  # noqa: E402
    import tests.test_river as test_river  # noqa: E402
    from lookahead.resolving import Resolving  # noqa: E402
    from terminal_equity.terminal_equity import TerminalEquity  # noqa: E402

    arguments.cfr_iters = args.iterations
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    import_seconds = time.perf_counter() - PROCESS_START
    setup_started = time.perf_counter()
    node, player_range, opponent_range = test_river.prepare_test()
    terminal_equity = TerminalEquity()
    terminal_equity.set_board(node.board)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_started
    setup_peak_allocated = torch.cuda.max_memory_allocated(device)
    setup_peak_reserved = torch.cuda.max_memory_reserved(device)

    repeats: list[dict[str, object]] = []
    reference_root: torch.Tensor | None = None
    reference_strategy: torch.Tensor | None = None
    for repeat_index in range(args.repeats):
        torch.cuda.reset_peak_memory_stats(device)
        resolver = Resolving(terminal_equity)
        torch.cuda.synchronize()
        started = time.perf_counter()
        results = resolver.resolve_first_node(node, player_range, opponent_range)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        root = results.root_cfvs.detach().clone()
        strategy = results.strategy.detach().clone()
        if reference_root is None:
            root_delta = 0.0
            strategy_delta = 0.0
            reference_root = root
            reference_strategy = strategy
        else:
            root_delta = float((root - reference_root).abs().max().item())
            strategy_delta = float(
                (strategy - reference_strategy).abs().max().item()
            )

        repeats.append(
            {
                "index": repeat_index,
                "seconds": elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "root_cfvs": finite_stats(root),
                "root_cfvs_sha256": tensor_sha256(root),
                "strategy": finite_stats(strategy),
                "strategy_sha256": tensor_sha256(strategy),
                "max_root_delta_from_repeat_0": root_delta,
                "max_strategy_delta_from_repeat_0": strategy_delta,
            }
        )

    if any(
        not math.isclose(float(item["max_root_delta_from_repeat_0"]), 0.0)
        or not math.isclose(float(item["max_strategy_delta_from_repeat_0"]), 0.0)
        for item in repeats
    ):
        raise RuntimeError("repeated solves were not bit-identical")

    root_sha256 = str(repeats[0]["root_cfvs_sha256"])
    strategy_sha256 = str(repeats[0]["strategy_sha256"])
    if args.expected_root_sha256 and root_sha256 != args.expected_root_sha256:
        raise RuntimeError(
            f"root CFV baseline changed: {root_sha256} != {args.expected_root_sha256}"
        )
    if (
        args.expected_strategy_sha256
        and strategy_sha256 != args.expected_strategy_sha256
    ):
        raise RuntimeError(
            "strategy baseline changed: "
            f"{strategy_sha256} != {args.expected_strategy_sha256}"
        )

    properties = torch.cuda.get_device_properties(device)
    summary = {
        "schema_version": 1,
        "benchmark": "dyypholdem-river-cuda-baseline",
        "source_commit": args.source_commit,
        "source_diff_sha256": args.source_diff_sha256,
        "street": args.street,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": properties.total_memory,
        },
        "startup": {
            "import_seconds": import_seconds,
            "setup_seconds": setup_seconds,
            "peak_allocated_bytes": setup_peak_allocated,
            "peak_reserved_bytes": setup_peak_reserved,
        },
        "actions": [float(value) for value in resolver.get_possible_actions().tolist()],
        "repeat_results": repeats,
        "parity": {
            "bit_identical_repeats": True,
            "matches_expected_baseline": bool(
                args.expected_root_sha256 and args.expected_strategy_sha256
            ),
            "max_root_delta": max(
                float(item["max_root_delta_from_repeat_0"]) for item in repeats
            ),
            "max_strategy_delta": max(
                float(item["max_strategy_delta_from_repeat_0"]) for item in repeats
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
