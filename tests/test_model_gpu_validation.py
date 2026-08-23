#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from model_gpu_validation import resolve_model_root  # noqa: E402

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from materialize_assets import ASSETS  # noqa: E402


class ModelGpuValidationTest(unittest.TestCase):
    def test_relative_model_root_is_repository_relative(self):
        self.assertEqual(
            resolve_model_root(Path("runs/model-recovery/compact")),
            (PROJECT_DIR / "runs/model-recovery/compact").resolve(),
        )

    def test_absolute_model_root_is_preserved(self):
        path = (PROJECT_DIR / "runs/model-recovery/compact").resolve()
        self.assertEqual(resolve_model_root(path), path)

    def test_play_profile_contains_preflop_transition_buckets(self):
        assets = {str(asset["path"]): asset for asset in ASSETS["play"]}
        preflop = assets["src/nn/bucketing/preflop_buckets.pt"]
        self.assertEqual(preflop["size"], 117_219_115)
        self.assertEqual(
            preflop["sha256"],
            "131814be7cec451cd4cdc894007db16b5c0eb83a9afc6ff7132e361ee2f4a1bc",
        )


if __name__ == "__main__":
    unittest.main()
