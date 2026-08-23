#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from model_gpu_validation import resolve_model_root  # noqa: E402


class ModelGpuValidationTest(unittest.TestCase):
    def test_relative_model_root_is_repository_relative(self):
        self.assertEqual(
            resolve_model_root(Path("runs/model-recovery/compact")),
            (PROJECT_DIR / "runs/model-recovery/compact").resolve(),
        )

    def test_absolute_model_root_is_preserved(self):
        path = (PROJECT_DIR / "runs/model-recovery/compact").resolve()
        self.assertEqual(resolve_model_root(path), path)


if __name__ == "__main__":
    unittest.main()
