#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from validate_random_benchmark import validate  # noqa: E402


class ValidateRandomBenchmarkTest(unittest.TestCase):
    def write_artifacts(self, root: Path, *, hands=100, random_winnings=-500):
        (root / "timing_report.json").write_text(
            json.dumps(
                {
                    "decision_count": 127,
                    "match": {
                        "hands_completed": hands,
                        "latest_hand_number": hands - 1,
                        "cumulative_winnings": 500,
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "random-summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "error": None,
                    "expected_hands": 100,
                    "hands_completed": hands,
                    "cumulative_winnings": random_winnings,
                }
            ),
            encoding="utf-8",
        )

    def test_accepts_exact_zero_sum_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root)
            result = validate(root, 100)
            self.assertTrue(result["valid"])
            self.assertEqual(result["decision_count"], 127)

    def test_rejects_missing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing or invalid"):
                validate(Path(directory), 100)

    def test_rejects_early_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root, hands=99)
            with self.assertRaisesRegex(ValueError, "99.*100"):
                validate(root, 100)

    def test_rejects_non_zero_sum_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root, random_winnings=-400)
            with self.assertRaisesRegex(ValueError, "zero-sum"):
                validate(root, 100)


if __name__ == "__main__":
    unittest.main()
