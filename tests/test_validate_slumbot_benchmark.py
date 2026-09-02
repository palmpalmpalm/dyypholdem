#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from validate_slumbot_benchmark import validate  # noqa: E402


class ValidateSlumbotBenchmarkTest(unittest.TestCase):
    def write_artifacts(self, root: Path, *, hands=1000, summary_hands=None, winnings=1250, status="complete", hand_errors=0):
        (root / "timing_report.json").write_text(
            json.dumps(
                {
                    "decision_count": 2900,
                    "match": {"hands_completed": hands, "latest_hand_number": hands - 1, "cumulative_winnings": winnings},
                }
            ),
            encoding="utf-8",
        )
        (root / "slumbot-summary.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "error": None,
                    "expected_hands": 1000,
                    "hands_completed": hands if summary_hands is None else summary_hands,
                    "cumulative_winnings": winnings,
                    "hand_errors": hand_errors,
                    "statistics": {"mbb_per_hand": 12.5},
                }
            ),
            encoding="utf-8",
        )

    def test_accepts_exact_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root)
            result = validate(root, 1000)
            self.assertTrue(result["valid"])
            self.assertEqual(result["decision_count"], 2900)
            self.assertEqual(result["mbb_per_hand"], 12.5)

    def test_rejects_missing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing or invalid"):
                validate(Path(directory), 1000)

    def test_rejects_early_completion_and_failed_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root, hands=999)
            with self.assertRaisesRegex(ValueError, "999.*1000"):
                validate(root, 1000)
            self.write_artifacts(root, status="failed")
            with self.assertRaisesRegex(ValueError, "clean completion"):
                validate(root, 1000)

    def test_rejects_disagreeing_winnings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root)
            summary = json.loads((root / "slumbot-summary.json").read_text())
            summary["cumulative_winnings"] = 1200
            (root / "slumbot-summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "disagree"):
                validate(root, 1000)


if __name__ == "__main__":
    unittest.main()
