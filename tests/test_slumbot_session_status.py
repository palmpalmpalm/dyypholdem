#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from slumbot_session_status import aggregate  # noqa: E402
from validate_slumbot_benchmark import validate_sessions  # noqa: E402


def write_session(root: Path, index: int, **fields):
    directory = root / f"session-{index}"
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "running", "hands_completed": 0, "hands_attempted": 0, "hand_errors": 0,
        "cumulative_winnings": 0, "expected_hands": 250, "error": None,
    }
    summary.update(fields)
    (directory / "slumbot-summary.json").write_text(json.dumps(summary))
    (directory / "timing_report.json").write_text(json.dumps({
        "decision_count": 3 * int(summary["hands_completed"]) or 1,
        "match": {"hands_completed": summary["hands_completed"], "cumulative_winnings": summary["cumulative_winnings"]},
    }))


class SessionStatusTest(unittest.TestCase):
    def test_aggregate_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(aggregate(root, 2)["readable_sessions"], 0)
            write_session(root, 0, hands_completed=10, hands_attempted=11, cumulative_winnings=500)
            partial = aggregate(root, 2)
            self.assertEqual((partial["status"], partial["hands_completed"], partial["hands_attempted"], partial["finished_sessions"]), ("running", 10, 11, 0))
            write_session(root, 1, status="complete", hands_completed=250, hands_attempted=250, cumulative_winnings=-200)
            running = aggregate(root, 2)
            self.assertEqual((running["status"], running["hands_completed"], running["finished_sessions"], running["cumulative_winnings"]), ("running", 260, 1, 300))
            write_session(root, 0, status="complete", hands_completed=250, hands_attempted=252, hand_errors=2, cumulative_winnings=500)
            done = aggregate(root, 2)
            self.assertEqual((done["status"], done["hands_completed"], done["finished_sessions"], done["hand_errors"]), ("complete", 500, 2, 2))
            write_session(root, 1, status="failed", hands_completed=100, error="boom")
            self.assertEqual(aggregate(root, 2)["status"], "failed")

    def test_validate_sessions_requires_every_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_session(root, 0, status="complete", hands_completed=250, hands_attempted=250, cumulative_winnings=500)
            write_session(root, 1, status="complete", hands_completed=250, hands_attempted=250, cumulative_winnings=-200)
            result = validate_sessions(root, 500, 2)
            self.assertTrue(result["valid"])
            self.assertEqual(result["bot_winnings"], 300)
            self.assertAlmostEqual(result["mbb_per_hand"], 6.0)
            with self.assertRaisesRegex(ValueError, "divide evenly"):
                validate_sessions(root, 500, 3)
            write_session(root, 1, status="complete", hands_completed=249, hands_attempted=250)
            with self.assertRaisesRegex(ValueError, "session 1"):
                validate_sessions(root, 500, 2)


if __name__ == "__main__":
    unittest.main()
