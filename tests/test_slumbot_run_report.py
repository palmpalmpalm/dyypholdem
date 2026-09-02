#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from slumbot_run_report import build_report, hand_statistics, render_markdown  # noqa: E402


class SlumbotRunReportTest(unittest.TestCase):
    def write_run(self, root: Path):
        events = [
            {"event": "hand_started", "hand_number": 0, "client_pos": 1, "timestamp": "2026-09-01T22:00:00+00:00"},
            {"event": "hand_result", "hand_number": 0, "client_pos": 1, "winnings": 100, "timestamp": "2026-09-01T22:00:10+00:00"},
            {"event": "hand_started", "hand_number": 1, "client_pos": 0, "timestamp": "2026-09-01T22:00:11+00:00"},
            {"event": "hand_result", "hand_number": 1, "client_pos": 0, "winnings": -50, "timestamp": "2026-09-01T22:00:30+00:00"},
        ]
        (root / "slumbot-events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
        decisions = [
            {"event": "initialization"},
            {"event": "decision", "street": "preflop", "total_response_seconds": 1.0, "cfr_seconds": 0.9},
            {"event": "decision", "street": "flop", "total_response_seconds": 5.0, "cfr_seconds": 3.5},
            {"event": "decision", "street": "flop", "total_response_seconds": 7.0, "cfr_seconds": 4.5},
            {"event": "hand_result", "hand_number": 1, "winnings": -50},
        ]
        (root / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions) + "\n")
        (root / "slumbot-summary.json").write_text(json.dumps({
            "status": "running", "expected_hands": 1000, "hand_errors": 0,
            "request_retries": 1, "bet_size_corrections": 0, "action_counts": {"call": 3},
        }))

    def test_partial_run_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_run(root)
            report = build_report(root)
            self.assertEqual(report["statistics"]["hands"], 2)
            self.assertEqual(report["statistics"]["chips"], 50)
            self.assertAlmostEqual(report["statistics"]["mbb_per_hand"], 250.0)
            self.assertEqual(report["decisions"], 3)
            self.assertAlmostEqual(report["match_wall_seconds"], 30.0)
            self.assertAlmostEqual(report["by_street"]["flop"]["response_mean"], 6.0)
            self.assertEqual(report["by_street"]["river"]["decisions"], 0)
            self.assertEqual(report["seat_chips"], {"small_blind": 100, "big_blind": -50})
            text = render_markdown(report)
            self.assertIn("| Hands completed / requested | 2 / 1,000 |", text)
            self.assertIn("250.00 mbb/hand", text)
            self.assertIn("| river | 0 | n/a | n/a | n/a | n/a |", text)

    def test_empty_run_does_not_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_report(Path(directory))
            self.assertEqual(report["statistics"]["hands"], 0)
            self.assertIn("n/a mbb/hand", render_markdown(report))
        self.assertEqual(hand_statistics([])["mbb_per_hand"], None)


if __name__ == "__main__":
    unittest.main()
