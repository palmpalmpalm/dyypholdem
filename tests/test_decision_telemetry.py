#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from utils.decision_telemetry import DecisionTelemetryWriter, build_report, percentile  # noqa: E402


class DecisionTelemetryTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1.0], 0.95), 1.0)
        self.assertAlmostEqual(percentile([1.0, 3.0], 0.50), 2.0)

    def test_report_groups_timings_by_street_without_private_strategy(self):
        events = [
            {"event": "initialization", "seconds": 7.5, "root_resolve": {"cfr_seconds": 6.0}},
            {"event": "decision", "street": "flop", "total_response_seconds": 3, "cfr_seconds": 2, "chance_reconstruction_seconds": 1.5, "chance_captured_flop": True, "bucketing_cache_hit": False, "bucketing_transform_bytes": 100, "lookahead_build_seconds": 0.4, "strategy": [{"probability": 1.0}]},
            {"event": "decision", "street": "flop", "total_response_seconds": 5, "cfr_seconds": 4, "chance_reconstruction_seconds": 2.5, "chance_replayed_flop": True, "bucketing_cache_hit": True, "bucketing_transform_bytes": 100, "lookahead_build_seconds": 0.01, "chosen_action": "call"},
            {"event": "decision", "street": "river", "total_response_seconds": 1, "cfr_seconds": 0.5},
            {"event": "hand_result", "hand_number": "0", "winnings": 150, "cumulative_winnings": 150},
            {"event": "hand_result", "hand_number": 1, "winnings": -50, "cumulative_winnings": 100},
        ]
        report = build_report(events, {"gpu_name": "test"})
        self.assertEqual(report["by_street"]["flop"]["decisions"], 2)
        self.assertEqual(report["initialization"]["seconds"], 7.5)
        self.assertEqual(report["by_street"]["flop"]["timing_seconds"]["total_response"]["p50"], 4)
        self.assertEqual(report["by_street"]["flop"]["timing_seconds"]["chance_reconstruction"]["max"], 2.5)
        self.assertEqual(report["chance_reconstruction"]["captured_flop"]["count"], 1)
        self.assertEqual(report["chance_reconstruction"]["replayed_flop"]["count"], 1)
        self.assertEqual(
            report["chance_reconstruction"]["captured_flop"]["timing_seconds"]["total"],
            1.5,
        )
        self.assertEqual(report["chance_reconstruction"]["unclassified"]["count"], 0)
        self.assertEqual(report["postflop_bucketing_cache"]["hits"], 1)
        self.assertEqual(report["postflop_bucketing_cache"]["misses"], 1)
        self.assertEqual(report["postflop_bucketing_cache"]["hit_rate"], 0.5)
        self.assertEqual(
            report["postflop_bucketing_cache"]["max_transform_bytes"],
            100,
        )
        self.assertNotIn("strategy", report["recent_decisions"][0])
        self.assertEqual(report["match"]["hands_completed"], 2)
        self.assertEqual(report["match"]["cumulative_winnings"], 100)

    def test_preflop_report_separates_cached_root_from_fresh_resolves(self):
        events = [
            {
                "event": "decision",
                "street": "preflop",
                "reused_root_precompute": True,
                "total_response_seconds": 0.02,
            },
            {
                "event": "decision",
                "street": "preflop",
                "reused_root_precompute": False,
                "total_response_seconds": 5.0,
            },
        ]
        report = build_report(events, {})
        self.assertEqual(report["preflop_solve_modes"]["cached_root"]["decisions"], 1)
        self.assertEqual(report["preflop_solve_modes"]["fresh_resolve"]["decisions"], 1)
        self.assertEqual(
            report["preflop_solve_modes"]["fresh_resolve"]["timing_seconds"]["total_response"]["max"],
            5.0,
        )

    def test_writer_refreshes_json_and_text_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = DecisionTelemetryWriter(root / "decisions.jsonl", root / "report.json", root / "report.txt", {"gpu_name": "test"})
            writer.append({"event": "decision", "street": "turn", "total_response_seconds": 2.5})
            writer.append({"event": "hand_result", "hand_number": 0, "cumulative_winnings": 125})
            self.assertIn('"decision_count": 1', (root / "report.json").read_text())
            self.assertIn("turn: n=1", (root / "report.txt").read_text())
            self.assertIn("Hands completed: 1", (root / "report.txt").read_text())
            self.assertIn("Bot winnings: 125 chips", (root / "report.txt").read_text())
            self.assertIn("captured_flop: n=0", (root / "report.txt").read_text())
            self.assertIn("Postflop bucketing-transform cache", (root / "report.txt").read_text())


if __name__ == "__main__":
    unittest.main()
