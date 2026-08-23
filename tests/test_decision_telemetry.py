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
            {"event": "decision", "street": "flop", "total_response_seconds": 3, "cfr_seconds": 2, "chance_reconstruction_seconds": 1.5, "strategy": [{"probability": 1.0}]},
            {"event": "decision", "street": "flop", "total_response_seconds": 5, "cfr_seconds": 4, "chosen_action": "call"},
            {"event": "decision", "street": "river", "total_response_seconds": 1, "cfr_seconds": 0.5},
        ]
        report = build_report(events, {"gpu_name": "test"})
        self.assertEqual(report["by_street"]["flop"]["decisions"], 2)
        self.assertEqual(report["initialization"]["seconds"], 7.5)
        self.assertEqual(report["by_street"]["flop"]["timing_seconds"]["total_response"]["p50"], 4)
        self.assertEqual(report["by_street"]["flop"]["timing_seconds"]["chance_reconstruction"]["max"], 1.5)
        self.assertNotIn("strategy", report["recent_decisions"][0])

    def test_writer_refreshes_json_and_text_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = DecisionTelemetryWriter(root / "decisions.jsonl", root / "report.json", root / "report.txt", {"gpu_name": "test"})
            writer.append({"event": "decision", "street": "turn", "total_response_seconds": 2.5})
            self.assertIn('"decision_count": 1', (root / "report.json").read_text())
            self.assertIn("turn: n=1", (root / "report.txt").read_text())


if __name__ == "__main__":
    unittest.main()
