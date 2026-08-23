#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from server.winnings import showdown_winnings  # noqa: E402


class AcpcGameTest(unittest.TestCase):
    def test_showdown_tie_is_a_split_pot(self):
        self.assertEqual(showdown_winnings(123, 123, 800, 800), 0)

    def test_lower_evaluator_strength_wins(self):
        self.assertEqual(showdown_winnings(10, 20, 600, 800), 800)
        self.assertEqual(showdown_winnings(20, 10, 600, 800), -600)


if __name__ == "__main__":
    unittest.main()
