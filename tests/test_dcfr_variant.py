#!/usr/bin/env python3

from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

evaluator_stub = ModuleType("game.evaluation.evaluator")
evaluator_stub.Evaluator = type("Evaluator", (), {})
with mock.patch.dict(sys.modules, {"game.evaluation.evaluator": evaluator_stub}):
    import settings.arguments as arguments  # noqa: E402
    from lookahead.lookahead import Lookahead  # noqa: E402


def make_lookahead():
    lookahead = Lookahead.__new__(Lookahead)
    lookahead.depth = 3
    lookahead.regrets_data = {2: torch.full((2, 3), 4.0), 3: torch.full((2, 2, 3), 8.0)}
    lookahead.reconstruction_opponent_cfvs = None
    lookahead._dcfr_iteration = None
    lookahead._dcfr_power = None
    lookahead._dcfr_discount = None
    return lookahead


class DcfrVariantTest(unittest.TestCase):
    def test_default_variant_allocates_nothing(self):
        lookahead = make_lookahead()
        with mock.patch.object(arguments, "cfr_variant", "cfr+"):
            lookahead._prepare_regret_discount()
        self.assertIsNone(lookahead._dcfr_iteration)

    def test_discount_sequence_follows_t_alpha_over_t_alpha_plus_one(self):
        lookahead = make_lookahead()
        gadget = mock.Mock()
        gadget.play_regrets = torch.full((3,), 2.0)
        gadget.terminate_regrets = torch.full((3,), 6.0)
        lookahead.reconstruction_gadget = gadget
        lookahead.reconstruction_opponent_cfvs = torch.zeros(3)
        with (
            mock.patch.object(arguments, "cfr_variant", "dcfr"),
            mock.patch.object(arguments, "dcfr_alpha", 1.5),
        ):
            lookahead._prepare_regret_discount()
            self.assertEqual(float(lookahead._dcfr_iteration), 0.0)
            lookahead._apply_regret_discount()
            first = 1.0 / 2.0
            self.assertTrue(torch.allclose(lookahead.regrets_data[2], torch.full((2, 3), 4.0 * first)))
            self.assertTrue(torch.allclose(gadget.play_regrets, torch.full((3,), 2.0 * first)))
            lookahead._apply_regret_discount()
            second = 2 ** 1.5 / (2 ** 1.5 + 1)
            self.assertTrue(torch.allclose(lookahead.regrets_data[3], torch.full((2, 2, 3), 8.0 * first * second)))
            self.assertTrue(torch.allclose(gadget.terminate_regrets, torch.full((3,), 6.0 * first * second)))
            self.assertEqual(float(lookahead._dcfr_iteration), 2.0)
            # A new solve restarts the counter without reallocating on the same device.
            counter = lookahead._dcfr_iteration
            lookahead._prepare_regret_discount()
            self.assertIs(counter, lookahead._dcfr_iteration)
            self.assertEqual(float(counter), 0.0)

    def test_iteration_calls_discount_only_for_dcfr(self):
        for variant, expected in (("cfr+", 0), ("dcfr", 1)):
            with self.subTest(variant=variant):
                lookahead = make_lookahead()
                calls = []
                for name in (
                    "_set_opponent_starting_range", "_compute_current_strategies", "_compute_ranges",
                    "_compute_update_average_strategies", "_compute_terminal_equities", "_compute_cfvs",
                    "_compute_regrets", "_compute_cumulate_average_cfvs",
                ):
                    setattr(lookahead, name, (lambda *_args, _name=name: calls.append(_name)))
                lookahead._apply_regret_discount = lambda: calls.append("discount")
                with mock.patch.object(arguments, "cfr_variant", variant):
                    lookahead._prepare_regret_discount()
                    lookahead._compute_iteration(7)
                self.assertEqual(calls.count("discount"), expected)
                if expected:
                    self.assertEqual(calls.index("discount"), calls.index("_compute_regrets") + 1)


if __name__ == "__main__":
    unittest.main()
