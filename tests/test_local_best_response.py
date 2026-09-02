#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

evaluator_stub = ModuleType("game.evaluation.evaluator")
evaluator_stub.Evaluator = type("Evaluator", (), {})
with mock.patch.dict(sys.modules, {"game.evaluation.evaluator": evaluator_stub}):
    import settings.arguments as arguments  # noqa: E402
    import settings.constants as constants  # noqa: E402
    import settings.game_settings as game_settings  # noqa: E402
    import game.card_tools as card_tools  # noqa: E402
    from player.local_best_response import (  # noqa: E402
        LocalBestResponse,
        StrategyChannel,
        action_to_bet,
        parse_raise_menu,
        summary_statistics,
    )
    from server.protocol_to_node import Action  # noqa: E402


class FixedEquity:
    """Terminal-equity stand-in: every opponent hand has the same win-minus-loss value."""

    def __init__(self, value):
        self.value = value
        self.boards = []

    def set_board(self, board):
        self.boards.append(board.clone() if hasattr(board, "clone") else board)

    def call_value(self, ranges, result):
        result.copy_(ranges.sum(1, keepdim=True).expand_as(result) * self.value)

    def fold_value(self, ranges, result):
        result.copy_(ranges.sum(1, keepdim=True).expand_as(result))


def state(position, bet1, bet2):
    return SimpleNamespace(position=position, bet1=bet1, bet2=bet2)


class StrategyChannelTest(unittest.TestCase):
    def test_incremental_reads_and_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bot-strategy.jsonl"
            channel = StrategyChannel(path, poll_seconds=0.01, timeout_seconds=0.2)
            with self.assertRaises(TimeoutError):
                channel.decisions(0, 1)
            with path.open("a") as stream:
                stream.write(json.dumps({"hand_number": 0, "decision_number": 1, "bets": [-2, -1, 300], "strategy": [[0.1], [0.2], [0.7]]}) + "\n")
                stream.write('{"hand_number": 0, "decision_number": 2, "bets": [-2, -1], "strategy": [[0.5], [0.5]]}\n{"hand_number": 1, "decision_number":')
            rows = channel.decisions(0, 2)
            self.assertEqual([row["decision_number"] for row in rows], [1, 2])
            with path.open("a") as stream:
                stream.write(' 1, "bets": [-1], "strategy": [[1.0]]}\n')
            self.assertEqual(channel.decisions(1, 1)[0]["decision_number"], 1)
            self.assertEqual(channel.waits, 1)


class RangeTrackingTest(unittest.TestCase):
    def test_bot_range_multiplies_by_taken_action_probability(self):
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            lbr = LocalBestResponse(FixedEquity(0.0))
            lbr.start_hand(3, "AhAs", 1)
            self.assertAlmostEqual(float(lbr.bot_range.sum()), 1.0, places=5)
            hands = lbr.bot_range.numel()
            record = {"bets": [-2, -1, 300], "strategy": [[0.0] * hands, [0.25] * hands, [0.75] * hands]}
            lbr.apply_bot_decision(record, action_to_bet(Action(action=constants.ACPCActions.rraise, raise_amount=300)))
            self.assertAlmostEqual(float(lbr.bot_range.sum()), 0.75, places=5)
            self.assertEqual(lbr.applied_bot_actions, 1)
            with self.assertRaisesRegex(ValueError, "not in its published menu"):
                lbr.apply_bot_decision(record, 600)

    def test_action_to_bet(self):
        self.assertEqual(action_to_bet(Action(action=constants.ACPCActions.fold)), -2)
        self.assertEqual(action_to_bet(Action(action=constants.ACPCActions.ccall, raise_amount=50)), -1)
        self.assertEqual(action_to_bet(Action(action=constants.ACPCActions.rraise, raise_amount=900)), 900)


class DecisionValueTest(unittest.TestCase):
    def decide(self, equity, position, bet1, bet2, menu=("pot", "all_in")):
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            lbr = LocalBestResponse(FixedEquity(equity), raise_menu=menu)
            lbr.start_hand(0, "AhAs", position)
            return lbr.decide(state(position, bet1, bet2), torch.tensor([0, 1, 2]))

    def test_call_and_raise_values_follow_call_down_formulas(self):
        # LBR is ACPC position 1 (bet1 is its own commitment): 100 in, facing 300.
        action, info = self.decide(0.2, 1, 100, 300)
        self.assertEqual(info["to_call"], 200)
        self.assertAlmostEqual(info["values"]["call"], 600 / 2 * 1.2 - 200)  # 160
        self.assertAlmostEqual(info["values"]["pot"], 900 * 0.2 + 100)         # 280
        self.assertAlmostEqual(info["values"]["all_in"], 20000 * 0.2 + 100)   # 4100
        self.assertEqual(action.action, constants.ACPCActions.rraise)
        self.assertEqual(action.raise_amount, 20000)

    def test_negative_equity_folds_when_facing_a_bet_but_checks_for_free(self):
        action, info = self.decide(-0.5, 0, 300, 100)  # position 0: bet2 is LBR's commitment
        self.assertEqual(info["chosen"], "fold")
        self.assertEqual(action.action, constants.ACPCActions.fold)
        action, info = self.decide(-0.5, 0, 300, 300)
        self.assertEqual(info["chosen"], "call")
        self.assertEqual(action.action, constants.ACPCActions.ccall)
        self.assertEqual(action.raise_amount, 0)

    def test_fold_call_only_menu_never_raises(self):
        action, info = self.decide(0.9, 1, 100, 100, menu=())
        self.assertEqual(sorted(info["values"]), ["call", "fold"])
        self.assertEqual(action.action, constants.ACPCActions.ccall)

    def test_raise_targets_respect_min_raise_and_stack(self):
        targets = LocalBestResponse.raise_targets(100, 300, ("half_pot", "pot", "double_pot", "all_in"))
        self.assertEqual(targets, {"half_pot": 600, "pot": 900, "double_pot": 1500, "all_in": 20000})
        self.assertEqual(LocalBestResponse.raise_targets(9000, 19000, ("pot", "all_in")), {"all_in": 20000})
        self.assertEqual(LocalBestResponse.raise_targets(20000, 20000, ("pot", "all_in")), {})


class RaiseMenuParsingTest(unittest.TestCase):
    def test_none_and_empty_mean_fold_call_only(self):
        for raw in ("none", "", "  ", None):
            self.assertEqual(parse_raise_menu(raw), [])

    def test_valid_menus_and_rejections(self):
        self.assertEqual(parse_raise_menu("pot,all_in"), ["pot", "all_in"])
        self.assertEqual(parse_raise_menu(" half_pot , pot "), ["half_pot", "pot"])
        for bad in ("Pot", "shove", "pot,pot"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_raise_menu(bad)


class SummaryStatisticsTest(unittest.TestCase):
    def test_mbb_and_interval(self):
        stats = summary_statistics([100, -50])
        self.assertAlmostEqual(stats["mbb_per_hand"], 250.0)
        self.assertAlmostEqual(stats["standard_error_mbb_per_hand"], 750.0)
        self.assertEqual(summary_statistics([])["mbb_per_hand"], None)


if __name__ == "__main__":
    unittest.main()
