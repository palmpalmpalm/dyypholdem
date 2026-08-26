#!/usr/bin/env python3

from pathlib import Path
from types import ModuleType
from unittest import mock
import sys
import unittest

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

evaluator_stub = ModuleType("game.evaluation.evaluator")
evaluator_stub.Evaluator = type("Evaluator", (), {})
with mock.patch.dict(
    sys.modules,
    {"game.evaluation.evaluator": evaluator_stub},
):
    from terminal_equity.terminal_equity import TerminalEquity  # noqa: E402
    import lookahead.resolving as resolving_module  # noqa: E402
    from lookahead.resolving import Resolving  # noqa: E402


class TerminalEquityBoardCacheTest(unittest.TestCase):
    def setUp(self):
        self.terminal_equity = TerminalEquity.__new__(TerminalEquity)
        self.terminal_equity._set_call_matrix = mock.Mock()
        self.terminal_equity._set_fold_matrix = mock.Mock()

    def test_identical_board_reuses_equity_matrices(self):
        board = torch.tensor([1.0, 2.0, 3.0])
        self.terminal_equity.set_board(board)
        self.terminal_equity.set_board(board.clone())

        self.terminal_equity._set_call_matrix.assert_called_once()
        self.terminal_equity._set_fold_matrix.assert_called_once()

    def test_board_snapshot_is_not_aliased_and_change_rebuilds(self):
        board = torch.tensor([1.0, 2.0, 3.0])
        self.terminal_equity.set_board(board)
        board[0] = 4.0

        self.assertEqual(self.terminal_equity.board.tolist(), [1.0, 2.0, 3.0])
        self.terminal_equity.set_board(board)

        self.assertEqual(self.terminal_equity._set_call_matrix.call_count, 2)
        self.assertEqual(self.terminal_equity._set_fold_matrix.call_count, 2)
        self.assertEqual(self.terminal_equity.board.tolist(), [4.0, 2.0, 3.0])

    def test_dtype_change_rebuilds_even_when_values_match(self):
        self.terminal_equity.set_board(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        )
        self.terminal_equity.set_board(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        )

        self.assertEqual(self.terminal_equity._set_call_matrix.call_count, 2)
        self.assertEqual(self.terminal_equity._set_fold_matrix.call_count, 2)
        self.assertEqual(self.terminal_equity.board.dtype, torch.float64)

    def test_failed_rebuild_preserves_previous_cached_state(self):
        first_board = torch.tensor([1.0, 2.0, 3.0])
        self.terminal_equity.set_board(first_board)
        previous_equity = torch.tensor([11.0])
        previous_fold = torch.tensor([12.0])
        self.terminal_equity.equity_matrix = previous_equity
        self.terminal_equity.fold_matrix = previous_fold
        self.terminal_equity._set_call_matrix.side_effect = RuntimeError(
            "synthetic rebuild failure"
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic rebuild failure"):
            self.terminal_equity.set_board(torch.tensor([4.0, 2.0, 3.0]))

        torch.testing.assert_close(self.terminal_equity.board, first_board)
        self.assertIs(self.terminal_equity.equity_matrix, previous_equity)
        self.assertIs(self.terminal_equity.fold_matrix, previous_fold)


class ChanceReplayBoardInvariantTest(unittest.TestCase):
    def test_flop_replay_restores_original_preflop_equity_board_first(self):
        resolver = Resolving.__new__(Resolving)
        resolver.terminal_equity = mock.Mock()
        resolver.lookahead = mock.Mock()
        resolver.lookahead.tree.board = torch.empty(0)
        resolver.lookahead.get_chance_action_cfv.return_value = torch.tensor(
            [1.0]
        )
        resolver.player_range = torch.tensor([0.4, 0.6])
        resolver.opponent_range = torch.tensor([0.5, 0.5])
        resolver.opponent_cfvs = None

        events = []
        resolver.terminal_equity.set_board.side_effect = (
            lambda _board: events.append("set_board")
        )
        resolver.lookahead.reset.side_effect = lambda: events.append("reset")
        resolver.lookahead.resolve_first_node.side_effect = (
            lambda *_args: events.append("resolve")
        )

        with mock.patch.object(
            resolving_module.card_tools,
            "get_flop_board_index",
            return_value=17,
        ):
            result = resolver.get_chance_action_cfv(
                action=-1,
                board=torch.tensor([1.0, 2.0, 3.0]),
            )

        self.assertEqual(events, ["set_board", "reset", "resolve"])
        torch.testing.assert_close(result, torch.tensor([1.0]))
        resolver.terminal_equity.set_board.assert_called_once()
        torch.testing.assert_close(
            resolver.terminal_equity.set_board.call_args.args[0],
            torch.empty(0),
        )
        self.assertIsNone(resolver.lookahead.next_board_idx)


if __name__ == "__main__":
    unittest.main()
