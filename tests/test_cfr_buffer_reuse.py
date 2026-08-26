#!/usr/bin/env python3

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock
import sys
import unittest

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SOURCE_DIR))

import settings.arguments as arguments  # noqa: E402
import settings.constants as constants  # noqa: E402
import settings.game_settings as game_settings  # noqa: E402

_ARGUMENT_DEVICE_STATE = (
    arguments.use_gpu,
    arguments.Tensor,
    arguments.LongTensor,
    arguments.device,
)
arguments.use_gpu = False
arguments.Tensor = torch.FloatTensor
arguments.LongTensor = torch.LongTensor
arguments.device = torch.device("cpu")

import nn.next_round_value_pre as next_round_value_pre_module  # noqa: E402
from nn.next_round_value import NextRoundValue  # noqa: E402
from nn.next_round_value_pre import NextRoundValuePre  # noqa: E402

evaluator_stub = ModuleType("game.evaluation.evaluator")
evaluator_stub.Evaluator = type("Evaluator", (), {})
with mock.patch.dict(
    sys.modules,
    {"game.evaluation.evaluator": evaluator_stub},
):
    from lookahead.lookahead import Lookahead  # noqa: E402

(
    arguments.use_gpu,
    arguments.Tensor,
    arguments.LongTensor,
    arguments.device,
) = _ARGUMENT_DEVICE_STATE


class CfrIterationBufferReuseTest(unittest.TestCase):
    def _make_lookahead(self):
        lookahead = Lookahead.__new__(Lookahead)
        lookahead.depth = 2
        lookahead.regret_epsilon = 1e-7
        lookahead.strategy_sum_data = {}
        lookahead.cfvs_sum_data = {}

        regrets = torch.tensor(
            [
                [[[[1.0, -2.0, 0.0, 4.0], [3.0, 0.0, 1.0, -1.0]]]],
                [[[[2.0, 3.0, -1.0, 0.0], [0.0, 2.0, 5.0, 1.0]]]],
                [[[[0.0, 1.0, 6.0, 2.0], [4.0, -3.0, 2.0, 3.0]]]],
            ]
        )
        lookahead.regrets_data = {2: regrets.clone()}
        lookahead.positive_regrets_data = {2: torch.empty_like(regrets)}
        lookahead.current_strategy_data = {2: torch.empty_like(regrets)}
        lookahead.empty_action_mask = {2: torch.ones_like(regrets)}
        lookahead.placeholder_data = {
            2: torch.empty(3, 1, 1, 2, 2, 4)
        }
        return lookahead

    def test_strategy_reduction_and_division_reuse_storage(self):
        lookahead = self._make_lookahead()
        lookahead._ensure_iteration_buffers()

        expected_positive = lookahead.regrets_data[2].clamp(1e-7, 999999)
        expected = expected_positive / expected_positive.sum(0).expand_as(
            expected_positive
        )
        sum_pointer = lookahead.strategy_sum_data[2].data_ptr()
        strategy_pointer = lookahead.current_strategy_data[2].data_ptr()

        lookahead._compute_current_strategies()
        torch.testing.assert_close(lookahead.current_strategy_data[2], expected)

        lookahead.regrets_data[2].add_(0.25)
        lookahead._compute_current_strategies()
        self.assertEqual(lookahead.strategy_sum_data[2].data_ptr(), sum_pointer)
        self.assertEqual(
            lookahead.current_strategy_data[2].data_ptr(), strategy_pointer
        )

    def test_cfv_reduction_reuses_storage_and_matches_legacy_sum(self):
        lookahead = self._make_lookahead()
        source_cfvs = torch.arange(48, dtype=torch.float32).view(
            3, 1, 1, 2, 2, 4
        )
        strategy = torch.tensor([0.2, 0.3, 0.5]).view(3, 1, 1, 1, 1)
        lookahead.current_strategy_data[2].copy_(strategy)
        lookahead.cfvs_data = {
            1: torch.zeros(1, 1, 1, 2, 2, 4),
            2: source_cfvs.clone(),
        }
        lookahead.acting_player = {2: 1}
        lookahead.terminal_actions_count = {0: 0}
        lookahead.nonallinbets_count = {-1: 1}
        lookahead.swap_data = {1: torch.empty(1, 1, 1, 2, 2, 4)}
        lookahead._ensure_iteration_buffers()

        legacy = source_cfvs.clone()
        legacy[:, :, :, :, 0, :].mul_(strategy)
        expected = legacy.sum(0).view(1, 1, 1, 2, 2, 4)
        sum_pointer = lookahead.cfvs_sum_data[2].data_ptr()

        lookahead._compute_cfvs()
        torch.testing.assert_close(lookahead.cfvs_data[1], expected)

        lookahead.cfvs_data[2].copy_(source_cfvs)
        lookahead._compute_cfvs()
        self.assertEqual(lookahead.cfvs_sum_data[2].data_ptr(), sum_pointer)

    def test_fold_terminal_negates_only_the_acting_player(self):
        class TerminalEquityStub:
            @staticmethod
            def call_value(_ranges, result):
                result.copy_(
                    torch.tensor([[5.0, 6.0], [7.0, 8.0]])
                )

            @staticmethod
            def fold_value(_ranges, result):
                result.copy_(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                )

        for acting_player, expected_fold in (
            (1, [[-1.0, -2.0], [3.0, 4.0]]),
            (2, [[1.0, 2.0], [-3.0, -4.0]]),
        ):
            with self.subTest(acting_player=acting_player), mock.patch.object(
                game_settings, "hand_count", 2
            ):
                lookahead = Lookahead.__new__(Lookahead)
                lookahead.depth = 2
                lookahead.tree = SimpleNamespace(
                    street=constants.streets_count
                )
                lookahead.first_call_terminal = True
                lookahead.acting_player = {2: acting_player}
                lookahead.term_call_indices = {2: [0, 1]}
                lookahead.term_fold_indices = {2: [0, 1]}
                lookahead.ranges_data = {
                    2: torch.ones(2, 1, 1, 1, 2, 2)
                }
                lookahead.ranges_data_call = torch.empty(1, 1, 2, 2)
                lookahead.ranges_data_fold = torch.empty(1, 1, 2, 2)
                lookahead.cfvs_data_call = torch.empty(1, 1, 2, 2)
                lookahead.cfvs_data_fold = torch.empty(1, 1, 2, 2)
                lookahead.cfvs_data = {
                    2: torch.empty(2, 1, 1, 1, 2, 2)
                }
                lookahead.terminal_equity = TerminalEquityStub()

                lookahead._compute_terminal_equities_terminal_equity()

                self.assertTrue(
                    torch.equal(
                        lookahead.cfvs_data[2][0, 0, 0, 0],
                        torch.tensor(expected_fold),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        lookahead.cfvs_data[2][1, 0, 0, 0],
                        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                    )
                )


class NextRoundValueBufferReuseTest(unittest.TestCase):
    def test_matrix_transforms_write_into_caller_storage(self):
        calculator = NextRoundValue.__new__(NextRoundValue)
        calculator._range_matrix = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]
        )
        calculator._reverse_value_matrix = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]]
        )
        card_range = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
        bucket_range = torch.empty(2, 3)
        bucket_pointer = bucket_range.data_ptr()

        result = calculator._card_range_to_bucket_range(
            card_range, bucket_range
        )
        self.assertIs(result, bucket_range)
        self.assertEqual(result.data_ptr(), bucket_pointer)
        torch.testing.assert_close(
            bucket_range, card_range @ calculator._range_matrix
        )

        card_value = torch.empty(2, 2)
        card_pointer = card_value.data_ptr()
        result = calculator._bucket_value_to_card_value(
            bucket_range, card_value
        )
        self.assertIs(result, card_value)
        self.assertEqual(result.data_ptr(), card_pointer)
        torch.testing.assert_close(
            card_value, bucket_range @ calculator._reverse_value_matrix
        )

    def test_preflop_scatter_and_gather_reuse_zero_based_indexes(self):
        calculator = NextRoundValuePre.__new__(NextRoundValuePre)
        calculator.board_count = 2
        calculator.bucket_count = 2
        calculator.weight_constant = 0.5
        calculator.board_indexes = torch.tensor(
            [[0, 1, 0], [1, 0, 1]], dtype=torch.long
        )
        calculator.board_indexes_scatter = calculator.board_indexes.clone()
        calculator.impossible_mask = torch.zeros(2, 3, dtype=torch.bool)
        calculator.values_per_board = torch.empty(2, 2, 3)

        card_range = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
        bucket_range = torch.empty(2, 2 * 3)
        index_pointer = calculator.board_indexes_scatter.data_ptr()
        with mock.patch.object(
            next_round_value_pre_module.game_settings, "hand_count", 3
        ):
            calculator._card_range_to_bucket_range(card_range, bucket_range)

            expected_bucket_range = torch.zeros(2, 2, 3)
            expected_indexes = calculator.board_indexes_scatter.view(
                1, 2, 3
            ).expand(2, 2, 3)
            expected_bucket_range.scatter_add_(
                2, expected_indexes, card_range.view(2, 1, 3).expand(2, 2, 3)
            )
            torch.testing.assert_close(
                bucket_range.view(2, 2, 3), expected_bucket_range
            )

            bucket_value = torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            )
            card_value = torch.empty(2, 3)
            gathered_pointer = calculator.values_per_board.data_ptr()
            calculator._bucket_value_to_card_value(bucket_value, card_value)

            expected_values = bucket_value.view(2, 2, 2).gather(
                2,
                calculator.board_indexes.view(1, 2, 3).expand(2, 2, 3),
            )
            torch.testing.assert_close(
                card_value, expected_values.sum(1).mul(0.5)
            )

            calculator._bucket_value_to_card_value(
                bucket_value.add(1), card_value
            )

        self.assertEqual(
            calculator.board_indexes_scatter.data_ptr(), index_pointer
        )
        self.assertEqual(
            calculator.values_per_board.data_ptr(), gathered_pointer
        )


if __name__ == "__main__":
    unittest.main()
