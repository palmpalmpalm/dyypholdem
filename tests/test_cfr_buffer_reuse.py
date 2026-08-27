#!/usr/bin/env python3

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock
import os
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

import nn.next_round_value as next_round_value_module  # noqa: E402
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
    def _make_lookahead(self, dtype=torch.float32):
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
            ],
            dtype=dtype,
        )
        lookahead.regrets_data = {2: regrets.clone()}
        lookahead.positive_regrets_data = {2: torch.empty_like(regrets)}
        lookahead.current_strategy_data = {2: torch.empty_like(regrets)}
        lookahead.empty_action_mask = {2: torch.ones_like(regrets)}
        lookahead.placeholder_data = {
            2: torch.empty(3, 1, 1, 2, 2, 4, dtype=dtype)
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

    def test_cfv_mask_broadcast_matches_legacy_player_plane_updates(self):
        for dtype in (torch.float32, torch.float64):
            for acting_player in (1, 2):
                with self.subTest(dtype=dtype, acting_player=acting_player):
                    lookahead = self._make_lookahead(dtype)
                    source_cfvs = (
                        torch.arange(48, dtype=dtype).add_(0.125).view(
                            3, 1, 1, 2, 2, 4
                        )
                    )
                    action_mask = torch.tensor(
                        [
                            [[[[1.0, 0.0, 0.5, 0.25], [2.0, 1.0, 0.5, 0.0]]]],
                            [[[[0.5, 1.0, 2.0, 0.0], [1.0, 0.25, 0.5, 2.0]]]],
                            [[[[0.0, 0.25, 1.0, 2.0], [0.5, 1.0, 0.0, 0.25]]]],
                        ],
                        dtype=dtype,
                    )
                    strategy = torch.tensor(
                        [0.25, 0.5, 0.25], dtype=dtype
                    ).view(3, 1, 1, 1, 1)
                    lookahead.empty_action_mask = {2: action_mask}
                    lookahead.current_strategy_data[2].copy_(strategy)
                    lookahead.cfvs_data = {
                        1: torch.zeros(
                            1, 1, 1, 2, 2, 4, dtype=dtype
                        ),
                        2: source_cfvs.clone(),
                    }
                    lookahead.acting_player = {2: acting_player}
                    lookahead.terminal_actions_count = {0: 0}
                    lookahead.nonallinbets_count = {-1: 1}
                    lookahead.swap_data = {
                        1: torch.empty(
                            1, 1, 1, 2, 2, 4, dtype=dtype
                        )
                    }
                    lookahead._ensure_iteration_buffers()

                    legacy = source_cfvs.clone()
                    legacy[:, :, :, :, 0, :].mul_(action_mask)
                    legacy[:, :, :, :, 1, :].mul_(action_mask)
                    expected_masked = legacy.clone()
                    legacy[
                        :, :, :, :, acting_player - 1, :
                    ].mul_(strategy)
                    expected_parent = legacy.sum(0).view(
                        1, 1, 1, 2, 2, 4
                    )
                    masked_pointer = lookahead.cfvs_data[2].data_ptr()

                    lookahead._compute_cfvs()

                    self.assertEqual(
                        lookahead.cfvs_data[2].data_ptr(), masked_pointer
                    )
                    self.assertTrue(
                        torch.equal(
                            lookahead.cfvs_data[2], expected_masked
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            lookahead.cfvs_data[1], expected_parent
                        )
                    )

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
    def tearDown(self):
        NextRoundValue._clear_bucketing_transform_cache()

    def test_invalid_explicit_cache_limit_fails_closed(self):
        with mock.patch.dict(
            os.environ,
            {"DYYPHOLDEM_NRV_CACHE_BYTES": "not-a-byte-count"},
        ):
            self.assertEqual(
                next_round_value_module._bucketing_transform_cache_limit(),
                0,
            )

    def test_same_board_shares_only_immutable_bucketing_transform(self):
        calls = []

        class IdentityValueNet:
            @staticmethod
            def get_value(inputs, output):
                output.copy_(inputs[:, 0:-1])

        def initialize(instance, _board):
            calls.append(instance)
            instance._street = 2
            instance.bucket_count = 3
            instance.board_count = 2
            instance._range_matrix = torch.arange(
                24, dtype=torch.float32
            ).view(4, 6)
            instance._range_matrix_board_view = (
                instance._range_matrix.view(4, 2, 3)
            )
            instance._reverse_value_matrix = (
                instance._range_matrix.t().clone().mul_(0.25)
            )

        board = torch.tensor([1.0, 2.0, 3.0])
        with (
            mock.patch.object(
                NextRoundValue,
                "_bucketing_transform_cache_key",
                return_value=("same-board",),
            ),
            mock.patch.object(
                NextRoundValue,
                "_init_bucketing",
                autospec=True,
                side_effect=initialize,
            ),
            mock.patch.object(game_settings, "hand_count", 4),
        ):
            first = NextRoundValue(IdentityValueNet(), board)
            second = NextRoundValue(IdentityValueNet(), board)

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            first._range_matrix.data_ptr(),
            second._range_matrix.data_ptr(),
        )
        self.assertEqual(
            first._reverse_value_matrix.data_ptr(),
            second._reverse_value_matrix.data_ptr(),
        )

        transform_before = first._range_matrix.clone()
        source = torch.ones(1, 4)
        first_output = first._card_range_to_bucket_range(source)
        second_output = second._card_range_to_bucket_range(source)
        self.assertTrue(torch.equal(first_output, second_output))
        self.assertTrue(torch.equal(first._range_matrix, transform_before))

        bucket_values = torch.arange(6, dtype=torch.float32).view(1, 6)
        first_cards = first._bucket_value_to_card_value(bucket_values)
        second_cards = second._bucket_value_to_card_value(bucket_values)
        self.assertTrue(torch.equal(first_cards, second_cards))
        self.assertTrue(torch.equal(first._range_matrix, transform_before))

        with (
            mock.patch.object(game_settings, "hand_count", 4),
            mock.patch.object(arguments, "Tensor", torch.FloatTensor),
        ):
            first.start_computation(torch.tensor([100.0]), 1)
            second.start_computation(torch.tensor([300.0]), 1)
            self.assertNotEqual(
                first.pot_sizes.data_ptr(), second.pot_sizes.data_ptr()
            )
            first.pot_sizes.fill_(999)
            self.assertTrue(
                torch.equal(second.pot_sizes, torch.tensor([[300.0]]))
            )

            ranges = torch.tensor(
                [[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]]
            )
            first_values = torch.empty_like(ranges)
            second_values = torch.empty_like(ranges)
            first.get_value(ranges, first_values)
            second.get_value(ranges, second_values)
            self.assertTrue(torch.equal(first_values, second_values))
            self.assertNotEqual(
                first.next_round_inputs.data_ptr(),
                second.next_round_inputs.data_ptr(),
            )
            self.assertTrue(torch.equal(first._range_matrix, transform_before))

    def test_lowered_byte_limit_evicts_existing_transform(self):
        calls = []

        def initialize(instance, _board):
            calls.append(instance)
            instance._street = 2
            instance.bucket_count = 1
            instance.board_count = 1
            instance._range_matrix = torch.ones(1, 1)
            instance._range_matrix_board_view = (
                instance._range_matrix.view(1, 1, 1)
            )
            instance._reverse_value_matrix = torch.ones(1, 1)

        board = torch.tensor([1.0, 2.0, 3.0])
        with (
            mock.patch.object(
                NextRoundValue,
                "_bucketing_transform_cache_key",
                return_value=("uncached",),
            ),
            mock.patch.object(
                NextRoundValue,
                "_init_bucketing",
                autospec=True,
                side_effect=initialize,
            ),
        ):
            NextRoundValue(object(), board)
            with mock.patch.object(
                next_round_value_module,
                "_bucketing_transform_cache_limit",
                return_value=0,
            ):
                NextRoundValue(object(), board)

        self.assertEqual(len(calls), 2)

    def test_cache_key_separates_board_dtype_and_backend(self):
        first_board = torch.tensor([1.0, 2.0, 3.0])
        second_board = torch.tensor([1.0, 2.0, 4.0])
        with (
            mock.patch.object(
                next_round_value_module.card_tools,
                "board_to_street",
                return_value=2,
            ),
            mock.patch.object(
                next_round_value_module.bucketer,
                "get_bucket_count",
                return_value=1000,
            ),
            mock.patch.object(arguments, "use_sqlite", False),
            mock.patch.object(arguments, "Tensor", torch.FloatTensor),
        ):
            first_key = NextRoundValue._bucketing_transform_cache_key(
                first_board
            )
            second_key = NextRoundValue._bucketing_transform_cache_key(
                second_board
            )
            with mock.patch.object(arguments, "Tensor", torch.DoubleTensor):
                double_key = NextRoundValue._bucketing_transform_cache_key(
                    first_board
                )
            with mock.patch.object(arguments, "use_sqlite", True):
                sqlite_key = NextRoundValue._bucketing_transform_cache_key(
                    first_board
                )

        self.assertNotEqual(first_key, second_key)
        self.assertNotEqual(first_key, double_key)
        self.assertNotEqual(first_key, sqlite_key)

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

    def test_preflop_bucket_source_is_released_after_index_construction(self):
        for source_dtype in (torch.float32, torch.long):
            with self.subTest(source_dtype=source_dtype):
                calculator = NextRoundValuePre.__new__(NextRoundValuePre)
                source_buckets = torch.tensor(
                    [[1, -1, 3], [2, 1, -1]], dtype=source_dtype
                )
                source_before = source_buckets.clone()

                with (
                    mock.patch.object(arguments, "use_gpu", False),
                    mock.patch.object(arguments, "Tensor", torch.FloatTensor),
                    mock.patch.object(game_settings, "hand_count", 3),
                    mock.patch.object(
                        game_settings, "board_card_count", [0, 3, 4, 5]
                    ),
                    mock.patch.object(game_settings, "card_count", 7),
                    mock.patch.object(game_settings, "hand_card_count", 2),
                    mock.patch.object(
                        next_round_value_pre_module.card_tools,
                        "board_to_street",
                        return_value=1,
                    ),
                    mock.patch.object(
                        next_round_value_pre_module.card_tools,
                        "get_next_round_boards",
                        return_value=torch.zeros(2, 3),
                    ),
                    mock.patch.object(
                        next_round_value_pre_module.bucketer,
                        "get_bucket_count",
                        side_effect=lambda street: {1: 3, 2: 3}[street],
                    ),
                    mock.patch.object(
                        next_round_value_pre_module.bucketer,
                        "compute_buckets",
                        return_value=torch.tensor([1.0, 2.0, 3.0]),
                    ),
                    mock.patch.object(
                        next_round_value_pre_module.torch,
                        "load",
                        return_value=source_buckets,
                    ),
                ):
                    calculator._init_bucketing(torch.tensor([]))

                self.assertFalse(hasattr(calculator, "board_buckets"))
                self.assertTrue(torch.equal(source_buckets, source_before))
                source_storage = source_buckets.untyped_storage().data_ptr()
                retained_source_storages = [
                    name
                    for name, value in vars(calculator).items()
                    if torch.is_tensor(value)
                    and value.numel() > 0
                    and value.untyped_storage().data_ptr() == source_storage
                ]
                self.assertEqual(retained_source_storages, [])
                self.assertEqual(
                    calculator.impossible_mask.dtype, torch.bool
                )
                self.assertEqual(calculator.board_indexes.dtype, torch.long)
                self.assertEqual(
                    calculator.board_indexes_scatter.dtype, torch.long
                )
                expected_gather = torch.tensor(
                    [[0, 0, 2], [1, 0, 0]]
                )
                expected_scatter = torch.tensor(
                    [[0, 3, 2], [1, 0, 3]]
                )
                self.assertTrue(
                    torch.equal(calculator.board_indexes, expected_gather)
                )
                self.assertTrue(
                    torch.equal(
                        calculator.board_indexes_scatter, expected_scatter
                    )
                )
                source_buckets.fill_(-99)
                self.assertTrue(
                    torch.equal(calculator.board_indexes, expected_gather)
                )
                self.assertTrue(
                    torch.equal(
                        calculator.board_indexes_scatter, expected_scatter
                    )
                )


class CudaGraphExecutionPlanningTest(unittest.TestCase):
    def test_standard_phase_plan_counts_every_iteration_once(self):
        plan = Lookahead._cuda_graph_phase_plan(1000, 500, 3)
        self.assertEqual(
            plan,
            [
                {
                    "name": "burn-in",
                    "iterations": 500,
                    "representative_iteration": 1,
                    "eager_iterations": 3,
                    "captures": 1,
                    "replays": 497,
                },
                {
                    "name": "averaging",
                    "iterations": 500,
                    "representative_iteration": 501,
                    "eager_iterations": 3,
                    "captures": 1,
                    "replays": 497,
                },
            ],
        )
        self.assertEqual(
            sum(
                phase["eager_iterations"]
                + phase["replays"]
                for phase in plan
            ),
            1000,
        )
        self.assertEqual(sum(phase["captures"] for phase in plan), 2)

    def test_phase_plan_rejects_empty_average_and_zero_warmup(self):
        with self.assertRaisesRegex(ValueError, "0 <= skip < iterations"):
            Lookahead._cuda_graph_phase_plan(1000, 1000, 3)
        with self.assertRaisesRegex(ValueError, "warmups must be positive"):
            Lookahead._cuda_graph_phase_plan(1000, 500, 0)

    def test_graph_runner_treats_capture_as_nonexecuting(self):
        state = {
            "capturing": False,
            "executed": [],
            "captured": [],
            "stream_synchronizations": 0,
        }

        class FakeStream:
            def wait_stream(self, _other):
                pass

            def synchronize(self):
                state["stream_synchronizations"] += 1

        graph_stream = FakeStream()
        caller_stream = FakeStream()

        class FakeGraph:
            representative = None

            def replay(self):
                state["executed"].append(self.representative)

        class StreamContext:
            def __enter__(self):
                return graph_stream

            def __exit__(self, _exc_type, _exc, _traceback):
                return False

        class GraphContext:
            def __init__(self, graph):
                self.graph = graph

            def __enter__(self):
                state["capturing"] = True
                return self.graph

            def __exit__(self, _exc_type, _exc, _traceback):
                state["capturing"] = False
                return False

        lookahead = Lookahead.__new__(Lookahead)
        lookahead.ranges_data = {
            1: SimpleNamespace(device=torch.device("cuda"))
        }
        lookahead._cuda_graph_handles = []
        lookahead._cuda_graph_stream = None
        lookahead.cuda_graph_telemetry = {}

        def iteration(representative):
            if state["capturing"]:
                state["captured"].append(representative)
                active_graph.representative = representative
            else:
                state["executed"].append(representative)

        active_graph = None

        def graph_context(graph, **_kwargs):
            nonlocal active_graph
            active_graph = graph
            return GraphContext(graph)

        lookahead._compute_iteration = iteration
        plan = Lookahead._cuda_graph_phase_plan(10, 5, 1)
        with (
            mock.patch.object(
                torch.cuda, "current_stream", return_value=caller_stream
            ),
            mock.patch.object(torch.cuda, "Stream", return_value=graph_stream),
            mock.patch.object(torch.cuda, "stream", return_value=StreamContext()),
            mock.patch.object(torch.cuda, "CUDAGraph", side_effect=FakeGraph),
            mock.patch.object(torch.cuda, "graph", side_effect=graph_context),
        ):
            lookahead._compute_with_cuda_graphs(plan)

        self.assertEqual(state["captured"], [1, 6])
        self.assertEqual(state["executed"], [1] * 5 + [6] * 5)
        self.assertEqual(state["stream_synchronizations"], 3)
        self.assertEqual(
            lookahead.cuda_graph_telemetry["cuda_graph_eager_iterations"], 2
        )
        self.assertEqual(
            lookahead.cuda_graph_telemetry["cuda_graph_captures"], 2
        )
        self.assertEqual(
            lookahead.cuda_graph_telemetry["cuda_graph_replays"], 8
        )

    def test_graph_runner_synchronizes_side_stream_after_replay_failure(self):
        synchronizations = []

        class FakeStream:
            def wait_stream(self, _other):
                pass

            def synchronize(self):
                synchronizations.append("sync")

        class FakeGraph:
            def replay(self):
                raise ValueError("replay failed")

        class Context:
            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _traceback):
                return False

        graph_stream = FakeStream()
        lookahead = Lookahead.__new__(Lookahead)
        lookahead.ranges_data = {
            1: SimpleNamespace(device=torch.device("cuda"))
        }
        lookahead._cuda_graph_handles = []
        lookahead._cuda_graph_stream = None
        lookahead.cuda_graph_telemetry = {}
        lookahead._compute_iteration = mock.Mock()
        plan = [
            {
                "representative_iteration": 1,
                "eager_iterations": 1,
                "captures": 1,
                "replays": 1,
            }
        ]

        with (
            mock.patch.object(
                torch.cuda, "current_stream", return_value=FakeStream()
            ),
            mock.patch.object(torch.cuda, "Stream", return_value=graph_stream),
            mock.patch.object(torch.cuda, "stream", return_value=Context()),
            mock.patch.object(torch.cuda, "CUDAGraph", side_effect=FakeGraph),
            mock.patch.object(torch.cuda, "graph", return_value=Context()),
        ):
            with self.assertRaisesRegex(ValueError, "replay failed"):
                lookahead._compute_with_cuda_graphs(plan)

        self.assertEqual(synchronizations, ["sync", "sync"])
        self.assertEqual(len(lookahead._cuda_graph_handles), 1)
        self.assertIs(lookahead._cuda_graph_stream, graph_stream)

    def test_compute_iteration_preserves_legacy_call_order(self):
        lookahead = Lookahead.__new__(Lookahead)
        calls = []
        for name in (
            "_set_opponent_starting_range",
            "_compute_current_strategies",
            "_compute_ranges",
            "_compute_terminal_equities",
            "_compute_cfvs",
            "_compute_regrets",
        ):
            setattr(
                lookahead,
                name,
                lambda name=name: calls.append(name),
            )
        lookahead._compute_update_average_strategies = (
            lambda iteration: calls.append(("strategy", iteration))
        )
        lookahead._compute_cumulate_average_cfvs = (
            lambda iteration: calls.append(("cfvs", iteration))
        )

        lookahead._compute_iteration(501)

        self.assertEqual(
            calls,
            [
                "_set_opponent_starting_range",
                "_compute_current_strategies",
                "_compute_ranges",
                ("strategy", 501),
                "_compute_terminal_equities",
                "_compute_cfvs",
                "_compute_regrets",
                ("cfvs", 501),
            ],
        )

    def _planning_lookahead(self):
        lookahead = Lookahead.__new__(Lookahead)
        lookahead._ensure_iteration_buffers = mock.Mock()
        lookahead._compute_normalize_average_strategies = mock.Mock()
        lookahead._compute_normalize_average_cfvs = mock.Mock()
        lookahead._cuda_graph_handles = []
        lookahead._cuda_graph_stream = None
        lookahead.cuda_graph_telemetry = {}
        return lookahead

    def test_auto_mode_falls_back_before_mutation_on_cpu(self):
        lookahead = self._planning_lookahead()
        iterations = []
        lookahead._compute_iteration = iterations.append
        with (
            mock.patch.object(arguments, "cuda_graph_mode", "auto"),
            mock.patch.object(arguments, "use_gpu", False),
            mock.patch.object(arguments, "cfr_iters", 4),
            mock.patch.object(arguments, "cfr_skip_iters", 2),
        ):
            lookahead._compute()

        self.assertEqual(iterations, [1, 2, 3, 4])
        self.assertFalse(lookahead.cuda_graph_telemetry["cuda_graph_used"])
        self.assertEqual(
            lookahead.cuda_graph_telemetry["cuda_graph_reason"],
            "gpu-disabled",
        )

    def test_required_mode_rejects_ineligible_solve_before_mutation(self):
        lookahead = self._planning_lookahead()
        lookahead._compute_iteration = mock.Mock()
        with (
            mock.patch.object(arguments, "cuda_graph_mode", "required"),
            mock.patch.object(arguments, "use_gpu", False),
            mock.patch.object(arguments, "cfr_iters", 4),
            mock.patch.object(arguments, "cfr_skip_iters", 2),
        ):
            with self.assertRaisesRegex(RuntimeError, "gpu-disabled"):
                lookahead._compute()

        lookahead._compute_iteration.assert_not_called()
        lookahead._compute_normalize_average_strategies.assert_not_called()
        lookahead._compute_normalize_average_cfvs.assert_not_called()

    def test_capture_failure_never_continues_with_eager_state(self):
        lookahead = self._planning_lookahead()
        mutations = []
        lookahead._cuda_graph_ineligibility = mock.Mock(return_value=None)

        def fail_after_mutation(_plan):
            mutations.append("captured")
            raise ValueError("capture failed")

        lookahead._compute_with_cuda_graphs = fail_after_mutation
        lookahead._compute_eager = mock.Mock()
        with (
            mock.patch.object(arguments, "cuda_graph_mode", "required"),
            mock.patch.object(arguments, "cfr_iters", 1000),
            mock.patch.object(arguments, "cfr_skip_iters", 500),
            mock.patch.object(arguments, "cuda_graph_eager_warmups", 3),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "without eager fallback"
            ):
                lookahead._compute()

        self.assertEqual(mutations, ["captured"])
        lookahead._compute_eager.assert_not_called()


if __name__ == "__main__":
    unittest.main()
