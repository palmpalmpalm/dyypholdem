#!/usr/bin/env python3

from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    import settings.arguments as arguments  # noqa: E402
    import settings.constants as constants  # noqa: E402
    import nn.next_round_value_pre as next_round_value_pre_module  # noqa: E402
    import lookahead.resolving as resolving_module  # noqa: E402
    from lookahead.lookahead import Lookahead  # noqa: E402
    from lookahead.lookahead_builder import LookaheadBuilder  # noqa: E402
    from lookahead.resolving import Resolving  # noqa: E402
    from nn.next_round_value_pre import NextRoundValuePre  # noqa: E402


class RowIndependentValueNet:
    def __init__(self):
        self.batch_sizes = []

    def get_value(self, inputs, output):
        self.batch_sizes.append(inputs.size(0))
        output.copy_(inputs[:, 0:-1])
        output.mul_(0.75)
        output.add_(inputs[:, -1:])


def legacy_full_board_values(
    calculator, board, trajectories, full_pot_sizes
):
    iteration_count, pot_count, batch_size, player_count, hand_count = (
        trajectories.shape
    )
    state_count = pot_count * batch_size
    bucket_count = calculator.bucket_count

    extended_range = trajectories.new_zeros(
        state_count, player_count, bucket_count + 1
    )
    serialized_range = extended_range.view(-1, bucket_count + 1)
    range_normalization = trajectories.new_empty(state_count * player_count)
    value_normalization = trajectories.new_empty(state_count, player_count)
    range_memory = trajectories.new_zeros(state_count * player_count, 1)
    cfv_memory = trajectories.new_zeros(
        state_count, player_count, bucket_count
    )
    inputs = trajectories.new_zeros(
        state_count, player_count * bucket_count + 1
    )
    inputs[:, -1].copy_(
        full_pot_sizes.view(-1, 1)
        .expand(pot_count, batch_size)
        .reshape(-1)
    )
    inputs[:, -1].mul_(
        float(1 / next_round_value_pre_module.game_settings.stack)
    )
    serialized_values = trajectories.new_empty(
        state_count, player_count * bucket_count
    )
    values = serialized_values.view(state_count, player_count, bucket_count)

    board_idx = next_round_value_pre_module.card_tools.get_flop_board_index(
        board
    )
    for iteration in range(iteration_count):
        ranges = trajectories[iteration].view(
            state_count, player_count, hand_count
        )
        calculator._card_range_to_bucket_range_on_board(
            board_idx,
            ranges.view(state_count * player_count, hand_count),
            extended_range.view(state_count * player_count, -1),
        )
        torch.sum(
            serialized_range[:, 0:bucket_count],
            1,
            out=range_normalization,
        )
        normalization_view = range_normalization.view(
            state_count, player_count
        )
        for player in range(player_count):
            value_normalization[:, player].copy_(
                normalization_view[:, 1 - player]
            )
        range_memory.add_(value_normalization.view(range_memory.shape))

        range_normalization[torch.eq(range_normalization, 0)] = 1
        serialized_range.div_(
            range_normalization.view(-1, 1).expand_as(serialized_range)
        )
        ranges_by_player = extended_range.view(
            state_count, player_count, bucket_count + 1
        )
        for player in range(player_count):
            start = player * bucket_count
            stop = (player + 1) * bucket_count
            inputs[:, start:stop].copy_(
                ranges_by_player[:, player, 0:bucket_count]
            )
        calculator.nn.get_value(inputs, serialized_values)
        values.mul_(value_normalization.view(state_count, player_count, 1))
        cfv_memory.add_(values)

    range_memory[torch.eq(range_memory, 0)] = 1
    cfv_memory.view(-1, bucket_count).div_(
        range_memory.expand(state_count * player_count, bucket_count)
    )
    card_values = trajectories.new_empty(
        state_count, player_count, hand_count
    )
    calculator._bucket_value_to_card_value_on_board(
        board,
        cfv_memory.view(state_count * player_count, bucket_count),
        card_values.view(state_count * player_count, hand_count),
    )
    return card_values.view(
        pot_count, batch_size, player_count, hand_count
    )


class PreflopTrajectoryNumericsTest(unittest.TestCase):
    def _calculator(self):
        calculator = NextRoundValuePre.__new__(NextRoundValuePre)
        calculator.bucket_count = 3
        calculator.board_count = 2
        calculator.nn = RowIndependentValueNet()
        calculator.board_indexes_scatter = torch.tensor(
            [[0, 1, 2, 3], [2, 1, 0, 3]], dtype=torch.long
        )
        calculator.board_indexes = torch.tensor(
            [[0, 1, 2, 0], [2, 1, 0, 0]], dtype=torch.long
        )
        calculator.impossible_mask = torch.tensor(
            [[False, False, False, True], [False, False, False, True]]
        )
        return calculator

    def test_all_action_rows_match_full_legacy_batch_exactly(self):
        calculator = self._calculator()
        iterations = 3
        pot_count = 10
        hand_count = 4
        full_trajectories = (
            torch.arange(
                iterations
                * pot_count
                * constants.players_count
                * hand_count,
                dtype=torch.float32,
            )
            .view(iterations, pot_count, 1, constants.players_count, hand_count)
            .remainder(17)
            .add(1)
        )
        full_pot_sizes = torch.tensor(
            [100, 300, 300, 900, 900, 2700, 2700, 8100, 8100, 20000],
            dtype=torch.float32,
        )
        action_indices = torch.tensor([0, 1, 2], dtype=torch.long)
        captured = full_trajectories.index_select(1, action_indices)
        board = torch.tensor([1.0, 2.0, 3.0])

        with (
            mock.patch.object(arguments, "cfr_iters", 6),
            mock.patch.object(arguments, "cfr_skip_iters", 3),
            mock.patch.object(
                next_round_value_pre_module.game_settings, "hand_count", 4
            ),
            mock.patch.object(
                next_round_value_pre_module.card_tools,
                "get_flop_board_index",
                return_value=1,
            ),
        ):
            legacy = legacy_full_board_values(
                calculator, board, full_trajectories, full_pot_sizes
            )
            calculator.nn.batch_sizes.clear()
            optimized = calculator.get_value_on_board_from_trajectories(
                board, captured, full_pot_sizes, action_indices
            )

        self.assertTrue(torch.equal(optimized, legacy[0:3]))
        self.assertEqual(calculator.nn.batch_sizes, [pot_count] * iterations)

        lookahead = Lookahead.__new__(Lookahead)
        lookahead.next_street_boxes = calculator
        lookahead.preflop_next_street_inputs = captured
        lookahead.preflop_next_street_action_indices = action_indices
        lookahead.preflop_next_street_action_slots = {-1: 0, 300: 1, 20000: 2}
        lookahead.preflop_next_street_input_count = iterations
        lookahead.next_round_pot_sizes = full_pot_sizes
        lookahead.tree = SimpleNamespace(current_player=constants.Players.P1)
        with (
            mock.patch.object(arguments, "cfr_iters", 6),
            mock.patch.object(arguments, "cfr_skip_iters", 3),
            mock.patch.object(
                next_round_value_pre_module.game_settings, "hand_count", 4
            ),
            mock.patch.object(
                next_round_value_pre_module.card_tools,
                "get_flop_board_index",
                return_value=1,
            ),
        ):
            for slot, action in enumerate((-1, 300, 20000)):
                result = lookahead.get_chance_action_cfv(action, board)
                expected = legacy[slot, 0, 0] * full_pot_sizes[slot]
                self.assertTrue(torch.equal(result, expected))

            lookahead.tree.current_player = constants.Players.P2
            for slot, action in enumerate((-1, 300, 20000)):
                result = lookahead.get_chance_action_cfv(action, board)
                # Board-specific memory is accumulated before the ordinary
                # per-iteration output swap, so final chance CFVs keep the raw
                # NN player ordering and P2 selects raw player 1.
                expected = legacy[slot, 0, 1] * full_pot_sizes[slot]
                self.assertTrue(torch.equal(result, expected))


class PreflopTrajectoryLifecycleTest(unittest.TestCase):
    @staticmethod
    def _lookahead(shared_box, inputs, hand_count):
        lookahead = Lookahead.__new__(Lookahead)
        lookahead.tree = SimpleNamespace(street=1)
        lookahead.next_street_boxes = shared_box
        lookahead.next_street_boxes_inputs = inputs
        lookahead.batch_size = 1
        lookahead.action_to_index = {-1: 0, 300: 1, 20000: 2}
        lookahead.num_pot_sizes = 10
        lookahead.next_board_idx = None
        lookahead.preflop_next_street_inputs = None
        lookahead.preflop_next_street_action_indices = None
        lookahead.preflop_next_street_action_slots = {}
        lookahead.preflop_next_street_input_count = 0
        return lookahead

    def test_capture_survives_shared_singleton_reuse(self):
        shared_box = SimpleNamespace(iter=2)
        first_inputs = torch.arange(80, dtype=torch.float32).view(10, 1, 2, 4)
        second_inputs = first_inputs.add(1000)

        with (
            mock.patch.object(arguments, "cfr_iters", 4),
            mock.patch.object(arguments, "cfr_skip_iters", 2),
            mock.patch.object(
                next_round_value_pre_module.game_settings, "hand_count", 4
            ),
        ):
            cached_root = self._lookahead(shared_box, first_inputs, 4)
            cached_root._prepare_preflop_next_street_inputs()
            cached_root._capture_preflop_next_street_inputs()
            shared_box.iter = 3
            cached_root.next_street_boxes_inputs.add_(100)
            cached_root._capture_preflop_next_street_inputs()
            cached_snapshot = cached_root.preflop_next_street_inputs.clone()
            cached_pointer = cached_root.preflop_next_street_inputs.data_ptr()

            shared_box.iter = 2
            fresh_resolve = self._lookahead(shared_box, second_inputs, 4)
            fresh_resolve._prepare_preflop_next_street_inputs()
            fresh_resolve._capture_preflop_next_street_inputs()
            shared_box.iter = 3
            fresh_resolve.next_street_boxes_inputs.add_(100)
            fresh_resolve._capture_preflop_next_street_inputs()

        self.assertTrue(
            torch.equal(cached_root.preflop_next_street_inputs, cached_snapshot)
        )
        self.assertNotEqual(
            fresh_resolve.preflop_next_street_inputs.data_ptr(), cached_pointer
        )
        self.assertTrue(cached_root.has_captured_preflop_inputs(-1))
        self.assertTrue(fresh_resolve.has_captured_preflop_inputs(20000))

    def test_standard_root_capture_has_strict_memory_bound(self):
        hand_count = next_round_value_pre_module.game_settings.hand_count
        inputs = torch.empty(10, 1, constants.players_count, hand_count)
        lookahead = self._lookahead(SimpleNamespace(iter=0), inputs, hand_count)

        with (
            mock.patch.object(arguments, "cfr_iters", 1000),
            mock.patch.object(arguments, "cfr_skip_iters", 500),
        ):
            lookahead._prepare_preflop_next_street_inputs()

        expected_bytes = 500 * 3 * 1 * 2 * 1326 * 4
        naive_full_transition_bytes = 500 * 10 * 1 * 2 * 1326 * 4
        self.assertEqual(lookahead.get_preflop_capture_bytes(), expected_bytes)
        self.assertEqual(expected_bytes, 15_912_000)
        self.assertLess(expected_bytes, 16 * 1024 * 1024)
        self.assertLess(expected_bytes, naive_full_transition_bytes / 3)


class LookaheadResetInitializationTest(unittest.TestCase):
    def test_reset_restores_fresh_constructor_state_exactly(self):
        class TransitionBox:
            def __init__(self):
                self.iter = 91
                self.start_calls = []

            def start_computation(self, pot_sizes, batch_size):
                self.iter = 0
                self.start_calls.append((pot_sizes.clone(), batch_size))

        zeros = lambda: torch.zeros(2, dtype=torch.float32)
        epsilon = 1.0 / 1_000_000_000
        uniform = 1.0 / 4
        state = SimpleNamespace(
            depth=4,
            batch_size=1,
            regret_epsilon=epsilon,
            ranges_data={
                1: torch.full((2,), uniform),
                2: torch.full((2,), uniform),
                3: zeros(),
                4: zeros(),
            },
            average_strategies_data={
                1: None,
                2: zeros(),
                3: zeros(),
                4: zeros(),
            },
            current_strategy_data={
                1: None,
                2: zeros(),
                3: zeros(),
                4: zeros(),
            },
            cfvs_data={d: zeros() for d in range(1, 5)},
            average_cfvs_data={1: zeros(), 2: zeros()},
            regrets_data={
                1: None,
                2: zeros(),
                3: torch.full((2,), epsilon),
                4: torch.full((2,), epsilon),
            },
            current_regrets_data={
                1: None,
                2: zeros(),
                3: zeros(),
                4: zeros(),
            },
            positive_regrets_data={
                1: None,
                2: zeros(),
                3: torch.full((2,), epsilon),
                4: torch.full((2,), epsilon),
            },
            placeholder_data={d: zeros() for d in range(1, 5)},
            regrets_sum={d: zeros() for d in range(1, 5)},
            inner_nodes={d: zeros() for d in range(1, 4)},
            inner_nodes_p1={d: zeros() for d in range(1, 4)},
            swap_data={d: zeros() for d in range(1, 4)},
            next_street_boxes=TransitionBox(),
            next_round_pot_sizes=torch.tensor([100.0, 300.0, 20_000.0]),
            next_street_boxes_inputs=torch.zeros(3, 1, 2, 4),
            next_street_boxes_outputs=torch.zeros(3, 1, 2, 4),
        )
        reset_dict_names = (
            "ranges_data",
            "average_strategies_data",
            "current_strategy_data",
            "cfvs_data",
            "average_cfvs_data",
            "regrets_data",
            "current_regrets_data",
            "positive_regrets_data",
            "placeholder_data",
            "regrets_sum",
            "inner_nodes",
            "inner_nodes_p1",
            "swap_data",
        )
        fresh_state = {
            name: {
                depth: None if value is None else value.clone()
                for depth, value in getattr(state, name).items()
            }
            for name in reset_dict_names
        }

        for name in reset_dict_names:
            for value in getattr(state, name).values():
                if value is not None:
                    value.fill_(37)
        state.next_street_boxes_inputs.fill_(37)
        state.next_street_boxes_outputs.fill_(37)

        with mock.patch.object(
            next_round_value_pre_module.game_settings, "hand_count", 4
        ):
            LookaheadBuilder(state).reset()

        for name, expected_by_depth in fresh_state.items():
            for depth, expected in expected_by_depth.items():
                actual = getattr(state, name)[depth]
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertTrue(
                        torch.equal(actual, expected),
                        msg=f"{name}[{depth}] differs after reset",
                    )
        self.assertEqual(state.next_street_boxes.iter, 0)
        self.assertEqual(len(state.next_street_boxes.start_calls), 1)
        self.assertTrue(torch.count_nonzero(state.next_street_boxes_inputs) == 0)
        self.assertTrue(torch.count_nonzero(state.next_street_boxes_outputs) == 0)


class ResolvingCapturedFlopTest(unittest.TestCase):
    def test_fresh_and_cached_preflop_resolves_skip_cfr_replay(self):
        board = torch.tensor([1.0, 2.0, 3.0])
        for opponent_cfvs in (None, torch.tensor([0.25, -0.25])):
            with self.subTest(cached_root=opponent_cfvs is None):
                resolver = Resolving.__new__(Resolving)
                resolver.terminal_equity = mock.Mock()
                resolver.lookahead = mock.Mock()
                resolver.lookahead.has_captured_preflop_inputs.return_value = True
                resolver.lookahead.get_chance_action_cfv.return_value = (
                    torch.tensor([3.0, 4.0])
                )
                resolver.player_range = torch.tensor([0.4, 0.6])
                resolver.opponent_range = torch.tensor([0.5, 0.5])
                resolver.opponent_cfvs = opponent_cfvs

                with mock.patch.object(
                    resolving_module.card_tools,
                    "get_flop_board_index",
                    side_effect=AssertionError("captured path replayed CFR"),
                ):
                    result = resolver.get_chance_action_cfv(-1, board)

                torch.testing.assert_close(result, torch.tensor([3.0, 4.0]))
                resolver.terminal_equity.set_board.assert_not_called()
                resolver.lookahead.reset.assert_not_called()
                resolver.lookahead.resolve.assert_not_called()
                resolver.lookahead.resolve_first_node.assert_not_called()
                self.assertFalse(resolver.last_chance_timing["replayed_flop"])
                self.assertTrue(resolver.last_chance_timing["captured_flop"])


if __name__ == "__main__":
    unittest.main()
