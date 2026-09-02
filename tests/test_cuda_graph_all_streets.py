#!/usr/bin/env python3
"""CUDA Graph support on every street: iteration accounting and eligibility."""

from pathlib import Path
import sys
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
    import settings.game_settings as game_settings  # noqa: E402
    from lookahead.lookahead import Lookahead  # noqa: E402
    from nn.next_round_value import NextRoundValue  # noqa: E402


class IdentityValueNet:
    calls = 0

    @classmethod
    def get_value(cls, inputs, output):
        cls.calls += 1
        output.copy_(inputs[:, 0:-1])


def make_next_round_value():
    def initialize(instance, _board):
        instance._street = 2
        instance.bucket_count = 3
        instance.board_count = 2
        instance._range_matrix = torch.arange(24, dtype=torch.float32).view(4, 6).div_(24.0)
        instance._range_matrix_board_view = instance._range_matrix.view(4, 2, 3)
        instance._reverse_value_matrix = instance._range_matrix.t().clone().mul_(0.25)

    with (
        mock.patch.object(NextRoundValue, "_bucketing_transform_cache_key", return_value=("iteration-test",)),
        mock.patch.object(NextRoundValue, "_init_bucketing", autospec=True, side_effect=initialize),
        mock.patch.object(arguments, "Tensor", torch.FloatTensor),
        mock.patch.object(game_settings, "hand_count", 4),
    ):
        nrv = NextRoundValue(IdentityValueNet, torch.tensor([1.0, 2.0, 3.0]))
        nrv.start_computation(torch.tensor([400.0]), 1)
    return nrv


class ExplicitIterationTest(unittest.TestCase):
    def tearDown(self):
        NextRoundValue._clear_bucketing_transform_cache()

    def run_iterations(self, iterations, explicit):
        torch.manual_seed(0)
        ranges = torch.rand(1, 2, 4)
        values = torch.zeros(1, 2, 4)
        nrv = make_next_round_value()
        with (
            mock.patch.object(arguments, "Tensor", torch.FloatTensor),
            mock.patch.object(arguments, "cfr_iters", 6),
            mock.patch.object(arguments, "cfr_skip_iters", 3),
            mock.patch.object(game_settings, "hand_count", 4),
        ):
            for iteration in iterations:
                if explicit:
                    nrv.get_value(ranges, values, iteration=iteration)
                else:
                    nrv.get_value(ranges, values)
            memory = nrv.counterfactual_value_memory.clone()
            normalization = nrv.range_normalization_memory.clone()
        return nrv, values.clone(), memory, normalization

    def test_explicit_iteration_matches_implicit_counting(self):
        implicit = self.run_iterations(range(1, 7), explicit=False)
        explicit = self.run_iterations(range(1, 7), explicit=True)
        self.assertEqual(implicit[0].iter, 6)
        self.assertEqual(explicit[0].iter, 6)
        self.assertTrue(torch.equal(implicit[1], explicit[1]))
        self.assertTrue(torch.equal(implicit[2], explicit[2]))
        self.assertTrue(torch.equal(implicit[3], explicit[3]))

    def test_repeated_representative_iteration_keeps_accumulating(self):
        # A CUDA Graph warmup replays the first averaging iteration three times;
        # the memory buffers must persist and keep accumulating across calls.
        nrv, _, memory, normalization = self.run_iterations([1, 4, 4, 4], explicit=True)
        _, _, memory_once, normalization_once = self.run_iterations([1, 4], explicit=True)
        self.assertTrue(torch.allclose(memory, memory_once * 3))
        self.assertTrue(torch.allclose(normalization, normalization_once * 3))

    def test_start_computation_resets_per_solve_buffers(self):
        nrv, *_ = self.run_iterations([1, 4], explicit=True)
        first = nrv.counterfactual_value_memory
        with (
            mock.patch.object(arguments, "Tensor", torch.FloatTensor),
            mock.patch.object(game_settings, "hand_count", 4),
        ):
            nrv.start_computation(torch.tensor([400.0]), 1)
        self.assertIsNone(nrv.counterfactual_value_memory)
        self.assertIsNone(nrv.next_round_inputs)
        self.assertIsNot(first, nrv.counterfactual_value_memory)
        self.assertTrue(NextRoundValue.supports_explicit_iteration)


class GraphEligibilityTest(unittest.TestCase):
    def _lookahead(self, boxes):
        lookahead = Lookahead.__new__(Lookahead)
        device = torch.device("cuda")
        lookahead.tree = SimpleNamespace(street=2)
        lookahead.next_street_boxes = boxes
        lookahead.ranges_data = {1: SimpleNamespace(device=device)}
        lookahead.terminal_equity = SimpleNamespace(
            equity_matrix=SimpleNamespace(device=device),
            fold_matrix=SimpleNamespace(device=device),
        )
        return lookahead

    def eligibility(self, boxes):
        lookahead = self._lookahead(boxes)
        with (
            mock.patch.object(arguments, "use_gpu", True),
            mock.patch.object(arguments, "cfr_iters", 1000),
            mock.patch.object(arguments, "cfr_skip_iters", 500),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False, create=True),
        ):
            return lookahead._cuda_graph_ineligibility()

    def test_flop_tree_with_capable_box_is_eligible(self):
        self.assertIsNone(self.eligibility(SimpleNamespace(supports_explicit_iteration=True, iter=0)))

    def test_box_without_iteration_api_is_rejected(self):
        self.assertEqual(self.eligibility(SimpleNamespace(iter=0)), "next-street-box-lacks-iteration-api")

    def test_river_tree_without_box_is_still_eligible(self):
        self.assertIsNone(self.eligibility(None))


class GraphRunnerAccountingTest(unittest.TestCase):
    def test_replays_drive_trajectory_capture_with_true_iterations(self):
        state = {"capturing": False, "compute": [], "captures": []}

        class FakeStream:
            def wait_stream(self, _other):
                pass

            def synchronize(self):
                pass

        stream = FakeStream()

        class FakeGraph:
            def replay(self):
                state["compute"].append("replay")

        class Context:
            def __init__(self, on_enter=None, on_exit=None):
                self.on_enter = on_enter
                self.on_exit = on_exit

            def __enter__(self):
                if self.on_enter:
                    self.on_enter()
                return self

            def __exit__(self, *_):
                if self.on_exit:
                    self.on_exit()
                return False

        lookahead = Lookahead.__new__(Lookahead)
        lookahead.ranges_data = {1: SimpleNamespace(device=torch.device("cuda"))}
        lookahead._cuda_graph_handles = []
        lookahead._cuda_graph_stream = None
        lookahead.cuda_graph_telemetry = {}
        lookahead.next_street_boxes = SimpleNamespace(supports_explicit_iteration=True, iter=0)

        def compute(representative, capture_iteration=None):
            state["compute"].append(("record" if state["capturing"] else "eager", representative, capture_iteration))

        lookahead._compute_iteration = compute
        lookahead._capture_preflop_next_street_inputs = lambda iteration: state["captures"].append(iteration)

        def set_capturing(value):
            state["capturing"] = value

        plan = Lookahead._cuda_graph_phase_plan(10, 5, 2)
        with (
            mock.patch.object(torch.cuda, "current_stream", return_value=stream),
            mock.patch.object(torch.cuda, "Stream", return_value=stream),
            mock.patch.object(torch.cuda, "stream", return_value=Context()),
            mock.patch.object(torch.cuda, "CUDAGraph", side_effect=FakeGraph),
            mock.patch.object(
                torch.cuda,
                "graph",
                side_effect=lambda graph, **_: Context(lambda: set_capturing(True), lambda: set_capturing(False)),
            ),
        ):
            lookahead._compute_with_cuda_graphs(plan)

        eager = [item for item in state["compute"] if item[0] == "eager"]
        self.assertEqual(eager, [("eager", 1, 1), ("eager", 1, 2), ("eager", 6, 6), ("eager", 6, 7)])
        records = [item for item in state["compute"] if item[0] == "record"]
        self.assertEqual(records, [("record", 1, 0), ("record", 6, 0)])
        self.assertEqual(state["compute"].count("replay"), 6)
        # Burn-in replays hand iterations 3..5 to the capture helper, which ignores
        # them; averaging replays hand 8..10 so every averaging slot is written.
        self.assertEqual(state["captures"], [3, 4, 5, 8, 9, 10])

        with mock.patch.object(arguments, "cfr_iters", 10):
            lookahead._finalize_iteration_accounting()
        self.assertEqual(lookahead.next_street_boxes.iter, 10)


class TrajectoryCaptureTest(unittest.TestCase):
    def _lookahead(self, averaging=3, actions=2):
        lookahead = Lookahead.__new__(Lookahead)
        lookahead.preflop_next_street_action_indices = torch.tensor([0, 2])[:actions]
        lookahead.preflop_next_street_inputs = torch.zeros(averaging, actions, 1, 2, 4)
        lookahead.preflop_next_street_action_slots = {}
        lookahead.preflop_next_street_input_count = 0
        lookahead.next_board_idx = None
        lookahead.next_street_boxes_inputs = torch.arange(3 * 1 * 2 * 4, dtype=torch.float32).view(3, 1, 2, 4)
        return lookahead

    def test_capture_uses_true_iteration_numbers(self):
        lookahead = self._lookahead()
        with mock.patch.object(arguments, "cfr_skip_iters", 5):
            for iteration in (1, 5):
                lookahead._capture_preflop_next_street_inputs(iteration)
            self.assertEqual(lookahead.preflop_next_street_input_count, 0)
            lookahead.next_street_boxes_inputs.add_(100)
            lookahead._capture_preflop_next_street_inputs(6)
            lookahead._capture_preflop_next_street_inputs(8)  # non-sequential: ignored
            lookahead.next_street_boxes_inputs.add_(100)
            lookahead._capture_preflop_next_street_inputs(7)
            lookahead._capture_preflop_next_street_inputs(8)
            lookahead._capture_preflop_next_street_inputs(9)  # beyond the buffer: ignored
        self.assertEqual(lookahead.preflop_next_street_input_count, 3)
        self.assertTrue(torch.equal(lookahead.preflop_next_street_inputs[0, 0, 0, 0], torch.tensor([100.0, 101.0, 102.0, 103.0])))
        self.assertTrue(torch.equal(lookahead.preflop_next_street_inputs[1, 1, 0, 0], torch.tensor([216.0, 217.0, 218.0, 219.0])))
        self.assertTrue(torch.equal(lookahead.preflop_next_street_inputs[2, 0, 0, 0], torch.tensor([200.0, 201.0, 202.0, 203.0])))

    def test_missing_capture_state_is_ignored(self):
        lookahead = Lookahead.__new__(Lookahead)
        lookahead._capture_preflop_next_street_inputs(7)  # no attributes at all: no-op


if __name__ == "__main__":
    unittest.main()
