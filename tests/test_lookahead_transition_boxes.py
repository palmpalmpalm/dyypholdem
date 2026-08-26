#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys
import unittest

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import settings.arguments as arguments  # noqa: E402
from lookahead.lookahead_builder import LookaheadBuilder  # noqa: E402
from server import protocol_to_node  # noqa: E402
from tree.tree_builder import PokerTreeBuilder  # noqa: E402
from tree.tree_node import BuildTreeParams  # noqa: E402


class LookaheadTransitionBoxesTest(unittest.TestCase):
    def test_all_terminal_preflop_response_does_not_load_value_networks(self):
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            state = protocol_to_node.parse_state(
                "MATCHSTATE:1:32:cr20000:|4d9h"
            )
            node = protocol_to_node.parsed_state_to_node(state)
            tree = PokerTreeBuilder().build_tree(
                BuildTreeParams(root_node=node, limit_to_street=True)
            )

            lookahead = SimpleNamespace(
                batch_size=1,
                terminal_equity=SimpleNamespace(board=node.board),
            )
            with mock.patch(
                "lookahead.lookahead_builder.ValueNn.load_for_street",
                side_effect=AssertionError("terminal tree loaded a value network"),
            ), mock.patch(
                "lookahead.lookahead_builder.NextRoundValuePre",
                side_effect=AssertionError("terminal tree built preflop buckets"),
            ):
                LookaheadBuilder(lookahead).build_from_tree(tree)

        self.assertEqual(lookahead.num_pot_sizes, 0)
        self.assertIsNone(lookahead.next_street_boxes)
        self.assertEqual(lookahead.indices, {})


if __name__ == "__main__":
    unittest.main()
