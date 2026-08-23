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
import settings.constants as constants  # noqa: E402
from lookahead.lookahead_builder import LookaheadBuilder  # noqa: E402
from server import protocol_to_node  # noqa: E402
from tree.tree_builder import PokerTreeBuilder  # noqa: E402
from tree.tree_node import BuildTreeParams  # noqa: E402


class ProtocolToNodeTest(unittest.TestCase):
    def _build_cpu_tree(self, message):
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            state = protocol_to_node.parse_state(message)
            node = protocol_to_node.parsed_state_to_node(state)
            tree = PokerTreeBuilder().build_tree(
                BuildTreeParams(root_node=node, limit_to_street=True)
            )
        return state, node, tree

    def test_initial_blinds_keep_preflop_check_option(self):
        _, node, tree = self._build_cpu_tree("MATCHSTATE:1:0::|4d9h")

        self.assertEqual(node.num_bets, 1)
        self.assertFalse(tree.children[1].terminal)
        self.assertEqual(tree.children[1].current_player, constants.Players.P2)

    def test_facing_preflop_all_in_has_only_terminal_fold_or_call(self):
        _, node, tree = self._build_cpu_tree(
            "MATCHSTATE:1:32:cr20000:|4d9h"
        )

        self.assertEqual(node.bets.tolist(), [100.0, 20000.0])
        self.assertEqual(node.num_bets, 0)
        self.assertEqual(tree.actions.tolist(), [-2.0, -1.0])
        self.assertEqual(len(tree.children), 2)
        self.assertTrue(all(child.terminal for child in tree.children))

        lookahead = SimpleNamespace(batch_size=1)
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor), \
                mock.patch.object(
                    LookaheadBuilder,
                    "_construct_transition_boxes",
                    lambda _self: None,
                ):
            LookaheadBuilder(lookahead).build_from_tree(tree)

        self.assertEqual(lookahead.actions_count[1], 2)
        self.assertEqual(lookahead.terminal_actions_count[1], 2)
        self.assertEqual(lookahead.nonallinbets_count[1], 0)
        self.assertEqual(tuple(lookahead.ranges_data[2].shape), (2, 1, 1, 1, 2, 1326))

    def test_tree_fails_safe_if_all_in_node_has_stale_bet_metadata(self):
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            state = protocol_to_node.parse_state(
                "MATCHSTATE:1:32:cr20000:|4d9h"
            )
            node = protocol_to_node.parsed_state_to_node(state)
            node.num_bets = 1
            tree = PokerTreeBuilder().build_tree(
                BuildTreeParams(root_node=node, limit_to_street=True)
            )

        self.assertEqual(len(tree.children), 2)
        self.assertTrue(all(child.terminal for child in tree.children))


if __name__ == "__main__":
    unittest.main()
