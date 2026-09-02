#!/usr/bin/env python3

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import settings.arguments as arguments  # noqa: E402
import settings.constants as constants  # noqa: E402
import settings.game_settings as game_settings  # noqa: E402
from game.bet_sizing import BetSizing  # noqa: E402
from server import protocol_to_node  # noqa: E402
from tree.tree_builder import PokerTreeBuilder  # noqa: E402
from tree.tree_node import BuildTreeParams  # noqa: E402


def flop_start_node(player):
    return SimpleNamespace(
        current_player=player,
        bets=torch.tensor([100.0, 100.0]),
        num_bets=0,
    )


class BetSizingMenuTest(unittest.TestCase):
    def test_opponent_menu_only_applies_when_requested(self):
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            sizing = BetSizing([[1], [1], [1]], [[0.5, 1, 2], [1], [1]])
            own = sizing.get_possible_bets(flop_start_node(constants.Players.P1))
            opponent = sizing.get_possible_bets(flop_start_node(constants.Players.P1), opponent_menu=True)
        self.assertEqual(own[:, 0].tolist(), [300.0, 20000.0])
        self.assertEqual(opponent[:, 0].tolist(), [200.0, 300.0, 500.0, 20000.0])

    def test_without_opponent_menu_both_players_share_the_default(self):
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            sizing = BetSizing([[1], [1], [1]])
            own = sizing.get_possible_bets(flop_start_node(constants.Players.P2))
            opponent = sizing.get_possible_bets(flop_start_node(constants.Players.P2), opponent_menu=True)
        self.assertEqual(own.tolist(), opponent.tolist())

    def test_env_parser(self):
        parse = game_settings._parse_opponent_bet_sizing
        self.assertIsNone(parse(None))
        self.assertIsNone(parse("  "))
        self.assertEqual(parse("0.5,1,2"), [[0.5, 1.0, 2.0], [1], [1]])
        for bad in ("half", "0,1", "2,1", "1,1", ","):
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    parse(bad)


class TreeBuilderOpponentMenuTest(unittest.TestCase):
    def build(self, message, opponent_menu):
        with (
            mock.patch.object(arguments, "Tensor", torch.FloatTensor),
            mock.patch.object(game_settings, "opponent_bet_sizing", opponent_menu),
        ):
            state = protocol_to_node.parse_state(message)
            node = protocol_to_node.parsed_state_to_node(state)
            tree = PokerTreeBuilder().build_tree(BuildTreeParams(root_node=node, limit_to_street=True))
        return tree

    def test_only_opponent_nodes_widen(self):
        # Big blind (ACPC position 0) acts first on the flop with 100 chips each in.
        tree = self.build("MATCHSTATE:0:0:cc/:Ad6h|/Ah7d2c", [[0.5, 1, 2], [1], [1]])
        root_children = [child.bets.tolist() for child in tree.children]
        self.assertEqual(len(tree.children), 4)  # fold, check, pot, all-in
        check_node = tree.children[1]
        self.assertNotEqual(check_node.current_player, tree.current_player)
        self.assertEqual(len(check_node.children), 6)  # fold, check, half, pot, double, all-in
        bets = [max(child.bets.tolist()) for child in check_node.children[2:]]
        self.assertEqual(bets, [200.0, 300.0, 500.0, 20000.0])
        pot_bet_node = tree.children[2]
        self.assertEqual(len(pot_bet_node.children), 4)  # opponent raises stay pot-only: fold, call, raise, all-in
        our_reraise_node = pot_bet_node.children[2]
        self.assertEqual(our_reraise_node.current_player, tree.current_player)
        self.assertLessEqual(len(our_reraise_node.children), 4)
        self.assertEqual(root_children[2], [100.0, 300.0][::-1] if tree.current_player == constants.Players.P1 else [100.0, 300.0])

    def test_default_tree_is_unchanged_without_the_menu(self):
        tree = self.build("MATCHSTATE:0:0:cc/:Ad6h|/Ah7d2c", None)
        self.assertEqual(len(tree.children), 4)
        self.assertEqual(len(tree.children[1].children), 4)


if __name__ == "__main__":
    unittest.main()
