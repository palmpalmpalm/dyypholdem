#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from player.web_acpc_player import HumanBridge, evaluate_seven, parse_matchstate  # noqa: E402


class WebAcpcPlayerTest(unittest.TestCase):
    def test_initial_state_and_pot_action_match_repository_abstraction(self):
        state = parse_matchstate("MATCHSTATE:1:7::AsKd|")
        self.assertEqual(state.current_street, 1)
        self.assertEqual((state.bet1, state.bet2), (50, 100))
        self.assertEqual(state.hero_player, 0)
        self.assertEqual(state.acting_player, state.hero_player)
        with tempfile.TemporaryDirectory() as directory:
            bridge = HumanBridge("unused", 1, Path(directory) / "events.jsonl")
            bridge.current = state
            bridge.status = "your_turn"
            bridge.state_nonce = 4
            actions = {item["id"]: item for item in bridge.available_actions(state)}
            self.assertEqual(actions["pot"]["raise_to"], 300)
            accepted, protocol = bridge.queue_action("pot", 4)
            self.assertTrue(accepted)
            self.assertEqual(protocol, "r300")

    def test_flop_transition_tracks_commitments_and_actor(self):
        state = parse_matchstate("MATCHSTATE:0:9:r300c/:|QhQs/2c7dJh")
        self.assertEqual(state.current_street, 2)
        self.assertEqual((state.bet1, state.bet2), (300, 300))
        self.assertEqual(state.acting_player, 1)
        self.assertEqual(state.hero_player, 1)

    def test_terminal_river_and_hand_evaluation(self):
        state = parse_matchstate("MATCHSTATE:1:11:cc/cc/cc/cc:AsAd|KsKd/2c7dJh/9s/3c")
        self.assertTrue(state.terminal)
        self.assertGreater(
            evaluate_seven(["As", "Ad", "2c", "7d", "Jh", "9s", "3c"]),
            evaluate_seven(["Ks", "Kd", "2c", "7d", "Jh", "9s", "3c"]),
        )

    def test_stale_action_is_rejected(self):
        state = parse_matchstate("MATCHSTATE:1:7::AsKd|")
        with tempfile.TemporaryDirectory() as directory:
            bridge = HumanBridge("unused", 1, Path(directory) / "events.jsonl")
            bridge.current = state
            bridge.status = "your_turn"
            bridge.state_nonce = 8
            accepted, message = bridge.queue_action("call", 7)
            self.assertFalse(accepted)
            self.assertIn("stale", message)


if __name__ == "__main__":
    unittest.main()
