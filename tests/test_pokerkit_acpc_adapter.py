#!/usr/bin/env python3

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from player.pokerkit_acpc_adapter import (  # noqa: E402
    POKERKIT_VERSION,
    PokerKitAcpcAdapter,
    PokerKitStateError,
)
from player.web_acpc_player import HumanBridge, parse_matchstate  # noqa: E402


def pokerkit_075_available():
    try:
        return version("PokerKit") == POKERKIT_VERSION
    except PackageNotFoundError:
        return False


@unittest.skipUnless(pokerkit_075_available(), "PokerKit 0.7.5 test environment required")
class PokerKitAcpcAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = PokerKitAcpcAdapter()

    def test_initial_preflop_actor_commitments_and_bounds(self):
        state = parse_matchstate("MATCHSTATE:1:7::AsKd|")
        pokerkit_state = self.adapter.replay(state)
        self.assertEqual(pokerkit_state.actor_index, 1)
        self.assertEqual(tuple(pokerkit_state.bets), (100, 50))
        legal = self.adapter.legal_actions(state)
        self.assertEqual(legal.call_amount, 50)
        self.assertEqual(legal.min_raise_to, 200)
        self.assertEqual(legal.pot_raise_to, 300)
        self.assertEqual(legal.max_raise_to, 20_000)

    def test_flop_cumulative_conversion_and_fractional_presets(self):
        state = parse_matchstate("MATCHSTATE:0:9:r300c/:|QhQs/2c7dJh")
        pokerkit_state = self.adapter.replay(state)
        self.assertEqual(pokerkit_state.actor_index, 0)
        self.assertEqual(state.hero_player, 1)
        self.assertEqual(state.acting_player, 1)
        self.assertEqual(tuple(pokerkit_state.bets), (0, 0))
        legal = self.adapter.legal_actions(state)
        self.assertTrue(legal.can_check)
        self.assertEqual(legal.min_raise_to, 400)
        self.assertEqual(legal.half_pot_raise_to, 600)
        self.assertEqual(legal.three_quarter_pot_raise_to, 750)
        self.assertEqual(legal.pot_raise_to, 900)
        self.assertEqual(legal.max_raise_to, 20_000)

    def test_flop_facing_bet_minimum_and_arbitrary_slider_raise(self):
        state = parse_matchstate("MATCHSTATE:0:10:r300c/cr500:|QhQs/2c7dJh")
        legal = self.adapter.legal_actions(state)
        self.assertEqual(legal.call_amount, 200)
        self.assertEqual(legal.min_raise_to, 700)
        self.assertFalse(legal.allows_raise_to(699))
        self.assertTrue(legal.allows_raise_to(875))

        with tempfile.TemporaryDirectory() as directory:
            bridge = HumanBridge(
                "unused",
                1,
                Path(directory) / "events.jsonl",
                legality_adapter=self.adapter,
            )
            bridge.current = state
            bridge.current_legality = bridge._legality_for_state(state)
            bridge.status = "your_turn"
            bridge.state_nonce = 3
            self.assertEqual(bridge.queue_action("raise", 3, 875), (True, "r875"))

    def test_short_all_in_is_single_point_and_does_not_reopen(self):
        short_state = parse_matchstate("MATCHSTATE:0:11:r19950:|QhQs")
        legal = self.adapter.legal_actions(short_state)
        self.assertTrue(legal.can_raise)
        self.assertTrue(legal.all_in_only)
        self.assertEqual(legal.nominal_min_raise_to, 39_800)
        self.assertEqual(legal.min_raise_to, 20_000)
        self.assertEqual(legal.max_raise_to, 20_000)
        self.assertFalse(legal.allows_raise_to(19_999))
        self.assertTrue(legal.allows_raise_to(20_000))

        not_reopened = parse_matchstate("MATCHSTATE:1:11:r19950r20000:AsKd|")
        reopened_legal = self.adapter.legal_actions(not_reopened)
        self.assertFalse(reopened_legal.can_raise)
        self.assertEqual(reopened_legal.call_amount, 50)

    def test_illegal_public_raise_fails_closed(self):
        state = parse_matchstate("MATCHSTATE:0:12:r150:|QhQs")
        with self.assertRaises(PokerKitStateError):
            self.adapter.legal_actions(state)


if __name__ == "__main__":
    unittest.main()
