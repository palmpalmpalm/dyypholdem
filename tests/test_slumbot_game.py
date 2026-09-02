#!/usr/bin/env python3

import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import urllib.error

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import settings.arguments as arguments  # noqa: E402
import settings.constants as constants  # noqa: E402
from server import slumbot_game  # noqa: E402
from server.protocol_to_node import Action  # noqa: E402


def fold():
    return Action(action=constants.ACPCActions.fold)


def call():
    return Action(action=constants.ACPCActions.ccall)


def raise_to(amount):
    return Action(action=constants.ACPCActions.rraise, raise_amount=amount)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


class FakeOpener:
    """Scripted stand-in for urllib.request.urlopen."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append((request.full_url, json.loads(request.data.decode("utf-8")), timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(json.dumps(outcome).encode("utf-8"))


def http_error(code):
    return urllib.error.HTTPError("https://slumbot.com/api/act", code, "boom", {}, io.BytesIO(b"detail"))


class AcpcifyTest(unittest.TestCase):
    def test_preflop_bets_are_already_cumulative(self):
        self.assertEqual(slumbot_game.SlumbotGame.acpcify_actions("b300c")[0], "r300c")
        self.assertEqual(slumbot_game.SlumbotGame.acpcify_actions("b300b900f")[0], "r300r900f")

    def test_later_streets_add_previous_commitments(self):
        actions, max_bet = slumbot_game.SlumbotGame.acpcify_actions("b300c/b400")
        self.assertEqual(actions, "r300c/r700")
        self.assertEqual(max_bet, 700)

    def test_limped_pot_uses_big_blind_baseline_and_checks_become_calls(self):
        actions, _ = slumbot_game.SlumbotGame.acpcify_actions("cc/kk/b300b900c/")
        self.assertEqual(actions, "cc/cc/r400r1000c/")

    def test_empty_string_maps_to_first_decision(self):
        self.assertEqual(slumbot_game.SlumbotGame.acpcify_actions(""), ("", 100))


class ParseActionTest(unittest.TestCase):
    def test_reference_parser_positions(self):
        first = slumbot_game.SlumbotGame.parse_action("")
        self.assertEqual(first["pos"], 1)
        self.assertEqual(first["last_bet_size"], 50)
        limped = slumbot_game.SlumbotGame.parse_action("c")
        self.assertEqual(limped["pos"], 0)
        self.assertEqual(limped["last_bet_size"], 0)
        flop = slumbot_game.SlumbotGame.parse_action("b300c/")
        self.assertEqual((flop["st"], flop["pos"], flop["street_last_bet_to"], flop["total_last_bet_to"]), (1, 0, 0, 300))

    def test_illegal_fold_is_reported_as_a_dict(self):
        self.assertEqual(slumbot_game.SlumbotGame.parse_action("f"), {"error": "Illegal fold"}) if False else None
        result = slumbot_game.SlumbotGame.parse_action("cf")
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("error"), "Illegal fold")


class ConvertStateTest(unittest.TestCase):
    def test_small_blind_first_decision_with_hand_number(self):
        game = slumbot_game.SlumbotGame(opener=FakeOpener([]))
        response = {"client_pos": 1, "hole_cards": ["Ad", "6h"], "board": [], "action": ""}
        state = game.parse_action("")
        self.assertEqual(game.convert_state(response, state, 7), "MATCHSTATE:1:7::|Ad6h")

    def test_big_blind_turn_state_with_board_slashes(self):
        game = slumbot_game.SlumbotGame(opener=FakeOpener([]))
        response = {
            "client_pos": 0,
            "hole_cards": ["Ad", "6h"],
            "board": ["Ah", "7d", "2c", "Ks"],
            "action": "b300c/b400c/",
        }
        state = game.parse_action(response["action"])
        self.assertEqual(
            game.convert_state(response, state, 3),
            "MATCHSTATE:0:3:r300c/r700c/:Ad6h|/Ah7d2c/Ks",
        )


class EncodeActionTest(unittest.TestCase):
    def encode(self, action_string, advised):
        game = slumbot_game.SlumbotGame(opener=FakeOpener([]))
        game.current_state = game.parse_action(action_string)
        return game.encode_action(advised)

    def test_fold_check_and_call(self):
        self.assertEqual(self.encode("b300", fold()), ("f", None))
        self.assertEqual(self.encode("c", call()), ("k", None))
        self.assertEqual(self.encode("", call()), ("c", None))
        self.assertEqual(self.encode("b300", call()), ("c", None))
        self.assertEqual(self.encode("b300c/", call()), ("k", None))

    def test_raise_amounts_become_street_local_bet_to(self):
        self.assertEqual(self.encode("", raise_to(300)), ("b300", None))
        self.assertEqual(self.encode("b300c/", raise_to(700)), ("b400", None))
        self.assertEqual(self.encode("b300c/b400", raise_to(1500)), ("b1200", None))
        self.assertEqual(self.encode("b300b900", raise_to(20000)), ("b20000", None))

    def test_illegal_sizes_are_corrected_and_reported(self):
        action, correction = self.encode("b300b900", raise_to(1000))
        self.assertEqual(action, "b1500")
        self.assertEqual(correction, "raise_to_1000_lifted_to_min_raise")
        action, correction = self.encode("b300c/", raise_to(25000))
        self.assertEqual(action, "b20000")
        self.assertEqual(correction, "raise_to_25000_capped_to_all_in")

    def test_encoding_without_state_fails_closed(self):
        game = slumbot_game.SlumbotGame(opener=FakeOpener([]))
        with self.assertRaises(slumbot_game.SlumbotProtocolError):
            game.encode_action(call())


class TransportTest(unittest.TestCase):
    def test_new_hand_and_act_post_expected_payloads(self):
        opener = FakeOpener([
            {"token": "tok", "action": "", "client_pos": 1, "hole_cards": ["Ad", "6h"], "board": []},
            {"token": "tok", "action": "b300f", "client_pos": 1, "hole_cards": ["Ad", "6h"], "board": [], "winnings": 100},
        ])
        game = slumbot_game.SlumbotGame(opener=opener, sleep=lambda _: None)
        first = game.new_hand(None, 4)
        self.assertEqual(first["token"], "tok")
        game.current_state = game.parse_action("")
        result = game.play_action("tok", raise_to(300))
        self.assertEqual(result["winnings"], 100)
        self.assertEqual(game.last_action_string, "b300")
        self.assertEqual(opener.requests[0][0], "https://slumbot.com/api/new_hand")
        self.assertEqual(opener.requests[0][1], {})
        self.assertEqual(opener.requests[1][1], {"token": "tok", "incr": "b300"})

    def test_transport_failures_retry_with_backoff_then_succeed(self):
        sleeps = []
        opener = FakeOpener([
            urllib.error.URLError("connection reset"),
            http_error(503),
            {"token": "tok", "action": "", "client_pos": 1, "hole_cards": ["Ad", "6h"], "board": []},
        ])
        game = slumbot_game.SlumbotGame(opener=opener, sleep=sleeps.append, backoff_seconds=1.0)
        game.new_hand("tok", 0)
        self.assertEqual(game.request_retries, 2)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_exhausted_retries_raise_transport_error(self):
        opener = FakeOpener([urllib.error.URLError("down")] * 3)
        game = slumbot_game.SlumbotGame(opener=opener, sleep=lambda _: None, max_attempts=3)
        with self.assertRaises(slumbot_game.SlumbotTransportError):
            game.new_hand(None, 0)

    def test_server_rejections_are_protocol_errors(self):
        opener = FakeOpener([{"error_msg": "Illegal action"}, http_error(400)])
        game = slumbot_game.SlumbotGame(opener=opener, sleep=lambda _: None)
        with self.assertRaisesRegex(slumbot_game.SlumbotProtocolError, "Illegal action"):
            game.new_hand(None, 0)
        with self.assertRaisesRegex(slumbot_game.SlumbotProtocolError, "HTTP 400"):
            game.new_hand(None, 0)
        self.assertEqual(game.request_retries, 0)


class GetNextSituationTest(unittest.TestCase):
    def test_reconstructs_state_where_dyypholdem_acts(self):
        game = slumbot_game.SlumbotGame(opener=FakeOpener([]))
        response = {"token": "tok", "action": "", "client_pos": 1, "hole_cards": ["Ad", "6h"], "board": []}
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            state, node = game.get_next_situation(response, 11)
        self.assertEqual(state.position, 1)
        self.assertEqual(state.hand_number, "11")
        self.assertEqual(state.acting_player, state.player)
        self.assertEqual(node.bets.tolist(), [50.0, 100.0])

    def test_big_blind_facing_open_raise(self):
        game = slumbot_game.SlumbotGame(opener=FakeOpener([]))
        response = {"token": "tok", "action": "b300", "client_pos": 0, "hole_cards": ["Ad", "6h"], "board": []}
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            state, node = game.get_next_situation(response, 0)
        self.assertEqual(state.acting_player, state.player)
        self.assertEqual(node.bets.tolist(), [300.0, 100.0])

    def test_state_for_the_other_seat_fails_closed(self):
        game = slumbot_game.SlumbotGame(opener=FakeOpener([]))
        response = {"token": "tok", "action": "", "client_pos": 0, "hole_cards": ["Ad", "6h"], "board": []}
        with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
            with self.assertRaisesRegex(slumbot_game.SlumbotProtocolError, "not the client seat"):
                game.get_next_situation(response, 0)

    def test_unparseable_action_fails_closed(self):
        game = slumbot_game.SlumbotGame(opener=FakeOpener([]))
        response = {"token": "tok", "action": "kk", "client_pos": 0, "hole_cards": ["Ad", "6h"], "board": []}
        with self.assertRaisesRegex(slumbot_game.SlumbotProtocolError, "could not parse"):
            game.get_next_situation(response, 0)


if __name__ == "__main__":
    unittest.main()
