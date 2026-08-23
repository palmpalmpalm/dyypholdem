#!/usr/bin/env python3

from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from player.pokerkit_acpc_adapter import LegalActionState  # noqa: E402
from player.web_acpc_player import (  # noqa: E402
    HumanBridge,
    PokerHTTPServer,
    SESSION_COOKIE,
    evaluate_seven,
    parse_matchstate,
)


DEFAULT_LEGALITY = LegalActionState(
    can_fold=True,
    can_call=True,
    can_raise=True,
    call_amount=50,
    min_raise_to=200,
    nominal_min_raise_to=200,
    half_pot_raise_to=200,
    three_quarter_pot_raise_to=250,
    pot_raise_to=300,
    max_raise_to=20_000,
)


class FixedLegalityAdapter:
    def __init__(self, legality=DEFAULT_LEGALITY):
        self.legality = legality
        self.calls = 0

    def legal_actions(self, _state):
        self.calls += 1
        return self.legality


class WebAcpcPlayerTest(unittest.TestCase):
    def make_bridge(self, directory, state, legality=DEFAULT_LEGALITY, nonce=4):
        adapter = FixedLegalityAdapter(legality)
        bridge = HumanBridge(
            "unused",
            1,
            Path(directory) / "events.jsonl",
            legality_adapter=adapter,
        )
        bridge.current = state
        bridge.current_legality = bridge._legality_for_state(state)
        bridge.status = "your_turn"
        bridge.state_nonce = nonce
        return bridge, adapter

    def test_initial_state_and_server_generated_presets(self):
        state = parse_matchstate("MATCHSTATE:1:7::AsKd|")
        self.assertEqual(state.current_street, 1)
        self.assertEqual((state.bet1, state.bet2), (50, 100))
        self.assertEqual(state.hero_player, 0)
        self.assertEqual(state.acting_player, state.hero_player)
        with tempfile.TemporaryDirectory() as directory:
            bridge, adapter = self.make_bridge(directory, state)
            actions = {item["id"]: item for item in bridge.available_actions(state)}
            presets = {item["id"]: item for item in actions["raise"]["presets"]}
            self.assertEqual(presets["pot"]["raise_to"], 300)
            self.assertEqual(adapter.calls, 1)
            bridge.snapshot()
            bridge.snapshot()
            self.assertEqual(adapter.calls, 1, "polling must use cached legality")
            accepted, protocol = bridge.queue_action("pot", 4)
            self.assertTrue(accepted)
            self.assertEqual(protocol, "r300")
            self.assertEqual(adapter.calls, 1)

    def test_arbitrary_raise_and_server_side_bounds(self):
        state = parse_matchstate("MATCHSTATE:1:7::AsKd|")
        with tempfile.TemporaryDirectory() as directory:
            bridge, _ = self.make_bridge(directory, state, nonce=8)
            accepted, protocol = bridge.queue_action("raise", 8, 750)
            self.assertTrue(accepted)
            self.assertEqual(protocol, "r750")

            bridge.status = "your_turn"
            self.assertFalse(bridge.queue_action("raise", 8, 199)[0])
            self.assertFalse(bridge.queue_action("raise", 8, 20_001)[0])
            self.assertFalse(bridge.queue_action("raise", 8, True)[0])

    def test_stale_action_is_rejected(self):
        state = parse_matchstate("MATCHSTATE:1:7::AsKd|")
        with tempfile.TemporaryDirectory() as directory:
            bridge, _ = self.make_bridge(directory, state, nonce=8)
            accepted, message = bridge.queue_action("call", 7)
            self.assertFalse(accepted)
            self.assertIn("stale", message)

    def test_missing_legality_fails_closed(self):
        state = parse_matchstate("MATCHSTATE:1:7::AsKd|")
        with tempfile.TemporaryDirectory() as directory:
            bridge = HumanBridge("unused", 1, Path(directory) / "events.jsonl")
            bridge.current = state
            bridge.status = "your_turn"
            bridge.state_nonce = 2
            self.assertEqual(bridge.available_actions(state), [])
            accepted, message = bridge.queue_action("call", 2)
            self.assertFalse(accepted)
            self.assertIn("validation", message)
            self.assertFalse(bridge.snapshot()["legal_actions"]["available"])

    def test_flop_transition_tracks_commitments_and_reversed_seat(self):
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

    def test_nonterminal_dealer_eof_is_an_error(self):
        state = parse_matchstate("MATCHSTATE:1:7::AsKd|")
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events.jsonl"
            bridge, _ = self.make_bridge(directory, state)
            bridge._handle_dealer_eof()
            self.assertEqual(bridge.status, "error")
            self.assertIn("before a terminal", bridge.error)
            event_text = events.read_text(encoding="utf-8")
            self.assertNotIn("AsKd", event_text)


class PokerHTTPServerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.dist = root / "dist"
        (self.dist / "assets").mkdir(parents=True)
        (self.dist / "index.html").write_text("<html>external build</html>", encoding="utf-8")
        (self.dist / "assets" / "app.js").write_text("window.app=true", encoding="utf-8")
        self.token = "test-session-token-0123456789abcdef"
        state = parse_matchstate("MATCHSTATE:1:7::AsKd|")
        adapter = FixedLegalityAdapter()
        self.bridge = HumanBridge(
            "unused",
            1,
            root / "events.jsonl",
            legality_adapter=adapter,
        )
        self.bridge.current = state
        self.bridge.current_legality = self.bridge._legality_for_state(state)
        self.bridge.status = "your_turn"
        self.bridge.state_nonce = 12
        self.server = PokerHTTPServer(
            ("127.0.0.1", 0),
            self.bridge,
            self.token,
            root / "missing-report.json",
            web_dist_path=self.dist,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.directory.cleanup()

    def request(self, method, path, body=None, headers=None):
        self.connection.request(method, path, body=body, headers=headers or {})
        response = self.connection.getresponse()
        data = response.read()
        return response, data

    def test_cookie_bootstrap_redirect_and_static_assets(self):
        response, _ = self.request("GET", f"/?token={self.token}")
        self.assertEqual(response.status, 303)
        self.assertEqual(response.getheader("Location"), "/")
        cookie = response.getheader("Set-Cookie")
        self.assertIn(f"{SESSION_COOKIE}=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("Secure", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertNotIn(self.token, response.getheader("Location"))

        cookie_header = {"Cookie": cookie.split(";", 1)[0]}
        response, body = self.request("GET", "/", headers=cookie_header)
        self.assertEqual(response.status, 200)
        self.assertIn(b"external build", body)
        response, body = self.request("GET", "/assets/app.js", headers=cookie_header)
        self.assertEqual(response.status, 200)
        self.assertEqual(body, b"window.app=true")

    def test_query_token_is_not_api_auth_and_header_token_remains_supported(self):
        response, _ = self.request("GET", f"/api/state?token={self.token}")
        self.assertEqual(response.status, 401)
        response, body = self.request(
            "GET", "/api/state", headers={"X-Session-Token": self.token}
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(body)["state_nonce"], 12)

    def test_post_accepts_arbitrary_raise_and_rejects_bad_type(self):
        headers = {
            "X-Session-Token": self.token,
            "Content-Type": "application/json",
        }
        response, _ = self.request(
            "POST",
            "/api/action",
            body=json.dumps({"action": "raise", "state_nonce": 12, "raise_to": 750}),
            headers=headers,
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(self.bridge.pending_action, "r750")

        self.bridge.status = "your_turn"
        response, body = self.request(
            "POST",
            "/api/action",
            body=json.dumps({"action": "raise", "state_nonce": 12, "raise_to": 199}),
            headers=headers,
        )
        self.assertEqual(response.status, 422)
        self.assertIn("not legal", json.loads(body)["error"])

        self.bridge.status = "your_turn"
        response, _ = self.request(
            "POST",
            "/api/action",
            body=json.dumps({"action": "raise", "state_nonce": 12, "raise_to": "750"}),
            headers=headers,
        )
        self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main()
