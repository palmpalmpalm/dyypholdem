#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from player.random_ui_player import RandomUIPlayer, choose_action  # noqa: E402


def snapshot(*, nonce=7, hand=3, facing_bet=True, presets=None):
    passive = (
        {"id": "call", "label": "Call"}
        if facing_bet
        else {"id": "check", "label": "Check"}
    )
    return {
        "state_nonce": nonce,
        "hand_number": hand,
        "available_actions": [
            {"id": "fold", "label": "Fold"},
            passive,
            {
                "id": "raise",
                "label": "Raise",
                "presets": presets
                if presets is not None
                else [
                    {"id": "min", "raise_to": 200},
                    {"id": "pot", "raise_to": 300},
                    {"id": "all_in", "raise_to": 20_000},
                ],
            },
        ],
    }


class RandomUIPlayerTest(unittest.TestCase):
    def test_same_state_and_seed_are_reproducible(self):
        first = choose_action(snapshot(), 20260824)
        second = choose_action(snapshot(), 20260824)
        self.assertEqual(first, second)

    def test_free_fold_is_not_a_sampled_action(self):
        seen = {
            choose_action(snapshot(nonce=nonce, facing_bet=False), 41)[0]["action"]
            for nonce in range(1, 100)
        }
        self.assertEqual(seen, {"check", "raise"})

    def test_facing_bet_samples_all_action_classes(self):
        seen = {
            choose_action(snapshot(nonce=nonce, facing_bet=True), 99)[0]["action"]
            for nonce in range(1, 200)
        }
        self.assertEqual(seen, {"fold", "call", "raise"})

    def test_raise_uses_only_server_generated_amounts(self):
        allowed = {275, 550, 20_000}
        for nonce in range(1, 300):
            payload, label = choose_action(
                snapshot(
                    nonce=nonce,
                    presets=[
                        {"id": "min", "raise_to": 275},
                        {"id": "pot", "raise_to": 550},
                        {"id": "all_in", "raise_to": 20_000},
                    ],
                ),
                17,
            )
            if payload["action"] == "raise":
                self.assertIn(payload["raise_to"], allowed)
                self.assertTrue(label.startswith("raise:"))

    def test_invalid_or_missing_raise_presets_fail_closed(self):
        state = snapshot(presets=[])
        state["available_actions"] = [state["available_actions"][-1]]
        with self.assertRaises(ValueError):
            choose_action(state, 1)

    def test_scripted_match_requires_exact_completion_and_logs_no_cards(self):
        class ScriptedPlayer(RandomUIPlayer):
            def __init__(self, states, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.states = list(states)
                self.posts = []

            def _request(self, method, path, payload=None):
                if method == "POST":
                    self.posts.append(payload)
                    return {"accepted": True}
                return self.states.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            turn = snapshot(nonce=12, hand=0)
            turn.update(
                {
                    "status": "your_turn",
                    "street": "preflop",
                    "pot": 150,
                    "hands_completed": 0,
                    "cumulative_winnings": 0,
                    "hero_hand": ["As", "Kd"],
                }
            )
            complete = {
                "status": "match_complete",
                "state_nonce": 13,
                "hands_completed": 1,
                "cumulative_winnings": -100,
            }
            player = ScriptedPlayer(
                [turn, complete],
                "http://unused",
                "x" * 32,
                7,
                1,
                root / "events.jsonl",
                root / "summary.json",
                poll_seconds=0.01,
            )
            self.assertEqual(player.run(), 0)
            self.assertEqual(len(player.posts), 1)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["hands_completed"], 1)
            self.assertNotIn("As", (root / "events.jsonl").read_text(encoding="utf-8"))

    def test_scripted_match_fails_when_dealer_finishes_early(self):
        class EarlyExitPlayer(RandomUIPlayer):
            def _request(self, method, path, payload=None):
                return {
                    "status": "match_complete",
                    "state_nonce": 1,
                    "hands_completed": 99,
                    "cumulative_winnings": 0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            player = EarlyExitPlayer(
                "http://unused",
                "x" * 32,
                7,
                100,
                root / "events.jsonl",
                root / "summary.json",
                poll_seconds=0.01,
            )
            self.assertEqual(player.run(), 1)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "error")
            self.assertIn("99 of 100", summary["error"])

    def test_action_rate_limit_retries_same_deterministic_state(self):
        class RateLimitedPlayer(RandomUIPlayer):
            def __init__(self, states, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.states = list(states)
                self.posts = []

            def _request(self, method, path, payload=None):
                if method == "POST":
                    self.posts.append(dict(payload))
                    if len(self.posts) == 1:
                        raise HTTPError("http://unused/api/action", 429, "limited", {}, None)
                    return {"accepted": True}
                return self.states.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            turn = snapshot(nonce=21, hand=0)
            turn.update(
                {
                    "status": "your_turn",
                    "street": "preflop",
                    "pot": 150,
                    "hands_completed": 0,
                    "cumulative_winnings": 0,
                }
            )
            complete = {
                "status": "match_complete",
                "state_nonce": 22,
                "hands_completed": 1,
                "cumulative_winnings": 100,
            }
            player = RateLimitedPlayer(
                [turn, turn, complete],
                "http://unused",
                "x" * 32,
                11,
                1,
                root / "events.jsonl",
                root / "summary.json",
                poll_seconds=0.01,
            )
            with patch("player.random_ui_player.time.sleep", return_value=None):
                self.assertEqual(player.run(), 0)
            self.assertEqual(player.posts[0], player.posts[1])
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["rate_limit_backoffs"], 1)


if __name__ == "__main__":
    unittest.main()
