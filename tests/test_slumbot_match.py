#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import settings.constants as constants  # noqa: E402
from player.slumbot_match import SlumbotMatch, match_statistics, paired_statistics  # noqa: E402
from server.protocol_to_node import Action  # noqa: E402
from server.slumbot_game import SlumbotProtocolError, SlumbotTransportError  # noqa: E402


class ScriptedGame:
    """Replays scripted Slumbot responses; each hand is a list of responses."""

    def __init__(self, hands):
        self.hands = [list(hand) for hand in hands]
        self.request_retries = 0
        self.last_action_string = None
        self.last_correction = None
        self.actions = []
        self.new_hand_calls = []
        self.current = None

    def reset_hand(self):
        self.current = None

    def new_hand(self, token, hand_number):
        self.new_hand_calls.append((token, hand_number))
        self.current = self.hands.pop(0)
        response = self.current.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get_next_situation(self, response, hand_number):
        street = len(response.get("board") or [])
        current_street = 1 if street == 0 else street - 1
        return SimpleNamespace(current_street=current_street, hand_number=str(hand_number)), object()

    def play_action(self, token, action):
        self.actions.append(action)
        self.last_action_string = "k" if action.action == constants.ACPCActions.ccall else "b300"
        self.last_correction = "raise_to_300_lifted_to_min_raise" if action.raise_amount == 300 else None
        response = self.current.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ScriptedResolver:
    def __init__(self, actions):
        self.actions = list(actions)
        self.started = []
        self.last_decision_telemetry = None

    def start_new_hand(self, state):
        self.started.append(state.hand_number)

    def compute_action(self, state, node):
        action = self.actions.pop(0)
        self.last_decision_telemetry = {
            "event": "decision",
            "hand_number": int(state.hand_number),
            "street": "preflop",
            "strategy": [0.5, 0.5],
            "chosen_action": "call",
        }
        return action


class ListWriter:
    def __init__(self):
        self.events = []

    def append(self, event):
        self.events.append(event)


def response(action, client_pos=1, board=None, winnings=None, token="tok",
             baseline_winnings=None, bot_hole_cards=None, won_pot=None):
    payload = {"token": token, "action": action, "client_pos": client_pos, "hole_cards": ["Ad", "6h"], "board": board or []}
    if winnings is not None:
        payload["winnings"] = winnings
        if baseline_winnings is not None:
            payload["baseline_winnings"] = baseline_winnings
        if bot_hole_cards is not None:
            payload["bot_hole_cards"] = bot_hole_cards
        if won_pot is not None:
            payload["won_pot"] = won_pot
    return payload


class SlumbotMatchTest(unittest.TestCase):
    def run_match(self, hands, actions, expected, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game = ScriptedGame(hands)
            resolver = ScriptedResolver(actions)
            writer = ListWriter()
            match = SlumbotMatch(
                game,
                resolver,
                expected,
                events_path=root / "slumbot-events.jsonl",
                summary_path=root / "slumbot-summary.json",
                telemetry_writer=writer,
                **kwargs,
            )
            rc = match.run()
            summary = json.loads((root / "slumbot-summary.json").read_text())
            events = [json.loads(line) for line in (root / "slumbot-events.jsonl").read_text().splitlines()]
            return rc, summary, events, writer.events, game, resolver

    def test_completes_exact_hand_count_with_zero_sum_bookkeeping(self):
        hands = [
            [response(""), response("b300f", winnings=100)],
            [response("b300", client_pos=0), response("b300f", client_pos=0, winnings=-100)],
            [response("f", client_pos=0, winnings=50)],
        ]
        actions = [Action(action=constants.ACPCActions.rraise, raise_amount=300), Action(action=constants.ACPCActions.fold)]
        rc, summary, events, telemetry, game, resolver = self.run_match(hands, actions, 3)

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["hands_completed"], 3)
        self.assertEqual(summary["hands_attempted"], 3)
        self.assertEqual(summary["cumulative_winnings"], 50)
        self.assertEqual(summary["decisions"], 2)
        self.assertEqual(summary["action_counts"], {"fold": 1, "raise": 1})
        self.assertEqual(summary["bet_size_corrections"], 1)
        self.assertEqual(summary["seat_counts"], {"big_blind": 2, "small_blind": 1})
        self.assertEqual(summary["error"], None)
        self.assertAlmostEqual(summary["statistics"]["mbb_per_hand"], 50 / 3 * 10)
        self.assertEqual(resolver.started, ["0", "1"])
        self.assertEqual(game.new_hand_calls, [(None, 0), ("tok", 1), ("tok", 2)])

        hand_results = [event for event in telemetry if event["event"] == "hand_result"]
        self.assertEqual([event["hand_number"] for event in hand_results], [0, 1, 2])
        self.assertEqual([event["cumulative_winnings"] for event in hand_results], [100, 0, 50])
        self.assertEqual(sum(1 for event in telemetry if event["event"] == "decision"), 2)
        self.assertEqual([event["event"] for event in events][:3], ["hand_started", "bet_size_correction", "hand_result"])
        self.assertNotIn("Ad", json.dumps(events))

    def test_baseline_score_is_recorded_alongside_the_raw_result(self):
        hands = [
            [response(""), response("b300f", winnings=100, baseline_winnings=250,
                                    bot_hole_cards=["Kc", "5h"], won_pot=300)],
            [response("b300", client_pos=0),
             response("b300f", client_pos=0, winnings=-100, baseline_winnings=-40,
                      bot_hole_cards=["7d", "6h"], won_pot=-300)],
        ]
        actions = [Action(action=constants.ACPCActions.rraise, raise_amount=300),
                   Action(action=constants.ACPCActions.fold)]
        _, summary, events, telemetry, _, _ = self.run_match(hands, actions, 2)

        self.assertEqual(summary["cumulative_winnings"], 0)
        self.assertAlmostEqual(summary["cumulative_baseline_winnings"], 210.0)
        self.assertAlmostEqual(summary["baseline_statistics"]["mbb_per_hand"], 105.0 * 10)
        self.assertEqual(summary["baseline_comparison"]["hands"], 2)

        # The opponent's cards stay in the private telemetry, never in events.
        hand_results = [event for event in telemetry if event["event"] == "hand_result"]
        self.assertEqual([event["baseline_winnings"] for event in hand_results], [250, -40])
        self.assertEqual(hand_results[0]["bot_hole_cards"], ["Kc", "5h"])
        self.assertEqual([event["won_pot"] for event in hand_results], [300, -300])
        safe = [event for event in events if event["event"] == "hand_result"]
        self.assertEqual([event["baseline_winnings"] for event in safe], [250, -40])
        self.assertNotIn("bot_hole_cards", json.dumps(events))
        self.assertNotIn("Kc", json.dumps(events))

    def test_missing_baseline_disables_the_baseline_score_instead_of_mixing(self):
        hands = [
            [response(""), response("b300f", winnings=100, baseline_winnings=250)],
            [response("b300", client_pos=0), response("b300f", client_pos=0, winnings=-100)],
        ]
        actions = [Action(action=constants.ACPCActions.rraise, raise_amount=300),
                   Action(action=constants.ACPCActions.fold)]
        _, summary, _, _, _, _ = self.run_match(hands, actions, 2)

        self.assertEqual(summary["cumulative_winnings"], 0)
        self.assertIsNone(summary["cumulative_baseline_winnings"])
        self.assertIsNone(summary["baseline_statistics"])
        self.assertIsNone(summary["baseline_comparison"])

    def test_paired_statistics_measure_agreement_and_variance_reduction(self):
        # A perfect control variate: baseline equals raw minus a zero-mean term.
        raw = [1000, -1000, 500, -500, 200, -200]
        baseline = [900, -900, 450, -450, 180, -180]
        stats = paired_statistics(raw, baseline)
        self.assertEqual(stats["hands"], 6)
        self.assertAlmostEqual(stats["correlation"], 1.0, places=9)
        self.assertGreater(stats["variance_ratio"], 1.0)
        self.assertAlmostEqual(stats["stdev_ratio"], 1 / 0.9, places=9)
        self.assertAlmostEqual(stats["mean_difference_mbb"], 0.0)

    def test_paired_statistics_are_unavailable_without_matching_samples(self):
        self.assertIsNone(paired_statistics([], [])["correlation"])
        self.assertIsNone(paired_statistics([1, 2, 3], [1, 2])["variance_ratio"])
        self.assertIsNone(paired_statistics([5], [5])["stdev_ratio"])

    def test_failed_hand_is_retried_and_counted(self):
        hands = [
            [response(""), SlumbotTransportError("network down")],
            [response("f", client_pos=0, winnings=50)],
            [response("f", client_pos=0, winnings=50)],
        ]
        actions = [Action(action=constants.ACPCActions.ccall)]
        rc, summary, events, telemetry, game, _ = self.run_match(hands, actions, 2)

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["hands_completed"], 2)
        self.assertEqual(summary["hands_attempted"], 3)
        self.assertEqual(summary["hand_errors"], 1)
        self.assertEqual(summary["cumulative_winnings"], 100)
        self.assertEqual([event["hand_number"] for event in events if event["event"] == "hand_error"], [0])
        self.assertEqual([event["hand_number"] for event in telemetry if event["event"] == "hand_result"], [1, 2])

    def test_consecutive_failures_abort_with_failed_status(self):
        hands = [
            [SlumbotProtocolError("Illegal action")],
            [SlumbotProtocolError("Illegal action")],
            [response("f", client_pos=0, winnings=50)],
        ]
        rc, summary, events, _, _, _ = self.run_match(hands, [], 5, max_consecutive_errors=2)

        self.assertEqual(rc, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["hands_completed"], 0)
        self.assertEqual(summary["hand_errors"], 2)
        self.assertIn("Illegal action", summary["error"])
        self.assertIsNotNone(summary["finished_at"])

    def test_runaway_hand_is_rejected(self):
        hands = [[response("")] + [response("") for _ in range(70)]]
        actions = [Action(action=constants.ACPCActions.ccall) for _ in range(70)]
        rc, summary, _, _, _, _ = self.run_match(hands, actions, 1, max_consecutive_errors=1)
        self.assertEqual(rc, 1)
        self.assertIn("exceeded 64 decisions", summary["error"])

    def test_match_statistics(self):
        stats = match_statistics([100, -50])
        self.assertEqual(stats["hands"], 2)
        self.assertAlmostEqual(stats["mbb_per_hand"], 250.0)
        self.assertEqual((stats["wins"], stats["losses"], stats["ties"]), (1, 1, 0))
        self.assertAlmostEqual(stats["standard_error_mbb_per_hand"], 750.0)
        self.assertEqual(match_statistics([])["mbb_per_hand"], None)


if __name__ == "__main__":
    unittest.main()
