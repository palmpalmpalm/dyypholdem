#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

evaluator_stub = ModuleType("game.evaluation.evaluator")
evaluator_stub.Evaluator = type("Evaluator", (), {})
with mock.patch.dict(sys.modules, {"game.evaluation.evaluator": evaluator_stub}):
    import settings.arguments as arguments  # noqa: E402
    import settings.constants as constants  # noqa: E402
    from player.lbr_acpc_player import LbrMatch  # noqa: E402
    from player.local_best_response import LocalBestResponse, StrategyChannel  # noqa: E402
    from server import protocol_to_node  # noqa: E402

from slumbot_session_status import aggregate  # noqa: E402
from validate_lbr_benchmark import validate  # noqa: E402


class FixedEquity:
    def __init__(self, value):
        self.value = value

    def set_board(self, board):
        pass

    def call_value(self, ranges, result):
        result.copy_(ranges.sum(1, keepdim=True).expand_as(result) * self.value)

    def fold_value(self, ranges, result):
        result.copy_(ranges.sum(1, keepdim=True).expand_as(result))


class ScriptedDealer:
    """Replays MATCHSTATE strings the way ACPCGame.get_next_situation reports them."""

    def __init__(self, script):
        self.script = list(script)
        self.actions = []

    def get_next_situation(self):
        if not self.script:
            return None, None, 0
        message, winnings = self.script.pop(0)
        state = protocol_to_node.parse_state(message)
        if winnings is not None:
            return state, None, winnings
        return state, protocol_to_node.parsed_state_to_node(state), 0

    def play_action(self, action):
        self.actions.append(action)


def hands(count):
    return 1326 if count is None else count


class LbrMatchTest(unittest.TestCase):
    def test_match_tracks_bot_actions_and_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            channel_path = root / "bot-strategy.jsonl"
            uniform = [1.0 / 1326] * 1326
            with channel_path.open("w") as stream:
                # hand 0: bot (big blind, ACPC position 0) calls preflop, then checks the flop
                stream.write(json.dumps({"hand_number": 0, "decision_number": 1, "bets": [-2, -1, 900, 20000], "chosen_bet": -1, "strategy": [[0.1] * 1326, [0.6] * 1326, [0.2] * 1326, [0.1] * 1326]}) + "\n")
                stream.write(json.dumps({"hand_number": 0, "decision_number": 2, "bets": [-2, -1, 600, 20000], "chosen_bet": -1, "strategy": [[0.0] * 1326, [0.5] * 1326, [0.4] * 1326, [0.1] * 1326]}) + "\n")
            script = [
                ("MATCHSTATE:1:0::|AhAs", None),                      # LBR is the small blind, first to act
                ("MATCHSTATE:1:0:r300c/c:|AhAs/2c3d7h", None),        # bot called, then checked the flop
                ("MATCHSTATE:1:0:r300c/cr600c/cc/cc:|AhAs/2c3d7h9sKd", 600),  # showdown, LBR won 600
                ("MATCHSTATE:0:1:f:|KcKh", 50),                        # hand 1: bot folded the small blind at once
            ]
            dealer = ScriptedDealer(script)
            with mock.patch.object(arguments, "Tensor", torch.FloatTensor):
                lbr = LocalBestResponse(FixedEquity(0.3), raise_menu=("pot",))
                match = LbrMatch(dealer, lbr, StrategyChannel(channel_path, timeout_seconds=1), 2,
                                 root / "lbr-events.jsonl", root / "lbr-summary.json", mock.Mock(), ["pot"])
                rc = match.run()

            self.assertEqual(rc, 0)
            summary = json.loads((root / "lbr-summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["hands_completed"], 2)
            self.assertEqual(summary["cumulative_winnings"], 650)
            self.assertEqual(summary["decisions"], 2)
            # Hand 1 ended before LBR acted, so its bookkeeping restarted for that hand.
            self.assertEqual(lbr.applied_bot_actions, 0)
            self.assertEqual(lbr.hand_number, 1)
            self.assertEqual([a.action for a in dealer.actions], [constants.ACPCActions.rraise, constants.ACPCActions.rraise])
            self.assertEqual(dealer.actions[0].raise_amount, 300)  # pot-sized open from the blinds
            events = [json.loads(line) for line in (root / "lbr-events.jsonl").read_text().splitlines()]
            self.assertEqual([e["event"] for e in events], ["hand_started", "decision", "decision", "hand_result", "hand_result"])
            self.assertEqual([e["applied_bot_actions"] for e in events if e["event"] == "decision"], [0, 2])
            # Range mass after the bot's call (0.6) and check (0.5) is 0.3 of the uniform start.
            # ... of the hands not blocked by AhAs and the 2c3d7h flop: C(47,2) of C(52,2).
            self.assertAlmostEqual(events[2]["bot_range_mass"], 0.3 * 1081 / 1326, places=4)

            # The headless controller reads the same summary through the session helper.
            session = root / "session-0"
            session.mkdir()
            (root / "lbr-summary.json").rename(session / "lbr-summary.json")
            status = aggregate(root, 1, "lbr-summary.json")
            self.assertEqual((status["status"], status["hands_completed"], status["finished_sessions"]), ("complete", 2, 1))

            (session / "timing_report.json").write_text(json.dumps({"decision_count": 4, "match": {"hands_completed": 2, "cumulative_winnings": -650}}))
            result = validate(root, 2)
            self.assertTrue(result["valid"])
            self.assertEqual(result["bot_winnings"], -650)
            (session / "timing_report.json").write_text(json.dumps({"decision_count": 4, "match": {"hands_completed": 2, "cumulative_winnings": -600}}))
            with self.assertRaisesRegex(ValueError, "zero-sum"):
                validate(root, 2)


if __name__ == "__main__":
    unittest.main()
