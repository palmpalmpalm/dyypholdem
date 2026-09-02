import argparse
import json
import os
from pathlib import Path
import sys
sys.path.append(os.getcwd())

import settings.constants as constants  # noqa: E402
from player.local_best_response import action_to_bet, summary_statistics  # noqa: E402


def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class LbrMatch:
    def __init__(self, acpc_game, lbr, channel, expected_hands, events_path, summary_path, logger, raise_menu):
        self.acpc_game = acpc_game
        self.lbr = lbr
        self.channel = channel
        self.expected_hands = int(expected_hands)
        self.events_path = Path(events_path) if events_path else None
        self.summary_path = Path(summary_path) if summary_path else None
        self.logger = logger
        self.raise_menu = list(raise_menu)
        self.started_at = utc_now()
        self.status = "starting"
        self.hands_completed = 0
        self.cumulative_winnings = 0
        self.hand_winnings = []
        self.decisions = 0
        self.action_counts = {}
        self.error = None
        self._write_summary()

    def _event(self, payload):
        if self.events_path is None:
            return
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": utc_now(), **payload}, sort_keys=True) + "\n")

    def _write_summary(self, finished_at=None):
        if self.summary_path is None:
            return
        payload = {
            "schema_version": 1,
            "updated_at": utc_now(),
            "started_at": self.started_at,
            "finished_at": finished_at,
            "status": self.status,
            "opponent": {"name": "local best response", "assumption": "call-down", "raise_menu": self.raise_menu},
            "expected_hands": self.expected_hands,
            "hands_completed": self.hands_completed,
            "hands_attempted": self.hands_completed,
            "hand_errors": 0,
            "cumulative_winnings": self.cumulative_winnings,
            "statistics": summary_statistics(self.hand_winnings),
            "decisions": self.decisions,
            "action_counts": dict(sorted(self.action_counts.items())),
            "channel_waits": self.channel.waits,
            "error": self.error,
        }
        temporary = self.summary_path.with_name(f".{self.summary_path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.summary_path)

    def run(self):
        self.status = "running"
        self._write_summary()
        current_hand = None
        bot_player = None
        try:
            while True:
                state, node, hand_winnings = self.acpc_game.get_next_situation()
                if state is None:
                    break
                hand_number = int(state.hand_number)
                if node is not None:
                    if current_hand != hand_number:
                        current_hand = hand_number
                        bot_player = constants.Players(1 - state.player.value)
                        self.lbr.start_hand(hand_number, state.my_hand_string, state.position)
                        self._event({"event": "hand_started", "hand_number": hand_number, "position": state.position})
                    # Apply every bot action since the last LBR decision using the bot's published strategy.
                    bot_actions = [action for action in state.all_actions[2:] if action.player == bot_player]
                    if len(bot_actions) > self.lbr.applied_bot_actions:
                        records = self.channel.decisions(hand_number, len(bot_actions))
                        for index in range(self.lbr.applied_bot_actions, len(bot_actions)):
                            self.lbr.apply_bot_decision(records[index], action_to_bet(bot_actions[index]))
                    action, telemetry = self.lbr.decide(state, node.board)
                    self.acpc_game.play_action(action)
                    self.decisions += 1
                    self.action_counts[telemetry["chosen"]] = self.action_counts.get(telemetry["chosen"], 0) + 1
                    self._event({"event": "decision", "hand_number": hand_number, "street": int(state.current_street),
                                 "board": state.board, "pot": int(state.bet1 + state.bet2),
                                 "applied_bot_actions": self.lbr.applied_bot_actions, **telemetry})
                else:
                    if current_hand != hand_number:
                        # The hand ended before LBR ever acted (bot folded or shoved).
                        current_hand = hand_number
                        self.lbr.start_hand(hand_number, state.my_hand_string, state.position)
                    self.hands_completed += 1
                    self.cumulative_winnings += int(hand_winnings)
                    self.hand_winnings.append(int(hand_winnings))
                    self._event({"event": "hand_result", "hand_number": hand_number, "winnings": int(hand_winnings),
                                 "cumulative_winnings": self.cumulative_winnings, "position": int(state.position)})
                    self.logger.success(f"Hand {hand_number} completed. LBR winnings: {hand_winnings}, total: {self.cumulative_winnings} ({self.hands_completed}/{self.expected_hands})")
                    self._write_summary()
                    if self.hands_completed >= self.expected_hands:
                        break
        except Exception as error:  # noqa: BLE001
            self.status = "failed"
            self.error = f"{type(error).__name__}: {error}"
            self._write_summary(finished_at=utc_now())
            self.logger.critical(f"LBR_MATCH_FAILED {self.error}")
            raise
        self.status = "complete" if self.hands_completed >= self.expected_hands else "incomplete"
        self._write_summary(finished_at=utc_now())
        stats = summary_statistics(self.hand_winnings)
        self.logger.success(f"LBR_MATCH_{self.status.upper()} hands={self.hands_completed} winnings={self.cumulative_winnings} mbb_per_hand={stats['mbb_per_hand']}")
        return 0 if self.status == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local best response opponent for DyypHoldem over ACPC")
    parser.add_argument("hostname", type=str)
    parser.add_argument("port", type=int)
    parser.add_argument("--hands", type=int, required=True)
    parser.add_argument("--strategy-channel", type=Path, required=True, help="JSONL written by the bot's --strategy-channel")
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--raise-menu", type=str, default="pot,all_in", help="comma list from half_pot,pot,double_pot,all_in or empty for fold/call only")
    parser.add_argument("--channel-timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.hands < 1:
        raise SystemExit("hands must be at least 1")
    raise_menu = [item for item in args.raise_menu.split(",") if item]
    for item in raise_menu:
        if item not in ("half_pot", "pot", "double_pot", "all_in"):
            raise SystemExit(f"unsupported raise size {item}")

    import settings.arguments as arguments
    from server.acpc_game import ACPCGame
    from terminal_equity.terminal_equity import TerminalEquity
    from player.local_best_response import LocalBestResponse, StrategyChannel

    acpc_game = ACPCGame()
    acpc_game.connect(args.hostname, args.port)
    lbr = LocalBestResponse(TerminalEquity(), raise_menu=raise_menu)
    channel = StrategyChannel(args.strategy_channel, timeout_seconds=args.channel_timeout)
    arguments.logger.success(f"LBR_READY raise_menu={raise_menu} device={arguments.device}")
    match = LbrMatch(acpc_game, lbr, channel, args.hands, args.events, args.summary, arguments.logger, raise_menu)
    sys.exit(match.run())
