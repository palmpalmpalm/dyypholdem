"""Local best response (LBR) against a DyypHoldem opponent.

The LBR agent tracks the bot's private-hand range from the bot's own full-hand
strategy at every bot decision (the bot publishes it through a strategy
channel), then values its own candidate actions under the call-down
assumption of Lisy and Bowling (2017): after the LBR action both players
check or call to showdown. Values are future chips relative to folding now:

    fold            0
    check / call    (T / 2) * (1 + eq) - C        T = pot after the call
    raise to R      R * eq + own_bet              bot assumed to call

where ``eq`` is the range-conditional showdown expectation (win minus loss)
of the LBR hand against the bot's current range on the current board, taken
from DyypHoldem's terminal-equity matrices (runouts averaged on the flop and
turn). LBR plays the argmax and folds when nothing beats zero. This is a
lower bound on exploitability, not an exact best response.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings
import game.card_tools as card_tools
import game.card_to_string_conversion as card_conversion
from server.protocol_to_node import Action

FOLD_BET = constants.Actions.fold.value
CALL_BET = constants.Actions.ccall.value


def action_to_bet(action: Action) -> int:
    """Map an observed ACPC action to the bet value used in the bot's strategy rows."""
    if action.action == constants.ACPCActions.fold:
        return FOLD_BET
    if action.action == constants.ACPCActions.ccall:
        return CALL_BET
    return int(action.raise_amount)


class StrategyChannel:
    """Tail a JSONL file of bot decisions keyed by hand number."""

    def __init__(self, path: Path, poll_seconds: float = 0.05, timeout_seconds: float = 900.0):
        self.path = Path(path)
        self.poll_seconds = float(poll_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._offset = 0
        self._buffer = ""
        self._by_hand: dict[int, list[dict]] = {}
        self.waits = 0

    def _read_new(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as stream:
            stream.seek(self._offset)
            chunk = stream.read()
            self._offset = stream.tell()
        if not chunk:
            return
        self._buffer += chunk
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()  # possibly incomplete trailing line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            self._by_hand.setdefault(int(record["hand_number"]), []).append(record)

    def decisions(self, hand_number: int, count: int) -> list[dict]:
        """Return at least ``count`` decisions for ``hand_number``, waiting for the bot."""
        deadline = time.monotonic() + self.timeout_seconds
        waited = False
        while True:
            self._read_new()
            rows = self._by_hand.get(int(hand_number), [])
            if len(rows) >= count:
                return rows
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"strategy channel has {len(rows)} decisions for hand {hand_number}, expected {count}"
                )
            if not waited:
                self.waits += 1
                waited = True
            time.sleep(self.poll_seconds)


class LocalBestResponse:
    def __init__(self, terminal_equity, raise_menu=("pot", "all_in")):
        self.terminal_equity = terminal_equity
        self.raise_menu = tuple(raise_menu)
        self.bot_range = None
        self.applied_bot_actions = 0
        self.hand_number = None
        self.my_cards = None
        self.my_hand_index = None
        self.position = None

    # -- hand lifecycle -----------------------------------------------------

    def start_hand(self, hand_number: int, my_hand_string: str, position: int) -> None:
        self.hand_number = int(hand_number)
        self.position = int(position)
        self.my_cards = card_conversion.string_to_board(my_hand_string)
        self.my_hand_index = card_tools.string_to_hole_index(my_hand_string)
        self.bot_range = card_tools.get_uniform_range(arguments.Tensor())
        self.applied_bot_actions = 0

    def apply_bot_decision(self, record: dict, observed_bet: int) -> None:
        """Multiply the bot's range by its probability of the action it actually took."""
        bets = [int(round(float(value))) for value in record["bets"]]
        if int(observed_bet) not in bets:
            raise ValueError(f"bot action {observed_bet} is not in its published menu {bets}")
        index = bets.index(int(observed_bet))
        row = torch.tensor(record["strategy"][index], dtype=self.bot_range.dtype)
        if row.numel() != self.bot_range.numel():
            raise ValueError("strategy row size does not match the hand count")
        self.bot_range = self.bot_range.mul(row)
        self.applied_bot_actions += 1

    def blocked_cards(self, board) -> torch.Tensor:
        cards = [int(card) for card in self.my_cards.tolist()]
        if board is not None and board.dim() > 0:
            cards.extend(int(card) for card in board.tolist())
        return torch.tensor(sorted(set(cards)), dtype=torch.long)

    def conditional_equity(self, board) -> tuple[float, float]:
        """Return (win-minus-loss expectation, bot range mass) for my hand on this board."""
        self.terminal_equity.set_board(board)
        mask = card_tools.get_possible_hand_indexes(self.blocked_cards(board))
        conditioned = self.bot_range.mul(mask)
        mass = float(conditioned.sum())
        if mass <= 0:
            return 0.0, 0.0
        conditioned = conditioned.div(mass).view(1, -1)
        values = torch.zeros_like(conditioned)
        weights = torch.zeros_like(conditioned)
        self.terminal_equity.call_value(conditioned, values)
        self.terminal_equity.fold_value(conditioned, weights)
        weight = float(weights[0, self.my_hand_index])
        if weight <= 0:
            return 0.0, mass
        return float(values[0, self.my_hand_index]) / weight, mass

    # -- decision ----------------------------------------------------------------

    @staticmethod
    def raise_targets(my_bet: int, opp_bet: int, menu) -> dict[str, int]:
        stack = game_settings.stack
        if opp_bet >= stack:
            return {}
        to_call = opp_bet - my_bet
        min_raise = opp_bet + max(to_call, game_settings.big_blind)
        pot_after_call = 2 * opp_bet
        targets = {}
        for name in menu:
            if name == "all_in":
                targets[name] = stack
                continue
            fraction = {"half_pot": 0.5, "pot": 1.0, "double_pot": 2.0}[name]
            target = opp_bet + int(round(pot_after_call * fraction))
            if target < min_raise:
                target = min_raise
            if target >= stack:
                continue
            targets[name] = target
        return targets

    def decide(self, state, board) -> tuple[Action, dict]:
        my_bet = int(state.bet2 if state.position == 0 else state.bet1)
        opp_bet = int(state.bet1 if state.position == 0 else state.bet2)
        to_call = opp_bet - my_bet
        equity, mass = self.conditional_equity(board)
        pot_after_call = my_bet + opp_bet + to_call
        values = {"fold": 0.0, "call": pot_after_call / 2.0 * (1.0 + equity) - to_call}
        targets = self.raise_targets(my_bet, opp_bet, self.raise_menu)
        for name, target in targets.items():
            values[name] = target * equity + my_bet
        best = max(values, key=lambda key: (values[key], key == "call"))
        if best == "fold" and to_call == 0:
            best = "call"
        if best == "fold":
            action = Action(action=constants.ACPCActions.fold)
        elif best == "call":
            action = Action(action=constants.ACPCActions.ccall, raise_amount=to_call)
        else:
            action = Action(action=constants.ACPCActions.rraise, raise_amount=targets[best])
        telemetry = {
            "equity": equity,
            "bot_range_mass": mass,
            "to_call": to_call,
            "values": {key: round(float(value), 3) for key, value in values.items()},
            "chosen": best,
            "raise_targets": targets,
        }
        return action, telemetry


def summary_statistics(hand_winnings: list[int]) -> dict:
    count = len(hand_winnings)
    if count == 0:
        return {"hands": 0, "mbb_per_hand": None, "standard_error_mbb_per_hand": None, "ci95_mbb_per_hand": None}
    mean = sum(hand_winnings) / count
    factor = 1000.0 / game_settings.big_blind
    variance = sum((value - mean) ** 2 for value in hand_winnings) / (count - 1) if count > 1 else 0.0
    se = math.sqrt(variance) / math.sqrt(count) * factor
    return {
        "hands": count,
        "mean_chips_per_hand": mean,
        "mbb_per_hand": mean * factor,
        "stdev_chips_per_hand": math.sqrt(variance),
        "standard_error_mbb_per_hand": se,
        "ci95_mbb_per_hand": 1.96 * se,
        "wins": sum(1 for value in hand_winnings if value > 0),
        "losses": sum(1 for value in hand_winnings if value < 0),
        "ties": sum(1 for value in hand_winnings if value == 0),
    }
