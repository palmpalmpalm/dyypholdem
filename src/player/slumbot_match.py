"""Fixed-length DyypHoldem versus Slumbot match with private and safe telemetry.

The match loop is deliberately independent of the network client and solver
so it can be exercised with scripted doubles. ``game`` must provide
``new_hand(token, hand_number)``, ``get_next_situation(response, hand_number)``,
``play_action(token, action)``, ``reset_hand()`` and the attributes
``request_retries``, ``last_action_string`` and ``last_correction``.
``resolver`` must provide ``start_new_hand(state)``, ``compute_action(state,
node)`` and ``last_decision_telemetry``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import settings.constants as constants
from server.slumbot_game import BIG_BLIND, SMALL_BLIND, STACK_SIZE, SlumbotError, SlumbotProtocolError

STREET_NAMES = {1: "preflop", 2: "flop", 3: "turn", 4: "river"}
MAX_DECISIONS_PER_HAND = 64


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def match_statistics(hand_winnings: list[int]) -> dict[str, float | int | None]:
    """Chip and milli-big-blind statistics for a list of per-hand results."""
    count = len(hand_winnings)
    if count == 0:
        return {
            "hands": 0,
            "mean_chips_per_hand": None,
            "mbb_per_hand": None,
            "stdev_chips_per_hand": None,
            "standard_error_mbb_per_hand": None,
            "ci95_mbb_per_hand": None,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    mean_chips = statistics.fmean(hand_winnings)
    mbb_factor = 1000.0 / BIG_BLIND
    stdev = statistics.stdev(hand_winnings) if count > 1 else None
    standard_error = stdev / math.sqrt(count) * mbb_factor if stdev is not None else None
    return {
        "hands": count,
        "mean_chips_per_hand": mean_chips,
        "mbb_per_hand": mean_chips * mbb_factor,
        "stdev_chips_per_hand": stdev,
        "standard_error_mbb_per_hand": standard_error,
        "ci95_mbb_per_hand": 1.96 * standard_error if standard_error is not None else None,
        "wins": sum(1 for value in hand_winnings if value > 0),
        "losses": sum(1 for value in hand_winnings if value < 0),
        "ties": sum(1 for value in hand_winnings if value == 0),
    }


def paired_statistics(
    hand_winnings: list[int], baseline_winnings: list[float]
) -> dict[str, float | int | None]:
    """Compare the raw chip result against Slumbot's variance-reduced baseline.

    ``mean_difference_mbb`` is the paired mean of raw minus baseline. The
    baseline is only usable as a score if that difference is consistent with
    zero, so it is reported with its own paired standard error rather than
    assumed.
    """
    count = len(hand_winnings)
    unavailable = {
        "hands": count,
        "correlation": None,
        "variance_ratio": None,
        "stdev_ratio": None,
        "mean_difference_mbb": None,
        "standard_error_difference_mbb": None,
        "ci95_difference_mbb": None,
    }
    if count < 2 or len(baseline_winnings) != count:
        return unavailable
    mbb_factor = 1000.0 / BIG_BLIND
    raw_stdev = statistics.stdev(hand_winnings)
    baseline_stdev = statistics.stdev(baseline_winnings)
    differences = [float(a) - float(b) for a, b in zip(hand_winnings, baseline_winnings)]
    difference_stdev = statistics.stdev(differences)
    standard_error = difference_stdev / math.sqrt(count) * mbb_factor
    raw_mean = statistics.fmean(hand_winnings)
    baseline_mean = statistics.fmean(baseline_winnings)
    covariance = sum(
        (float(a) - raw_mean) * (float(b) - baseline_mean)
        for a, b in zip(hand_winnings, baseline_winnings)
    ) / (count - 1)
    denominator = raw_stdev * baseline_stdev
    correlation = covariance / denominator if denominator > 0 else None
    return {
        "hands": count,
        "correlation": correlation,
        "variance_ratio": (raw_stdev ** 2) / (baseline_stdev ** 2) if baseline_stdev > 0 else None,
        "stdev_ratio": raw_stdev / baseline_stdev if baseline_stdev > 0 else None,
        "mean_difference_mbb": statistics.fmean(differences) * mbb_factor,
        "standard_error_difference_mbb": standard_error,
        "ci95_difference_mbb": 1.96 * standard_error,
    }


class SlumbotMatch:
    def __init__(
        self,
        game,
        resolver,
        expected_hands: int,
        events_path: Path | None = None,
        summary_path: Path | None = None,
        telemetry_writer=None,
        logger=None,
        after_hand=None,
        max_consecutive_errors: int = 3,
        max_hand_errors: int = 10,
        host: str = "slumbot.com",
        seed: int | None = None,
    ) -> None:
        if int(expected_hands) < 1:
            raise ValueError("expected_hands must be at least 1")
        self.game = game
        self.resolver = resolver
        self.expected_hands = int(expected_hands)
        self.events_path = Path(events_path) if events_path is not None else None
        self.summary_path = Path(summary_path) if summary_path is not None else None
        self.telemetry_writer = telemetry_writer
        self.logger = logger
        self.after_hand = after_hand
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.max_hand_errors = int(max_hand_errors)
        self.host = host
        self.seed = seed
        self.token: str | None = None
        self.started_at = utc_now()
        self.status = "starting"
        self.hands_completed = 0
        self.hands_attempted = 0
        self.hand_errors = 0
        self.cumulative_winnings = 0
        self.hand_winnings: list[int] = []
        self.cumulative_baseline_winnings = 0.0
        self.hand_baseline_winnings: list[float] = []
        self.baseline_available = True
        self.decisions = 0
        self.action_counts: Counter[str] = Counter()
        self.street_counts: Counter[str] = Counter()
        self.corrections = 0
        self.correction_counts: Counter[str] = Counter()
        self.error_messages: list[str] = []
        self.seat_counts: Counter[str] = Counter()
        self.last_error: str | None = None
        for path in (self.events_path, self.summary_path):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
        self._write_summary()

    # -- logging helpers ---------------------------------------------------

    def _log(self, level: str, message: str) -> None:
        if self.logger is None:
            return
        method = getattr(self.logger, level, None) or getattr(self.logger, "info")
        method(message)

    def _append_event(self, payload: dict[str, object]) -> None:
        if self.events_path is None:
            return
        record = {"timestamp": utc_now(), **payload}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()

    def summary(self, *, finished_at: str | None = None) -> dict[str, object]:
        return {
            "schema_version": 1,
            "updated_at": utc_now(),
            "started_at": self.started_at,
            "finished_at": finished_at,
            "status": self.status,
            "opponent": {"name": "Slumbot", "host": self.host, "api": f"https://{self.host}/api"},
            "game": {"stack": STACK_SIZE, "small_blind": SMALL_BLIND, "big_blind": BIG_BLIND},
            "expected_hands": self.expected_hands,
            "hands_completed": self.hands_completed,
            "hands_attempted": self.hands_attempted,
            "hand_errors": self.hand_errors,
            "cumulative_winnings": self.cumulative_winnings,
            "statistics": match_statistics(self.hand_winnings),
            "cumulative_baseline_winnings": (
                self.cumulative_baseline_winnings if self.baseline_available else None
            ),
            "baseline_statistics": (
                match_statistics(self.hand_baseline_winnings) if self.baseline_available else None
            ),
            "baseline_comparison": (
                paired_statistics(self.hand_winnings, self.hand_baseline_winnings)
                if self.baseline_available
                else None
            ),
            "decisions": self.decisions,
            "action_counts": dict(sorted(self.action_counts.items())),
            "street_decision_counts": dict(sorted(self.street_counts.items())),
            "seat_counts": dict(sorted(self.seat_counts.items())),
            "bet_size_corrections": self.corrections,
            "bet_size_correction_counts": dict(sorted(self.correction_counts.items())),
            "request_retries": int(getattr(self.game, "request_retries", 0)),
            "max_consecutive_errors": self.max_consecutive_errors,
            "max_hand_errors": self.max_hand_errors,
            "bot_seed": self.seed,
            "error": self.last_error,
            "recent_errors": self.error_messages[-5:],
        }

    def _write_summary(self, *, finished_at: str | None = None) -> None:
        if self.summary_path is None:
            return
        payload = self.summary(finished_at=finished_at)
        temporary = self.summary_path.with_name(f".{self.summary_path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.summary_path)

    # -- hand loop ---------------------------------------------------------

    @staticmethod
    def _classify(action, action_string: str | None) -> str:
        if action.action == constants.ACPCActions.fold:
            return "fold"
        if action.action == constants.ACPCActions.ccall:
            return "check" if action_string == "k" else "call"
        if int(getattr(action, "raise_amount", 0)) >= STACK_SIZE:
            return "all_in"
        return "raise"

    def play_hand(self, hand_number: int) -> dict[str, object]:
        started = time.monotonic()
        response = self.game.new_hand(self.token, hand_number)
        token = response.get("token")
        if isinstance(token, str) and token:
            self.token = token
        client_pos = response.get("client_pos")
        self._append_event({"event": "hand_started", "hand_number": hand_number, "client_pos": client_pos})

        decisions = 0
        corrections = 0
        while response.get("winnings") is None:
            if decisions >= MAX_DECISIONS_PER_HAND:
                raise SlumbotProtocolError(f"hand {hand_number} exceeded {MAX_DECISIONS_PER_HAND} decisions")
            state, node = self.game.get_next_situation(response, hand_number)
            if decisions == 0:
                self.resolver.start_new_hand(state)
            advised = self.resolver.compute_action(state, node)
            telemetry = getattr(self.resolver, "last_decision_telemetry", None)
            if self.telemetry_writer is not None and telemetry:
                self.telemetry_writer.append(telemetry)
            response = self.game.play_action(self.token, advised)
            decisions += 1
            action_class = self._classify(advised, getattr(self.game, "last_action_string", None))
            self.action_counts[action_class] += 1
            street = STREET_NAMES.get(int(getattr(state, "current_street", 0)), "unknown")
            self.street_counts[street] += 1
            correction = getattr(self.game, "last_correction", None)
            if correction:
                corrections += 1
                self.correction_counts[str(correction)] += 1
                self._append_event(
                    {
                        "event": "bet_size_correction",
                        "hand_number": hand_number,
                        "street": street,
                        "correction": str(correction),
                    }
                )
            token = response.get("token")
            if isinstance(token, str) and token:
                self.token = token

        winnings = response.get("winnings")
        if type(winnings) is not int:
            raise SlumbotProtocolError(f"hand {hand_number} ended without integer winnings: {winnings!r}")
        return {
            "hand_number": hand_number,
            "winnings": winnings,
            "decisions": decisions,
            "corrections": corrections,
            "seconds": time.monotonic() - started,
            "client_pos": client_pos,
            "final_action": response.get("action"),
            "board": "".join(str(card) for card in (response.get("board") or [])),
            "bot_hole_cards": response.get("bot_hole_cards"),
            "baseline_winnings": response.get("baseline_winnings"),
            "won_pot": response.get("won_pot"),
        }

    def run(self) -> int:
        self.status = "running"
        self._write_summary()
        consecutive_errors = 0
        hand_number = 0
        try:
            while self.hands_completed < self.expected_hands:
                self.hands_attempted += 1
                self._write_summary()
                try:
                    result = self.play_hand(hand_number)
                except SlumbotError as error:
                    message = f"hand {hand_number}: {error}"
                    self.hand_errors += 1
                    consecutive_errors += 1
                    self.last_error = message
                    self.error_messages.append(message)
                    self._append_event({"event": "hand_error", "hand_number": hand_number, "error": str(error)})
                    self._log("error", f"Slumbot hand {hand_number} failed: {error}")
                    self.game.reset_hand()
                    if (
                        consecutive_errors >= self.max_consecutive_errors
                        or self.hand_errors > self.max_hand_errors
                    ):
                        self.status = "failed"
                        self._write_summary(finished_at=utc_now())
                        self._log("critical", f"SLUMBOT_MATCH_FAILED hand_errors={self.hand_errors}")
                        return 1
                    hand_number += 1
                    if self.after_hand is not None:
                        self.after_hand()
                    continue

                consecutive_errors = 0
                self.last_error = None
                self.hands_completed += 1
                self.decisions += int(result["decisions"])
                self.cumulative_winnings += int(result["winnings"])
                self.hand_winnings.append(int(result["winnings"]))
                baseline = result["baseline_winnings"]
                if isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
                    self.cumulative_baseline_winnings += float(baseline)
                    self.hand_baseline_winnings.append(float(baseline))
                else:
                    self.baseline_available = False
                self.corrections += int(result["corrections"])
                self.seat_counts["small_blind" if result["client_pos"] == 1 else "big_blind"] += 1
                if self.telemetry_writer is not None:
                    self.telemetry_writer.append(
                        {
                            "event": "hand_result",
                            "timestamp": utc_now(),
                            "hand_number": int(hand_number),
                            "winnings": int(result["winnings"]),
                            "cumulative_winnings": int(self.cumulative_winnings),
                            "decisions": int(result["decisions"]),
                            "client_pos": result["client_pos"],
                            "final_action": result["final_action"],
                            "board": result["board"],
                            "bot_hole_cards": result["bot_hole_cards"],
                            "baseline_winnings": result["baseline_winnings"],
                            "won_pot": result["won_pot"],
                            "hand_seconds": float(result["seconds"]),
                        }
                    )
                self._append_event(
                    {
                        "event": "hand_result",
                        "hand_number": int(hand_number),
                        "winnings": int(result["winnings"]),
                        "cumulative_winnings": int(self.cumulative_winnings),
                        "decisions": int(result["decisions"]),
                        "client_pos": result["client_pos"],
                        "final_action": result["final_action"],
                        "board": result["board"],
                        "baseline_winnings": result["baseline_winnings"],
                        "won_pot": result["won_pot"],
                        "hand_seconds": round(float(result["seconds"]), 6),
                    }
                )
                self._log(
                    "success",
                    f"Hand {hand_number} completed. Hand winnings: {result['winnings']}, "
                    f"Total winnings: {self.cumulative_winnings} ({self.hands_completed}/{self.expected_hands})",
                )
                self._write_summary()
                hand_number += 1
                if self.after_hand is not None:
                    self.after_hand()
        except Exception as error:  # noqa: BLE001 - record then re-raise unexpected failures
            self.status = "failed"
            self.last_error = f"unexpected {type(error).__name__}: {error}"
            self.error_messages.append(self.last_error)
            self._write_summary(finished_at=utc_now())
            self._log("critical", f"SLUMBOT_MATCH_FAILED {self.last_error}")
            raise

        self.status = "complete"
        self._write_summary(finished_at=utc_now())
        stats = match_statistics(self.hand_winnings)
        self._log(
            "success",
            f"SLUMBOT_MATCH_COMPLETE hands={self.hands_completed} winnings={self.cumulative_winnings} "
            f"mbb_per_hand={stats['mbb_per_hand']:.2f}",
        )
        return 0
