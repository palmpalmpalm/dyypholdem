#!/usr/bin/env python3
"""Seeded random legal-action opponent for the local DyypHoldem web seat.

The web bridge and ACPC dealer remain authoritative.  This client only reads
the bridge's allowlisted state, samples one of the actions the bridge exposes,
and submits the state nonce with the selected action.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
import json
from pathlib import Path
import random
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decision_rng(seed: int, hand_number: int, state_nonce: int) -> random.Random:
    """Return a stable per-state RNG so HTTP retries cannot change an action."""

    mixed = (
        (int(seed) * 0x9E3779B185EBCA87)
        ^ (int(hand_number) * 0xC2B2AE3D27D4EB4F)
        ^ (int(state_nonce) * 0x165667B19E3779F9)
    ) & ((1 << 64) - 1)
    return random.Random(mixed)


def choose_action(snapshot: dict[str, Any], seed: int) -> tuple[dict[str, int | str], str]:
    """Uniformly sample a strategic action class and, for raises, a preset.

    Facing a wager, the classes are fold/call/raise.  When checking is free,
    the classes are check/raise; free folds are intentionally excluded as a
    dominated action.  Raise sizes are sampled uniformly from the bridge's
    deduplicated Min, half-pot, three-quarter-pot, pot, and all-in presets.
    """

    nonce = snapshot.get("state_nonce")
    hand_number = snapshot.get("hand_number")
    if type(nonce) is not int or type(hand_number) is not int:
        raise ValueError("state nonce and hand number must be integers")

    available = snapshot.get("available_actions")
    if not isinstance(available, list):
        raise ValueError("available_actions must be a list")
    by_id = {
        str(item.get("id")): item
        for item in available
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }

    classes: list[str] = []
    if "call" in by_id:
        if "fold" in by_id:
            classes.append("fold")
        classes.append("call")
    elif "check" in by_id:
        classes.append("check")
    if "raise" in by_id:
        classes.append("raise")
    if not classes:
        raise ValueError("the bridge exposed no usable legal action")

    rng = _decision_rng(seed, hand_number, nonce)
    action_id = rng.choice(classes)
    payload: dict[str, int | str] = {
        "action": action_id,
        "state_nonce": nonce,
    }
    policy_label = action_id

    if action_id == "raise":
        raise_action = by_id["raise"]
        presets_raw = raise_action.get("presets")
        if not isinstance(presets_raw, list):
            raise ValueError("raise action has no server-generated presets")
        presets = [
            item
            for item in presets_raw
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and type(item.get("raise_to")) is int
        ]
        if not presets:
            raise ValueError("raise action has no usable server-generated preset")
        preset = rng.choice(presets)
        payload["raise_to"] = int(preset["raise_to"])
        policy_label = f"raise:{preset['id']}"

    return payload, policy_label


class RandomUIPlayer:
    def __init__(
        self,
        base_url: str,
        token: str,
        seed: int,
        expected_hands: int,
        events_path: Path,
        summary_path: Path,
        poll_seconds: float = 0.1,
        idle_timeout_seconds: float = 900.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.seed = int(seed)
        self.expected_hands = int(expected_hands)
        self.events_path = Path(events_path)
        self.summary_path = Path(summary_path)
        self.poll_seconds = float(poll_seconds)
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        self.started_at = utc_now()
        self.last_progress = time.monotonic()
        self.last_signature: tuple[object, ...] | None = None
        self.accepted_nonces: set[int] = set()
        self.action_counts: Counter[str] = Counter()
        self.street_counts: Counter[str] = Counter()
        self.conflicts = 0
        self.rate_limit_backoffs = 0
        self.request_retries = 0
        self.hands_completed = 0
        self.cumulative_winnings = 0
        self.status = "starting"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    def _request(self, method: str, path: str, payload: dict[str, object] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Session-Token": self.token,
            },
        )
        with urlopen(request, timeout=15) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def _append_event(self, payload: dict[str, object]) -> None:
        record = {"timestamp": utc_now(), **payload}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()

    def _write_summary(self, *, finished_at: str | None = None, error: str | None = None) -> None:
        payload = {
            "schema_version": 1,
            "updated_at": utc_now(),
            "started_at": self.started_at,
            "finished_at": finished_at,
            "status": self.status,
            "expected_hands": self.expected_hands,
            "hands_completed": self.hands_completed,
            "cumulative_winnings": self.cumulative_winnings,
            "seed": self.seed,
            "decisions": sum(self.action_counts.values()),
            "action_counts": dict(sorted(self.action_counts.items())),
            "street_action_counts": dict(sorted(self.street_counts.items())),
            "stale_state_conflicts": self.conflicts,
            "rate_limit_backoffs": self.rate_limit_backoffs,
            "request_retries": self.request_retries,
            "error": error,
            "policy": {
                "action_classes": "uniform fold/call/raise when facing a wager; uniform check/raise otherwise",
                "raise_sizes": "uniform over deduplicated server-generated min/half-pot/three-quarter-pot/pot/all-in presets",
                "free_folds": "excluded",
                "legality": "PokerKit 0.7.5 validation plus authoritative ACPC dealer",
            },
        }
        temporary = self.summary_path.with_name(f".{self.summary_path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.summary_path)

    def _observe(self, snapshot: dict[str, Any]) -> None:
        status = snapshot.get("status")
        hands = snapshot.get("hands_completed")
        winnings = snapshot.get("cumulative_winnings")
        nonce = snapshot.get("state_nonce")
        signature = (status, hands, winnings, nonce)
        changed = signature != self.last_signature
        if changed:
            self.last_signature = signature
            self.last_progress = time.monotonic()
        if isinstance(status, str):
            self.status = status
        if type(hands) is int:
            self.hands_completed = hands
        if type(winnings) is int:
            self.cumulative_winnings = winnings
        if changed:
            self._write_summary()

    def run(self) -> int:
        self._append_event(
            {
                "event": "random_player_started",
                "seed": self.seed,
                "expected_hands": self.expected_hands,
            }
        )
        try:
            while True:
                try:
                    snapshot = self._request("GET", "/api/state")
                except HTTPError as error:
                    if error.code == HTTPStatus.TOO_MANY_REQUESTS:
                        self.rate_limit_backoffs += 1
                    elif error.code < 500:
                        raise
                    self.request_retries += 1
                    if time.monotonic() - self.last_progress > self.idle_timeout_seconds:
                        raise RuntimeError("bridge did not provide progress before the idle timeout")
                    time.sleep(min(1.0, max(self.poll_seconds, 0.1)))
                    continue
                except (URLError, TimeoutError, json.JSONDecodeError):
                    self.request_retries += 1
                    if time.monotonic() - self.last_progress > self.idle_timeout_seconds:
                        raise RuntimeError("bridge did not provide progress before the idle timeout")
                    time.sleep(min(1.0, max(self.poll_seconds, 0.1)))
                    continue

                if not isinstance(snapshot, dict):
                    raise RuntimeError("bridge returned a non-object state")
                self._observe(snapshot)
                if self.status == "match_complete":
                    if self.hands_completed != self.expected_hands:
                        raise RuntimeError(
                            f"match ended after {self.hands_completed} of {self.expected_hands} hands"
                        )
                    finished_at = utc_now()
                    self.status = "complete"
                    self._append_event(
                        {
                            "event": "random_player_complete",
                            "hands_completed": self.hands_completed,
                            "cumulative_winnings": self.cumulative_winnings,
                        }
                    )
                    self._write_summary(finished_at=finished_at)
                    return 0
                if self.status == "error":
                    raise RuntimeError(str(snapshot.get("error") or "bridge entered error state"))
                if time.monotonic() - self.last_progress > self.idle_timeout_seconds:
                    raise RuntimeError("match state did not progress before the idle timeout")
                if self.status != "your_turn":
                    time.sleep(self.poll_seconds)
                    continue

                nonce = snapshot.get("state_nonce")
                if type(nonce) is not int:
                    raise RuntimeError("bridge returned an invalid state nonce")
                if nonce in self.accepted_nonces:
                    time.sleep(self.poll_seconds)
                    continue

                payload, policy_label = choose_action(snapshot, self.seed)
                try:
                    self._request("POST", "/api/action", payload)
                except HTTPError as error:
                    if error.code == HTTPStatus.CONFLICT:
                        self.conflicts += 1
                        time.sleep(self.poll_seconds)
                        continue
                    if error.code == HTTPStatus.TOO_MANY_REQUESTS:
                        self.rate_limit_backoffs += 1
                        time.sleep(2.0)
                        continue
                    raise

                self.accepted_nonces.add(nonce)
                street = str(snapshot.get("street") or "unknown")
                self.action_counts[policy_label] += 1
                self.street_counts[street] += 1
                event = {
                    "event": "random_action",
                    "hand_number": snapshot.get("hand_number"),
                    "state_nonce": nonce,
                    "street": street,
                    "pot": snapshot.get("pot"),
                    "action": payload["action"],
                    "policy_action": policy_label,
                }
                if "raise_to" in payload:
                    event["raise_to"] = payload["raise_to"]
                self._append_event(event)
                self.last_progress = time.monotonic()
                self._write_summary()
        except Exception as error:
            self.status = "error"
            safe_error = f"{type(error).__name__}: {error}"
            self._append_event({"event": "random_player_error", "error": safe_error})
            self._write_summary(finished_at=utc_now(), error=safe_error)
            return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hands", type=int, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--idle-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    if not 0 <= args.seed <= 2_147_483_647:
        raise SystemExit("seed must be between 0 and 2147483647")
    if not 1 <= args.hands <= 1000:
        raise SystemExit("hands must be between 1 and 1000")
    if not 0.01 <= args.poll_seconds <= 10:
        raise SystemExit("poll-seconds must be between 0.01 and 10")
    if not 30 <= args.idle_timeout_seconds <= 3600:
        raise SystemExit("idle-timeout-seconds must be between 30 and 3600")
    token = args.token_file.read_text(encoding="utf-8").strip()
    if len(token) < 24:
        raise SystemExit("session token is missing or too short")

    player = RandomUIPlayer(
        args.base_url,
        token,
        args.seed,
        args.hands,
        args.events,
        args.summary,
        poll_seconds=args.poll_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )
    raise SystemExit(player.run())


if __name__ == "__main__":
    main()
