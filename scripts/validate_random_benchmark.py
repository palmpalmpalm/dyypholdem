#!/usr/bin/env python3
"""Fail closed unless a seeded random benchmark completed exactly as requested."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"missing or invalid benchmark artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"benchmark artifact is not an object: {path.name}")
    return value


def validate(run_dir: Path, expected_hands: int) -> dict[str, int | bool]:
    timing = load_json(run_dir / "timing_report.json")
    random_summary = load_json(run_dir / "random-summary.json")
    match = timing.get("match")
    if not isinstance(match, dict):
        raise ValueError("timing report has no match summary")

    bot_hands = match.get("hands_completed")
    random_hands = random_summary.get("hands_completed")
    configured_hands = random_summary.get("expected_hands")
    latest_hand = match.get("latest_hand_number")
    if type(bot_hands) is not int or bot_hands != expected_hands:
        raise ValueError(f"bot telemetry completed {bot_hands!r} of {expected_hands} hands")
    if type(random_hands) is not int or random_hands != expected_hands:
        raise ValueError(f"random opponent completed {random_hands!r} of {expected_hands} hands")
    if type(configured_hands) is not int or configured_hands != expected_hands:
        raise ValueError("random opponent expected-hand count does not match the request")
    if latest_hand != expected_hands - 1:
        raise ValueError(f"latest hand is {latest_hand!r}, expected {expected_hands - 1}")
    if random_summary.get("status") != "complete" or random_summary.get("error") is not None:
        raise ValueError("random opponent did not record clean completion")

    decision_count = timing.get("decision_count")
    if type(decision_count) is not int or decision_count <= 0:
        raise ValueError("bot telemetry contains no decisions")
    bot_winnings = match.get("cumulative_winnings")
    random_winnings = random_summary.get("cumulative_winnings")
    if type(bot_winnings) is not int or type(random_winnings) is not int:
        raise ValueError("match winnings are missing")
    if bot_winnings != -random_winnings:
        raise ValueError("bot and random-opponent winnings are not zero-sum")

    return {
        "valid": True,
        "hands_completed": expected_hands,
        "decision_count": decision_count,
        "bot_winnings": bot_winnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hands", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.hands <= 1000:
        raise SystemExit("hands must be between 1 and 1000")
    try:
        result = validate(args.run_dir, args.hands)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
