#!/usr/bin/env python3
"""Fail closed unless a Slumbot benchmark completed exactly as requested."""

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


def validate(run_dir: Path, expected_hands: int) -> dict[str, object]:
    timing = load_json(run_dir / "timing_report.json")
    summary = load_json(run_dir / "slumbot-summary.json")
    match = timing.get("match")
    if not isinstance(match, dict):
        raise ValueError("timing report has no match summary")

    bot_hands = match.get("hands_completed")
    summary_hands = summary.get("hands_completed")
    configured_hands = summary.get("expected_hands")
    if type(bot_hands) is not int or bot_hands != expected_hands:
        raise ValueError(f"bot telemetry completed {bot_hands!r} of {expected_hands} hands")
    if type(summary_hands) is not int or summary_hands != expected_hands:
        raise ValueError(f"Slumbot summary completed {summary_hands!r} of {expected_hands} hands")
    if type(configured_hands) is not int or configured_hands != expected_hands:
        raise ValueError("Slumbot summary expected-hand count does not match the request")
    if summary.get("status") != "complete" or summary.get("error") is not None:
        raise ValueError("Slumbot match did not record clean completion")

    decision_count = timing.get("decision_count")
    if type(decision_count) is not int or decision_count <= 0:
        raise ValueError("bot telemetry contains no decisions")
    bot_winnings = match.get("cumulative_winnings")
    summary_winnings = summary.get("cumulative_winnings")
    if type(bot_winnings) is not int or type(summary_winnings) is not int:
        raise ValueError("match winnings are missing")
    if bot_winnings != summary_winnings:
        raise ValueError("bot telemetry and Slumbot summary winnings disagree")
    hand_errors = summary.get("hand_errors")
    if type(hand_errors) is not int or hand_errors < 0:
        raise ValueError("Slumbot summary has no hand error count")
    statistics = summary.get("statistics")
    mbb_per_hand = statistics.get("mbb_per_hand") if isinstance(statistics, dict) else None

    return {
        "valid": True,
        "hands_completed": expected_hands,
        "decision_count": decision_count,
        "bot_winnings": bot_winnings,
        "mbb_per_hand": mbb_per_hand,
        "hand_errors": hand_errors,
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
