#!/usr/bin/env python3
"""Aggregate per-session Slumbot summaries into one status line.

Prints ``<status> <hands_completed> <hands_attempted> <finished_sessions>``.
``status`` is ``failed`` if any session failed, ``complete`` when every
expected session reports completion, otherwise ``running``. Missing session
summaries count as not started. Exit status 1 means nothing is readable yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def session_dirs(run_dir: Path, sessions: int) -> list[Path]:
    return [run_dir / f"session-{index}" for index in range(sessions)]


def load_summary(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def aggregate(run_dir: Path, sessions: int, summary_name: str = "slumbot-summary.json") -> dict[str, object]:
    summaries = []
    for directory in session_dirs(run_dir, sessions):
        summary = load_summary(directory / summary_name)
        if summary is not None:
            summaries.append(summary)
    statuses = [str(summary.get("status")) for summary in summaries]
    if any(status == "failed" for status in statuses):
        status = "failed"
    elif len(summaries) == sessions and all(status == "complete" for status in statuses):
        status = "complete"
    else:
        status = "running"
    return {
        "status": status,
        "readable_sessions": len(summaries),
        "expected_sessions": sessions,
        "hands_completed": sum(int(summary.get("hands_completed") or 0) for summary in summaries),
        "hands_attempted": sum(int(summary.get("hands_attempted") or 0) for summary in summaries),
        "hand_errors": sum(int(summary.get("hand_errors") or 0) for summary in summaries),
        "cumulative_winnings": sum(int(summary.get("cumulative_winnings") or 0) for summary in summaries),
        "finished_sessions": sum(1 for status in statuses if status in ("complete", "failed")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--summary-name", default="slumbot-summary.json")
    args = parser.parse_args()
    if not 1 <= args.sessions <= 16:
        raise SystemExit("sessions must be between 1 and 16")
    result = aggregate(args.run_dir, args.sessions, args.summary_name)
    if result["readable_sessions"] == 0:
        raise SystemExit(1)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result["status"], result["hands_completed"], result["hands_attempted"], result["finished_sessions"])


if __name__ == "__main__":
    main()
