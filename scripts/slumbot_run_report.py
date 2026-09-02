#!/usr/bin/env python3
"""Summarize a DyypHoldem versus Slumbot run directory as Markdown.

Works on partial runs (for example one cut off by the paid-resource guard):
every figure is computed from completed hands only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

BIG_BLIND = 100
STREETS = ("preflop", "flop", "turn", "river")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def hand_statistics(winnings: list[int]) -> dict[str, float | int | None]:
    count = len(winnings)
    if count == 0:
        return {"hands": 0, "chips": 0, "mbb_per_hand": None, "se_mbb": None, "ci95_mbb": None}
    mean = statistics.fmean(winnings)
    factor = 1000.0 / BIG_BLIND
    stdev = statistics.stdev(winnings) if count > 1 else 0.0
    se = stdev / math.sqrt(count) * factor
    return {
        "hands": count,
        "chips": sum(winnings),
        "mean_chips": mean,
        "mbb_per_hand": mean * factor,
        "stdev_chips": stdev,
        "se_mbb": se,
        "ci95_mbb": 1.96 * se,
        "wins": sum(1 for w in winnings if w > 0),
        "losses": sum(1 for w in winnings if w < 0),
        "ties": sum(1 for w in winnings if w == 0),
    }


def build_report(run_dir: Path) -> dict:
    events = load_jsonl(run_dir / "slumbot-events.jsonl")
    decisions = [row for row in load_jsonl(run_dir / "decisions.jsonl") if row.get("event") == "decision"]
    summary_path = run_dir / "slumbot-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    results = [row for row in events if row.get("event") == "hand_result"]
    started = [row for row in events if row.get("event") == "hand_started"]
    winnings = [int(row["winnings"]) for row in results]
    stats = hand_statistics(winnings)

    wall_seconds = None
    if started and results:
        first = datetime.fromisoformat(started[0]["timestamp"])
        last = datetime.fromisoformat(results[-1]["timestamp"])
        wall_seconds = (last - first).total_seconds()

    by_street = {}
    for street in STREETS:
        rows = [row for row in decisions if row.get("street") == street]
        response = [float(row.get("total_response_seconds", 0.0)) for row in rows]
        cfr = [float(row.get("cfr_seconds", 0.0)) for row in rows]
        by_street[street] = {
            "decisions": len(rows),
            "response_mean": statistics.fmean(response) if response else None,
            "response_p95": sorted(response)[max(0, math.ceil(0.95 * len(response)) - 1)] if response else None,
            "response_max": max(response) if response else None,
            "cfr_mean": statistics.fmean(cfr) if cfr else None,
        }
    seat_counts = {"small_blind": 0, "big_blind": 0}
    for row in results:
        seat_counts["small_blind" if row.get("client_pos") == 1 else "big_blind"] += 1
    seat_chips = {"small_blind": 0, "big_blind": 0}
    for row in results:
        seat_chips["small_blind" if row.get("client_pos") == 1 else "big_blind"] += int(row["winnings"])

    return {
        "run_dir": str(run_dir),
        "status": summary.get("status"),
        "expected_hands": summary.get("expected_hands"),
        "hand_errors": summary.get("hand_errors"),
        "request_retries": summary.get("request_retries"),
        "bet_size_corrections": summary.get("bet_size_corrections"),
        "action_counts": summary.get("action_counts"),
        "statistics": stats,
        "decisions": len(decisions),
        "decisions_per_hand": len(decisions) / len(results) if results else None,
        "match_wall_seconds": wall_seconds,
        "seconds_per_hand": wall_seconds / len(results) if wall_seconds and results else None,
        "by_street": by_street,
        "seat_counts": seat_counts,
        "seat_chips": seat_chips,
    }


def fmt(value, digits=2, suffix=""):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value:,}{suffix}"


def render_markdown(report: dict) -> str:
    stats = report["statistics"]
    lines = [
        f"Run: `{Path(report['run_dir']).name}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {report['status']} |",
        f"| Hands completed / requested | {fmt(stats['hands'])} / {fmt(report['expected_hands'])} |",
        f"| Bot decisions (per hand) | {fmt(report['decisions'])} ({fmt(report['decisions_per_hand'])}) |",
        f"| Net chips | {fmt(stats['chips'])} |",
        f"| Result | {fmt(stats.get('mbb_per_hand'))} mbb/hand, SE {fmt(stats.get('se_mbb'))}, 95% CI ±{fmt(stats.get('ci95_mbb'))} |",
        f"| Hands won / lost / tied | {fmt(stats.get('wins'))} / {fmt(stats.get('losses'))} / {fmt(stats.get('ties'))} |",
        f"| Small blind hands (chips) | {fmt(report['seat_counts']['small_blind'])} ({fmt(report['seat_chips']['small_blind'])}) |",
        f"| Big blind hands (chips) | {fmt(report['seat_counts']['big_blind'])} ({fmt(report['seat_chips']['big_blind'])}) |",
        f"| Match wall time | {fmt(report['match_wall_seconds'], 1, ' s')} ({fmt(report['seconds_per_hand'], 1, ' s/hand')}) |",
        f"| Hand errors / request retries / bet corrections | {fmt(report['hand_errors'])} / {fmt(report['request_retries'])} / {fmt(report['bet_size_corrections'])} |",
        "",
        "| Street | Decisions | Response mean | p95 | Max | CFR mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for street in STREETS:
        row = report["by_street"][street]
        lines.append(
            f"| {street} | {fmt(row['decisions'])} | {fmt(row['response_mean'], 3, ' s')} | "
            f"{fmt(row['response_p95'], 3, ' s')} | {fmt(row['response_max'], 3, ' s')} | {fmt(row['cfr_mean'], 3, ' s')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="print the raw report as JSON")
    args = parser.parse_args()
    report = build_report(args.run_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
