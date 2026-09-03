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


def session_directories(run_dir: Path) -> list[Path]:
    """Concurrent-session layout: run_dir/session-<i>/...; otherwise the run root."""
    found = sorted(
        (path for path in run_dir.glob("session-*") if path.is_dir() and path.name[8:].isdigit()),
        key=lambda path: int(path.name[8:]),
    )
    return found or [run_dir]


def merged_summary(directories: list[Path], summary_name: str = "slumbot-summary.json") -> dict:
    summaries = []
    for directory in directories:
        path = directory / summary_name
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        return {}
    if len(summaries) == 1:
        return summaries[0]
    statuses = [str(item.get("status")) for item in summaries]
    if any(status == "failed" for status in statuses):
        status = "failed"
    elif all(status == "complete" for status in statuses) and len(summaries) == len(directories):
        status = "complete"
    else:
        status = "running"
    actions: dict[str, int] = {}
    for item in summaries:
        for key, value in (item.get("action_counts") or {}).items():
            actions[key] = actions.get(key, 0) + int(value)
    return {
        "status": status,
        "sessions": len(summaries),
        "expected_hands": sum(int(item.get("expected_hands") or 0) for item in summaries),
        "hand_errors": sum(int(item.get("hand_errors") or 0) for item in summaries),
        "request_retries": sum(int(item.get("request_retries") or 0) for item in summaries),
        "bet_size_corrections": sum(int(item.get("bet_size_corrections") or 0) for item in summaries),
        "action_counts": dict(sorted(actions.items())),
    }


def paired_comparison(winnings: list[int], baselines: list[float]) -> dict[str, float | int | None]:
    """Agreement and variance reduction of the server baseline against raw chips."""
    count = len(winnings)
    if count < 2 or len(baselines) != count:
        return {"hands": count, "correlation": None, "variance_ratio": None,
                "mean_difference_mbb": None, "ci95_difference_mbb": None}
    factor = 1000.0 / BIG_BLIND
    raw_stdev = statistics.stdev(winnings)
    baseline_stdev = statistics.stdev(baselines)
    raw_mean = statistics.fmean(winnings)
    baseline_mean = statistics.fmean(baselines)
    covariance = sum((float(a) - raw_mean) * (b - baseline_mean) for a, b in zip(winnings, baselines)) / (count - 1)
    differences = [float(a) - b for a, b in zip(winnings, baselines)]
    difference_se = statistics.stdev(differences) / math.sqrt(count) * factor
    denominator = raw_stdev * baseline_stdev
    return {
        "hands": count,
        "correlation": covariance / denominator if denominator > 0 else None,
        "variance_ratio": (raw_stdev ** 2) / (baseline_stdev ** 2) if baseline_stdev > 0 else None,
        "mean_difference_mbb": statistics.fmean(differences) * factor,
        "ci95_difference_mbb": 1.96 * difference_se,
    }


def build_report(run_dir: Path, summary_name: str = "slumbot-summary.json", events_name: str = "slumbot-events.jsonl") -> dict:
    directories = session_directories(run_dir)
    events = []
    decisions = []
    for directory in directories:
        events.extend(load_jsonl(directory / events_name))
        decisions.extend(row for row in load_jsonl(directory / "decisions.jsonl") if row.get("event") == "decision")
    events.sort(key=lambda row: str(row.get("timestamp", "")))
    summary = merged_summary(directories, summary_name)
    results = [row for row in events if row.get("event") == "hand_result"]
    started = [row for row in events if row.get("event") == "hand_started"]
    winnings = [int(row["winnings"]) for row in results]
    stats = hand_statistics(winnings)
    baselines = [row.get("baseline_winnings") for row in results]
    baseline_available = bool(results) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in baselines
    )
    baseline_stats = hand_statistics([float(value) for value in baselines]) if baseline_available else None
    baseline_comparison = (
        paired_comparison(winnings, [float(value) for value in baselines]) if baseline_available else None
    )

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
    def seat(row):
        position = row.get("client_pos", row.get("position"))
        return "small_blind" if position == 1 else "big_blind"

    seat_counts = {"small_blind": 0, "big_blind": 0}
    seat_chips = {"small_blind": 0, "big_blind": 0}
    for row in results:
        seat_counts[seat(row)] += 1
        seat_chips[seat(row)] += int(row["winnings"])

    return {
        "run_dir": str(run_dir),
        "sessions": len(directories),
        "status": summary.get("status"),
        "expected_hands": summary.get("expected_hands"),
        "hand_errors": summary.get("hand_errors"),
        "request_retries": summary.get("request_retries"),
        "bet_size_corrections": summary.get("bet_size_corrections"),
        "action_counts": summary.get("action_counts"),
        "statistics": stats,
        "baseline_statistics": baseline_stats,
        "baseline_comparison": baseline_comparison,
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
        f"| Status | {report['status']} ({fmt(report.get('sessions', 1))} session(s)) |",
        f"| Hands completed / requested | {fmt(stats['hands'])} / {fmt(report['expected_hands'])} |",
        f"| Bot decisions (per hand) | {fmt(report['decisions'])} ({fmt(report['decisions_per_hand'])}) |",
        f"| Net chips | {fmt(stats['chips'])} |",
        f"| Result | {fmt(stats.get('mbb_per_hand'))} mbb/hand, SE {fmt(stats.get('se_mbb'))}, 95% CI ±{fmt(stats.get('ci95_mbb'))} |",
    ]
    baseline = report.get("baseline_statistics")
    comparison = report.get("baseline_comparison") or {}
    if baseline:
        lines += [
            f"| Baseline score | {fmt(baseline.get('mbb_per_hand'))} mbb/hand, "
            f"95% CI ±{fmt(baseline.get('ci95_mbb'))} |",
            f"| Baseline vs raw | variance ratio {fmt(comparison.get('variance_ratio'), 2)}, "
            f"correlation {fmt(comparison.get('correlation'), 3)} |",
            f"| Raw minus baseline | {fmt(comparison.get('mean_difference_mbb'))} mbb/hand, "
            f"95% CI ±{fmt(comparison.get('ci95_difference_mbb'))} |",
        ]
    lines += [
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
    parser.add_argument("--summary-name", default="slumbot-summary.json")
    parser.add_argument("--events-name", default="slumbot-events.jsonl")
    args = parser.parse_args()
    report = build_report(args.run_dir, args.summary_name, args.events_name)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
