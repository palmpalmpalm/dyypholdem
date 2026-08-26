"""Structured, privacy-aware telemetry for live DyypHoldem decisions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


STREETS = ("preflop", "flop", "turn", "river")
PHASE_FIELDS = (
    "invariant_seconds",
    "chance_reconstruction_seconds",
    "terminal_equity_seconds",
    "public_tree_seconds",
    "lookahead_tensor_seconds",
    "lookahead_build_seconds",
    "cfr_seconds",
    "results_seconds",
    "resolve_total_seconds",
    "sampling_seconds",
    "total_response_seconds",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(model_root: Path | None) -> list[dict[str, object]]:
    if model_root is None:
        return []
    out = []
    for street in ("preflop-aux", "flop", "turn", "river"):
        path = model_root / street / "final_compact.pt"
        if path.is_file():
            out.append(
                {
                    "street": street,
                    "file": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return out


def percentile(values: list[float], fraction: float) -> float:
    """Linear percentile matching common monitoring dashboards."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        return {"count": 0, "total": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(samples),
        "total": sum(samples),
        "mean": statistics.fmean(samples),
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "max": max(samples),
    }


def normalized_hand_number(value: object) -> int | None:
    """Normalize legacy ACPC string hand ids without accepting loose coercions."""

    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return None


def build_report(events: Iterable[dict[str, object]], metadata: dict[str, object]) -> dict[str, object]:
    event_list = list(events)
    decisions = [event for event in event_list if event.get("event") == "decision"]
    hand_results = [event for event in event_list if event.get("event") == "hand_result"]
    initializations = [event for event in event_list if event.get("event") == "initialization"]
    initialization = initializations[-1] if initializations else None
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for decision in decisions:
        grouped[str(decision.get("street", "unknown"))].append(decision)

    def summarize_decisions(street_events: list[dict[str, object]]) -> dict[str, object]:
        phase_summary = {
            field.removesuffix("_seconds"): summarize_values(
                float(item.get(field, 0.0)) for item in street_events
            )
            for field in PHASE_FIELDS
        }
        return {
            "decisions": len(street_events),
            "timing_seconds": phase_summary,
            "latest_action": street_events[-1].get("chosen_action") if street_events else None,
        }

    by_street: dict[str, object] = {}
    for street in STREETS:
        street_events = grouped.get(street, [])
        by_street[street] = summarize_decisions(street_events)

    preflop_events = grouped.get("preflop", [])
    preflop_modes = {
        "cached_root": summarize_decisions(
            [item for item in preflop_events if item.get("reused_root_precompute") is True]
        ),
        "fresh_resolve": summarize_decisions(
            [item for item in preflop_events if item.get("reused_root_precompute") is not True]
        ),
    }

    captured_chance = [
        item for item in decisions if item.get("chance_captured_flop") is True
    ]
    replayed_chance = [
        item for item in decisions if item.get("chance_replayed_flop") is True
    ]
    unclassified_chance = [
        item
        for item in decisions
        if float(item.get("chance_reconstruction_seconds", 0.0)) > 0
        and item.get("chance_captured_flop") is not True
        and item.get("chance_replayed_flop") is not True
    ]
    chance_reconstruction = {
        "captured_flop": {
            "count": len(captured_chance),
            "timing_seconds": summarize_values(
                item.get("chance_reconstruction_seconds", 0.0)
                for item in captured_chance
            ),
        },
        "replayed_flop": {
            "count": len(replayed_chance),
            "timing_seconds": summarize_values(
                item.get("chance_reconstruction_seconds", 0.0)
                for item in replayed_chance
            ),
        },
        "unclassified": {
            "count": len(unclassified_chance),
            "timing_seconds": summarize_values(
                item.get("chance_reconstruction_seconds", 0.0)
                for item in unclassified_chance
            ),
        },
    }

    bucketing_cache_events = [
        item
        for item in decisions
        if item.get("bucketing_cache_hit") is True
        or item.get("bucketing_cache_hit") is False
    ]
    bucketing_cache_hits = [
        item
        for item in bucketing_cache_events
        if item.get("bucketing_cache_hit") is True
    ]
    bucketing_cache_misses = [
        item
        for item in bucketing_cache_events
        if item.get("bucketing_cache_hit") is False
    ]
    postflop_bucketing_cache = {
        "eligible_decisions": len(bucketing_cache_events),
        "hits": len(bucketing_cache_hits),
        "misses": len(bucketing_cache_misses),
        "hit_rate": (
            len(bucketing_cache_hits) / len(bucketing_cache_events)
            if bucketing_cache_events
            else 0.0
        ),
        "hit_lookahead_build_seconds": summarize_values(
            item.get("lookahead_build_seconds", 0.0)
            for item in bucketing_cache_hits
        ),
        "miss_lookahead_build_seconds": summarize_values(
            item.get("lookahead_build_seconds", 0.0)
            for item in bucketing_cache_misses
        ),
        "max_transform_bytes": max(
            (
                int(item.get("bucketing_transform_bytes", 0))
                for item in bucketing_cache_events
            ),
            default=0,
        ),
    }

    completed_hand_numbers = {
        hand_number
        for item in hand_results
        if (hand_number := normalized_hand_number(item.get("hand_number"))) is not None
    }
    latest_hand_result = hand_results[-1] if hand_results else {}

    recent = []
    for item in decisions[-12:]:
        recent.append(
            {
                "timestamp": item.get("timestamp"),
                "hand_number": item.get("hand_number"),
                "decision_number": item.get("decision_number"),
                "street": item.get("street"),
                "board": item.get("board"),
                "pot": item.get("pot"),
                "chosen_action": item.get("chosen_action"),
                "cfr_iterations": item.get("cfr_iterations"),
                "total_response_seconds": item.get("total_response_seconds"),
                "cfr_seconds": item.get("cfr_seconds"),
                "chance_reconstruction_seconds": item.get(
                    "chance_reconstruction_seconds"
                ),
                "chance_captured_flop": item.get("chance_captured_flop"),
                "chance_replayed_flop": item.get("chance_replayed_flop"),
                "bucketing_cache_hit": item.get("bucketing_cache_hit"),
                "bucketing_transform_bytes": item.get(
                    "bucketing_transform_bytes"
                ),
                "peak_cuda_allocated_bytes": item.get("peak_cuda_allocated_bytes"),
            }
        )

    return {
        "schema_version": 1,
        "updated_at": utc_now(),
        "metadata": metadata,
        "initialization": initialization,
        "decision_count": len(decisions),
        "match": {
            "hands_completed": len(completed_hand_numbers),
            "latest_hand_number": max(completed_hand_numbers) if completed_hand_numbers else None,
            "cumulative_winnings": int(latest_hand_result.get("cumulative_winnings", 0)),
        },
        "by_street": by_street,
        "preflop_solve_modes": preflop_modes,
        "chance_reconstruction": chance_reconstruction,
        "postflop_bucketing_cache": postflop_bucketing_cache,
        "recent_decisions": recent,
    }


def render_text_report(report: dict[str, object]) -> str:
    metadata = report.get("metadata", {})
    lines = [
        "DyypHoldem live calculation report",
        f"Updated: {report.get('updated_at')}",
        f"Decisions: {report.get('decision_count', 0)}",
        f"Hands completed: {int((report.get('match') or {}).get('hands_completed', 0))}",
        f"Bot winnings: {int((report.get('match') or {}).get('cumulative_winnings', 0))} chips",
        f"GPU: {metadata.get('gpu_name', 'unknown')}",
        f"CFR: {metadata.get('cfr_iterations', 'unknown')} iterations "
        f"({metadata.get('cfr_skip_iterations', 'unknown')} skipped)",
        f"Root precompute: {float((report.get('initialization') or {}).get('seconds', 0.0)):.6f} seconds",
        "",
        "Street timing (seconds)",
    ]
    for street in STREETS:
        data = report["by_street"][street]
        total = data["timing_seconds"]["total_response"]
        cfr = data["timing_seconds"]["cfr"]
        lines.append(
            f"- {street}: n={data['decisions']}, response mean={total['mean']:.6f}, "
            f"p50={total['p50']:.6f}, p95={total['p95']:.6f}, max={total['max']:.6f}, "
            f"CFR mean={cfr['mean']:.6f}"
        )
    lines.extend(["", "Preflop timing split (seconds)"])
    for mode in ("cached_root", "fresh_resolve"):
        data = report["preflop_solve_modes"][mode]
        total = data["timing_seconds"]["total_response"]
        lines.append(
            f"- {mode}: n={data['decisions']}, response mean={total['mean']:.6f}, "
            f"p50={total['p50']:.6f}, p95={total['p95']:.6f}, max={total['max']:.6f}"
        )
    lines.extend(["", "Preflop-to-flop chance reconstruction (seconds)"])
    for mode in ("captured_flop", "replayed_flop", "unclassified"):
        data = report["chance_reconstruction"][mode]
        timing = data["timing_seconds"]
        lines.append(
            f"- {mode}: n={data['count']}, total={timing['total']:.6f}, "
            f"mean={timing['mean']:.6f}, p95={timing['p95']:.6f}, "
            f"max={timing['max']:.6f}"
        )
    cache = report["postflop_bucketing_cache"]
    lines.extend(
        [
            "",
            "Postflop bucketing-transform cache",
            f"- eligible={cache['eligible_decisions']}, hits={cache['hits']}, "
            f"misses={cache['misses']}, hit rate={cache['hit_rate']:.3f}",
            f"- hit build mean={cache['hit_lookahead_build_seconds']['mean']:.6f}s, "
            f"miss build mean={cache['miss_lookahead_build_seconds']['mean']:.6f}s, "
            f"max transform bytes={cache['max_transform_bytes']}",
        ]
    )
    return "\n".join(lines) + "\n"


class DecisionTelemetryWriter:
    """Append full private records and atomically refresh safe reports."""

    def __init__(self, jsonl_path: Path, report_path: Path, text_report_path: Path, metadata: dict[str, object]):
        self.jsonl_path = Path(jsonl_path)
        self.report_path = Path(report_path)
        self.text_report_path = Path(text_report_path)
        self.metadata = dict(metadata)
        self.events: list[dict[str, object]] = []
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.text_report_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, object]) -> None:
        record = dict(event)
        record.setdefault("timestamp", utc_now())
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        self.events.append(record)
        self.refresh_report()

    def refresh_report(self) -> dict[str, object]:
        report = build_report(self.events, self.metadata)
        self._atomic_text(self.report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        self._atomic_text(self.text_report_path, render_text_report(report))
        return report

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
