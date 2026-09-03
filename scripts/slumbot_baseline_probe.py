#!/usr/bin/env python3
"""Measure Slumbot's server-side baseline score against the raw chip result.

Slumbot's API returns ``baseline_winnings`` next to ``winnings`` on every hand.
If that baseline is an unbiased estimate of the same quantity, it can replace
the raw chip count as the score and shrink the confidence interval for free.
Neither property can be assumed, so this probe measures both:

* agreement -- the paired mean of raw minus baseline must be consistent with
  zero, otherwise the baseline is measuring something else and using it would
  bias every strength claim;
* variance reduction -- the ratio of variances says how many hands the baseline
  saves.

The probe plays with a fixed trivial policy and never invokes the solver, so it
costs no GPU time. The variance-reduction factor it measures is a property of
the baseline and the game rather than of a particular strategy, but the
agreement check only licenses the baseline for the policy played here; a real
match re-checks it with the same code path in the match summary.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import pathlib
import sys
import time
import urllib.error
import urllib.request

BIG_BLIND = 100
DEFAULT_HOST = "slumbot.com"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from server.slumbot_game import SlumbotGame  # noqa: E402


def post(host: str, path: str, payload: dict, retries: int = 4) -> dict:
    url = f"https://{host}/api{path}"
    body = json.dumps(payload).encode()
    delay = 1.0
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after {type(error).__name__}: {error}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def facing_bet(action: str) -> bool:
    """True when the acting player must call or fold rather than check.

    Uses Slumbot's own parser instead of a string heuristic: preflop the small
    blind faces the big blind even though the action string is empty.
    """
    state = SlumbotGame.parse_action(action)
    if state.get("error"):
        raise RuntimeError(f"could not parse action {action!r}: {state['error']}")
    return int(state["last_bet_size"]) > 0


def play_hand(host: str, token: str | None, policy: str) -> dict:
    response = post(host, "/new_hand", {"token": token} if token else {})
    if "error_msg" in response:
        raise RuntimeError(response["error_msg"])
    # The token only reappears on some responses; carry the last one seen.
    current = response.get("token") or token
    steps = 0
    while response.get("winnings") is None:
        steps += 1
        if steps > 64:
            raise RuntimeError("hand exceeded 64 decisions")
        if facing_bet(response.get("action", "")):
            increment = "f" if policy == "fold" else "c"
        else:
            increment = "k"
        payload = {"incr": increment}
        if current:
            payload["token"] = current
        response = post(host, "/act", payload)
        if "error_msg" in response:
            raise RuntimeError(response["error_msg"])
        current = response.get("token") or current
    response["token"] = current
    return response


def summarise(raw: list[float], baseline: list[float]) -> dict:
    count = len(raw)
    factor = 1000.0 / BIG_BLIND
    raw_sd = statistics.stdev(raw)
    baseline_sd = statistics.stdev(baseline)
    differences = [a - b for a, b in zip(raw, baseline)]
    difference_sd = statistics.stdev(differences)
    raw_mean, baseline_mean = statistics.fmean(raw), statistics.fmean(baseline)
    covariance = sum((a - raw_mean) * (b - baseline_mean) for a, b in zip(raw, baseline)) / (count - 1)
    return {
        "hands": count,
        "raw_mbb": raw_mean * factor,
        "raw_ci95_mbb": 1.96 * raw_sd / math.sqrt(count) * factor,
        "baseline_mbb": baseline_mean * factor,
        "baseline_ci95_mbb": 1.96 * baseline_sd / math.sqrt(count) * factor,
        "difference_mbb": statistics.fmean(differences) * factor,
        "difference_ci95_mbb": 1.96 * difference_sd / math.sqrt(count) * factor,
        "correlation": covariance / (raw_sd * baseline_sd) if raw_sd > 0 and baseline_sd > 0 else None,
        "variance_ratio": (raw_sd ** 2) / (baseline_sd ** 2) if baseline_sd > 0 else None,
        "raw_sd_chips": raw_sd,
        "baseline_sd_chips": baseline_sd,
        "difference_sd_chips": difference_sd,
        # raw - baseline is the control-variate form. It only beats the raw
        # score when the two are correlated enough to pay for the baseline's
        # own variance: rho > sd_baseline / (2 * sd_raw).
        "correlation_needed": baseline_sd / (2 * raw_sd) if raw_sd > 0 else None,
        "control_variate_variance_ratio": (raw_sd ** 2) / (difference_sd ** 2) if difference_sd > 0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hands", type=int, help="number of hands to play")
    parser.add_argument("--policy", choices=("call", "fold"), default="call")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--sleep", type=float, default=0.05, help="pause between hands in seconds")
    parser.add_argument("--out", help="write per-hand rows as JSONL")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    arguments = parser.parse_args()

    raw: list[float] = []
    baseline: list[float] = []
    token = None
    rows = open(arguments.out, "w") if arguments.out else None
    started = time.monotonic()
    try:
        for index in range(arguments.hands):
            result = play_hand(arguments.host, token, arguments.policy)
            token = result.get("token") or token
            value = result.get("baseline_winnings")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                print(f"hand {index}: no baseline_winnings in response; aborting", file=sys.stderr)
                return 1
            raw.append(float(result["winnings"]))
            baseline.append(float(value))
            if rows is not None:
                rows.write(json.dumps({
                    "hand": index,
                    "winnings": result["winnings"],
                    "baseline_winnings": value,
                    "action": result.get("action"),
                    "board": result.get("board"),
                    "client_pos": result.get("client_pos"),
                }) + "\n")
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{arguments.hands} hands, {time.monotonic() - started:.0f}s", file=sys.stderr)
            if arguments.sleep:
                time.sleep(arguments.sleep)
    finally:
        if rows is not None:
            rows.close()

    if len(raw) < 2:
        print("need at least two hands", file=sys.stderr)
        return 1
    report = summarise(raw, baseline)
    if arguments.json:
        print(json.dumps(report, indent=2))
        return 0
    print()
    print(f"policy={arguments.policy} hands={report['hands']}")
    print(f"  raw           {report['raw_mbb']:+10.1f} mbb/hand  +-{report['raw_ci95_mbb']:.1f}   sd {report['raw_sd_chips']:.0f} chips")
    print(f"  baseline      {report['baseline_mbb']:+10.1f} mbb/hand  +-{report['baseline_ci95_mbb']:.1f}   sd {report['baseline_sd_chips']:.0f} chips")
    print(f"  raw-baseline  {report['difference_mbb']:+10.1f} mbb/hand  +-{report['difference_ci95_mbb']:.1f}  <- must contain zero")
    print(f"  correlation   {report['correlation']:.3f} (needs > {report['correlation_needed']:.3f} for raw-baseline to help)")
    for label, ratio in (("baseline as the score", report["variance_ratio"]),
                         ("raw-baseline as the score", report["control_variate_variance_ratio"])):
        if ratio is None:
            continue
        verdict = f"{ratio:.2f}x fewer hands" if ratio > 1 else f"{1 / ratio:.2f}x MORE hands"
        print(f"  {label:26s} variance ratio {ratio:5.2f}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
