#!/usr/bin/env python3
"""Download and verify the minimal large assets needed by GPU benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]

RIVER_ASSETS = (
        {
            "drive_id": "1aDIOsaDROQBaMtpXetThSmduGY46FwNT",
            "path": "src/game/evaluation/hand_ranks.pt",
            "sha256": "f896304f2dde706945978fed38069dfc9a9a06d3f2970afb702f1514f9587a68",
            "size": 259_903_403,
        },
        {
            "drive_id": "1VixteYtYtdsorWc039Pyl7ZWn6Uq8lTN",
            "path": "src/terminal_equity/block_matrix.pt",
            "sha256": "d28b9561b182e43dc86901f713d60c7e94cd4a69f76bf5a27c825d5b3333e80d",
            "size": 7_033_835,
        },
        {
            "drive_id": "1oePwh3S27UM-URi8bZUqTp_lAYal4RZS",
            "path": "src/terminal_equity/preflop_equity.pt",
            "sha256": "ad47a518612c5a0c92d44fbef570fb2c005cd96b76536e7ca4d420663cfba7c8",
            "size": 7_033_835,
        },
)

EAGER_BUCKET_ASSETS = (
        {
            "drive_id": "19VUnYVzRzHmicGA-P1tQkoNtGdNHvULA",
            "path": "src/nn/bucketing/ihr_pair_to_bucket.pkl",
            "sha256": "8f6df2a556c25f6e5f59417cc7a99558d4300520278844411523894698c24857",
            "size": 3_041,
        },
        {
            "drive_id": "1AjD1utFjn04v5IHWx1QUFXZWdwNxbgz7",
            "path": "src/nn/bucketing/flop_dist_cats.pkl",
            "sha256": "335dfa94cfe79d77db64ea226493a601ed293c803188f780dc1bd67ebcbf5392",
            "size": 10_499_158,
        },
        {
            "drive_id": "1gK82FqtSIghEPnkfvPmyoQxaE5O30rzZ",
            "path": "src/nn/bucketing/turn_dist_cats.pkl",
            "sha256": "4697af9bfc5e17e243557d74092326b669148fb35fd10d151cf130f7493037f7",
            "size": 116_507_098,
        },
        {
            "drive_id": "1X6PbbT2m7Dhr--IesIDy3kyPs0mDVuT-",
            "path": "src/nn/bucketing/river_ihr.pkl",
            "sha256": "cbe82220f1ea5082e9f3f6daa525c2ac4df89e6043f8928f2eb134859ac50d33",
            "size": 188_711_781,
        },
)

PREFLOP_PLAY_ASSETS = (
        {
            "drive_id": "1VQnqGBDwY39oDdgJjsrAk0RuVNfShs6y",
            "path": "src/nn/bucketing/preflop_buckets.pt",
            "sha256": "131814be7cec451cd4cdc894007db16b5c0eb83a9afc6ff7132e361ee2f4a1bc",
            "size": 117_219_115,
        },
)

ASSETS = {
    "river": RIVER_ASSETS,
    "legacy-eager-river": RIVER_ASSETS + EAGER_BUCKET_ASSETS,
    # Full continual play also loads a precomputed hand-to-flop-bucket matrix
    # from NextRoundValuePre; the 169 current-street buckets are computed.
    "play": RIVER_ASSETS + PREFLOP_PLAY_ASSETS + EAGER_BUCKET_ASSETS,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and file_sha256(path) == expected_sha256
    )


def materialize(asset: dict[str, object]) -> dict[str, object]:
    relative_path = str(asset["path"])
    destination = PROJECT_DIR / relative_path
    expected_size = int(asset["size"])
    expected_sha256 = str(asset["sha256"])

    if is_verified(destination, expected_size, expected_sha256):
        return {"path": relative_path, "status": "reused", "size": expected_size}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gdown",
                str(asset["drive_id"]),
                "-O",
                str(temporary),
            ],
            check=True,
        )
        if not is_verified(temporary, expected_size, expected_sha256):
            actual_size = temporary.stat().st_size if temporary.exists() else 0
            actual_sha256 = file_sha256(temporary) if temporary.exists() else None
            raise RuntimeError(
                f"asset verification failed for {relative_path}: "
                f"size={actual_size}, sha256={actual_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {"path": relative_path, "status": "downloaded", "size": expected_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(ASSETS), default="river")
    args = parser.parse_args()

    results = [materialize(asset) for asset in ASSETS[args.profile]]
    print(
        json.dumps(
            {
                "profile": args.profile,
                "verified": True,
                "assets": results,
                "total_bytes": sum(int(item["size"]) for item in results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
