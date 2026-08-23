#!/usr/bin/env python3
"""Recover the original public DeepHoldem networks after Git LFS loss.

The DyypHoldem Git history still contains model pointers, but the corresponding
GitHub LFS objects currently return HTTP 410. The original Torch7 files linked
from DeepHoldem issue #28 remain publicly downloadable. This script downloads
them with resume support and accepts a file only after exact size and SHA-256
verification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "runs" / "model-recovery" / "original"
SOURCE_ISSUE = (
    "https://github.com/happypepper/DeepHoldem/issues/28#issuecomment-689021950"
)


@dataclass(frozen=True)
class ModelAsset:
    street: str
    kind: str
    drive_id: str
    size: int
    sha256: str

    @property
    def filename(self) -> str:
        return f"{self.street}.{self.kind}"

    @property
    def download_url(self) -> str:
        return (
            "https://drive.usercontent.google.com/download"
            f"?id={self.drive_id}&export=download&confirm=t"
        )


MODEL_ASSETS = (
    ModelAsset(
        "preflop",
        "info",
        "1Vk2B3bhnQ3tbPAdRYWYRfZSZjxHiM5wt",
        86,
        "94ce5180f08bf3919f8c6bb063ba3d67fc34ae3a6fb6f3b4b19f7df9327e8072",
    ),
    ModelAsset(
        "preflop",
        "model",
        "1G1gZNwo8yzvcUU9jgxbD46H3OBCS8s2e",
        65_805_286,
        "a761dfc985cbe93d1cc2fb470fa70b90354aecf402b8f25fe36d651b41c39817",
    ),
    ModelAsset(
        "flop",
        "info",
        "1k5EiO-lkrerf2B3EGZbSjFUu5J24Z3Qj",
        86,
        "48a3f3464975a806e800dd0e20e4e7a9b2d858649612243805cfefd41c171c0b",
    ),
    ModelAsset(
        "flop",
        "model",
        "1skQEwQ2i7rEXTywoRwkdXNmim23etfDL",
        192_130_574,
        "4164d7026f86efe7fc63a2f1c7e6f8eeb9de52381f40f84525923a2ed288766d",
    ),
    ModelAsset(
        "turn",
        "info",
        "17jV2qaLLhdHJjQ_BrpEoaw364TcZqbrj",
        86,
        "7794de4e01f4a38332a80584a3e48c186e21505fbd0e3f679f77069e80d96a67",
    ),
    ModelAsset(
        "turn",
        "model",
        "1xkapGGFSdXT643OGRHH8XyXn0jXKgpE4",
        192_130_574,
        "213b4ea517b57ef3a8b389c0f49f7e3222822db0ae3c9fc19561386c19d9134a",
    ),
    ModelAsset(
        "river",
        "info",
        "1MPpXdOa8Q6NB_Ue547e6MutghRpfBNmv",
        86,
        "5026f66d439f96453125ae1c0f9ef034e5b2cc2912fdae33efb579397573d36d",
    ),
    ModelAsset(
        "river",
        "model",
        "1lFZQDQwtroKvmZSm3u7BVTmCUgGFH3Yk",
        116_730_638,
        "b8c619349de35f7427aa3dc768d80d32a7f96edd5106baf606e28fa87668e493",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified(path: Path, asset: ModelAsset) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == asset.size
        and file_sha256(path) == asset.sha256
    )


def selected_assets(streets: Sequence[str]) -> Iterable[ModelAsset]:
    requested = set(streets)
    if "all" in requested:
        return MODEL_ASSETS
    return tuple(asset for asset in MODEL_ASSETS if asset.street in requested)


def download(asset: ModelAsset, output_root: Path) -> dict:
    destination = output_root / asset.filename
    if is_verified(destination, asset):
        return _asset_result(asset, destination, "reused")

    output_root.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and partial.stat().st_size >= asset.size:
        partial.unlink()

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "DyypHoldem-model-recovery/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(asset.download_url, headers=headers)

    with urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        append = bool(offset and status == 206)
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)

    if not is_verified(partial, asset):
        actual_size = partial.stat().st_size if partial.exists() else 0
        actual_sha256 = file_sha256(partial) if partial.exists() else None
        raise RuntimeError(
            f"verification failed for {asset.filename}: "
            f"size={actual_size}, sha256={actual_sha256}"
        )
    os.replace(partial, destination)
    return _asset_result(asset, destination, "downloaded")


def _asset_result(asset: ModelAsset, path: Path, status: str) -> dict:
    return {
        "street": asset.street,
        "kind": asset.kind,
        "path": str(path),
        "status": status,
        "size": asset.size,
        "sha256": asset.sha256,
    }


def progress_report(output_root: Path, assets: Iterable[ModelAsset]) -> dict:
    rows = []
    for asset in assets:
        path = output_root / asset.filename
        verified = is_verified(path, asset)
        rows.append(
            {
                **asdict(asset),
                "path": str(path),
                "present": path.is_file(),
                "verified": verified,
                "actual_size": path.stat().st_size if path.is_file() else 0,
            }
        )
    return {
        "source_issue": SOURCE_ISSUE,
        "output_root": str(output_root),
        "files": rows,
        "expected_files": len(rows),
        "verified_files": sum(bool(row["verified"]) for row in rows),
        "expected_bytes": sum(int(row["size"]) for row in rows),
        "verified": bool(rows) and all(bool(row["verified"]) for row in rows),
    }


def write_manifest(output_root: Path, report: dict) -> None:
    payload = {
        **report,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = output_root / ".manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_root / "manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--street",
        action="append",
        choices=("all", "preflop", "flop", "turn", "river"),
        default=None,
    )
    parser.add_argument("--progress-report", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assets = tuple(selected_assets(args.street or ("all",)))
    if args.progress_report:
        print(json.dumps(progress_report(args.output_root, assets), indent=2, sort_keys=True))
        return

    results = [download(asset, args.output_root) for asset in assets]
    report = progress_report(args.output_root, assets)
    report["results"] = results
    write_manifest(args.output_root, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
