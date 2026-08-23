#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from recover_models import MODEL_ASSETS, ModelAsset, is_verified, selected_assets  # noqa: E402


class ModelRecoveryTest(unittest.TestCase):
    def test_manifest_has_all_original_network_files(self):
        self.assertEqual(len(MODEL_ASSETS), 8)
        self.assertEqual({asset.street for asset in MODEL_ASSETS}, {"preflop", "flop", "turn", "river"})
        self.assertEqual({asset.kind for asset in MODEL_ASSETS}, {"info", "model"})
        self.assertEqual(
            sum(asset.size for asset in MODEL_ASSETS if asset.kind == "model"),
            566_797_072,
        )
        self.assertEqual(len({asset.sha256 for asset in MODEL_ASSETS}), 8)

    def test_street_filter_keeps_info_and_model(self):
        assets = tuple(selected_assets(("turn",)))
        self.assertEqual([(asset.street, asset.kind) for asset in assets], [("turn", "info"), ("turn", "model")])

    def test_verification_requires_exact_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.model"
            path.write_bytes(b"model")
            asset = ModelAsset(
                street="tiny",
                kind="model",
                drive_id="unused",
                size=5,
                sha256="9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
            )
            self.assertTrue(is_verified(path, asset))
            path.write_bytes(b"changed")
            self.assertFalse(is_verified(path, asset))


if __name__ == "__main__":
    unittest.main()
