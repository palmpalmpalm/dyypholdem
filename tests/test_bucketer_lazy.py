#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
os.chdir(SOURCE_DIR)
sys.path.insert(0, str(SOURCE_DIR))

import nn.bucketer as bucketer  # noqa: E402


class BucketerLazyLoadingTest(unittest.TestCase):
    def setUp(self):
        bucketer._preflop_buckets = None
        bucketer._flop_cats = None
        bucketer._turn_cats = None
        bucketer._ihr_pair_to_bucket = None
        bucketer._river_ihr = None
        bucketer._river_buckets = None

    def test_import_does_not_load_lookup_tables(self):
        self.assertIsNone(bucketer._flop_cats)
        self.assertIsNone(bucketer._turn_cats)
        self.assertIsNone(bucketer._ihr_pair_to_bucket)
        self.assertIsNone(bucketer._river_ihr)

    def test_flop_initialization_loads_only_flop_table(self):
        with patch.object(bucketer, "_load_pickle", return_value={1: 2}) as loader:
            bucketer.initialize(2)

        loader.assert_called_once_with("./nn/bucketing/flop_dist_cats.pkl")
        self.assertEqual(bucketer._flop_cats, {1: 2})
        self.assertIsNone(bucketer._turn_cats)
        self.assertIsNone(bucketer._ihr_pair_to_bucket)
        self.assertIsNone(bucketer._river_ihr)

    def test_lazy_initialization_uses_configured_logger_log_method(self):
        logger = Mock(spec_set=["log"])
        with (
            patch.object(bucketer.arguments, "logger", logger),
            patch.object(bucketer, "_load_pickle", return_value={1: 2}),
        ):
            bucketer.initialize(2)

        logger.log.assert_called_once()
        level, message = logger.log.call_args.args
        self.assertEqual(level, "LOADING")
        self.assertIn("Flop categories initialized in:", message)

    def test_river_count_does_not_load_river_category_table(self):
        with patch.object(
            bucketer, "_load_pickle", return_value={10: 1, 20: 2}
        ) as loader:
            self.assertEqual(bucketer.get_bucket_count(4), 2)

        loader.assert_called_once_with("./nn/bucketing/ihr_pair_to_bucket.pkl")
        self.assertIsNone(bucketer._river_ihr)
        self.assertIsNone(bucketer._flop_cats)
        self.assertIsNone(bucketer._turn_cats)

    def test_river_initialization_avoids_flop_and_turn_tables(self):
        values = {
            "./nn/bucketing/ihr_pair_to_bucket.pkl": {10: 1, 20: 2},
            "./nn/bucketing/river_ihr.pkl": {30: (40, 50)},
        }
        with patch.object(
            bucketer, "_load_pickle", side_effect=lambda path: values[path]
        ) as loader:
            bucketer.initialize(4)

        self.assertEqual(
            [call.args[0] for call in loader.call_args_list],
            [
                "./nn/bucketing/ihr_pair_to_bucket.pkl",
                "./nn/bucketing/river_ihr.pkl",
            ],
        )
        self.assertIsNone(bucketer._flop_cats)
        self.assertIsNone(bucketer._turn_cats)


if __name__ == "__main__":
    unittest.main()
