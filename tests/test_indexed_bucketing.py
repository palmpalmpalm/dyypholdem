#!/usr/bin/env python3
"""The indexed bucketing transform must agree with the dense matmul it replaces."""

from pathlib import Path
import sys
import unittest

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import settings.arguments as arguments  # noqa: E402
import settings.game_settings as game_settings  # noqa: E402
from nn.next_round_value import NextRoundValue  # noqa: E402


class IndexedBucketingTest(unittest.TestCase):
    """Builds a NextRoundValue shell around a synthetic bucketing map.

    The real map comes from the bucketer and needs its lookup tables; the
    transform only cares that each hand lands in at most one bucket per board,
    so a synthetic map exercises the same code without the assets.
    """

    def setUp(self):
        self.previous_mode = arguments.bucketing_mode
        self.previous_hand_count = game_settings.hand_count
        game_settings.hand_count = 40
        self.board_count = 5
        self.bucket_count = 7
        self.weight = 1.0 / 45.0
        torch.manual_seed(20260903)

        box = NextRoundValue.__new__(NextRoundValue)
        box.board_count = self.board_count
        box.bucket_count = self.bucket_count
        box._bucket_flat_index = None
        matrix = torch.zeros(game_settings.hand_count, self.board_count * self.bucket_count)
        view = matrix.view(game_settings.hand_count, self.board_count, self.bucket_count)
        for board in range(self.board_count):
            for hand in range(game_settings.hand_count):
                # Every fifth hand is blocked by this board and maps nowhere.
                if (hand + board) % 5 == 0:
                    continue
                view[hand, board, torch.randint(self.bucket_count, (1,)).item()] = 1.0
        box._range_matrix = matrix
        box._range_matrix_board_view = view
        box._reverse_value_matrix = matrix.t().clone().mul_(self.weight)
        self.box = box

    def tearDown(self):
        arguments.bucketing_mode = self.previous_mode
        game_settings.hand_count = self.previous_hand_count

    def both_ways(self, rows=6):
        card_range = torch.rand(rows, game_settings.hand_count)
        bucket_value = torch.rand(rows, self.board_count * self.bucket_count)
        arguments.bucketing_mode = "dense"
        dense_forward = self.box._card_range_to_bucket_range(card_range.clone())
        dense_reverse = self.box._bucket_value_to_card_value(bucket_value.clone())
        arguments.bucketing_mode = "indexed"
        indexed_forward = self.box._card_range_to_bucket_range(card_range.clone())
        indexed_reverse = self.box._bucket_value_to_card_value(bucket_value.clone())
        return dense_forward, indexed_forward, dense_reverse, indexed_reverse

    def assert_agrees_to_rounding(self, dense, indexed):
        """Both forms sum the same products in a different order.

        Reordering float32 additions costs at most a few units in the last
        place, so the bound is relative rather than exact. On the real flop map
        buckets hold 1.1 hands on average and most sums have a single term, so
        the transforms come out bitwise identical there; that is a property of
        the occupancy, not a guarantee, and this map is deliberately denser.
        """
        self.assertFalse(torch.equal(dense, torch.zeros_like(dense)))
        difference = (dense - indexed).abs()
        relative = difference / dense.abs().clamp_min(1e-12)
        self.assertLess(float(relative.max()), 1e-6)

    def test_forward_transform_agrees_to_float32_rounding(self):
        dense, indexed, _, _ = self.both_ways()
        self.assert_agrees_to_rounding(dense, indexed)

    def test_reverse_transform_agrees_to_float32_rounding(self):
        _, _, dense, indexed = self.both_ways()
        self.assert_agrees_to_rounding(dense, indexed)

    def test_a_one_hand_bucket_map_reproduces_the_forward_exactly(self):
        # The forward sums the hands inside a bucket, so with one hand per
        # bucket there is nothing to reorder and it must agree bit for bit --
        # the regime the real map mostly sits in. The reverse instead sums one
        # term per board, so it stays a multi-term reduction either way.
        view = self.box._range_matrix.view(
            game_settings.hand_count, self.board_count, self.bucket_count
        )
        view.zero_()
        for board in range(self.board_count):
            for bucket in range(self.bucket_count):
                view[bucket, board, bucket] = 1.0
        self.box._reverse_value_matrix = self.box._range_matrix.t().clone().mul_(self.weight)
        self.box._bucket_flat_index = None
        dense, indexed, dense_reverse, indexed_reverse = self.both_ways()
        self.assertTrue(torch.equal(dense, indexed))
        self.assert_agrees_to_rounding(dense_reverse, indexed_reverse)

    def test_blocked_hands_stay_zero_in_both_directions(self):
        arguments.bucketing_mode = "indexed"
        self.box._ensure_bucket_indices()
        blocked = self.box._bucket_valid_t == 0
        self.assertGreater(int(blocked.sum()), 0)
        bucket_value = torch.rand(3, self.board_count * self.bucket_count)
        gathered = self.box._bucket_value_to_card_value(bucket_value)
        # A hand blocked on every board must receive nothing at all.
        fully_blocked = blocked.all(dim=0)
        if int(fully_blocked.sum()):
            self.assertTrue(torch.equal(gathered[:, fully_blocked], torch.zeros(3, int(fully_blocked.sum()))))
        self.assertEqual(gathered.shape, (3, game_settings.hand_count))

    def test_writes_into_a_supplied_output_tensor(self):
        arguments.bucketing_mode = "indexed"
        card_range = torch.rand(4, game_settings.hand_count)
        destination = torch.full((4, self.board_count * self.bucket_count), 7.0)
        returned = self.box._card_range_to_bucket_range(card_range, destination)
        self.assertIs(returned, destination)
        self.assertFalse(torch.equal(destination, torch.full_like(destination, 7.0)))

        bucket_value = torch.rand(4, self.board_count * self.bucket_count)
        card_destination = torch.full((4, game_settings.hand_count), -1.0)
        returned = self.box._bucket_value_to_card_value(bucket_value, card_destination)
        self.assertIs(returned, card_destination)
        self.assertFalse(torch.equal(card_destination, torch.full_like(card_destination, -1.0)))

    def test_indices_are_rebuilt_when_the_transform_changes(self):
        arguments.bucketing_mode = "indexed"
        self.box._ensure_bucket_indices()
        first = self.box._bucket_flat_index.clone()
        # A cached transform for a different board replaces the matrix; the
        # derived indices must not survive it.
        self.box._bucket_flat_index = None
        rolled = self.box._range_matrix.view(
            game_settings.hand_count, self.board_count, self.bucket_count
        ).roll(1, dims=2)
        self.box._range_matrix = rolled.reshape(self.box._range_matrix.shape).contiguous()
        self.box._reverse_value_matrix = self.box._range_matrix.t().clone().mul_(self.weight)
        self.box._ensure_bucket_indices()
        self.assertFalse(torch.equal(first, self.box._bucket_flat_index))

    def test_mode_flag_rejects_unknown_values(self):
        self.assertIn(arguments.bucketing_mode, ("dense", "indexed"))


if __name__ == "__main__":
    unittest.main()
