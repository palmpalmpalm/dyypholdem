#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
os.chdir(SOURCE_DIR)
sys.path.insert(0, str(SOURCE_DIR))

import settings.arguments as arguments  # noqa: E402
from nn.compact_value_net import (  # noqa: E402
    checkpoint_payload,
    from_legacy_model,
    load_compact_checkpoint,
)
from nn.net_builder import TrainingNetwork  # noqa: E402
from nn.value_nn import ValueNn  # noqa: E402


class CompactValueNetTest(unittest.TestCase):
    def setUp(self):
        self.original_arguments = {
            "use_gpu": arguments.use_gpu,
            "Tensor": arguments.Tensor,
            "LongTensor": arguments.LongTensor,
            "device": arguments.device,
        }
        arguments.use_gpu = False
        arguments.Tensor = torch.FloatTensor
        arguments.LongTensor = torch.LongTensor
        arguments.device = torch.device("cpu")
        torch.manual_seed(7)

    def tearDown(self):
        for key, value in self.original_arguments.items():
            setattr(arguments, key, value)

    @staticmethod
    def _inputs(input_size: int, output_size: int) -> torch.Tensor:
        buckets = output_size // 2
        inputs = torch.zeros(2, input_size)
        inputs[:, :buckets] = 1.0 / buckets
        inputs[:, buckets:output_size] = 1.0 / buckets
        inputs[0, -1] = 0.25
        inputs[1, -1] = 0.75
        return inputs

    def test_legacy_conversion_preserves_outputs_and_zero_sum(self):
        legacy = TrainingNetwork.build_net(1)
        legacy.evaluate()
        compact = from_legacy_model(legacy)
        inputs = self._inputs(compact.input_size, compact.output_size)

        with torch.inference_mode():
            expected = legacy.forward(inputs).clone()
            actual = compact(inputs)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        residual = torch.sum(actual * inputs[:, : compact.output_size], dim=1)
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-6, rtol=0)

    def test_checkpoint_round_trip_and_value_nn_loader(self):
        compact = from_legacy_model(TrainingNetwork.build_net(1))
        compact.eval()
        inputs = self._inputs(compact.input_size, compact.output_size)
        payload = checkpoint_payload(
            compact,
            {
                "street": 1,
                "epoch": 2,
                "valid_loss": 0.125,
                "device": "cpu",
                "datatype": "float32",
            },
            {"model_sha256": "test"},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(payload, path)
            loaded_payload = torch.load(path, map_location="cpu", weights_only=True)
            direct_model = load_compact_checkpoint(loaded_payload)
            value_nn = ValueNn().load_from_file(str(path))

            with torch.inference_mode():
                expected = direct_model(inputs)
            output = torch.empty_like(expected)
            value_nn.get_value(inputs, output)

        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(value_nn.model_info["street"], 1)
        self.assertEqual(value_nn.model_info["device"], torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
