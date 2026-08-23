"""Device-neutral inference format for DyypHoldem counterfactual value nets."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn


CHECKPOINT_FORMAT = "dyypholdem.compact-value-net"
CHECKPOINT_VERSION = 1


class CompactValueNet(nn.Module):
    """Native PyTorch equivalent of the legacy Torch7 value-network graph.

    The final correction enforces the same range-weighted zero-sum identity as
    the original ``ConcatTable``/``DotProduct`` graph.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 500,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if input_size != output_size + 1:
            raise ValueError("value-net input size must equal output size plus pot input")
        if output_size <= 0 or output_size % 2:
            raise ValueError("value-net output size must be a positive even number")

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_size = int(hidden_size)
        self.batch_norm_eps = float(batch_norm_eps)
        self.batch_norm_momentum = float(batch_norm_momentum)

        layer_inputs = (self.input_size, self.hidden_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(size, self.hidden_size) for size in layer_inputs
        )
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(
                self.hidden_size,
                eps=self.batch_norm_eps,
                momentum=self.batch_norm_momentum,
            )
            for _ in range(3)
        )
        self.activations = nn.ModuleList(nn.PReLU(1) for _ in range(3))
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 2 or inputs.size(1) != self.input_size:
            raise ValueError(
                f"expected [batch, {self.input_size}] inputs, got {tuple(inputs.shape)}"
            )

        values = inputs
        for linear, batch_norm, activation in zip(
            self.hidden_layers, self.batch_norms, self.activations
        ):
            values = activation(batch_norm(linear(values)))
        values = self.output_layer(values)

        ranges = inputs.narrow(1, 0, self.output_size)
        correction = -0.5 * torch.sum(values * ranges, dim=1, keepdim=True)
        return values + correction

    def architecture(self) -> Mapping[str, object]:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "batch_norm_eps": self.batch_norm_eps,
            "batch_norm_momentum": self.batch_norm_momentum,
        }


def from_legacy_model(legacy_model) -> CompactValueNet:
    """Copy a parsed legacy Torch7 graph into a native inference module."""

    try:
        forward_part = legacy_model.modules[0].modules[0]
        modules = forward_part.modules
    except (AttributeError, IndexError) as exc:
        raise ValueError("unexpected legacy value-network graph") from exc

    if len(modules) != 10:
        raise ValueError(f"expected 10 feed-forward modules, found {len(modules)}")

    legacy_linears = [modules[index] for index in (0, 3, 6, 9)]
    legacy_batch_norms = [modules[index] for index in (1, 4, 7)]
    legacy_activations = [modules[index] for index in (2, 5, 8)]

    input_size = int(legacy_linears[0].weight.size(1))
    hidden_size = int(legacy_linears[0].weight.size(0))
    output_size = int(legacy_linears[-1].weight.size(0))
    eps_values = {float(module.eps) for module in legacy_batch_norms}
    momentum_values = {float(module.momentum) for module in legacy_batch_norms}
    if len(eps_values) != 1 or len(momentum_values) != 1:
        raise ValueError("legacy batch-normalization settings are inconsistent")

    compact = CompactValueNet(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        batch_norm_eps=eps_values.pop(),
        batch_norm_momentum=momentum_values.pop(),
    )

    with torch.no_grad():
        for destination, source in zip(
            list(compact.hidden_layers) + [compact.output_layer], legacy_linears
        ):
            destination.weight.copy_(source.weight)
            destination.bias.copy_(source.bias)

        for destination, source in zip(compact.batch_norms, legacy_batch_norms):
            destination.weight.copy_(source.weight)
            destination.bias.copy_(source.bias)
            destination.running_mean.copy_(source.running_mean)
            destination.running_var.copy_(source.running_var)
            destination.num_batches_tracked.zero_()

        for destination, source in zip(compact.activations, legacy_activations):
            destination.weight.copy_(source.weight)

    compact.eval()
    return compact


def checkpoint_payload(
    model: CompactValueNet,
    model_info: Mapping[str, object],
    source: Mapping[str, object],
) -> Mapping[str, object]:
    """Build a weights-only payload accepted by recent safe ``torch.load``."""

    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "architecture": dict(model.architecture()),
        "model_info": dict(model_info),
        "source": dict(source),
        "state_dict": OrderedDict(
            (name, tensor.detach().cpu().contiguous())
            for name, tensor in model.state_dict().items()
        ),
    }


def is_compact_checkpoint(payload: object) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("format") == CHECKPOINT_FORMAT
        and payload.get("version") == CHECKPOINT_VERSION
    )


def load_compact_checkpoint(payload: Mapping[str, object]) -> CompactValueNet:
    if not is_compact_checkpoint(payload):
        raise ValueError("unsupported compact value-network checkpoint")
    architecture = payload.get("architecture")
    state_dict = payload.get("state_dict")
    if not isinstance(architecture, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("compact checkpoint is missing architecture or state_dict")

    model = CompactValueNet(**dict(architecture))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
