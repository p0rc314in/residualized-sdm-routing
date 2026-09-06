"""Residualized read/write routing for Sparse Delta Memory."""

from __future__ import annotations

from typing import Iterator

import torch
from torch import nn
import torch.nn.functional as F


class ResidualizedRouter(nn.Module):
    """Replace two affine routes with a common base plus role residuals.

    The common basis starts at the mean of the already initialized native read
    and write projections. Both residual branches start at zero. The native
    projection tensors stay registered for checkpoint compatibility but are
    frozen and bypassed by forward hooks.
    """

    def __init__(self, read: nn.Linear, write: nn.Linear) -> None:
        super().__init__()
        if read.weight.shape != write.weight.shape:
            raise ValueError("read and write routing weights must have equal shapes")
        if read.bias is None or write.bias is None:
            raise ValueError("router residualization requires biased projections")
        if read.bias.shape != write.bias.shape:
            raise ValueError("read and write routing biases must have equal shapes")

        shared_weight = 0.5 * (read.weight.detach() + write.weight.detach())
        shared_bias = 0.5 * (read.bias.detach() + write.bias.detach())
        self.shared_weight = nn.Parameter(shared_weight.clone())
        self.shared_bias = nn.Parameter(shared_bias.clone())
        self.read_residual_weight = nn.Parameter(torch.zeros_like(shared_weight))
        self.read_residual_bias = nn.Parameter(torch.zeros_like(shared_bias))
        self.write_residual_weight = nn.Parameter(torch.zeros_like(shared_weight))
        self.write_residual_bias = nn.Parameter(torch.zeros_like(shared_bias))

        for projection in (read, write):
            for parameter in projection.parameters():
                parameter.requires_grad_(False)

        self._handles = [
            read.register_forward_hook(self._hook("read")),
            write.register_forward_hook(self._hook("write")),
        ]

    def _hook(self, role: str):
        def route(
            _projection: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            _native_output: torch.Tensor,
        ) -> torch.Tensor:
            if not inputs:
                raise RuntimeError("routing projection received no input")
            if role == "read":
                residual_weight = self.read_residual_weight
                residual_bias = self.read_residual_bias
            else:
                residual_weight = self.write_residual_weight
                residual_bias = self.write_residual_bias
            return F.linear(
                inputs[0],
                self.shared_weight + residual_weight,
                self.shared_bias + residual_bias,
            )

        return route

    def effective_parameters(self, role: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the effective affine map for an audit or diagnostic."""

        if role == "read":
            return (
                self.shared_weight + self.read_residual_weight,
                self.shared_bias + self.read_residual_bias,
            )
        if role == "write":
            return (
                self.shared_weight + self.write_residual_weight,
                self.shared_bias + self.write_residual_bias,
            )
        raise ValueError("role must be 'read' or 'write'")


def install_residualized_routing(model: nn.Module) -> tuple[ResidualizedRouter, ...]:
    """Install one residualized read/write router on every SDM mixer."""

    from reproduction.model import iter_sdm_mixers

    routers: list[ResidualizedRouter] = []
    for mixer in iter_sdm_mixers(model):
        if hasattr(mixer, "residualized_router"):
            raise RuntimeError("router residualization is already installed")
        router = ResidualizedRouter(mixer.Wq_read, mixer.Wk_write)
        mixer.add_module("residualized_router", router)
        routers.append(router)
    if not routers:
        raise ValueError("router residualization requires at least one SDM layer")
    model._residualized_routing = True
    return tuple(routers)


def iter_residualized_routers(model: nn.Module) -> Iterator[ResidualizedRouter]:
    for module in model.modules():
        if isinstance(module, ResidualizedRouter):
            yield module


__all__ = [
    "ResidualizedRouter",
    "install_residualized_routing",
    "iter_residualized_routers",
]
