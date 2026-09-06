"""Untouched-state providers evaluated only at selected logical tuples."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .addresses import encode_product_addresses, logical_capacity


class SelectedRowPrior(nn.Module, ABC):
    """Interface for model-owned initial values at selected addresses."""

    codebook_size: int
    factors: int
    value_width: int
    templates: int

    @abstractmethod
    def forward(
        self,
        addresses: torch.Tensor,
        template_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B,K,V] values for integer tuple addresses [B,K,p]."""

    def _validate_request(
        self,
        addresses: torch.Tensor,
        template_indices: torch.Tensor,
    ) -> torch.Tensor:
        if addresses.ndim != 3 or addresses.shape[-1] != self.factors:
            raise ValueError(f"prior addresses must be [B,K,{self.factors}]")
        if addresses.dtype not in (torch.int32, torch.int64):
            raise ValueError("prior addresses must use int32 or int64")
        if template_indices.shape != (addresses.shape[0],):
            raise ValueError("template indices must be [B]")
        if template_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("template indices must be integer")
        if template_indices.device != addresses.device:
            raise ValueError("template indices and addresses must share a device")
        if addresses.device.type == "cpu":
            valid = addresses >= 0
            if bool(valid.any()) and int(addresses[valid].max()) >= self.codebook_size:
                raise ValueError("selected address lies outside the prior codebook")
            if template_indices.numel() and (
                int(template_indices.min()) < 0
                or int(template_indices.max()) >= self.templates
            ):
                raise ValueError("template index lies outside the prior")
        return addresses.ge(0).all(dim=-1)

    def parameter_elements(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class ZeroPrior(SelectedRowPrior):
    """A zero untouched state with no logical-row parameter table."""

    def __init__(
        self,
        *,
        templates: int,
        factors: int,
        codebook_size: int,
        value_width: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if templates <= 0 or value_width <= 0:
            raise ValueError("templates and value width must be positive")
        logical_capacity(codebook_size, factors)
        self.templates = templates
        self.factors = factors
        self.codebook_size = codebook_size
        self.value_width = value_width
        self.register_buffer("_anchor", torch.zeros((), dtype=dtype), persistent=False)

    def forward(
        self,
        addresses: torch.Tensor,
        template_indices: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_request(addresses, template_indices)
        return torch.zeros(
            *addresses.shape[:-1],
            self.value_width,
            device=addresses.device,
            dtype=self._anchor.dtype,
        )


class DenseTablePrior(SelectedRowPrior):
    """A small-S semantic oracle that gathers a conventional full table."""

    def __init__(
        self,
        table: torch.Tensor,
        *,
        factors: int,
        codebook_size: int,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if table.ndim != 3:
            raise ValueError("dense prior table must be [H,S,V]")
        capacity = logical_capacity(codebook_size, factors)
        if table.shape[1] != capacity:
            raise ValueError("dense table does not match its logical capacity")
        self.templates = table.shape[0]
        self.factors = factors
        self.codebook_size = codebook_size
        self.value_width = table.shape[2]
        if trainable:
            self.table = nn.Parameter(table)
        else:
            self.register_buffer("table", table)

    def forward(
        self,
        addresses: torch.Tensor,
        template_indices: torch.Tensor,
    ) -> torch.Tensor:
        valid = self._validate_request(addresses, template_indices)
        safe = addresses.clamp_min(0)
        encoded = encode_product_addresses(
            safe,
            codebook_size=self.codebook_size,
        )
        result = self.table[
            template_indices.to(torch.int64).unsqueeze(1),
            encoded,
        ]
        return result * valid.unsqueeze(-1).to(result.dtype)


def materialize_prior(
    prior: SelectedRowPrior, *, maximum_rows: int = 65_536
) -> torch.Tensor:
    """Materialize a prior only for bounded oracle tests."""

    capacity = logical_capacity(prior.codebook_size, prior.factors)
    if capacity > maximum_rows:
        raise ValueError("refusing to materialize a large logical prior")
    parameter = next(prior.parameters(), None)
    if parameter is not None:
        device = parameter.device
    else:
        buffer = next(prior.buffers(), None)
        device = buffer.device if buffer is not None else torch.device("cpu")
    indices = torch.arange(capacity, device=device)
    addresses = []
    remaining = indices
    for _ in range(prior.factors):
        addresses.append(torch.remainder(remaining, prior.codebook_size))
        remaining = torch.div(remaining, prior.codebook_size, rounding_mode="floor")
    tuples = torch.stack(tuple(reversed(addresses)), dim=-1)
    tuples = tuples.unsqueeze(0).expand(prior.templates, -1, -1)
    templates = torch.arange(prior.templates, device=tuples.device)
    return prior(tuples, templates)


__all__ = [
    "DenseTablePrior",
    "SelectedRowPrior",
    "ZeroPrior",
    "materialize_prior",
]
