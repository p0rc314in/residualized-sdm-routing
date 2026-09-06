"""M4 additive untouched-state provider evaluated at selected tuples only."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..addresses import logical_capacity
from ..priors import SelectedRowPrior


class AdditiveProductPrior(SelectedRowPrior):
    """Sum one learned factor row per coordinate of a selected address."""

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
        self.factor_tables = nn.Parameter(
            torch.empty(
                templates,
                factors,
                codebook_size,
                value_width,
                dtype=dtype,
            )
        )
        self.reset_parameters()

    def reset_parameters(
        self,
        full_table_std: float | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        """Match the released full-table target variance after summation."""

        target_std = (
            self.value_width**-0.5 if full_table_std is None else full_table_std
        )
        factor_std = target_std / math.sqrt(self.factors)
        nn.init.trunc_normal_(
            self.factor_tables,
            mean=0.0,
            std=factor_std,
            a=-3.0 * factor_std,
            b=3.0 * factor_std,
            generator=generator,
        )

    def forward(
        self,
        addresses: torch.Tensor,
        template_indices: torch.Tensor,
    ) -> torch.Tensor:
        valid = self._validate_request(addresses, template_indices)
        safe = addresses.to(torch.int64).clamp_min(0)
        templates = template_indices.to(torch.int64).unsqueeze(1)
        result = torch.zeros(
            *addresses.shape[:-1],
            self.value_width,
            device=addresses.device,
            dtype=self.factor_tables.dtype,
        )
        for factor in range(self.factors):
            result = result + self.factor_tables[
                templates,
                factor,
                safe[..., factor],
            ]
        return result * valid.unsqueeze(-1).to(result.dtype)


__all__ = ["AdditiveProductPrior"]
