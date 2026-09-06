# Copyright (c) 2026 p0rc314in

"""Learned initial-memory representations for Sparse Delta Memory."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import nn
from torch.nn.utils import parametrize


INITIAL_MEMORY_MODES = ("zero", "full", "product_key")


def _prefer_fused_selected_rows(
    dtype: torch.dtype,
    compute_capability_major: int,
) -> bool:
    """Choose the faster gradient scatter for the active GPU generation."""
    return dtype == torch.float32 or compute_capability_major >= 9


@triton.jit
def _selected_product_key_forward_kernel(
    row_factor,
    column_factor,
    logical_rows,
    template_indices,
    output,
    codebook_size: tl.constexpr,
    value_width: tl.constexpr,
    block_width: tl.constexpr,
):
    selected = tl.program_id(0)
    width_block = tl.program_id(1)
    dimension = width_block * block_width + tl.arange(0, block_width)
    valid = dimension < value_width
    logical = tl.load(logical_rows + selected).to(tl.int64)
    template = tl.load(template_indices + selected).to(tl.int64)
    row = logical // codebook_size
    column = logical % codebook_size
    factor_base = template * codebook_size * value_width
    row_offset = factor_base + row * value_width + dimension
    column_offset = factor_base + column * value_width + dimension
    value = tl.load(row_factor + row_offset, mask=valid, other=0.0)
    value += tl.load(column_factor + column_offset, mask=valid, other=0.0)
    tl.store(output + selected * value_width + dimension, value, mask=valid)


@triton.jit
def _selected_product_key_backward_kernel(
    output_gradient,
    logical_rows,
    template_indices,
    row_gradient,
    column_gradient,
    codebook_size: tl.constexpr,
    value_width: tl.constexpr,
    block_width: tl.constexpr,
):
    selected = tl.program_id(0)
    width_block = tl.program_id(1)
    dimension = width_block * block_width + tl.arange(0, block_width)
    valid = dimension < value_width
    logical = tl.load(logical_rows + selected).to(tl.int64)
    template = tl.load(template_indices + selected).to(tl.int64)
    row = logical // codebook_size
    column = logical % codebook_size
    factor_base = template * codebook_size * value_width
    gradient = tl.load(
        output_gradient + selected * value_width + dimension,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        row_gradient + factor_base + row * value_width + dimension,
        gradient,
        mask=valid,
        sem="relaxed",
    )
    tl.atomic_add(
        column_gradient + factor_base + column * value_width + dimension,
        gradient,
        mask=valid,
        sem="relaxed",
    )


class _SelectedProductKeyMemory(torch.autograd.Function):
    """Fuse selected factor gathers and their two sparse gradient scatters."""

    @staticmethod
    def forward(
        ctx,
        row_factor: torch.Tensor,
        column_factor: torch.Tensor,
        logical_rows: torch.Tensor,
        template_indices: torch.Tensor,
        codebook_size: int,
    ) -> torch.Tensor:
        rows = logical_rows.numel()
        value_width = row_factor.shape[-1]
        output = torch.empty(
            rows,
            value_width,
            device=row_factor.device,
            dtype=row_factor.dtype,
        )
        block_width = min(256, triton.next_power_of_2(value_width))
        _selected_product_key_forward_kernel[
            (rows, triton.cdiv(value_width, block_width))
        ](
            row_factor,
            column_factor,
            logical_rows,
            template_indices,
            output,
            codebook_size=codebook_size,
            value_width=value_width,
            block_width=block_width,
            num_warps=4,
        )
        ctx.save_for_backward(logical_rows, template_indices)
        ctx.factor_shape = row_factor.shape
        ctx.codebook_size = codebook_size
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        logical_rows, template_indices = ctx.saved_tensors
        rows = logical_rows.numel()
        value_width = ctx.factor_shape[-1]
        row_gradient = torch.zeros(
            ctx.factor_shape,
            device=output_gradient.device,
            dtype=output_gradient.dtype,
        )
        column_gradient = torch.zeros_like(row_gradient)
        block_width = min(256, triton.next_power_of_2(value_width))
        _selected_product_key_backward_kernel[
            (rows, triton.cdiv(value_width, block_width))
        ](
            output_gradient.contiguous(),
            logical_rows,
            template_indices,
            row_gradient,
            column_gradient,
            codebook_size=ctx.codebook_size,
            value_width=value_width,
            block_width=block_width,
            num_warps=4,
        )
        return row_gradient, column_gradient, None, None, None


def _selected_product_key_memory(
    row_factor: torch.Tensor,
    column_factor: torch.Tensor,
    logical_rows: torch.Tensor,
    template_indices: torch.Tensor,
    codebook_size: int,
) -> torch.Tensor:
    compute_capability_major = 0
    if row_factor.is_cuda:
        compute_capability_major, _minor = torch.cuda.get_device_capability(
            row_factor.device
        )
    if (
        row_factor.is_cuda
        and row_factor.dtype
        in (torch.float32, torch.bfloat16, torch.float16)
        and _prefer_fused_selected_rows(
            row_factor.dtype,
            compute_capability_major,
        )
        and row_factor.shape == column_factor.shape
        and row_factor.is_contiguous()
        and column_factor.is_contiguous()
    ):
        return _SelectedProductKeyMemory.apply(
            row_factor,
            column_factor,
            logical_rows.contiguous(),
            template_indices.contiguous(),
            codebook_size,
        )
    row = torch.div(logical_rows, codebook_size, rounding_mode="floor")
    column = torch.remainder(logical_rows, codebook_size)
    return row_factor[template_indices, row] + column_factor[
        template_indices,
        column,
    ]


class ProductKeyInitialMemory(nn.Module):
    """Compose each initial value from its two native product-key entries."""

    def __init__(
        self,
        *,
        heads: int,
        slots_per_head: int,
        value_width: int,
    ) -> None:
        super().__init__()
        codebook_size = math.isqrt(slots_per_head)
        if heads <= 0 or value_width <= 0:
            raise ValueError("initial-memory geometry must be positive")
        if codebook_size * codebook_size != slots_per_head:
            raise ValueError("logical rows must form a square product-key table")
        self.heads = heads
        self.slots_per_head = slots_per_head
        self.codebook_size = codebook_size
        self.value_width = value_width

    def _reshape_table(self, value: torch.Tensor) -> torch.Tensor:
        expected = (self.heads * self.slots_per_head, self.value_width)
        if value.shape != expected:
            raise ValueError("initial-memory table shape changed")
        return value.reshape(
            self.heads,
            self.codebook_size,
            self.codebook_size,
            self.value_width,
        )

    def forward(self, row: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
        expected = (self.heads, self.codebook_size, self.value_width)
        if row.shape != expected or column.shape != expected:
            raise ValueError("product-key initial-memory factor shape changed")
        return (row[:, :, None, :] + column[:, None, :, :]).reshape(
            self.heads * self.slots_per_head,
            self.value_width,
        )

    def right_inverse(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a full table onto the additive product-key representation."""

        table = self._reshape_table(value)
        grand = table.mean(dim=(1, 2), keepdim=True)
        row = table.mean(dim=2) - 0.5 * grand.squeeze(2)
        column = table.mean(dim=1) - 0.5 * grand.squeeze(1)
        return row, column


def attach_product_key_initial_memory(
    layer: nn.Module,
    *,
    heads: int,
    slots_per_head: int,
    value_width: int,
) -> None:
    """Replace a full initial table parameter with two additive factor tables."""

    memory = getattr(layer, "memory", None)
    if not isinstance(memory, nn.Parameter):
        raise ValueError("product-key initialization requires learned memory")
    representation = ProductKeyInitialMemory(
        heads=heads,
        slots_per_head=slots_per_head,
        value_width=value_width,
    )
    parametrize.register_parametrization(
        layer,
        "memory",
        representation,
        unsafe=True,
    )
    for parameter in initial_memory_parameters(layer):
        parameter._sdm_memory_bank = True


def initial_memory_parameters(layer: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return the trainable tensors that represent a layer's initial memory."""

    if parametrize.is_parametrized(layer, "memory"):
        originals = layer.parametrizations.memory
        return originals.original0, originals.original1
    memory = getattr(layer, "memory", None)
    return (memory,) if isinstance(memory, nn.Parameter) else ()


def selected_initial_memory(
    layer: nn.Module,
    logical_rows: torch.Tensor,
    template_indices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate initial values for selected flat product-key addresses only."""

    if logical_rows.ndim != 1 or template_indices.shape != logical_rows.shape:
        raise ValueError("selected rows and templates must be aligned vectors")
    if parametrize.is_parametrized(layer, "memory"):
        representation = layer.parametrizations.memory[0]
        if not isinstance(representation, ProductKeyInitialMemory):
            raise TypeError("unknown initial-memory parametrization")
        row_factor, column_factor = initial_memory_parameters(layer)
        return _selected_product_key_memory(
            row_factor,
            column_factor,
            logical_rows,
            template_indices,
            representation.codebook_size,
        )
    memory = getattr(layer, "memory", None)
    if memory is None:
        width = int(layer.head_dim)
        return torch.zeros(
            logical_rows.numel(),
            width,
            device=logical_rows.device,
            dtype=next(layer.parameters()).dtype,
        )
    return memory.reshape(
        int(layer.num_heads),
        int(layer.slots_per_head),
        int(layer.head_dim),
    )[template_indices, logical_rows]


def initialize_product_key_memory(
    layer: nn.Module,
    *,
    table_std: float,
    generators: tuple[torch.Generator | None, torch.Generator | None] = (
        None,
        None,
    ),
) -> None:
    """Initialize factor sums to match the target full-table variance."""

    factor_std = table_std / math.sqrt(2.0)
    with torch.no_grad():
        factors = initial_memory_parameters(layer)
        if len(factors) != len(generators):
            raise ValueError("product-key memory requires two initialization streams")
        for factor, generator in zip(factors, generators):
            nn.init.trunc_normal_(
                factor,
                mean=0.0,
                std=factor_std,
                a=-3.0 * factor_std,
                b=3.0 * factor_std,
                generator=generator,
            )
