"""Product-key candidate merging that returns explicit selected tuples."""

from __future__ import annotations

from collections.abc import Callable

import torch


def _lexicographic_order(addresses: torch.Tensor) -> torch.Tensor:
    """Return a stable lexicographic order for [...,K,p] integer tuples."""

    selected = addresses.shape[-2]
    order = torch.arange(selected, device=addresses.device, dtype=torch.int64)
    order = order.expand(*addresses.shape[:-2], selected)
    for factor in range(addresses.shape[-1] - 1, -1, -1):
        values = torch.gather(addresses[..., factor], -1, order)
        local = torch.argsort(values, dim=-1, stable=True)
        order = torch.gather(order, -1, local)
    return order


def product_key_topk(
    factor_scores: torch.Tensor,
    *,
    selected: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select top product-key tuples without enumerating C**p addresses.

    Args:
        factor_scores: [B,T,p,C] scores.
        selected: final route count R or W.

    Returns:
        selected softmax weights [B,T,K], sorted tuples [B,T,K,p], and
        corresponding unnormalized scores [B,T,K].
    """

    if factor_scores.ndim != 4:
        raise ValueError("factor scores must be [B,T,p,C]")
    if selected <= 0:
        raise ValueError("selected route count must be positive")
    _, _, factors, codebook_size = factor_scores.shape
    if factors <= 0 or codebook_size <= 0:
        raise ValueError("factor count and codebook size must be positive")
    per_factor = min(selected, codebook_size)
    top_values, top_indices = torch.topk(
        factor_scores,
        k=per_factor,
        dim=-1,
    )
    values = top_values[:, :, 0]
    addresses = top_indices[:, :, 0].unsqueeze(-1).to(torch.int64)

    for factor in range(1, factors):
        next_values = top_values[:, :, factor]
        next_indices = top_indices[:, :, factor].to(torch.int64)
        pair_scores = values.unsqueeze(-1) + next_values.unsqueeze(-2)
        previous_count = pair_scores.shape[-2]
        next_count = pair_scores.shape[-1]
        flattened = pair_scores.flatten(-2)
        keep = min(selected, flattened.shape[-1])
        values, positions = torch.topk(flattened, k=keep, dim=-1)
        previous_position = torch.div(
            positions,
            next_count,
            rounding_mode="floor",
        )
        next_position = torch.remainder(positions, next_count)
        previous_tuple = torch.gather(
            addresses,
            -2,
            previous_position.unsqueeze(-1).expand(
                *previous_position.shape,
                addresses.shape[-1],
            ),
        )
        selected_next = torch.gather(next_indices, -1, next_position)
        addresses = torch.cat((previous_tuple, selected_next.unsqueeze(-1)), dim=-1)
        if previous_count <= 0:
            raise AssertionError("candidate merge produced an empty frontier")

    order = _lexicographic_order(addresses)
    addresses = torch.gather(
        addresses,
        -2,
        order.unsqueeze(-1).expand(*order.shape, factors),
    )
    values = torch.gather(values, -1, order)
    weights = torch.softmax(values, dim=-1)
    return weights, addresses, values


def fused_product_key_scores(
    factor_scores: torch.Tensor,
    *,
    selected: int,
    outer_add_topk: Callable[
        [torch.Tensor, torch.Tensor, int],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    index_sort: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iteratively reuse released SDM's fused outer-add/top-k primitive.

    The frontier contains at most ``selected`` candidates. Each additional
    factor therefore contributes one fused candidate merge over at most
    ``selected²`` pairs. The function returns unnormalized selected scores and
    row-major int64 logical IDs for the released host; execution immediately
    decodes those IDs back to explicit tuples.
    """

    if factor_scores.ndim != 4:
        raise ValueError("factor scores must be [B,T,p,C]")
    if selected <= 0:
        raise ValueError("selected route count must be positive")
    _, _, factors, codebook_size = factor_scores.shape
    if factors <= 0 or codebook_size <= 0:
        raise ValueError("factor count and codebook size must be positive")
    if selected > codebook_size:
        raise ValueError("the released fused merge requires selected routes <= C")

    per_factor = min(selected, codebook_size)
    top_values, top_indices = torch.topk(
        factor_scores,
        k=per_factor,
        dim=-1,
    )
    values = top_values[:, :, 0]
    flat = top_indices[:, :, 0].to(torch.int64)

    for factor in range(1, factors):
        next_values = top_values[:, :, factor]
        next_indices = top_indices[:, :, factor].to(torch.int64)
        keep = min(selected, values.shape[-1] * next_values.shape[-1])
        values, previous_position, next_position = outer_add_topk(
            values,
            next_values,
            keep,
        )
        flat = (
            torch.gather(flat, -1, previous_position.to(torch.int64))
            * codebook_size
            + torch.gather(next_indices, -1, next_position.to(torch.int64))
        )

    original_shape = flat.shape
    flattened = flat.reshape(-1, original_shape[-1])
    if index_sort is not None and original_shape[-1] <= 128:
        sorted_flat, permutation = index_sort(flattened)
    else:
        sorted_flat, permutation = flattened.sort(dim=-1)
    flat = sorted_flat.reshape(original_shape)
    values = values.reshape(-1, original_shape[-1]).gather(
        -1, permutation
    ).reshape(original_shape)
    return values, flat


__all__ = ["fused_product_key_scores", "product_key_topk"]
