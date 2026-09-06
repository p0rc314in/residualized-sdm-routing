"""Faithful read-after-write SDM recurrence over a compact row namespace."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .addresses import encode_product_addresses


@dataclass(frozen=True)
class RecurrenceResult:
    readings: torch.Tensor
    final_memory: torch.Tensor
    maximum_workspace_elements: int


def _validate_recurrence(
    memory: torch.Tensor,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if memory.ndim != 3:
        raise ValueError("memory must be [B,K,V]")
    banks, rows, width = memory.shape
    if write_indices.ndim != 3 or write_indices.shape[0] != banks:
        raise ValueError("write indices must be [B,T,W]")
    if read_indices.ndim != 3 or read_indices.shape[:2] != write_indices.shape[:2]:
        raise ValueError("read indices must align with write banks and time")
    if write_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("write indices must be integer")
    if read_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("read indices must be integer")
    if write_weights.shape != write_indices.shape:
        raise ValueError("write weights must align with write indices")
    if read_weights.shape != read_indices.shape:
        raise ValueError("read weights must align with read indices")
    time, writes = write_indices.shape[1:]
    if values.shape != (banks, time, width):
        raise ValueError("write values must be [B,T,V]")
    if beta.shape not in ((banks, time, 1), (banks, time, width)):
        raise ValueError("beta must be scalar or value-channelwise per bank and token")
    if log_decay.shape not in ((banks, time, 1), (banks, time, writes)):
        raise ValueError("log decay must be scalar or write-routewise")
    tensors = (
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
    )
    if any(tensor.device != memory.device for tensor in tensors):
        raise ValueError("recurrence tensors must share a device")
    if write_indices.device.type == "cpu":
        if int(write_indices.min()) < 0 or int(write_indices.max()) >= rows:
            raise ValueError("compact write index lies outside the table")
        if int(read_indices.min()) < 0 or int(read_indices.max()) >= rows:
            raise ValueError("compact read index lies outside the table")
    return banks, rows, width, time, writes


def compact_gated_delta_recurrence(
    memory: torch.Tensor,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
) -> RecurrenceResult:
    """Execute native SDM math while touching only compact selected rows.

    This implementation is the portable correctness path. CUDA integration may
    replace its scatter operations with fused kernels while retaining the same
    compact namespace and equations.
    """

    banks, rows, width, time, writes = _validate_recurrence(
        memory,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
    )
    state = memory
    outputs: list[torch.Tensor] = []
    maximum_workspace = 0

    for position in range(time):
        write_index = write_indices[:, position].to(torch.int64)
        write_weight = write_weights[:, position].to(state.dtype)
        gather_index = write_index.unsqueeze(-1).expand(banks, writes, width)
        selected = torch.gather(state, 1, gather_index)
        decay = torch.exp(log_decay[:, position].float()).to(state.dtype)
        if decay.shape[-1] == 1:
            decay = decay.expand(banks, writes)
        decayed_selected = selected * decay.unsqueeze(-1)
        retrieved = (write_weight.unsqueeze(-1) * decayed_selected).sum(dim=1)
        update = beta[:, position].to(state.dtype) * (
            values[:, position].to(state.dtype) - retrieved
        )

        decay_by_row = torch.ones(
            banks,
            rows,
            device=state.device,
            dtype=state.dtype,
        ).scatter_reduce(
            1,
            write_index,
            decay,
            reduce="prod",
            include_self=True,
        )
        write_by_row = torch.zeros(
            banks,
            rows,
            width,
            device=state.device,
            dtype=state.dtype,
        ).scatter_add(
            1,
            gather_index,
            write_weight.unsqueeze(-1) * update.unsqueeze(1),
        )
        state = state * decay_by_row.unsqueeze(-1) + write_by_row

        read_index = read_indices[:, position].to(torch.int64)
        read_count = read_index.shape[-1]
        read_rows = torch.gather(
            state,
            1,
            read_index.unsqueeze(-1).expand(banks, read_count, width),
        )
        reading = (
            read_weights[:, position].to(state.dtype).unsqueeze(-1) * read_rows
        ).sum(dim=1)
        outputs.append(reading)
        maximum_workspace = max(
            maximum_workspace,
            decay_by_row.numel()
            + write_by_row.numel()
            + selected.numel()
            + read_rows.numel(),
        )

    return RecurrenceResult(
        readings=torch.stack(outputs, dim=1),
        final_memory=state,
        maximum_workspace_elements=maximum_workspace,
    )


def dense_gated_delta_oracle(
    initial_memory: torch.Tensor,
    write_addresses: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_addresses: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    codebook_size: int,
) -> RecurrenceResult:
    """Run the same recurrence on a full small-S table for semantic tests."""

    write_indices = encode_product_addresses(
        write_addresses,
        codebook_size=codebook_size,
    )
    read_indices = encode_product_addresses(
        read_addresses,
        codebook_size=codebook_size,
    )
    return compact_gated_delta_recurrence(
        initial_memory,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
    )


__all__ = [
    "RecurrenceResult",
    "compact_gated_delta_recurrence",
    "dense_gated_delta_oracle",
]
