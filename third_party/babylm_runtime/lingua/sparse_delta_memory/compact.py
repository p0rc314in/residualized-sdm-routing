# Copyright (c) 2026 p0rc314in

"""Selected-row execution for native Sparse Delta Memory routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .cache import CompactSDMLayerState


@dataclass(frozen=True)
class CompactExecution:
    """A padded per-bank selected union and its physical route remaps."""

    keys: torch.Tensor
    counts: torch.Tensor
    write_remap: torch.Tensor
    read_remap: torch.Tensor
    memory: torch.Tensor
    written: torch.Tensor
    template_indices: torch.Tensor

    @property
    def banks(self) -> int:
        return self.keys.shape[0]

    @property
    def rows_per_bank(self) -> int:
        return self.keys.shape[1]

    def flattened_memory(self) -> torch.Tensor:
        return self.memory.reshape(
            self.banks * self.rows_per_bank,
            self.memory.shape[-1],
        )

    def offset_routes(self) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = (
            torch.arange(
                self.banks,
                device=self.keys.device,
                dtype=torch.int64,
            ).view(self.banks, 1, 1)
            * self.rows_per_bank
        )
        return self.write_remap + offsets, self.read_remap + offsets

    def finalize(
        self,
        final_memory: torch.Tensor,
        *,
        seq_len: int,
    ) -> CompactSDMLayerState:
        """Retain only rows that have received at least one write."""

        if final_memory.shape != (
            self.banks * self.rows_per_bank,
            self.memory.shape[-1],
        ):
            raise ValueError("terminal compact memory has the wrong shape")
        final = final_memory.reshape_as(self.memory)
        counts = self.written.sum(dim=1, dtype=torch.int64)
        maximum = int(counts.max().item()) if counts.numel() else 0
        keys = torch.full(
            (self.banks, maximum),
            -1,
            device=self.keys.device,
            dtype=torch.int64,
        )
        values = torch.empty(
            self.banks,
            maximum,
            final.shape[-1],
            device=final.device,
            dtype=final.dtype,
        )
        bank, row = self.written.nonzero(as_tuple=True)
        if bank.numel():
            starts = counts.cumsum(0) - counts
            local = (
                torch.arange(bank.numel(), device=bank.device, dtype=torch.int64)
                - starts[bank]
            )
            keys[bank, local] = self.keys[bank, row]
            values[bank, local] = final[bank, row]
        return CompactSDMLayerState(
            keys,
            values,
            counts,
            self.template_indices,
            seq_len=seq_len,
        )

    def storage_elements(self) -> dict[str, int]:
        return {
            "logical_keys": self.keys.numel(),
            "row_counts": self.counts.numel(),
            "write_remap": self.write_remap.numel(),
            "read_remap": self.read_remap.numel(),
            "selected_values": self.memory.numel(),
            "written_mask": self.written.numel(),
            "logical_capacity_tensor": 0,
        }


def _validate_routes(
    name: str,
    routes: torch.Tensor,
    *,
    banks: int,
    time: int,
    slots: int,
) -> None:
    if (
        routes.ndim != 3
        or routes.shape[:2] != (banks, time)
        or routes.shape[2] == 0
        or routes.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(f"{name} routes must be integer [banks,time,routes]")
    valid = ((routes >= 0) & (routes < slots)).all()
    if valid.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(valid, f"{name} route lies outside logical memory")
    elif not bool(valid):
        raise ValueError(f"{name} route lies outside logical memory")


def prepare_compact_execution(
    write_indices: torch.Tensor,
    read_indices: torch.Tensor,
    *,
    slots: int,
    width: int,
    dtype: torch.dtype,
    template_indices: torch.Tensor,
    initial_values: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    previous: CompactSDMLayerState | None = None,
) -> CompactExecution:
    """Deduplicate selected routes without allocating a slots-wide tensor."""

    if write_indices.ndim != 3:
        raise ValueError("write routes must be [banks,time,routes]")
    banks, time, writes = write_indices.shape
    if banks <= 0 or time <= 0 or writes <= 0 or slots <= 0 or width <= 0:
        raise ValueError("compact execution geometry must be positive")
    _validate_routes(
        "write",
        write_indices,
        banks=banks,
        time=time,
        slots=slots,
    )
    _validate_routes(
        "read",
        read_indices,
        banks=banks,
        time=time,
        slots=slots,
    )
    if write_indices.device != read_indices.device:
        raise ValueError("read and write routes must share a device")
    if template_indices.shape != (banks,) or template_indices.device != write_indices.device:
        raise ValueError("template indices must be [banks] on the route device")
    if previous is not None:
        if (
            previous.keys.shape[0] != banks
            or previous.values.shape[-1] != width
            or previous.keys.device != write_indices.device
            or not torch.equal(previous.template_indices, template_indices)
        ):
            raise ValueError("previous compact state geometry changed")

    reads = read_indices.shape[2]
    if banks > torch.iinfo(torch.int64).max // slots:
        raise OverflowError("compact route encoding exceeds int64")
    current = torch.cat(
        (
            write_indices.reshape(banks, time * writes),
            read_indices.reshape(banks, time * reads),
        ),
        dim=1,
    ).to(torch.int64)
    # A bank-major scalar code preserves lexicographic (bank, row) order while
    # allowing the selected union to use the substantially cheaper 1-D unique.
    bank_offsets = (
        torch.arange(banks, device=current.device, dtype=torch.int64)
        .view(banks, 1)
        * slots
    )
    current_codes = (current + bank_offsets).reshape(-1)

    previous_codes = current_codes[:0]
    previous_banks = current_codes[:0]
    previous_values = None
    if previous is not None and previous.keys.shape[1]:
        old_row = torch.arange(previous.keys.shape[1], device=current.device)
        old_valid = old_row.unsqueeze(0) < previous.counts.unsqueeze(1)
        old_bank, old_position = old_valid.nonzero(as_tuple=True)
        previous_banks = old_bank.to(torch.int64)
        previous_codes = (
            previous_banks * slots
            + previous.keys[old_bank, old_position]
        )
        previous_values = previous.values[old_bank, old_position]

    candidates = torch.cat((current_codes, previous_codes), dim=0)
    unique_codes, inverse = torch.unique(
        candidates,
        sorted=True,
        return_inverse=True,
    )
    unique_banks = torch.div(unique_codes, slots, rounding_mode="floor")
    unique_rows = torch.remainder(unique_codes, slots)
    counts = torch.bincount(unique_banks, minlength=banks).to(torch.int64)
    starts = counts.cumsum(0) - counts
    local_unique = (
        torch.arange(
            unique_codes.shape[0],
            device=current.device,
            dtype=torch.int64,
        )
        - starts[unique_banks]
    )
    local_inverse = local_unique[inverse[: current_codes.shape[0]]].reshape_as(current)
    write_remap = local_inverse[:, : time * writes].reshape(banks, time, writes)
    read_remap = local_inverse[:, time * writes :].reshape(banks, time, reads)

    maximum = int(counts.max().item())
    keys = torch.full(
        (banks, maximum),
        -1,
        device=current.device,
        dtype=torch.int64,
    )
    keys[unique_banks, local_unique] = unique_rows
    memory = torch.zeros(
        banks,
        maximum,
        width,
        device=current.device,
        dtype=dtype,
    )
    if initial_values is not None:
        selected = initial_values(
            unique_rows,
            template_indices[unique_banks],
        )
        if selected.shape != (unique_codes.shape[0], width):
            raise ValueError("selected initial values have the wrong shape")
        memory[unique_banks, local_unique] = selected.to(dtype)

    written = torch.zeros(
        banks,
        maximum,
        device=current.device,
        dtype=torch.bool,
    )
    if previous_values is not None and previous_codes.shape[0]:
        old_local = local_unique[inverse[current_codes.shape[0] :]]
        memory[previous_banks, old_local] = previous_values.to(dtype)
        written[previous_banks, old_local] = True
    written.scatter_(1, write_remap.reshape(banks, -1), True)

    return CompactExecution(
        keys=keys,
        counts=counts,
        write_remap=write_remap,
        read_remap=read_remap,
        memory=memory,
        written=written,
        template_indices=template_indices.to(torch.int64),
    )
