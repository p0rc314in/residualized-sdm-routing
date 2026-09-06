# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Minimal cache/state classes for SparseDeltaMemory.

Extracted (self-contained) from the internal memory-controller layer so this
open-source package has no dependency on the original framework. ``Cache`` is a
tiny ABC; ``SDMLayerState`` holds the memory bank plus the running sequence
length for inference/decode.
"""

from abc import ABC, abstractmethod

import torch


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize a tensor along a given dimension."""
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


class Cache(ABC):
    """Base class for all caches. A model's full cache is a ``list[Cache]``."""

    @abstractmethod
    def __getitem__(self, *args, **kwargs) -> torch.Tensor | int:
        pass

    @property
    @abstractmethod
    def cache_len(self) -> int:
        pass

    @abstractmethod
    def update_(self, *args, **kwargs) -> "Cache":
        """Update the cache in place."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the state."""
        pass


class SDMLayerState(Cache):
    """Inference/decode state for a ``SparseDeltaMemory`` layer: the memory bank
    plus the number of tokens consumed so far."""

    def __init__(self, memory: torch.Tensor, seq_len: int = 0) -> None:
        self.memory = memory
        self._cache_len = seq_len

    def __getitem__(self, key_idx: int) -> torch.Tensor | None:
        if key_idx == 0:
            return self.memory
        raise KeyError

    def __len__(self) -> int:
        return 1

    @property
    def cache_len(self) -> int:
        return self._cache_len

    def reset(self) -> None:
        self.memory.zero_()
        self._cache_len = 0

    def update_(self, new_memory: torch.Tensor, seq_len: int = 1) -> Cache:
        assert self.memory is not None
        self.memory = new_memory.detach()
        self._cache_len += seq_len
        return self


class CompactSDMLayerState(Cache):
    """Inference state containing only rows written by each recurrent bank."""

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        counts: torch.Tensor,
        template_indices: torch.Tensor,
        *,
        seq_len: int = 0,
    ) -> None:
        if keys.ndim != 2 or keys.dtype not in (torch.int32, torch.int64):
            raise ValueError("compact keys must be integer [banks,rows]")
        if values.ndim != 3 or values.shape[:2] != keys.shape:
            raise ValueError("compact values must be [banks,rows,width]")
        banks = keys.shape[0]
        if counts.shape != (banks,) or counts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("compact row counts must be integer [banks]")
        if template_indices.shape != (banks,) or template_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("compact template indices must be integer [banks]")
        if any(
            tensor.device != keys.device
            for tensor in (values, counts, template_indices)
        ):
            raise ValueError("compact state tensors must share a device")
        if seq_len < 0:
            raise ValueError("cache sequence length cannot be negative")

        self.keys = keys
        self.values = values
        self.counts = counts.to(torch.int64)
        self.template_indices = template_indices.to(torch.int64)
        self._cache_len = seq_len

    @classmethod
    def empty(
        cls,
        template_indices: torch.Tensor,
        *,
        width: int,
        dtype: torch.dtype,
        seq_len: int = 0,
    ) -> "CompactSDMLayerState":
        if width <= 0:
            raise ValueError("compact value width must be positive")
        banks = template_indices.numel()
        return cls(
            torch.empty(
                banks,
                0,
                device=template_indices.device,
                dtype=torch.int64,
            ),
            torch.empty(
                banks,
                0,
                width,
                device=template_indices.device,
                dtype=dtype,
            ),
            torch.zeros(banks, device=template_indices.device, dtype=torch.int64),
            template_indices,
            seq_len=seq_len,
        )

    def __getitem__(self, key_idx: int) -> torch.Tensor:
        if key_idx == 0:
            return self.values
        raise KeyError(key_idx)

    def __len__(self) -> int:
        return 1

    @property
    def memory(self) -> torch.Tensor:
        return self.values

    @property
    def cache_len(self) -> int:
        return self._cache_len

    def update_(
        self,
        new_memory: "CompactSDMLayerState",
        seq_len: int = 1,
    ) -> Cache:
        if not isinstance(new_memory, CompactSDMLayerState):
            raise TypeError("compact cache updates require compact state")
        if seq_len < 0:
            raise ValueError("cache sequence increment cannot be negative")
        self.keys = new_memory.keys.detach()
        self.values = new_memory.values.detach()
        self.counts = new_memory.counts.detach()
        self.template_indices = new_memory.template_indices
        self._cache_len += seq_len
        return self

    def reset(self) -> None:
        self.keys = self.keys[:, :0]
        self.values = self.values[:, :0]
        self.counts.zero_()
        self._cache_len = 0

    def storage_elements(self) -> dict[str, int]:
        """Report retained state without counting scalar logical capacity."""

        return {
            "logical_keys": self.keys.numel(),
            "private_values": self.values.numel(),
            "row_counts": self.counts.numel(),
            "template_indices": self.template_indices.numel(),
            "logical_capacity_tensor": 0,
        }


class CopyOnWriteSDMLayerState(Cache):
    """Inference state that materializes an SDM row on its first write.

    ``shared_initial_memory`` remains model-owned and is indexed by
    ``template_indices`` for each independent batch/head bank. Mutable rows
    live in one shared physical slab. ``logical_to_physical`` resolves a
    logical address to that slab, or remains ``-1`` while reads should observe
    the shared initial row.
    """

    def __init__(
        self,
        shared_initial_memory: torch.Tensor,
        template_indices: torch.Tensor,
        *,
        capacity_rows: int,
        state_dtype: torch.dtype,
        growth_quantum_rows: int,
        seq_len: int = 0,
    ) -> None:
        if shared_initial_memory.ndim != 3:
            raise ValueError("shared initial memory must be [heads,slots,width]")
        heads, slots, width = shared_initial_memory.shape
        if (
            template_indices.ndim != 1
            or template_indices.dtype not in (torch.int32, torch.int64)
            or template_indices.device != shared_initial_memory.device
        ):
            raise ValueError("template indices must be integer [banks] on device")
        if capacity_rows <= 0 or growth_quantum_rows <= 0:
            raise ValueError("copy-on-write row capacities must be positive")
        if capacity_rows > template_indices.numel() * slots:
            raise ValueError("physical capacity exceeds the logical cache")
        if state_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("copy-on-write state must use a floating-point dtype")

        templates = template_indices.to(torch.int64)
        if templates.device.type == "cpu":
            if templates.numel() and (
                int(templates.min()) < 0 or int(templates.max()) >= heads
            ):
                raise ValueError("template index lies outside initial memory")
        elif templates.numel() and hasattr(torch, "_assert_async"):
            torch._assert_async(
                ((templates >= 0) & (templates < heads)).all(),
                "template index lies outside initial memory",
            )

        device = shared_initial_memory.device
        self.shared_initial_memory = shared_initial_memory
        self.template_indices = templates.contiguous()
        self.values = torch.empty(
            capacity_rows,
            width,
            device=device,
            dtype=state_dtype,
        )
        self.logical_to_physical = torch.full(
            (templates.numel(), slots),
            -1,
            device=device,
            dtype=torch.int32,
        )
        self.free_rows = torch.arange(
            capacity_rows,
            device=device,
            dtype=torch.int32,
        )
        self.free_count = torch.tensor(
            capacity_rows,
            device=device,
            dtype=torch.int32,
        )
        self.overflow = torch.zeros((), device=device, dtype=torch.int32)
        self.growth_quantum_rows = growth_quantum_rows
        self.growth_events = 0
        self.rows_added = 0
        self.rows_copied = 0
        self._guaranteed_free_rows = capacity_rows
        self._cache_len = seq_len

    @property
    def banks(self) -> int:
        return self.logical_to_physical.shape[0]

    @property
    def slots(self) -> int:
        return self.logical_to_physical.shape[1]

    @property
    def width(self) -> int:
        return self.values.shape[1]

    @property
    def capacity_rows(self) -> int:
        return self.values.shape[0]

    @property
    def cache_len(self) -> int:
        return self._cache_len

    @property
    def materialized_rows(self) -> int:
        return self.capacity_rows - int(self.free_count.item())

    def __getitem__(self, key_idx: int) -> torch.Tensor:
        if key_idx == 0:
            return self.values
        raise KeyError

    def __len__(self) -> int:
        return 1

    @torch.no_grad()
    def grow(self, minimum_capacity_rows: int) -> int:
        """Grow the pooled value slab without changing logical row ownership."""

        maximum_rows = self.banks * self.slots
        if minimum_capacity_rows <= self.capacity_rows:
            return 0
        if minimum_capacity_rows > maximum_rows:
            raise ValueError("copy-on-write cache exceeds its logical capacity")
        quantum = self.growth_quantum_rows
        selected_capacity = min(
            maximum_rows,
            ((minimum_capacity_rows + quantum - 1) // quantum) * quantum,
        )
        old_capacity = self.capacity_rows
        old_free_count = int(self.free_count.item())
        added = selected_capacity - old_capacity

        values = torch.empty(
            selected_capacity,
            self.width,
            device=self.values.device,
            dtype=self.values.dtype,
        )
        values[:old_capacity].copy_(self.values)
        free_rows = torch.empty(
            selected_capacity,
            device=self.free_rows.device,
            dtype=self.free_rows.dtype,
        )
        if old_free_count:
            free_rows[:old_free_count].copy_(self.free_rows[:old_free_count])
        free_rows[old_free_count : old_free_count + added].copy_(
            torch.arange(
                old_capacity,
                selected_capacity,
                device=self.free_rows.device,
                dtype=self.free_rows.dtype,
            )
        )
        self.values = values
        self.free_rows = free_rows
        self.free_count.fill_(old_free_count + added)
        self.overflow.zero_()
        self.growth_events += 1
        self.rows_added += added
        self.rows_copied += old_capacity
        self._guaranteed_free_rows += added
        return added

    @torch.no_grad()
    def prepare_step(self, maximum_new_rows: int) -> None:
        """Ensure a decode step cannot exhaust the physical row slab."""

        if maximum_new_rows < 0:
            raise ValueError("maximum new rows cannot be negative")
        if self._guaranteed_free_rows < maximum_new_rows:
            free_rows = int(self.free_count.item())
            maximum_rows = self.banks * self.slots
            if free_rows < maximum_new_rows and self.capacity_rows < maximum_rows:
                required = min(
                    maximum_rows,
                    self.capacity_rows + maximum_new_rows - free_rows,
                )
                self.grow(required)
                free_rows = int(self.free_count.item())
            self._guaranteed_free_rows = max(free_rows, maximum_new_rows)
        self._guaranteed_free_rows -= maximum_new_rows

    @torch.no_grad()
    def replace_from_dense(
        self,
        dense_memory: torch.Tensor,
        written_slots: torch.Tensor,
    ) -> None:
        """Pack a dense prefill result using its hard-written address union."""

        if dense_memory.shape != (self.banks, self.slots, self.width):
            raise ValueError("dense prefill memory does not match the cache")
        if (
            written_slots.shape != (self.banks, self.slots)
            or written_slots.dtype != torch.bool
            or written_slots.device != self.values.device
        ):
            raise ValueError("written-slot mask does not match the cache")
        active_count = int(written_slots.sum().item())
        if active_count > self.capacity_rows:
            self.grow(active_count)

        self.logical_to_physical.fill_(-1)
        if active_count:
            bank, slot = written_slots.nonzero(as_tuple=True)
            physical = torch.arange(
                active_count,
                device=self.values.device,
                dtype=torch.int32,
            )
            self.logical_to_physical[bank, slot] = physical
            self.values[:active_count].copy_(
                dense_memory[bank, slot].to(self.values.dtype)
            )
        free_count = self.capacity_rows - active_count
        if free_count:
            self.free_rows[:free_count].copy_(
                torch.arange(
                    active_count,
                    self.capacity_rows,
                    device=self.free_rows.device,
                    dtype=self.free_rows.dtype,
                )
            )
        self.free_count.fill_(free_count)
        self.overflow.zero_()
        self._guaranteed_free_rows = free_count

    @torch.no_grad()
    def materialize(self) -> torch.Tensor:
        """Build the effective dense table for validation or diagnostics."""

        dense = self.shared_initial_memory[self.template_indices].to(
            self.values.dtype
        ).clone()
        private = self.logical_to_physical >= 0
        if bool(private.any()):
            dense[private] = self.values[
                self.logical_to_physical[private].to(torch.int64)
            ]
        return dense

    def storage_bytes(self) -> dict[str, int]:
        """Report private cache storage separately from shared model state."""

        def size(tensor: torch.Tensor) -> int:
            return tensor.numel() * tensor.element_size()

        private = {
            "value_slab": size(self.values),
            "logical_to_physical": size(self.logical_to_physical),
            "free_rows": size(self.free_rows),
            "allocator_scalars": size(self.free_count) + size(self.overflow),
            "template_indices": size(self.template_indices),
        }
        return {
            **private,
            "private_total": sum(private.values()),
            "shared_initial_memory": size(self.shared_initial_memory),
        }

    def update_(
        self,
        new_memory: "CopyOnWriteSDMLayerState",
        seq_len: int = 1,
    ) -> Cache:
        if new_memory is not self:
            raise ValueError("copy-on-write cache cannot replace its row pool")
        self._cache_len += seq_len
        return self

    @torch.no_grad()
    def reset(self) -> None:
        self.logical_to_physical.fill_(-1)
        self.free_rows.copy_(
            torch.arange(
                self.capacity_rows,
                device=self.free_rows.device,
                dtype=self.free_rows.dtype,
            )
        )
        self.free_count.fill_(self.capacity_rows)
        self.overflow.zero_()
        self._guaranteed_free_rows = self.capacity_rows
        self._cache_len = 0
