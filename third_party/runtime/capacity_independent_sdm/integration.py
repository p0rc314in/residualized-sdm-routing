"""Host-neutral adapter from released SDM routes to compact execution state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from .addresses import (
    SelectedAddressLayout,
    compact_selected_addresses,
    decode_product_addresses,
)
from .accounting import SparseOccupancy, sparse_occupancy
from .priors import SelectedRowPrior
from .kernels import CompactKernelBackend, add_bank_offsets
from .state import (
    DEFAULT_GROWTH_QUANTUM_ROWS,
    CapacityIndependentLayerState,
    PackedMemoryState,
)


@dataclass
class PreparedMemoryExecution:
    """Private context carried across the three released-SDM host hooks."""

    write_addresses: torch.Tensor
    read_addresses: torch.Tensor
    template_indices: torch.Tensor
    layout: SelectedAddressLayout | None
    initial_selected_memory: torch.Tensor | None
    cache: CapacityIndependentLayerState | None
    physical_write_remap: torch.Tensor
    physical_read_remap: torch.Tensor
    context_parallel_world: int = 1
    context_parallel_rank: int = 0
    context_parallel_group: object | None = None
    final_selected_memory: torch.Tensor | None = None
    occupancy: SparseOccupancy | None = None


def _module_device(module: torch.nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


class CapacityIndependentExecutionAdapter:
    """Translate flat native-SDM routes into selected tuples and packed state."""

    def __init__(
        self,
        prior: SelectedRowPrior,
        *,
        num_heads: int,
        growth_quantum_rows: int = DEFAULT_GROWTH_QUANTUM_ROWS,
        memory_block_size: int = 256,
        snapshot_quant: str = "none",
        key_weighted_decay: bool = False,
        collect_sparse_telemetry: bool = False,
    ) -> None:
        if num_heads <= 0:
            raise ValueError("the host must expose at least one memory head")
        if prior.templates != num_heads:
            raise ValueError("the prior needs one untouched-state template per head")
        if growth_quantum_rows < 0:
            raise ValueError("growth quantum cannot be negative")
        self.prior = prior
        self.num_heads = num_heads
        self.growth_quantum_rows = growth_quantum_rows
        self.collect_sparse_telemetry = collect_sparse_telemetry
        self.kernel_backend = CompactKernelBackend(
            memory_block_size=memory_block_size,
            snapshot_quant=snapshot_quant,
            key_weighted_decay=key_weighted_decay,
        )

    def _template_indices(
        self,
        banks: int,
        device: torch.device,
        *,
        local_heads: int | None = None,
        head_start: int = 0,
    ) -> torch.Tensor:
        heads = self.num_heads if local_heads is None else local_heads
        if heads <= 0 or banks % heads:
            raise ValueError("memory banks do not form complete local-head groups")
        if head_start < 0 or head_start + heads > self.num_heads:
            raise ValueError("local head slice lies outside the prior templates")
        return (
            torch.arange(banks, device=device, dtype=torch.int64).remainder(heads)
            + head_start
        )

    def _validate_cache(
        self,
        cache: CapacityIndependentLayerState,
        *,
        banks: int,
        template_indices: torch.Tensor,
        device: torch.device,
    ) -> None:
        packed = cache.packed
        if packed.device != device:
            raise ValueError("cache and selected routes must share a device")
        if packed.banks != banks:
            raise ValueError("cache bank count does not match the current batch")
        if (
            packed.codebook_size != self.prior.codebook_size
            or packed.factors != self.prior.factors
            or packed.value_width != self.prior.value_width
        ):
            raise ValueError("cache geometry does not match the selected-row prior")
        if not torch.equal(packed.template_indices, template_indices):
            raise ValueError("cache head templates do not match the current batch")
        if not bool(packed.active_banks.all()):
            raise ValueError("released cache banks must be compacted before execution")

    def prepare(
        self,
        *,
        k_idx: torch.Tensor,
        q_idx: torch.Tensor,
        cache: CapacityIndependentLayerState | None,
        context_parallel_active: bool,
        context_parallel_group=None,
        local_heads: int | None = None,
        head_start: int = 0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        PreparedMemoryExecution,
    ]:
        """Decode native flat routes and prepare compact or packed physical state."""

        if k_idx.ndim != 3 or q_idx.ndim != 3:
            raise ValueError("released SDM routes must be [BH,T,N]")
        if k_idx.shape[:2] != q_idx.shape[:2]:
            raise ValueError("read and write routes must share banks and time")
        if k_idx.device != q_idx.device:
            raise ValueError("read and write routes must share a device")
        if k_idx.dtype not in (torch.int32, torch.int64) or q_idx.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("released SDM routes must be integer")

        banks = k_idx.shape[0]
        templates = self._template_indices(
            banks,
            k_idx.device,
            local_heads=local_heads,
            head_start=head_start,
        )
        cp_world = 1
        cp_rank = 0
        global_k_idx = k_idx
        global_q_idx = q_idx
        if context_parallel_group is not None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "context-parallel execution needs initialized torch.distributed"
                )
            cp_world = dist.get_world_size(context_parallel_group)
            cp_rank = dist.get_rank(context_parallel_group)
            gathered_k = [torch.empty_like(k_idx) for _ in range(cp_world)]
            gathered_q = [torch.empty_like(q_idx) for _ in range(cp_world)]
            dist.all_gather(
                gathered_k, k_idx.contiguous(), group=context_parallel_group
            )
            dist.all_gather(
                gathered_q, q_idx.contiguous(), group=context_parallel_group
            )
            global_k_idx = torch.cat(gathered_k, dim=1)
            global_q_idx = torch.cat(gathered_q, dim=1)
        elif context_parallel_active and local_heads is None:
            # A boolean without a group is retained for host compatibility.  It
            # denotes Ulysses-only execution only when the local-head geometry
            # is supplied explicitly.
            raise ValueError(
                "sequential context parallelism requires its process group"
            )

        writes = decode_product_addresses(
            global_k_idx,
            codebook_size=self.prior.codebook_size,
            factors=self.prior.factors,
        )
        reads = decode_product_addresses(
            global_q_idx,
            codebook_size=self.prior.codebook_size,
            factors=self.prior.factors,
        )

        if cache is not None:
            if cp_world != 1:
                raise ValueError("mutable decode caches are not sequence-sharded")
            if not isinstance(cache, CapacityIndependentLayerState):
                raise TypeError(
                    "capacity-independent execution requires its packed cache"
                )
            self._validate_cache(
                cache,
                banks=banks,
                template_indices=templates,
                device=k_idx.device,
            )
            context = PreparedMemoryExecution(
                write_addresses=writes,
                read_addresses=reads,
                template_indices=templates,
                layout=None,
                initial_selected_memory=None,
                cache=cache,
                physical_write_remap=k_idx,
                physical_read_remap=q_idx,
            )
            return cache.memory, k_idx, q_idx, context

        layout = compact_selected_addresses(
            writes,
            reads,
            codebook_size=self.prior.codebook_size,
        )
        initial = self.prior(layout.keys, templates)
        time = k_idx.shape[1]
        start = cp_rank * time
        stop = start + time
        local_write_remap = layout.write_remap[:, start:stop]
        local_read_remap = layout.read_remap[:, start:stop]
        physical_writes = add_bank_offsets(local_write_remap, layout.max_rows)
        physical_reads = add_bank_offsets(local_read_remap, layout.max_rows)
        context = PreparedMemoryExecution(
            write_addresses=writes,
            read_addresses=reads,
            template_indices=templates,
            layout=layout,
            initial_selected_memory=initial,
            cache=None,
            physical_write_remap=physical_writes,
            physical_read_remap=physical_reads,
            context_parallel_world=cp_world,
            context_parallel_rank=cp_rank,
            context_parallel_group=context_parallel_group,
        )
        return (
            initial.reshape(-1, self.prior.value_width),
            physical_writes,
            physical_reads,
            context,
        )

    def execute(
        self,
        memory: torch.Tensor,
        k_idx: torch.Tensor,
        k_val: torch.Tensor,
        values: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        q_idx: torch.Tensor,
        q_val: torch.Tensor,
        context: PreparedMemoryExecution,
        *,
        training: bool,
        grad_final_memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run compact prefill/training or mutate an existing packed cache."""

        if context.cache is None:
            if context.layout is None or context.initial_selected_memory is None:
                raise RuntimeError("compact execution context is incomplete")
            layout = context.layout
            expected = (
                layout.banks * layout.max_rows,
                self.prior.value_width,
            )
            if memory.shape != expected:
                raise ValueError(f"compact memory must be {expected}")
            if not torch.equal(k_idx, context.physical_write_remap) or not torch.equal(
                q_idx, context.physical_read_remap
            ):
                raise ValueError("host changed compact route remaps before recurrence")
            readings, terminal = self.kernel_backend.execute(
                memory,
                k_idx,
                k_val,
                values,
                beta,
                log_decay,
                q_idx,
                q_val,
                rows_per_bank=layout.max_rows,
                training=training,
                grad_final_memory=grad_final_memory,
            )
            context.final_selected_memory = terminal.view(
                layout.banks, layout.max_rows, self.prior.value_width
            )
            return (
                readings,
                terminal,
            )

        if training:
            raise ValueError("mutable packed caches are inference-only")
        packed = context.cache.packed
        outputs: list[torch.Tensor] = []
        for position in range(k_idx.shape[1]):
            step = packed.decode_step(
                self.prior,
                context.write_addresses[:, position],
                k_val[:, position],
                values[:, position],
                beta[:, position],
                log_decay[:, position],
                context.read_addresses[:, position],
                q_val[:, position],
            )
            outputs.append(step.readings)
        return torch.stack(outputs, dim=1), packed.values

    def prepare_telemetry(
        self,
        context: PreparedMemoryExecution,
        write_weights: torch.Tensor,
        read_weights: torch.Tensor,
    ) -> SparseOccupancy | None:
        """Collect selected-route telemetry while every CP rank participates."""

        if not self.collect_sparse_telemetry:
            return None
        telemetry_write = write_weights
        telemetry_read = read_weights
        if context.context_parallel_world > 1:
            group = context.context_parallel_group
            if group is None:
                raise RuntimeError("missing sequential CP telemetry group")
            telemetry_write = torch.cat(
                tuple(dist_nn.all_gather(write_weights, group=group)), dim=1
            )
            telemetry_read = torch.cat(
                tuple(dist_nn.all_gather(read_weights, group=group)), dim=1
            )
        context.occupancy = sparse_occupancy(
            context.write_addresses,
            telemetry_write,
            context.read_addresses,
            telemetry_read,
            codebook_size=self.prior.codebook_size,
        )
        return context.occupancy

    def finalize(
        self,
        context: PreparedMemoryExecution,
        cache: CapacityIndependentLayerState | None,
        *,
        seq_len: int,
    ) -> CapacityIndependentLayerState:
        """Pack written prefill rows or advance the already-mutated decode cache."""

        if seq_len <= 0:
            raise ValueError("cache sequence increments must be positive")
        if cache is not None:
            if cache is not context.cache:
                raise ValueError("host replaced the packed cache during execution")
            cache.seq_len += seq_len
            return cache
        if context.layout is None or context.final_selected_memory is None:
            raise RuntimeError("compact execution did not produce a terminal state")
        packed = PackedMemoryState.from_compact(
            context.layout,
            context.final_selected_memory,
            context.template_indices,
            self.prior,
            growth_quantum_rows=self.growth_quantum_rows,
            state_dtype=context.final_selected_memory.dtype,
        )
        return CapacityIndependentLayerState(packed=packed, seq_len=seq_len)

    def create_cache(
        self,
        *,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: str | torch.device | None,
    ) -> CapacityIndependentLayerState:
        """Create an empty physical slab without materializing untouched rows."""

        if batch_size <= 0 or seq_len < 0:
            raise ValueError("cache batch size must be positive and length nonnegative")
        prior_device = _module_device(self.prior)
        requested_device = prior_device if device is None else torch.device(device)
        if requested_device.type != prior_device.type or requested_device.index not in (
            None,
            prior_device.index,
        ):
            raise ValueError("move the layer prior to the requested cache device first")
        selected_device = prior_device
        banks = batch_size * self.num_heads
        templates = self._template_indices(banks, selected_device)
        packed = PackedMemoryState.empty(
            self.prior,
            banks=banks,
            template_indices=templates,
            growth_quantum_rows=self.growth_quantum_rows,
            state_dtype=dtype,
        )
        return CapacityIndependentLayerState(packed=packed, seq_len=seq_len)


__all__ = [
    "CapacityIndependentExecutionAdapter",
    "PreparedMemoryExecution",
]
