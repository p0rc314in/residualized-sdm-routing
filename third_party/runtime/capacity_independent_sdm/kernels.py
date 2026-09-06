"""Production dispatch into released SDM kernels on compact physical rows.

The released CUDA kernels are already parameterized by the number of physical
slots per recurrent bank.  This module supplies the selected-union width K in
that parameter and remaps every route into the flattened B*K slab.  As a
result, the kernels retain their optimized WY forward/backward and fused
inference implementations while their slot accumulators scale with K, not the
logical address-space size S.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .recurrence import compact_gated_delta_recurrence


@dataclass(frozen=True)
class CompactKernelReport:
    backend: str
    banks: int
    selected_rows_per_bank: int
    selected_value_elements: int
    logical_capacity_tensor_elements: int = 0


class _AttachTerminalGradient(torch.autograd.Function):
    """Inject CP's downstream-state cotangent into a portable recurrence."""

    @staticmethod
    def forward(ctx, readings, terminal, terminal_gradient):
        ctx.save_for_backward(terminal_gradient)
        return readings

    @staticmethod
    def backward(ctx, grad_readings):
        (terminal_gradient,) = ctx.saved_tensors
        return grad_readings, terminal_gradient, None


def add_bank_offsets(indices: torch.Tensor, rows_per_bank: int) -> torch.Tensor:
    """Map bank-local compact IDs to one flattened physical value slab."""

    if indices.ndim != 3:
        raise ValueError("compact routes must be [B,T,N]")
    if rows_per_bank <= 0:
        raise ValueError("compact execution needs at least one selected row")
    offsets = (
        torch.arange(indices.shape[0], device=indices.device, dtype=torch.int64)
        .view(-1, 1, 1)
        .mul(rows_per_bank)
    )
    return indices.to(torch.int64) + offsets


def remove_bank_offsets(indices: torch.Tensor, rows_per_bank: int) -> torch.Tensor:
    """Recover bank-local IDs for the portable oracle backend."""

    offsets = (
        torch.arange(indices.shape[0], device=indices.device, dtype=torch.int64)
        .view(-1, 1, 1)
        .mul(rows_per_bank)
    )
    return indices.to(torch.int64) - offsets


def _pad_parallel_time(
    tensors: tuple[torch.Tensor, ...],
    *,
    chunk_size: int,
) -> tuple[tuple[torch.Tensor, ...], int]:
    """Pad each bank independently so released chunks never cross banks."""

    time = tensors[0].shape[1]
    padded_time = math.ceil(time / chunk_size) * chunk_size
    if padded_time == time:
        return tensors, time
    padded: list[torch.Tensor] = []
    for tensor in tensors:
        shape = list(tensor.shape)
        shape[1] = padded_time - time
        fill = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
        padded.append(torch.cat((tensor, fill), dim=1))
    return tuple(padded), padded_time


class CompactKernelBackend:
    """Run native SDM recurrence on the selected physical namespace."""

    def __init__(
        self,
        *,
        memory_block_size: int,
        snapshot_quant: str = "none",
        key_weighted_decay: bool = False,
    ) -> None:
        if memory_block_size < 2:
            raise ValueError("memory block size must be at least two")
        self.memory_block_size = memory_block_size
        self.snapshot_quant = snapshot_quant
        self.key_weighted_decay = key_weighted_decay
        self.last_report: CompactKernelReport | None = None

    def _report(
        self,
        backend: str,
        memory: torch.Tensor,
        banks: int,
        rows_per_bank: int,
    ) -> None:
        self.last_report = CompactKernelReport(
            backend=backend,
            banks=banks,
            selected_rows_per_bank=rows_per_bank,
            selected_value_elements=memory.numel(),
        )

    def execute(
        self,
        memory: torch.Tensor,
        write_indices: torch.Tensor,
        write_weights: torch.Tensor,
        values: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        read_indices: torch.Tensor,
        read_weights: torch.Tensor,
        *,
        rows_per_bank: int,
        training: bool,
        grad_final_memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        banks, time = write_indices.shape[:2]
        expected_memory = (banks * rows_per_bank, values.shape[-1])
        if memory.shape != expected_memory:
            raise ValueError(f"compact physical memory must be {expected_memory}")
        if read_indices.shape[:2] != (banks, time):
            raise ValueError("compact read and write routes must align")

        if memory.device.type != "cuda":
            local_writes = remove_bank_offsets(write_indices, rows_per_bank)
            local_reads = remove_bank_offsets(read_indices, rows_per_bank)
            result = compact_gated_delta_recurrence(
                memory.view(banks, rows_per_bank, values.shape[-1]),
                local_writes,
                write_weights,
                values,
                beta,
                log_decay,
                local_reads,
                read_weights,
            )
            terminal = result.final_memory.reshape_as(memory)
            readings = result.readings
            if grad_final_memory is not None:
                readings = _AttachTerminalGradient.apply(
                    readings,
                    terminal,
                    grad_final_memory,
                )
            self._report("torch-oracle", terminal, banks, rows_per_bank)
            return readings, terminal

        if training:
            readings, terminal = self._training(
                memory,
                write_indices,
                write_weights,
                values,
                beta,
                log_decay,
                read_indices,
                read_weights,
                rows_per_bank=rows_per_bank,
                grad_final_memory=grad_final_memory,
            )
            self._report("released-wy-cuda", terminal, banks, rows_per_bank)
            return readings, terminal

        readings, terminal, backend = self._inference(
            memory,
            write_indices,
            write_weights,
            values,
            beta,
            log_decay,
            read_indices,
            read_weights,
            rows_per_bank=rows_per_bank,
        )
        self._report(backend, terminal, banks, rows_per_bank)
        return readings, terminal

    def _training(
        self,
        memory: torch.Tensor,
        write_indices: torch.Tensor,
        write_weights: torch.Tensor,
        values: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        read_indices: torch.Tensor,
        read_weights: torch.Tensor,
        *,
        rows_per_bank: int,
        grad_final_memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from lingua.sparse_delta_memory.memory_ops import GatedSparseMemoryWriteRead

        banks, time = write_indices.shape[:2]
        # Released WY matmul kernels require K >= 16.  Short sequences and CP
        # shards are padded independently with zero gates/weights, so choosing
        # 16 here is state-neutral and avoids cross-bank chunks.
        chunk_size = max(16, min(self.memory_block_size, time))
        inputs, padded_time = _pad_parallel_time(
            (
                write_indices,
                write_weights,
                values,
                beta,
                log_decay,
                read_indices,
                read_weights,
            ),
            chunk_size=chunk_size,
        )
        (
            padded_w_idx,
            padded_w_val,
            padded_values,
            padded_beta,
            padded_decay,
            padded_q_idx,
            padded_q_val,
        ) = inputs
        if padded_time != time:
            bank_base = (
                torch.arange(banks, device=memory.device, dtype=torch.int64)
                .view(banks, 1, 1)
                .mul(rows_per_bank)
            )
            padded_w_idx[:, time:] = bank_base
            padded_q_idx[:, time:] = bank_base
        flat_readings, _ = GatedSparseMemoryWriteRead.apply(
            memory,
            padded_w_idx,
            padded_w_val,
            padded_values,
            padded_beta,
            padded_decay,
            padded_q_idx,
            padded_q_val,
            chunk_size,
            True,
            rows_per_bank,
            banks,
            False,
            self.snapshot_quant,
            grad_final_memory,
        )
        readings = flat_readings.view(banks, padded_time, -1)[:, :time]
        return readings, memory

    @torch.no_grad()
    def _inference(
        self,
        memory: torch.Tensor,
        write_indices: torch.Tensor,
        write_weights: torch.Tensor,
        values: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        read_indices: torch.Tensor,
        read_weights: torch.Tensor,
        *,
        rows_per_bank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        from lingua.sparse_delta_memory.memory_ops import (
            fused_decode_step,
            gated_sparse_memory_write_read_inference_v2,
        )

        banks, time = write_indices.shape[:2]
        if time <= 64:
            outputs = []
            for position in range(time):
                outputs.append(
                    fused_decode_step(
                        memory,
                        write_indices[:, position],
                        write_weights[:, position],
                        values[:, position],
                        beta[:, position],
                        log_decay[:, position],
                        read_indices[:, position],
                        read_weights[:, position],
                        use_delta_rule=True,
                        normalize_memory=False,
                        key_weighted_decay=self.key_weighted_decay,
                    )
                )
            return torch.stack(outputs, dim=1), memory, "released-fused-decode-cuda"

        chunk_size = max(16, min(self.memory_block_size, time))
        inputs, padded_time = _pad_parallel_time(
            (
                write_indices,
                write_weights,
                values,
                beta,
                log_decay,
                read_indices,
                read_weights,
            ),
            chunk_size=chunk_size,
        )
        if padded_time != time:
            inputs = list(inputs)
            bank_base = (
                torch.arange(banks, device=memory.device, dtype=torch.int64)
                .view(banks, 1, 1)
                .mul(rows_per_bank)
            )
            inputs[0][:, time:] = bank_base
            inputs[5][:, time:] = bank_base
            inputs = tuple(inputs)
        flat_readings = gated_sparse_memory_write_read_inference_v2(
            memory,
            *inputs,
            chunk_size,
            rows_per_bank,
            banks,
        )
        readings = flat_readings.view(banks, padded_time, -1)[:, :time]
        return readings, memory, "released-prefill-cuda"


__all__ = [
    "CompactKernelBackend",
    "CompactKernelReport",
    "add_bank_offsets",
    "remove_bank_offsets",
]
