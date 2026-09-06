"""End-to-end compact training or prefill execution from selected tuples."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .addresses import SelectedAddressLayout, compact_selected_addresses
from .priors import SelectedRowPrior
from .recurrence import RecurrenceResult, compact_gated_delta_recurrence


@dataclass(frozen=True)
class ExecutionTrace:
    logical_capacity: int
    banks: int
    time: int
    factors: int
    reads: int
    writes: int
    value_width: int
    selected_rows: int
    materialized_rows: int
    tensor_elements: dict[str, int]

    def capacity_dependent_runtime_elements(self) -> int:
        """Return zero by construction; logical capacity is scalar metadata."""

        return self.tensor_elements.get("logical_capacity_tensor", 0)


@dataclass(frozen=True)
class ExecutionResult:
    readings: torch.Tensor
    final_selected_memory: torch.Tensor
    initial_selected_memory: torch.Tensor
    layout: SelectedAddressLayout
    template_indices: torch.Tensor
    trace: ExecutionTrace

    def written_keys(self, bank: int) -> torch.Tensor:
        count = int(self.layout.counts[bank])
        return self.layout.keys[bank, :count][self.layout.written[bank, :count]]

    def written_values(self, bank: int) -> torch.Tensor:
        count = int(self.layout.counts[bank])
        return self.final_selected_memory[bank, :count][
            self.layout.written[bank, :count]
        ]


def _default_templates(
    prior: SelectedRowPrior, banks: int, device: torch.device
) -> torch.Tensor:
    if banks % prior.templates:
        raise ValueError("banks require explicit prior template indices")
    return torch.arange(banks, device=device, dtype=torch.int64).remainder(
        prior.templates
    )


def execute_compact(
    prior: SelectedRowPrior,
    write_addresses: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_addresses: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    template_indices: torch.Tensor | None = None,
) -> ExecutionResult:
    """Compact selected rows, evaluate their prior, and execute native SDM."""

    if write_addresses.shape[:-1] != write_weights.shape:
        raise ValueError("write tuple and weight shapes do not align")
    if read_addresses.shape[:-1] != read_weights.shape:
        raise ValueError("read tuple and weight shapes do not align")
    if write_addresses.shape[-1] != prior.factors:
        raise ValueError("write tuple width does not match the prior")
    if prior.codebook_size <= 0:
        raise ValueError("prior codebook is invalid")
    banks, time, writes, _ = write_addresses.shape
    reads = read_addresses.shape[2]
    if template_indices is None:
        template_indices = _default_templates(prior, banks, write_addresses.device)
    layout = compact_selected_addresses(
        write_addresses,
        read_addresses,
        codebook_size=prior.codebook_size,
    )
    initial = prior(layout.keys, template_indices)
    recurrence: RecurrenceResult = compact_gated_delta_recurrence(
        initial,
        layout.write_remap,
        write_weights,
        values,
        beta,
        log_decay,
        layout.read_remap,
        read_weights,
    )
    address_elements = sum(layout.storage_elements().values())
    trace = ExecutionTrace(
        logical_capacity=layout.logical_capacity,
        banks=banks,
        time=time,
        factors=layout.factors,
        reads=reads,
        writes=writes,
        value_width=values.shape[-1],
        selected_rows=layout.selected_rows,
        materialized_rows=layout.materialized_rows,
        tensor_elements={
            "logical_capacity_tensor": 0,
            "address_metadata": address_elements,
            "selected_prior_rows": initial.numel(),
            "recurrent_table": recurrence.final_memory.numel(),
            "recurrence_workspace_maximum": recurrence.maximum_workspace_elements,
            "readings": recurrence.readings.numel(),
        },
    )
    return ExecutionResult(
        readings=recurrence.readings,
        final_selected_memory=recurrence.final_memory,
        initial_selected_memory=initial,
        layout=layout,
        template_indices=template_indices,
        trace=trace,
    )


__all__ = ["ExecutionResult", "ExecutionTrace", "execute_compact"]
