"""Capacity-independent hard occupancy and differentiable noisy-OR accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .addresses import encode_product_addresses, logical_capacity


@dataclass(frozen=True)
class SparseOccupancy:
    hard_unique_by_position: torch.Tensor
    first_touch_count_by_position: torch.Tensor
    repeated_write_count_by_position: torch.Tensor
    private_read_count_by_position: torch.Tensor
    private_read_weight_by_position: torch.Tensor
    write_route_entropy_by_position: torch.Tensor
    sparse_write_addresses: tuple[torch.Tensor, ...]
    sparse_write_counts: tuple[torch.Tensor, ...]
    hard_final_fraction: torch.Tensor
    read_final_fraction: torch.Tensor
    soft_final_fraction: torch.Tensor
    straight_through_fraction: torch.Tensor
    read_address_entropy_normalized: torch.Tensor
    write_address_entropy_normalized: torch.Tensor
    logical_capacity: int

    def storage_elements(self) -> int:
        fixed = (
            self.hard_unique_by_position,
            self.first_touch_count_by_position,
            self.repeated_write_count_by_position,
            self.private_read_count_by_position,
            self.private_read_weight_by_position,
            self.write_route_entropy_by_position,
            self.hard_final_fraction,
            self.read_final_fraction,
            self.soft_final_fraction,
            self.straight_through_fraction,
            self.read_address_entropy_normalized,
            self.write_address_entropy_normalized,
        )
        return sum(tensor.numel() for tensor in fixed) + sum(
            tensor.numel()
            for pair in zip(self.sparse_write_addresses, self.sparse_write_counts)
            for tensor in pair
        )


def _validate_routes(
    write_addresses: torch.Tensor,
    write_weights: torch.Tensor,
    read_addresses: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    codebook_size: int,
) -> tuple[int, int, int, int, int]:
    if write_addresses.ndim != 4 or read_addresses.ndim != 4:
        raise ValueError("route addresses must be [B,T,N,p]")
    if write_addresses.shape[:-1] != write_weights.shape:
        raise ValueError("write route weights do not align")
    if read_addresses.shape[:-1] != read_weights.shape:
        raise ValueError("read route weights do not align")
    if write_addresses.shape[:2] != read_addresses.shape[:2]:
        raise ValueError("read and write routes must share banks and time")
    if write_addresses.shape[-1] != read_addresses.shape[-1]:
        raise ValueError("read and write tuple widths differ")
    if write_addresses.device != read_addresses.device:
        raise ValueError("route address devices differ")
    if (
        write_weights.device != write_addresses.device
        or read_weights.device != write_addresses.device
    ):
        raise ValueError("route weights and addresses must share a device")
    if write_addresses.dtype not in (torch.int32, torch.int64):
        raise ValueError("write addresses must be integer")
    if read_addresses.dtype not in (torch.int32, torch.int64):
        raise ValueError("read addresses must be integer")
    banks, time, writes, factors = write_addresses.shape
    reads = read_addresses.shape[2]
    if min(banks, time, writes, reads, factors, codebook_size) <= 0:
        raise ValueError("occupancy route geometry must be nonempty")
    if write_addresses.device.type == "cpu":
        for name, addresses in (("write", write_addresses), ("read", read_addresses)):
            if int(addresses.min()) < 0 or int(addresses.max()) >= codebook_size:
                raise ValueError(f"{name} address lies outside the factor codebook")
    return banks, time, writes, reads, factors


def sparse_occupancy(
    write_addresses: torch.Tensor,
    write_weights: torch.Tensor,
    read_addresses: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    codebook_size: int,
) -> SparseOccupancy:
    """Compute first-touch accounting over selected tuple event groups only.

    This is a representation primitive. Objectives such as Elastic SDM may
    consume its straight-through estimate, but no loss policy lives here.
    """

    banks, time, writes, reads, factors = _validate_routes(
        write_addresses,
        write_weights,
        read_addresses,
        read_weights,
        codebook_size=codebook_size,
    )
    capacity = logical_capacity(codebook_size, factors)
    device = write_addresses.device
    write_times = (
        torch.arange(time, device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(time, writes)
        .reshape(-1)
    )
    read_times = (
        torch.arange(time, device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(time, reads)
        .reshape(-1)
    )

    hard_rows: list[torch.Tensor] = []
    first_rows: list[torch.Tensor] = []
    repeated_rows: list[torch.Tensor] = []
    private_count_rows: list[torch.Tensor] = []
    private_weight_rows: list[torch.Tensor] = []
    entropy_rows: list[torch.Tensor] = []
    sparse_addresses: list[torch.Tensor] = []
    sparse_counts: list[torch.Tensor] = []
    hard_fractions: list[torch.Tensor] = []
    read_fractions: list[torch.Tensor] = []
    soft_fractions: list[torch.Tensor] = []
    read_address_entropies: list[torch.Tensor] = []
    write_address_entropies: list[torch.Tensor] = []

    for bank in range(banks):
        flat_write = (
            write_addresses[bank].reshape(time * writes, factors).to(torch.int64)
        )
        flat_read = read_addresses[bank].reshape(time * reads, factors).to(torch.int64)
        unique, inverse = torch.unique(
            torch.cat((flat_write, flat_read), dim=0),
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        write_inverse = inverse[: time * writes]
        read_inverse = inverse[time * writes :]
        selected_rows = unique.shape[0]

        first_write = torch.full(
            (selected_rows,),
            time,
            device=device,
            dtype=torch.int64,
        ).scatter_reduce(
            0,
            write_inverse,
            write_times,
            reduce="amin",
            include_self=True,
        )
        first_count = torch.bincount(
            first_write[first_write < time],
            minlength=time,
        )
        hard_unique = first_count.cumsum(0)
        repeated = torch.full_like(first_count, writes) - first_count

        private_read = first_write[read_inverse] <= read_times
        private_read = private_read.reshape(time, reads)
        private_count = private_read.sum(dim=-1)
        private_weight = (private_read.to(read_weights.dtype) * read_weights[bank]).sum(
            dim=-1
        )

        composite = write_times * selected_rows + write_inverse
        event_groups, event_inverse = torch.unique(
            composite,
            sorted=True,
            return_inverse=True,
        )
        grouped_mass = torch.zeros(
            event_groups.shape[0],
            device=device,
            dtype=torch.float32,
        ).scatter_add(
            0,
            event_inverse,
            write_weights[bank].reshape(-1).float(),
        )
        grouped_mass = grouped_mass.clamp(min=0.0, max=1.0 - 1e-6)
        grouped_address = torch.remainder(event_groups, selected_rows)
        log_unoccupied = torch.zeros(
            selected_rows,
            device=device,
            dtype=torch.float32,
        ).scatter_add(
            0,
            grouped_address,
            torch.log1p(-grouped_mass),
        )
        soft_occupied = -torch.expm1(log_unoccupied)
        soft_fraction = soft_occupied.sum() / float(capacity)
        hard_fraction = hard_unique[-1].float() / float(capacity)

        write_count = torch.bincount(write_inverse, minlength=selected_rows)
        read_count = torch.bincount(read_inverse, minlength=selected_rows)
        was_written = write_count > 0
        read_fraction = read_count.ne(0).sum().float() / float(capacity)
        entropy_denominator = math.log(capacity) if capacity > 1 else 1.0

        def normalized_address_entropy(
            inverse_rows: torch.Tensor, weights: torch.Tensor
        ) -> torch.Tensor:
            mass = torch.zeros(
                selected_rows,
                device=device,
                dtype=torch.float32,
            ).scatter_add(0, inverse_rows, weights.reshape(-1).float().clamp_min(0.0))
            probabilities = mass / mass.sum().clamp_min(1e-12)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            return entropy / entropy_denominator

        sparse_addresses.append(unique[was_written])
        sparse_counts.append(write_count[was_written])
        hard_rows.append(hard_unique)
        first_rows.append(first_count)
        repeated_rows.append(repeated)
        private_count_rows.append(private_count)
        private_weight_rows.append(private_weight)
        entropy_rows.append(
            -(
                write_weights[bank].float().clamp_min(1e-12)
                * write_weights[bank].float().clamp_min(1e-12).log()
            ).sum(dim=-1)
        )
        hard_fractions.append(hard_fraction)
        read_fractions.append(read_fraction)
        soft_fractions.append(soft_fraction)
        read_address_entropies.append(
            normalized_address_entropy(read_inverse, read_weights[bank])
        )
        write_address_entropies.append(
            normalized_address_entropy(write_inverse, write_weights[bank])
        )

    hard_final = torch.stack(hard_fractions)
    soft_final = torch.stack(soft_fractions)
    straight_through = hard_final.detach() - soft_final.detach() + soft_final
    return SparseOccupancy(
        hard_unique_by_position=torch.stack(hard_rows),
        first_touch_count_by_position=torch.stack(first_rows),
        repeated_write_count_by_position=torch.stack(repeated_rows),
        private_read_count_by_position=torch.stack(private_count_rows),
        private_read_weight_by_position=torch.stack(private_weight_rows),
        write_route_entropy_by_position=torch.stack(entropy_rows),
        sparse_write_addresses=tuple(sparse_addresses),
        sparse_write_counts=tuple(sparse_counts),
        hard_final_fraction=hard_final,
        read_final_fraction=torch.stack(read_fractions),
        soft_final_fraction=soft_final,
        straight_through_fraction=straight_through,
        read_address_entropy_normalized=torch.stack(read_address_entropies),
        write_address_entropy_normalized=torch.stack(write_address_entropies),
        logical_capacity=capacity,
    )


def dense_occupancy_oracle(
    write_addresses: torch.Tensor,
    write_weights: torch.Tensor,
    read_addresses: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    codebook_size: int,
    maximum_rows: int = 65_536,
) -> dict[str, torch.Tensor | int]:
    """The historical S-wide accounting, restricted to bounded oracle tests."""

    banks, time, writes, _, factors = _validate_routes(
        write_addresses,
        write_weights,
        read_addresses,
        read_weights,
        codebook_size=codebook_size,
    )
    capacity = logical_capacity(codebook_size, factors)
    if capacity > maximum_rows:
        raise ValueError("refusing to allocate a large dense occupancy oracle")
    write_indices = encode_product_addresses(
        write_addresses,
        codebook_size=codebook_size,
    )
    read_indices = encode_product_addresses(
        read_addresses,
        codebook_size=codebook_size,
    )
    hard_events = torch.zeros(
        banks,
        time,
        capacity,
        device=write_addresses.device,
        dtype=torch.int64,
    ).scatter_add(
        -1,
        write_indices,
        torch.ones_like(write_indices, dtype=torch.int64),
    )
    hard_touches = hard_events.ne(0)
    active = hard_touches.to(torch.int64).cumsum(dim=1).ne(0)
    hard_unique = active.sum(dim=-1)
    prior = torch.cat(
        (torch.zeros_like(hard_unique[:, :1]), hard_unique[:, :-1]), dim=1
    )
    first = hard_unique - prior
    repeated = hard_events.sum(dim=-1) - first
    soft_routes = torch.zeros(
        banks,
        time,
        capacity,
        device=write_addresses.device,
        dtype=torch.float32,
    ).scatter_add(-1, write_indices, write_weights.float())
    soft_routes = soft_routes.clamp(min=0.0, max=1.0 - 1e-6)
    soft_occupied = -torch.expm1(torch.log1p(-soft_routes).sum(dim=1))
    soft_fraction = soft_occupied.mean(dim=-1)
    hard_fraction = hard_unique[:, -1].float() / float(capacity)
    private = torch.gather(active, -1, read_indices)
    return {
        "hard_unique_by_position": hard_unique,
        "first_touch_count_by_position": first,
        "repeated_write_count_by_position": repeated,
        "private_read_count_by_position": private.sum(dim=-1),
        "private_read_weight_by_position": (
            private.to(read_weights.dtype) * read_weights
        ).sum(dim=-1),
        "hard_final_fraction": hard_fraction,
        "soft_final_fraction": soft_fraction,
        "straight_through_fraction": (
            hard_fraction.detach() - soft_fraction.detach() + soft_fraction
        ),
        "logical_capacity": capacity,
    }


__all__ = ["SparseOccupancy", "dense_occupancy_oracle", "sparse_occupancy"]
