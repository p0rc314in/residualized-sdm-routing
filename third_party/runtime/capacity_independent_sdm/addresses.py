"""Compact selected product-key tuples without allocating logical-width maps."""

from __future__ import annotations

from dataclasses import dataclass

import torch


INTEGER_DTYPES = (torch.int32, torch.int64)


def logical_capacity(codebook_size: int, factors: int) -> int:
    """Return C**p as a Python integer, without creating a capacity-sized tensor."""

    if codebook_size <= 0 or factors <= 0:
        raise ValueError("codebook size and factor count must be positive")
    return codebook_size**factors


def _validate_addresses(
    name: str,
    addresses: torch.Tensor,
    *,
    codebook_size: int,
) -> None:
    if addresses.ndim != 4:
        raise ValueError(f"{name} addresses must be [B,T,N,p]")
    if addresses.dtype not in INTEGER_DTYPES:
        raise ValueError(f"{name} addresses must use int32 or int64")
    if addresses.shape[-1] <= 0:
        raise ValueError("addresses need at least one product-key factor")
    if addresses.device.type == "cpu" and addresses.numel():
        if int(addresses.min()) < 0 or int(addresses.max()) >= codebook_size:
            raise ValueError(f"{name} address lies outside the factor codebook")


def encode_product_addresses(
    addresses: torch.Tensor,
    *,
    codebook_size: int,
) -> torch.Tensor:
    """Encode tuples in row-major order for small dense-oracle operations.

    Production lookup retains explicit tuples. This helper deliberately refuses
    capacities that cannot be represented by a signed int64 scalar.
    """

    if addresses.shape[-1] <= 0 or addresses.dtype not in INTEGER_DTYPES:
        raise ValueError("addresses must end in a nonempty integer tuple axis")
    capacity = logical_capacity(codebook_size, addresses.shape[-1])
    if capacity - 1 > torch.iinfo(torch.int64).max:
        raise OverflowError("logical address does not fit in signed int64")
    encoded = torch.zeros(
        addresses.shape[:-1], device=addresses.device, dtype=torch.int64
    )
    for factor in range(addresses.shape[-1]):
        encoded = encoded * codebook_size + addresses[..., factor].to(torch.int64)
    return encoded


def decode_product_addresses(
    indices: torch.Tensor,
    *,
    codebook_size: int,
    factors: int,
) -> torch.Tensor:
    """Decode row-major int64 addresses into explicit product-key tuples."""

    capacity = logical_capacity(codebook_size, factors)
    if indices.dtype not in INTEGER_DTYPES:
        raise ValueError("logical indices must use int32 or int64")
    if capacity - 1 > torch.iinfo(torch.int64).max:
        raise OverflowError("logical address does not fit in signed int64")
    if indices.device.type == "cpu" and indices.numel():
        if int(indices.min()) < 0 or int(indices.max()) >= capacity:
            raise ValueError("logical index lies outside the address space")
    remaining = indices.to(torch.int64)
    decoded: list[torch.Tensor] = []
    for _ in range(factors):
        decoded.append(torch.remainder(remaining, codebook_size))
        remaining = torch.div(remaining, codebook_size, rounding_mode="floor")
    return torch.stack(tuple(reversed(decoded)), dim=-1)


@dataclass(frozen=True)
class SelectedAddressLayout:
    """One padded compact namespace per independent recurrent bank.

    Padding follows the largest selected union in the current batch, never the
    logical address space. Remaps are local to each bank in [0, counts[b]).
    """

    keys: torch.Tensor
    counts: torch.Tensor
    write_remap: torch.Tensor
    read_remap: torch.Tensor
    written: torch.Tensor
    first_write_position: torch.Tensor
    codebook_size: int
    logical_capacity: int

    def __post_init__(self) -> None:
        if self.keys.ndim != 3 or self.keys.dtype not in INTEGER_DTYPES:
            raise ValueError("compact keys must be integer [B,K,p]")
        banks, rows, _ = self.keys.shape
        if self.counts.shape != (banks,) or self.counts.dtype not in INTEGER_DTYPES:
            raise ValueError("compact counts must be integer [B]")
        if self.write_remap.ndim != 3 or self.write_remap.shape[0] != banks:
            raise ValueError("write remap must be [B,T,W]")
        if (
            self.read_remap.ndim != 3
            or self.read_remap.shape[:2] != self.write_remap.shape[:2]
        ):
            raise ValueError("read remap must align with write time and banks")
        if self.written.shape != (banks, rows) or self.written.dtype != torch.bool:
            raise ValueError("written mask must be bool [B,K]")
        if self.first_write_position.shape != (banks, rows):
            raise ValueError("first-write positions must be [B,K]")
        if any(
            tensor.device != self.keys.device
            for tensor in (
                self.counts,
                self.write_remap,
                self.read_remap,
                self.written,
                self.first_write_position,
            )
        ):
            raise ValueError("all compact address tensors must share a device")
        if self.logical_capacity != logical_capacity(self.codebook_size, self.factors):
            raise ValueError("logical capacity does not match tuple geometry")

    @property
    def banks(self) -> int:
        return self.keys.shape[0]

    @property
    def max_rows(self) -> int:
        return self.keys.shape[1]

    @property
    def factors(self) -> int:
        return self.keys.shape[2]

    @property
    def selected_rows(self) -> int:
        return int(self.counts.sum().item())

    @property
    def materialized_rows(self) -> int:
        return int(self.written.sum().item())

    def valid_mask(self) -> torch.Tensor:
        positions = torch.arange(self.max_rows, device=self.keys.device)
        return positions.unsqueeze(0) < self.counts.to(torch.int64).unsqueeze(1)

    def storage_elements(self) -> dict[str, int]:
        """Report actual compact tensor elements by semantic surface."""

        return {
            "tuple_keys": self.keys.numel(),
            "row_counts": self.counts.numel(),
            "write_remap": self.write_remap.numel(),
            "read_remap": self.read_remap.numel(),
            "written_mask": self.written.numel(),
            "first_write_position": self.first_write_position.numel(),
        }


def compact_selected_addresses(
    write_addresses: torch.Tensor,
    read_addresses: torch.Tensor,
    *,
    codebook_size: int,
) -> SelectedAddressLayout:
    """Deduplicate the selected read/write union independently for each bank."""

    _validate_addresses("write", write_addresses, codebook_size=codebook_size)
    _validate_addresses("read", read_addresses, codebook_size=codebook_size)
    if write_addresses.shape[:2] != read_addresses.shape[:2]:
        raise ValueError("read and write addresses must share banks and time")
    if write_addresses.shape[-1] != read_addresses.shape[-1]:
        raise ValueError("read and write tuple widths must match")
    if write_addresses.device != read_addresses.device:
        raise ValueError("read and write addresses must share a device")

    banks, time, writes, factors = write_addresses.shape
    reads = read_addresses.shape[2]
    if banks <= 0 or time <= 0 or writes <= 0 or reads <= 0:
        raise ValueError(
            "compact execution requires nonempty banks, time, reads, and writes"
        )
    device = write_addresses.device

    # Merge every bank in one device sort.  Prefixing the explicit tuple with
    # its bank ID preserves independent namespaces without encoding the tuple
    # into a logical-capacity scalar.  This is the production candidate-merge
    # surface: its input is B*T*(W+R)*(p+1), never B*S.
    flat_write = write_addresses.reshape(banks, time * writes, factors).to(torch.int64)
    flat_read = read_addresses.reshape(banks, time * reads, factors).to(torch.int64)
    candidates = torch.cat((flat_write, flat_read), dim=1)
    bank_ids = (
        torch.arange(banks, device=device, dtype=torch.int64)
        .view(banks, 1, 1)
        .expand(banks, candidates.shape[1], 1)
    )
    composite = torch.cat((bank_ids, candidates), dim=-1).reshape(-1, factors + 1)
    unique, inverse = torch.unique(
        composite,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    unique_banks = unique[:, 0]
    counts = torch.bincount(unique_banks, minlength=banks)
    bank_starts = counts.cumsum(0) - counts
    local_unique = (
        torch.arange(unique.shape[0], device=device, dtype=torch.int64)
        - bank_starts[unique_banks]
    )
    local_inverse = local_unique[inverse].reshape(banks, candidates.shape[1])
    write_remap = local_inverse[:, : time * writes].reshape(banks, time, writes)
    read_remap = local_inverse[:, time * writes :].reshape(banks, time, reads)

    max_rows = int(counts.max().item())
    keys = torch.full(
        (banks, max_rows, factors),
        -1,
        device=device,
        dtype=torch.int64,
    )
    keys[unique_banks, local_unique] = unique[:, 1:]
    written = torch.zeros((banks, max_rows), device=device, dtype=torch.bool)
    written.scatter_(1, write_remap.reshape(banks, -1), True)
    write_times = (
        torch.arange(time, device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(time, writes)
        .reshape(-1)
    )
    first_write = torch.full(
        (banks, max_rows),
        time,
        device=device,
        dtype=torch.int64,
    )
    first_write.scatter_reduce_(
        1,
        write_remap.reshape(banks, -1),
        write_times.view(1, -1).expand(banks, -1),
        reduce="amin",
        include_self=True,
    )
    return SelectedAddressLayout(
        keys=keys,
        counts=counts,
        write_remap=write_remap,
        read_remap=read_remap,
        written=written,
        first_write_position=first_write,
        codebook_size=codebook_size,
        logical_capacity=logical_capacity(codebook_size, factors),
    )


__all__ = [
    "SelectedAddressLayout",
    "compact_selected_addresses",
    "decode_product_addresses",
    "encode_product_addresses",
    "logical_capacity",
]
