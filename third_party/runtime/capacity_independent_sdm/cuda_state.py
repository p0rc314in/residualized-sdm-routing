"""Triton sparse-tuple map and fused copy-on-write decode.

The map stores one physical-row ID per occupied hash bucket and verifies the
bank plus every tuple factor on lookup.  Hash collisions therefore affect only
probe work, never address identity.  Its storage is proportional to the value
slab capacity and contains no logical-capacity dimension.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - Triton is exercised by remote CUDA acceptance.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU development hosts.
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None


if triton is not None:

    @triton.jit
    def _tuple_hash(owner, address_ptr, FACTORS: tl.constexpr):
        value = owner.to(tl.uint64) + 1442695040888963407
        for factor in tl.static_range(0, FACTORS):
            key = tl.load(address_ptr + factor).to(tl.uint64) + 1
            value = value * 6364136223846793005 + key * 1442695040888963407
        value = value ^ (value >> 33)
        value = value * 6364136223846793005
        return value ^ (value >> 29)

    @triton.jit
    def _row_matches(keys, owners, row, owner, address_ptr, FACTORS: tl.constexpr):
        matches = tl.load(owners + row) == owner
        for factor in tl.static_range(0, FACTORS):
            matches = matches & (
                tl.load(keys + row * FACTORS + factor) == tl.load(address_ptr + factor)
            )
        return matches

    @triton.jit
    def _hash_build_kernel(
        keys,
        owners,
        hash_rows,
        hash_size,
        capacity,
        overflow_flag,
        FACTORS: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= capacity:
            return
        owner = tl.load(owners + row)
        if owner < 0:
            return
        address = keys + row * FACTORS
        slot = (_tuple_hash(owner, address, FACTORS) & (hash_size - 1)).to(tl.int64)
        probe = 0
        inserted = False
        while (probe < hash_size) & (~inserted):
            previous = tl.atomic_cas(hash_rows + slot, -1, row)
            inserted = (previous == -1) | (previous == row)
            slot = (slot + 1) & (hash_size - 1)
            probe += 1
        tl.atomic_max(overflow_flag, (~inserted).to(tl.int32))

    @triton.jit
    def _hash_lookup_kernel(
        keys,
        owners,
        hash_rows,
        addresses,
        resolved,
        hash_size,
        routes,
        FACTORS: tl.constexpr,
        ROUTES_PER_BANK: tl.constexpr,
    ):
        route = tl.program_id(0)
        if route >= routes:
            return
        owner = route // ROUTES_PER_BANK
        address = addresses + route * FACTORS
        slot = (_tuple_hash(owner, address, FACTORS) & (hash_size - 1)).to(tl.int64)
        probe = 0
        result = -1
        searching = True
        while (probe < hash_size) & searching:
            row = tl.load(hash_rows + slot)
            empty = row < 0
            matched = False
            if row >= 0:
                matched = _row_matches(keys, owners, row, owner, address, FACTORS)
            result = tl.where(matched, row, result)
            searching = (~empty) & (~matched)
            slot = (slot + 1) & (hash_size - 1)
            probe += 1
        tl.store(resolved + route, result)

    @triton.jit
    def _hash_resolve_write_kernel(
        keys,
        owners,
        hash_rows,
        free_rows,
        free_count,
        overflow_flag,
        addresses,
        resolved,
        first_touches,
        hash_size,
        routes,
        FACTORS: tl.constexpr,
        WRITES: tl.constexpr,
    ):
        route = tl.program_id(0)
        if route >= routes:
            return
        owner = route // WRITES
        address = addresses + route * FACTORS
        slot = (_tuple_hash(owner, address, FACTORS) & (hash_size - 1)).to(tl.int64)
        probe = 0
        result = -1
        first = False
        searching = True
        while (probe < hash_size) & searching:
            previous = tl.load(hash_rows + slot)
            if previous >= 0:
                matched = _row_matches(keys, owners, previous, owner, address, FACTORS)
                result = tl.where(matched, previous, result)
                searching = ~matched
            else:
                prior_free = tl.atomic_add(free_count, -1)
                free_index = prior_free - 1
                if free_index < 0:
                    tl.atomic_add(free_count, 1)
                    tl.atomic_max(overflow_flag, 1)
                    searching = False
                else:
                    candidate = tl.load(free_rows + free_index)
                    tl.store(owners + candidate, owner)
                    for factor in tl.static_range(0, FACTORS):
                        tl.store(
                            keys + candidate * FACTORS + factor,
                            tl.load(address + factor),
                        )
                    claimed = tl.atomic_cas(hash_rows + slot, -1, candidate)
                    if claimed == -1:
                        result = candidate
                        first = True
                        searching = False
                    else:
                        # Another route won this bucket. Return the speculative
                        # row to the shared stack, then inspect/probe normally.
                        tl.store(owners + candidate, -1)
                        for factor in tl.static_range(0, FACTORS):
                            tl.store(keys + candidate * FACTORS + factor, -1)
                        push = tl.atomic_add(free_count, 1)
                        tl.store(free_rows + push, candidate)
                        if claimed >= 0:
                            matched = _row_matches(
                                keys, owners, claimed, owner, address, FACTORS
                            )
                            result = tl.where(matched, claimed, result)
                            searching = ~matched
            slot = (slot + 1) & (hash_size - 1)
            probe += 1
        tl.atomic_max(overflow_flag, (result < 0).to(tl.int32))
        tl.store(resolved + route, result)
        tl.store(first_touches + route, first)

    @triton.jit
    def _packed_tuple_update_read_kernel(
        values_slab,
        write_rows,
        first_touches,
        prior_write,
        write_weights,
        values,
        beta,
        log_decay,
        read_rows,
        prior_read,
        read_weights,
        output,
        width,
        WRITES: tl.constexpr,
        READS: tl.constexpr,
        BETA_WIDTH: tl.constexpr,
        DECAY_WIDTH: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bank = tl.program_id(0)
        width_block = tl.program_id(1)
        offsets_w = tl.arange(0, BLOCK_W)
        offsets_r = tl.arange(0, BLOCK_R)
        offsets_d = width_block * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_w = offsets_w < WRITES
        valid_r = offsets_r < READS
        valid_d = offsets_d < width

        write_offset = bank * WRITES + offsets_w
        physical = tl.load(write_rows + write_offset, mask=valid_w, other=0)
        first = tl.load(first_touches + write_offset, mask=valid_w, other=0).to(tl.int1)
        slab_offset = physical[:, None] * width + offsets_d[None, :]
        private = tl.load(
            values_slab + slab_offset,
            mask=valid_w[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        selected_prior = tl.load(
            prior_write + (write_offset[:, None] * width + offsets_d[None, :]),
            mask=valid_w[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        current = tl.where(first[:, None], selected_prior, private)
        if DECAY_WIDTH == 1:
            decay = tl.exp(tl.load(log_decay + bank).to(tl.float32))
        else:
            decay = tl.exp(
                tl.load(log_decay + write_offset, mask=valid_w, other=0.0).to(
                    tl.float32
                )
            )
        decayed = current * decay
        weights = tl.load(write_weights + write_offset, mask=valid_w, other=0.0).to(
            tl.float32
        )
        retrieved = tl.sum(weights[:, None] * decayed, axis=0)
        value = tl.load(values + bank * width + offsets_d, mask=valid_d, other=0.0).to(
            tl.float32
        )
        if BETA_WIDTH == 1:
            input_gate = tl.load(beta + bank).to(tl.float32)
        else:
            input_gate = tl.load(
                beta + bank * width + offsets_d, mask=valid_d, other=0.0
            ).to(tl.float32)
        update = input_gate * (value - retrieved)
        updated = decayed + weights[:, None] * update[None, :]
        tl.store(
            values_slab + slab_offset,
            updated,
            mask=valid_w[:, None] & valid_d[None, :],
        )
        tl.debug_barrier()

        read_offset = bank * READS + offsets_r
        read_physical = tl.load(read_rows + read_offset, mask=valid_r, other=-1)
        private_mask = read_physical >= 0
        read_slab_offset = (
            tl.maximum(read_physical, 0)[:, None] * width + offsets_d[None, :]
        )
        private_read = tl.load(
            values_slab + read_slab_offset,
            mask=valid_r[:, None] & private_mask[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        untouched_read = tl.load(
            prior_read + read_offset[:, None] * width + offsets_d[None, :],
            mask=valid_r[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        visible = tl.where(private_mask[:, None], private_read, untouched_read)
        q_weight = tl.load(read_weights + read_offset, mask=valid_r, other=0.0).to(
            tl.float32
        )
        reading = tl.sum(q_weight[:, None] * visible, axis=0)
        tl.store(output + bank * width + offsets_d, reading, mask=valid_d)


def _require_cuda_hash(state) -> None:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if state.values.device.type != "cuda":
        raise ValueError("sparse CUDA tuple map requires CUDA state")
    tensors = (
        state.keys,
        state.values,
        state.owners,
        state.hash_rows,
        state.free_rows,
        state.free_count,
        state.overflow_flag,
    )
    if any(tensor is None or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("sparse CUDA state tensors must be contiguous")


@torch.no_grad()
def rebuild_hash_table(state) -> None:
    """Rebuild hash metadata after lifecycle operations or slab growth."""

    _require_cuda_hash(state)
    state.hash_rows.fill_(-1)
    state.overflow_flag.zero_()
    if state.capacity_rows:
        _hash_build_kernel[(state.capacity_rows,)](
            state.keys,
            state.owners,
            state.hash_rows,
            state.hash_rows.numel(),
            state.capacity_rows,
            state.overflow_flag,
            FACTORS=state.factors,
            num_warps=1,
        )
        torch._assert_async(
            state.overflow_flag == 0,
            "sparse tuple hash rebuild exhausted its table",
        )


@torch.no_grad()
def lookup_hash_rows(
    state, addresses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve [B,N,p] tuples without broadcasting against all stored rows."""

    _require_cuda_hash(state)
    banks, routes, factors = addresses.shape
    if banks != state.banks or factors != state.factors:
        raise ValueError("sparse tuple lookup geometry changed")
    result = torch.empty(
        banks,
        routes,
        device=addresses.device,
        dtype=torch.int32,
    )
    _hash_lookup_kernel[(banks * routes,)](
        state.keys,
        state.owners,
        state.hash_rows,
        addresses.contiguous(),
        result,
        state.hash_rows.numel(),
        banks * routes,
        FACTORS=factors,
        ROUTES_PER_BANK=routes,
        num_warps=1,
    )
    return result.to(torch.int64), result >= 0


@torch.no_grad()
def fused_packed_decode_step(
    state,
    prior,
    write_addresses: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_addresses: torch.Tensor,
    read_weights: torch.Tensor,
):
    """Resolve first touches and execute one exact native-SDM decode token."""

    _require_cuda_hash(state)
    banks, writes, factors = write_addresses.shape
    reads = read_addresses.shape[1]
    width = values.shape[1]
    state.prepare_step_capacity(banks * writes)
    prior_write = prior(write_addresses, state.template_indices).to(state.values.dtype)
    prior_read = prior(read_addresses, state.template_indices).to(state.values.dtype)

    write_rows = torch.empty(banks, writes, device=state.device, dtype=torch.int32)
    first = torch.empty(banks, writes, device=state.device, dtype=torch.bool)
    state.overflow_flag.zero_()
    _hash_resolve_write_kernel[(banks * writes,)](
        state.keys,
        state.owners,
        state.hash_rows,
        state.free_rows,
        state.free_count,
        state.overflow_flag,
        write_addresses.contiguous(),
        write_rows,
        first,
        state.hash_rows.numel(),
        banks * writes,
        FACTORS=factors,
        WRITES=writes,
        num_warps=1,
    )
    torch._assert_async(
        state.overflow_flag == 0,
        "sparse tuple value slab or hash table is exhausted",
    )
    read_rows, _ = lookup_hash_rows(state, read_addresses)
    read_rows = read_rows.to(torch.int32)
    output = torch.empty_like(values)
    block_w = triton.next_power_of_2(writes)
    block_r = triton.next_power_of_2(reads)
    block_d = 32
    _packed_tuple_update_read_kernel[(banks, triton.cdiv(width, block_d))](
        state.values,
        write_rows,
        first,
        prior_write.contiguous(),
        write_weights.contiguous(),
        values.contiguous(),
        beta.contiguous(),
        log_decay.contiguous(),
        read_rows.contiguous(),
        prior_read.contiguous(),
        read_weights.contiguous(),
        output,
        width,
        WRITES=writes,
        READS=reads,
        BETA_WIDTH=beta.shape[-1],
        DECAY_WIDTH=log_decay.shape[-1],
        BLOCK_W=block_w,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    first_by_bank = first.sum(dim=-1)
    return output, first_by_bank


__all__ = [
    "fused_packed_decode_step",
    "lookup_hash_rows",
    "rebuild_hash_table",
    "triton_available",
]
