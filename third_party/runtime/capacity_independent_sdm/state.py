"""Packed copy-on-write state keyed by materialized product-key tuples."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .addresses import SelectedAddressLayout
from .execution import ExecutionResult
from .priors import SelectedRowPrior, materialize_prior


DEFAULT_GROWTH_QUANTUM_ROWS = 256


@dataclass(frozen=True)
class DecodeStep:
    readings: torch.Tensor
    first_touches_by_bank: torch.Tensor
    repeated_writes_by_bank: torch.Tensor
    materialized_rows_by_bank: torch.Tensor


@dataclass
class PackedMemoryState:
    """One shared physical slab with one tuple key per materialized value row."""

    keys: torch.Tensor
    values: torch.Tensor
    owners: torch.Tensor
    template_indices: torch.Tensor
    active_banks: torch.Tensor
    codebook_size: int
    factors: int
    value_width: int
    prior_templates: int
    growth_quantum_rows: int = DEFAULT_GROWTH_QUANTUM_ROWS
    growth_events: int = 0
    growth_rows_added: int = 0
    growth_rows_copied: int = 0
    hash_rows: torch.Tensor | None = None
    free_rows: torch.Tensor | None = None
    free_count: torch.Tensor | None = None
    overflow_flag: torch.Tensor | None = None
    bank_row_counts: torch.Tensor | None = None
    proven_additional_rows: int = 0

    def __post_init__(self) -> None:
        capacity = self.values.shape[0]
        if (
            self.codebook_size <= 0
            or self.factors <= 0
            or self.value_width <= 0
            or self.prior_templates <= 0
        ):
            raise ValueError("packed tuple and value geometry must be positive")
        if self.keys.shape != (capacity, self.factors) or self.keys.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("packed keys must be integer [C,p]")
        if self.values.ndim != 2 or self.values.shape[1] != self.value_width:
            raise ValueError("packed values must be [C,V]")
        if self.owners.shape != (capacity,) or self.owners.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("packed owners must be integer [C]")
        if self.template_indices.ndim != 1 or self.template_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("template indices must be integer [B]")
        if self.active_banks.shape != self.template_indices.shape:
            raise ValueError("active-bank mask must align with templates")
        if self.active_banks.dtype != torch.bool:
            raise ValueError("active-bank mask must be boolean")
        tensors = (self.values, self.owners, self.template_indices, self.active_banks)
        if any(tensor.device != self.keys.device for tensor in tensors):
            raise ValueError("packed state tensors must share a device")
        if self.growth_quantum_rows < 0:
            raise ValueError("growth quantum cannot be negative")
        if self.template_indices.numel() and (
            int(self.template_indices.min()) < 0
            or int(self.template_indices.max()) >= self.prior_templates
        ):
            raise ValueError("packed template index lies outside the prior")
        self.validate_invariants()
        if any(
            tensor is None
            for tensor in (
                self.hash_rows,
                self.free_rows,
                self.free_count,
                self.overflow_flag,
                self.bank_row_counts,
            )
        ):
            self._rebuild_allocator_metadata()

    @staticmethod
    def _hash_capacity(capacity_rows: int) -> int:
        target = max(8, capacity_rows * 2)
        return 1 << (target - 1).bit_length()

    @staticmethod
    def _python_tuple_hash(owner: int, key: torch.Tensor) -> int:
        mask = (1 << 64) - 1
        value = (owner + 1442695040888963407) & mask
        for component in key.tolist():
            value = (
                value * 6364136223846793005 + (int(component) + 1) * 1442695040888963407
            ) & mask
        value ^= value >> 33
        value = (value * 6364136223846793005) & mask
        return (value ^ (value >> 29)) & mask

    @torch.no_grad()
    def _rebuild_allocator_metadata(self) -> None:
        """Rebuild O(U) map/free metadata at a lifecycle boundary."""

        if self.capacity_rows > torch.iinfo(torch.int32).max:
            raise ValueError("packed physical row IDs exceed int32")
        free = self.owners.lt(0).nonzero(as_tuple=False).flatten().to(torch.int32)
        self.free_rows = torch.empty(
            self.capacity_rows, device=self.device, dtype=torch.int32
        )
        if free.numel():
            self.free_rows[: free.numel()].copy_(free)
        self.free_count = torch.tensor(
            free.numel(), device=self.device, dtype=torch.int32
        )
        self.overflow_flag = torch.zeros((), device=self.device, dtype=torch.int32)
        live = self.owners >= 0
        self.bank_row_counts = torch.bincount(
            self.owners[live], minlength=self.banks
        ).to(torch.int64)
        self.hash_rows = torch.full(
            (self._hash_capacity(self.capacity_rows),),
            -1,
            device=self.device,
            dtype=torch.int32,
        )
        self.proven_additional_rows = int(free.numel())
        if self.device.type == "cuda":
            from .cuda_state import rebuild_hash_table

            rebuild_hash_table(self)
            return
        live_rows = self.owners.ge(0).nonzero(as_tuple=False).flatten()
        for row_tensor in live_rows:
            row = int(row_tensor)
            owner = int(self.owners[row])
            slot = self._python_tuple_hash(owner, self.keys[row]) & (
                self.hash_rows.numel() - 1
            )
            while int(self.hash_rows[slot]) >= 0:
                slot = (slot + 1) & (self.hash_rows.numel() - 1)
            self.hash_rows[slot] = row

    @classmethod
    def empty(
        cls,
        prior: SelectedRowPrior,
        *,
        banks: int,
        template_indices: torch.Tensor | None = None,
        capacity_rows: int = 0,
        growth_quantum_rows: int = DEFAULT_GROWTH_QUANTUM_ROWS,
        state_dtype: torch.dtype | None = None,
    ) -> "PackedMemoryState":
        if banks <= 0 or capacity_rows < 0:
            raise ValueError("banks must be positive and capacity nonnegative")
        if capacity_rows == 0 and growth_quantum_rows > 0:
            capacity_rows = growth_quantum_rows
        parameter = next(prior.parameters(), None)
        if parameter is not None:
            device = parameter.device
            prior_dtype = parameter.dtype
        else:
            buffers = tuple(prior.buffers())
            device = buffers[0].device if buffers else torch.device("cpu")
            prior_dtype = buffers[0].dtype if buffers else torch.float32
        dtype = prior_dtype if state_dtype is None else state_dtype
        if template_indices is None:
            if banks % prior.templates:
                raise ValueError("banks require explicit template indices")
            template_indices = torch.arange(
                banks,
                device=device,
                dtype=torch.int64,
            ).remainder(prior.templates)
        else:
            template_indices = template_indices.to(device=device, dtype=torch.int64)
        return cls(
            keys=torch.full(
                (capacity_rows, prior.factors),
                -1,
                device=device,
                dtype=torch.int64,
            ),
            values=torch.empty(
                capacity_rows,
                prior.value_width,
                device=device,
                dtype=dtype,
            ),
            owners=torch.full(
                (capacity_rows,),
                -1,
                device=device,
                dtype=torch.int64,
            ),
            template_indices=template_indices,
            active_banks=torch.ones(banks, device=device, dtype=torch.bool),
            codebook_size=prior.codebook_size,
            factors=prior.factors,
            value_width=prior.value_width,
            prior_templates=prior.templates,
            growth_quantum_rows=growth_quantum_rows,
        )

    @classmethod
    @torch.no_grad()
    def from_execution(
        cls,
        result: ExecutionResult,
        prior: SelectedRowPrior,
        *,
        capacity_rows: int | None = None,
        growth_quantum_rows: int = DEFAULT_GROWTH_QUANTUM_ROWS,
        state_dtype: torch.dtype | None = None,
    ) -> "PackedMemoryState":
        return cls.from_compact(
            result.layout,
            result.final_selected_memory,
            result.template_indices,
            prior,
            capacity_rows=capacity_rows,
            growth_quantum_rows=growth_quantum_rows,
            state_dtype=state_dtype,
        )

    @classmethod
    @torch.no_grad()
    def from_compact(
        cls,
        layout: SelectedAddressLayout,
        final_selected_memory: torch.Tensor,
        template_indices: torch.Tensor,
        prior: SelectedRowPrior,
        *,
        capacity_rows: int | None = None,
        growth_quantum_rows: int = DEFAULT_GROWTH_QUANTUM_ROWS,
        state_dtype: torch.dtype | None = None,
    ) -> "PackedMemoryState":
        """Retain only rows first-written during a compact execution."""

        if layout.codebook_size != prior.codebook_size:
            raise ValueError("execution layout and prior codebooks differ")
        if layout.factors != prior.factors:
            raise ValueError("execution layout and prior tuple widths differ")
        expected = (layout.banks, layout.max_rows, prior.value_width)
        if final_selected_memory.shape != expected:
            raise ValueError(f"final compact memory must be {expected}")
        if template_indices.shape != (layout.banks,):
            raise ValueError("compact template indices must be [B]")
        rows = layout.materialized_rows
        if capacity_rows is None:
            if growth_quantum_rows:
                capacity_rows = max(
                    growth_quantum_rows,
                    ((rows + growth_quantum_rows - 1) // growth_quantum_rows)
                    * growth_quantum_rows,
                )
            else:
                capacity_rows = rows
        if capacity_rows < rows:
            raise ValueError("packed capacity is smaller than the written working set")
        parameter = next(prior.parameters(), None)
        if parameter is not None:
            prior_dtype = parameter.dtype
        else:
            buffers = tuple(prior.buffers())
            prior_dtype = buffers[0].dtype if buffers else torch.float32
        dtype = prior_dtype if state_dtype is None else state_dtype
        keys = torch.full(
            (capacity_rows, prior.factors),
            -1,
            device=layout.keys.device,
            dtype=torch.int64,
        )
        values = torch.empty(
            capacity_rows,
            prior.value_width,
            device=layout.keys.device,
            dtype=dtype,
        )
        owners = torch.full(
            (capacity_rows,),
            -1,
            device=layout.keys.device,
            dtype=torch.int64,
        )
        live = layout.written & layout.valid_mask()
        bank_ids = (
            torch.arange(layout.banks, device=layout.keys.device, dtype=torch.int64)
            .unsqueeze(1)
            .expand_as(live)
        )
        if rows:
            keys[:rows].copy_(layout.keys[live])
            values[:rows].copy_(final_selected_memory[live].to(dtype))
            owners[:rows].copy_(bank_ids[live])
        # Construct once from the already-packed tensors.  __post_init__ builds
        # the free stack, per-bank counts, and sparse hash exactly once rather
        # than building an empty hash and immediately rebuilding it.
        return cls(
            keys=keys,
            values=values,
            owners=owners,
            template_indices=template_indices.to(
                device=layout.keys.device, dtype=torch.int64
            ),
            active_banks=torch.ones(
                layout.banks, device=layout.keys.device, dtype=torch.bool
            ),
            codebook_size=prior.codebook_size,
            factors=prior.factors,
            value_width=prior.value_width,
            prior_templates=prior.templates,
            growth_quantum_rows=growth_quantum_rows,
        )

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def capacity_rows(self) -> int:
        return self.values.shape[0]

    @property
    def banks(self) -> int:
        return self.template_indices.shape[0]

    @property
    def materialized_rows(self) -> int:
        return int(self.owners.ge(0).sum().item())

    def materialized_rows_by_bank(self) -> torch.Tensor:
        if self.bank_row_counts is None:
            self._rebuild_allocator_metadata()
        return self.bank_row_counts.clone()

    def validate_invariants(self) -> None:
        if bool((self.owners < -1).any()):
            raise ValueError("free packed rows must use owner sentinel -1")
        valid = self.owners >= 0
        if bool(valid.any()):
            observed = self.owners[valid]
            if int(observed.max()) >= self.banks:
                raise ValueError("packed owner lies outside the bank table")
            if bool((~self.active_banks[observed]).any()):
                raise ValueError("released bank still owns packed rows")
            selected_keys = torch.cat(
                (observed.unsqueeze(-1), self.keys[valid]),
                dim=-1,
            )
            if torch.unique(selected_keys, dim=0).shape[0] != selected_keys.shape[0]:
                raise ValueError("packed state contains duplicate bank/tuple keys")
            if bool((self.keys[valid] < 0).any()) or bool(
                (self.keys[valid] >= self.codebook_size).any()
            ):
                raise ValueError("live packed tuple lies outside the factor codebook")
        if bool((self.keys[~valid] != -1).any()):
            raise ValueError("free rows must have sentinel tuple keys")

    def lookup(self, addresses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve selected [B,N,p] tuples against the materialized slab."""

        if addresses.ndim != 3 or addresses.shape != (
            self.banks,
            addresses.shape[1],
            self.factors,
        ):
            raise ValueError(f"lookup addresses must be [B,N,{self.factors}]")
        if addresses.device != self.device:
            raise ValueError("lookup addresses must share the state device")
        if addresses.dtype not in (torch.int32, torch.int64):
            raise ValueError("lookup addresses must be integer tuples")
        if addresses.device.type == "cpu":
            if bool((addresses < 0).any()) or bool(
                (addresses >= self.codebook_size).any()
            ):
                raise ValueError("lookup tuple lies outside the factor codebook")
        elif hasattr(torch, "_assert_async"):
            torch._assert_async(
                ((addresses >= 0) & (addresses < self.codebook_size)).all(),
                "lookup tuple lies outside the factor codebook",
            )
        if not self.capacity_rows:
            positions = torch.zeros(
                addresses.shape[:-1],
                device=self.device,
                dtype=torch.int64,
            )
            return positions, torch.zeros_like(positions, dtype=torch.bool)
        if self.device.type == "cuda":
            from .cuda_state import lookup_hash_rows

            return lookup_hash_rows(self, addresses)
        bank_ids = torch.arange(self.banks, device=self.device, dtype=torch.int64)
        owner_match = self.owners.view(1, 1, -1) == bank_ids.view(-1, 1, 1)
        key_match = (
            addresses.to(torch.int64)
            .unsqueeze(2)
            .eq(self.keys.view(1, 1, self.capacity_rows, self.factors))
            .all(dim=-1)
        )
        matches = owner_match & key_match
        found = matches.any(dim=-1)
        positions = matches.to(torch.int64).argmax(dim=-1)
        return positions, found

    @torch.no_grad()
    def grow(self, minimum_capacity_rows: int) -> int:
        if minimum_capacity_rows <= self.capacity_rows:
            return 0
        if self.growth_quantum_rows:
            # Pool-sized quanta bound excess reservation; geometric expansion
            # bounds total rows copied across a long decode trace by O(U).
            target = max(
                minimum_capacity_rows,
                self.growth_quantum_rows,
                self.capacity_rows * 2,
            )
            selected = (
                (target + self.growth_quantum_rows - 1) // self.growth_quantum_rows
            ) * self.growth_quantum_rows
        else:
            selected = minimum_capacity_rows
        old = self.capacity_rows
        added = selected - old
        new_keys = torch.full(
            (selected, self.factors),
            -1,
            device=self.device,
            dtype=torch.int64,
        )
        new_values = torch.empty(
            selected,
            self.value_width,
            device=self.device,
            dtype=self.values.dtype,
        )
        new_owners = torch.full(
            (selected,),
            -1,
            device=self.device,
            dtype=torch.int64,
        )
        if old:
            new_keys[:old].copy_(self.keys)
            new_values[:old].copy_(self.values)
            new_owners[:old].copy_(self.owners)
        self.keys = new_keys
        self.values = new_values
        self.owners = new_owners
        self.growth_events += 1
        self.growth_rows_added += added
        self.growth_rows_copied += old
        self._rebuild_allocator_metadata()
        return added

    @torch.no_grad()
    def prepare_step_capacity(self, maximum_new_rows: int) -> int:
        """Prove aggregate slab headroom for one fused decode launch."""

        if maximum_new_rows < 0:
            raise ValueError("maximum new rows cannot be negative")
        if self.proven_additional_rows < maximum_new_rows:
            if self.free_count is None:
                self._rebuild_allocator_metadata()
            free = int(self.free_count.item())
            if free < maximum_new_rows:
                if self.growth_quantum_rows <= 0:
                    raise ValueError(
                        "packed decode capacity is exhausted and growth is disabled"
                    )
                self.grow(self.capacity_rows + maximum_new_rows - free)
                free = int(self.free_count.item())
            self.proven_additional_rows = free
        self.proven_additional_rows -= maximum_new_rows
        return self.capacity_rows

    @torch.no_grad()
    def _allocate_first_touches(
        self,
        prior: SelectedRowPrior,
        write_addresses: torch.Tensor,
    ) -> torch.Tensor:
        positions, found = self.lookup(write_addresses)
        new_by_bank: list[tuple[int, torch.Tensor]] = []
        total_new = 0
        for bank in range(self.banks):
            missing = write_addresses[bank][~found[bank]]
            if missing.numel():
                missing = torch.unique(missing.to(torch.int64), dim=0, sorted=True)
            else:
                missing = missing.reshape(0, self.factors).to(torch.int64)
            new_by_bank.append((bank, missing))
            total_new += missing.shape[0]
        free = int(self.owners.lt(0).sum().item())
        if free < total_new:
            self.grow(self.capacity_rows + total_new - free)
        free_rows = self.owners.lt(0).nonzero(as_tuple=False).flatten()
        cursor = 0
        for bank, missing in new_by_bank:
            count = missing.shape[0]
            if not count:
                continue
            target = free_rows[cursor : cursor + count]
            templates = self.template_indices[bank : bank + 1]
            prior_values = prior(missing.unsqueeze(0), templates).squeeze(0)
            self.keys[target] = missing
            self.values[target] = prior_values.to(self.values.dtype)
            self.owners[target] = bank
            cursor += count
        self._rebuild_allocator_metadata()
        resolved, resolved_found = self.lookup(write_addresses)
        if not bool(resolved_found.all()):
            raise AssertionError("first-touch allocation did not resolve every write")
        return resolved

    @torch.no_grad()
    def decode_step(
        self,
        prior: SelectedRowPrior,
        write_addresses: torch.Tensor,
        write_weights: torch.Tensor,
        values: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        read_addresses: torch.Tensor,
        read_weights: torch.Tensor,
    ) -> DecodeStep:
        """Execute one native SDM decode position on selected tuple state."""

        if not bool(self.active_banks.all()):
            raise ValueError("released banks must be compacted before decode")
        if write_addresses.ndim != 3 or write_addresses.shape[0] != self.banks:
            raise ValueError("decode writes must be [B,W,p]")
        if write_addresses.shape[-1] != self.factors:
            raise ValueError("decode write tuple width changed")
        if read_addresses.ndim != 3 or read_addresses.shape[0] != self.banks:
            raise ValueError("decode reads must be [B,R,p]")
        if read_addresses.shape[-1] != self.factors:
            raise ValueError("decode read tuple width changed")
        writes = write_addresses.shape[1]
        reads = read_addresses.shape[1]
        if write_weights.shape != (self.banks, writes):
            raise ValueError("decode write weights do not align")
        if read_weights.shape != (self.banks, reads):
            raise ValueError("decode read weights do not align")
        if values.shape != (self.banks, self.value_width):
            raise ValueError("decode values must be [B,V]")
        if beta.shape not in ((self.banks, 1), (self.banks, self.value_width)):
            raise ValueError("decode beta must be scalar or channelwise")
        if log_decay.shape not in ((self.banks, 1), (self.banks, writes)):
            raise ValueError("decode log decay must be scalar or routewise")

        if self.device.type == "cuda":
            if hasattr(torch, "_assert_async"):
                torch._assert_async(
                    (
                        (write_addresses >= 0) & (write_addresses < self.codebook_size)
                    ).all(),
                    "decode write tuple lies outside the codebook",
                )
                torch._assert_async(
                    (
                        (read_addresses >= 0) & (read_addresses < self.codebook_size)
                    ).all(),
                    "decode read tuple lies outside the codebook",
                )
                duplicate = (
                    write_addresses.unsqueeze(2)
                    .eq(write_addresses.unsqueeze(1))
                    .all(dim=-1)
                )
                duplicate = torch.triu(duplicate, diagonal=1).any()
                torch._assert_async(
                    ~duplicate,
                    "released product-key writes must be unique within a bank",
                )
            from .cuda_state import fused_packed_decode_step

            readings, first_counts = fused_packed_decode_step(
                self,
                prior,
                write_addresses,
                write_weights,
                values,
                beta,
                log_decay,
                read_addresses,
                read_weights,
            )
            self.bank_row_counts.add_(first_counts.to(torch.int64))
            return DecodeStep(
                readings=readings,
                first_touches_by_bank=first_counts,
                repeated_writes_by_bank=torch.full_like(first_counts, writes)
                - first_counts,
                materialized_rows_by_bank=self.materialized_rows_by_bank(),
            )

        _, existed = self.lookup(write_addresses)
        write_positions = self._allocate_first_touches(prior, write_addresses)
        first_counts = torch.empty(self.banks, device=self.device, dtype=torch.int64)
        for bank in range(self.banks):
            missing = write_addresses[bank][~existed[bank]]
            first_counts[bank] = (
                torch.unique(missing, dim=0).shape[0] if missing.numel() else 0
            )

        selected = self.values[write_positions]
        decay = torch.exp(log_decay.float()).to(self.values.dtype)
        if decay.shape[-1] == 1:
            decay = decay.expand(self.banks, writes)
        retrieved = (
            write_weights.to(self.values.dtype).unsqueeze(-1)
            * selected
            * decay.unsqueeze(-1)
        ).sum(dim=1)
        update = beta.to(self.values.dtype) * (values.to(self.values.dtype) - retrieved)
        flat_positions = write_positions.reshape(-1)
        decay_by_row = torch.ones(
            self.capacity_rows,
            device=self.device,
            dtype=self.values.dtype,
        ).scatter_reduce(
            0,
            flat_positions,
            decay.reshape(-1),
            reduce="prod",
            include_self=True,
        )
        bank_updates = update.unsqueeze(1).expand(self.banks, writes, self.value_width)
        weighted_updates = (
            write_weights.to(self.values.dtype).unsqueeze(-1) * bank_updates
        )
        write_by_row = torch.zeros_like(self.values).scatter_add(
            0,
            flat_positions.unsqueeze(-1).expand(-1, self.value_width),
            weighted_updates.reshape(-1, self.value_width),
        )
        self.values.mul_(decay_by_row.unsqueeze(-1)).add_(write_by_row)

        read_positions, private = self.lookup(read_addresses)
        prior_reads = prior(read_addresses, self.template_indices).to(self.values.dtype)
        private_reads = self.values[read_positions]
        selected_reads = torch.where(private.unsqueeze(-1), private_reads, prior_reads)
        readings = (
            read_weights.to(self.values.dtype).unsqueeze(-1) * selected_reads
        ).sum(dim=1)
        return DecodeStep(
            readings=readings,
            first_touches_by_bank=first_counts,
            repeated_writes_by_bank=torch.full_like(first_counts, writes)
            - first_counts,
            materialized_rows_by_bank=self.materialized_rows_by_bank(),
        )

    @torch.no_grad()
    def release(self, bank_indices: torch.Tensor) -> int:
        if bank_indices.ndim != 1 or bank_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("released bank indices must be integer [N]")
        indices = bank_indices.to(device=self.device, dtype=torch.int64)
        if indices.numel() and (
            int(indices.min()) < 0
            or int(indices.max()) >= self.banks
            or torch.unique(indices).numel() != indices.numel()
        ):
            raise ValueError("released bank index is invalid or duplicated")
        if indices.numel() and not bool(self.active_banks[indices].all()):
            raise ValueError("bank is already released")
        selected = torch.zeros(self.banks, device=self.device, dtype=torch.bool)
        selected[indices] = True
        rows = self.owners.ge(0) & selected[self.owners.clamp_min(0)]
        released = int(rows.sum().item())
        self.owners[rows] = -1
        self.keys[rows] = -1
        self.active_banks[indices] = False
        self._rebuild_allocator_metadata()
        return released

    @torch.no_grad()
    def compact_survivors(self, bank_indices: torch.Tensor) -> "PackedMemoryState":
        if bank_indices.ndim != 1 or bank_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("survivor indices must be integer [N]")
        indices = bank_indices.to(device=self.device, dtype=torch.int64)
        if indices.numel() and (
            int(indices.min()) < 0
            or int(indices.max()) >= self.banks
            or torch.unique(indices).numel() != indices.numel()
        ):
            raise ValueError("survivor index is invalid or duplicated")
        if indices.numel() and not bool(self.active_banks[indices].all()):
            raise ValueError("survivors must refer to active banks")
        selected_banks = torch.zeros(self.banks, device=self.device, dtype=torch.bool)
        selected_banks[indices] = True
        if bool(self.active_banks[~selected_banks].any()):
            raise ValueError("discarded banks must be released before compaction")
        old_to_new = torch.full(
            (self.banks,), -1, device=self.device, dtype=torch.int64
        )
        old_to_new[indices] = torch.arange(indices.numel(), device=self.device)
        live = self.owners.ge(0) & old_to_new[self.owners.clamp_min(0)].ge(0)
        live_count = int(live.sum().item())
        capacity = live_count
        if self.growth_quantum_rows and live_count:
            capacity = max(
                self.growth_quantum_rows,
                (
                    (live_count + self.growth_quantum_rows - 1)
                    // self.growth_quantum_rows
                )
                * self.growth_quantum_rows,
            )
        keys = torch.full(
            (capacity, self.factors), -1, device=self.device, dtype=torch.int64
        )
        values = torch.empty(
            capacity, self.value_width, device=self.device, dtype=self.values.dtype
        )
        owners = torch.full((capacity,), -1, device=self.device, dtype=torch.int64)
        if live_count:
            keys[:live_count] = self.keys[live]
            values[:live_count] = self.values[live]
            owners[:live_count] = old_to_new[self.owners[live]]
        return PackedMemoryState(
            keys=keys,
            values=values,
            owners=owners,
            template_indices=self.template_indices[indices].clone(),
            active_banks=torch.ones(
                indices.numel(), device=self.device, dtype=torch.bool
            ),
            codebook_size=self.codebook_size,
            factors=self.factors,
            value_width=self.value_width,
            prior_templates=self.prior_templates,
            growth_quantum_rows=self.growth_quantum_rows,
            growth_events=self.growth_events,
            growth_rows_added=self.growth_rows_added,
            growth_rows_copied=self.growth_rows_copied,
        )

    @torch.no_grad()
    def admit(self, template_indices: torch.Tensor) -> torch.Tensor:
        templates = template_indices.to(device=self.device, dtype=torch.int64)
        if templates.ndim != 1:
            raise ValueError("admitted templates must be [N]")
        if templates.numel() and (
            int(templates.min()) < 0 or int(templates.max()) >= self.prior_templates
        ):
            raise ValueError("admitted template index lies outside the prior")
        start = self.banks
        self.template_indices = torch.cat((self.template_indices, templates))
        self.active_banks = torch.cat(
            (
                self.active_banks,
                torch.ones(templates.numel(), device=self.device, dtype=torch.bool),
            )
        )
        if self.bank_row_counts is None:
            self._rebuild_allocator_metadata()
        else:
            self.bank_row_counts = torch.cat(
                (
                    self.bank_row_counts,
                    torch.zeros(
                        templates.numel(), device=self.device, dtype=torch.int64
                    ),
                )
            )
        return torch.arange(start, self.banks, device=self.device, dtype=torch.int64)

    def storage_elements(self) -> dict[str, int]:
        return {
            "tuple_keys": self.keys.numel(),
            "private_values": self.values.numel(),
            "row_owners": self.owners.numel(),
            "sparse_hash_rows": 0 if self.hash_rows is None else self.hash_rows.numel(),
            "free_row_stack": 0 if self.free_rows is None else self.free_rows.numel(),
            "allocator_scalars": 2,
            "bank_row_counts": (
                0 if self.bank_row_counts is None else self.bank_row_counts.numel()
            ),
            "template_indices": self.template_indices.numel(),
            "active_banks": self.active_banks.numel(),
            "logical_capacity_tensor": 0,
        }

    @torch.no_grad()
    def materialize_dense(
        self,
        prior: SelectedRowPrior,
        *,
        maximum_rows: int = 65_536,
    ) -> torch.Tensor:
        base = materialize_prior(prior, maximum_rows=maximum_rows)
        dense = base[self.template_indices].clone().to(self.values.dtype)
        live = self.owners >= 0
        if bool(live.any()):
            from .addresses import encode_product_addresses

            logical = encode_product_addresses(
                self.keys[live],
                codebook_size=self.codebook_size,
            )
            dense[self.owners[live], logical] = self.values[live]
        return dense


@dataclass
class CapacityIndependentLayerState:
    """Released-SDM-compatible cache shell around packed tuple state."""

    packed: PackedMemoryState
    seq_len: int = 0

    @property
    def memory(self) -> torch.Tensor:
        """Expose physical rows for diagnostics, never a logical table."""

        return self.packed.values

    def update_(
        self,
        packed: PackedMemoryState,
        seq_len: int,
    ) -> "CapacityIndependentLayerState":
        if seq_len < 0:
            raise ValueError("cache sequence increments cannot be negative")
        self.packed = packed
        self.seq_len += seq_len
        return self

    def __getitem__(self, key_idx: int) -> torch.Tensor:
        if key_idx == 0:
            return self.memory
        raise KeyError(key_idx)

    def __len__(self) -> int:
        return 1

    @property
    def cache_len(self) -> int:
        return self.seq_len

    @torch.no_grad()
    def reset(self) -> None:
        """Release every private row while retaining allocator capacity."""

        self.packed.keys.fill_(-1)
        self.packed.owners.fill_(-1)
        self.packed.active_banks.fill_(True)
        self.packed._rebuild_allocator_metadata()
        self.seq_len = 0

    def get_seq_length(self) -> int:
        return self.seq_len


__all__ = [
    "CapacityIndependentLayerState",
    "DEFAULT_GROWTH_QUANTUM_ROWS",
    "DecodeStep",
    "PackedMemoryState",
]
