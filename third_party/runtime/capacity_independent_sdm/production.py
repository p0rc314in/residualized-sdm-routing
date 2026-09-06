"""Capacity-independent Sparse Delta Memory on the released SDM host layer."""

from __future__ import annotations

from dataclasses import replace
import math

import torch

try:
    from lingua.sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs
except ModuleNotFoundError as error:  # pragma: no cover - exercised in the CUDA overlay
    raise ModuleNotFoundError(
        "production requires the vendored SDM runtime and its CUDA dependencies; "
        "install the locked environment with `uv sync --frozen`"
    ) from error

from .integration import CapacityIndependentExecutionAdapter
from .extensions.factorized_prior import AdditiveProductPrior
from .extensions.geometry import ProductKeyGeometry
from .extensions.routing import fused_product_key_scores
from .priors import DenseTablePrior, SelectedRowPrior, ZeroPrior
from .state import DEFAULT_GROWTH_QUANTUM_ROWS, CapacityIndependentLayerState


SDM_EXECUTION_SEAMS = (
    "_product_key_geometry",
    "_select_product_keys_for_head",
    "_prepare_memory_execution",
    "_prepare_memory_telemetry",
    "_execute_memory",
    "_finalize_memory_execution",
)
for seam in SDM_EXECUTION_SEAMS:
    if not hasattr(SparseDeltaMemory, seam):
        raise RuntimeError(
            "capacity-independent SDM requires the reviewed execution-seam patch; "
            f"SparseDeltaMemory is missing {seam}"
        )


class CapacityIndependentSparseDeltaMemory(SparseDeltaMemory):
    """Released native SDM with selected-row recurrence and packed request state."""

    def _product_key_geometry(self, args: SparseDeltaMemoryArgs) -> tuple[int, int]:
        del args
        geometry = self._selected_product_key_geometry
        return geometry.factors, geometry.codebook_size

    def __init__(
        self,
        args: SparseDeltaMemoryArgs,
        layer_id: int,
        *,
        selected_row_prior: SelectedRowPrior | None = None,
        product_key_factors: int = 2,
        product_key_codebook_size: int | None = None,
        growth_quantum_rows: int = DEFAULT_GROWTH_QUANTUM_ROWS,
    ) -> None:
        if product_key_codebook_size is None:
            if product_key_factors != 2:
                raise ValueError("p > 2 requires an explicit product-key codebook size")
            product_key_codebook_size = math.isqrt(args.slots_per_head)
        geometry = ProductKeyGeometry(
            factors=product_key_factors,
            codebook_size=product_key_codebook_size,
        )
        if geometry.logical_capacity != args.slots_per_head:
            raise ValueError("slots_per_head must equal C**p for the selected geometry")
        geometry.validate_access(
            reads=args.num_reads,
            writes=args.num_writes,
        )
        if (
            geometry.factors > 2
            and max(args.num_reads, args.num_writes) > geometry.codebook_size
        ):
            raise ValueError("p > 2 fused routing requires R and W <= C")
        self._selected_product_key_geometry = geometry

        requested_learned_prior = args.backprop_on_memory
        if (
            geometry.factors > 2
            and requested_learned_prior
            and selected_row_prior is None
        ):
            raise ValueError(
                "p > 2 learned initialization requires an explicit selected-row "
                "prior so a full logical table is never allocated"
            )
        self._sparse_access_stats_enabled = args.log_memory_access_stats
        self._sparse_memory_norms_enabled = args.log_memory_norms
        host_args = replace(
            args,
            backprop_on_memory=False,
            log_memory_access_stats=False,
            log_memory_norms=False,
        )
        super().__init__(host_args, layer_id)
        if self.memory is not None:
            raise AssertionError("released host allocated a logical recurrent table")

        if selected_row_prior is None and requested_learned_prior:
            selected_row_prior = DenseTablePrior(
                torch.empty(
                    self.num_heads,
                    args.slots_per_head,
                    self.head_dim,
                ),
                factors=geometry.factors,
                codebook_size=geometry.codebook_size,
                trainable=True,
            )
        elif selected_row_prior is None:
            selected_row_prior = ZeroPrior(
                templates=self.num_heads,
                factors=geometry.factors,
                codebook_size=geometry.codebook_size,
                value_width=self.head_dim,
            )
        expected = (
            self.num_heads,
            geometry.factors,
            geometry.codebook_size,
            self.head_dim,
        )
        observed = (
            selected_row_prior.templates,
            selected_row_prior.factors,
            selected_row_prior.codebook_size,
            selected_row_prior.value_width,
        )
        if observed != expected:
            raise ValueError(
                "selected-row prior geometry must be "
                f"(templates,factors,codebook,width)={expected}; got {observed}"
            )
        self.selected_row_prior = selected_row_prior
        self.growth_quantum_rows = growth_quantum_rows
        self.last_sparse_occupancy = None
        self._sparse_objective_accounting_enabled = False
        self._sparse_stat_rows: dict[str, list[float]] = {}
        self._active_memory_execution = None
        self._capacity_execution = CapacityIndependentExecutionAdapter(
            self.selected_row_prior,
            num_heads=self.num_heads,
            growth_quantum_rows=growth_quantum_rows,
            memory_block_size=args.memory_block_size,
            snapshot_quant=args.snapshot_quant,
            key_weighted_decay=args.key_weighted_decay,
            collect_sparse_telemetry=(
                self._sparse_access_stats_enabled or self._sparse_memory_norms_enabled
            ),
        )

    def _select_product_keys_for_head(
        self,
        scores: torch.Tensor,
        selected: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self._selected_product_key_geometry
        if geometry.factors == 2:
            self._last_selector_backend = "released-p2"
            return super()._select_product_keys_for_head(scores, selected)
        expected_width = geometry.score_width_per_head
        if scores.shape[-1] != expected_width:
            raise ValueError("released host changed the p-factor score width")
        from lingua.sparse_delta_memory.memory_ops import fused_outer_add_topk
        from lingua.sparse_delta_memory.triton_argsort import triton_argsort

        selected_scores, flat = fused_product_key_scores(
            scores.reshape(
                *scores.shape[:-1],
                geometry.factors,
                geometry.codebook_size,
            ),
            selected=selected,
            outer_add_topk=fused_outer_add_topk,
            index_sort=triton_argsort,
        )
        self._last_selector_backend = "released-fused-iterative"
        return selected_scores.to(scores.dtype), flat

    def _prepare_memory_execution(
        self,
        *,
        x: torch.Tensor,
        cache,
        k_idx: torch.Tensor,
        q_idx: torch.Tensor,
        batch_size: int,
        heads: int,
        slots_per_head: int,
        head_dim: int,
        batch_heads: int,
        ulysses_active: bool,
        head_start: int,
        context_parallel_active: bool,
    ):
        del x
        if slots_per_head != self.slots_per_head or head_dim != self.head_dim:
            raise ValueError("released host changed the memory geometry")
        if batch_heads != batch_size * heads or batch_heads != k_idx.shape[0]:
            raise ValueError("released host changed the memory-bank layout")
        prepared = self._capacity_execution.prepare(
            k_idx=k_idx,
            q_idx=q_idx,
            cache=cache,
            context_parallel_active=context_parallel_active or ulysses_active,
            context_parallel_group=getattr(self, "cp_group", None),
            local_heads=heads if ulysses_active else None,
            head_start=head_start,
        )
        self._active_memory_execution = prepared[3]
        return prepared

    def _execute_memory(
        self,
        memory: torch.Tensor,
        k_idx: torch.Tensor,
        k_val: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        q_idx: torch.Tensor,
        q_val: torch.Tensor,
        execution_context,
        grad_final_memory: torch.Tensor | None = None,
    ):
        result = self._capacity_execution.execute(
            memory,
            k_idx,
            k_val,
            v,
            beta,
            g,
            q_idx,
            q_val,
            execution_context,
            training=self.training,
            grad_final_memory=grad_final_memory,
        )
        return result

    def set_sparse_objective_accounting(self, enabled: bool) -> None:
        """Retain differentiable selected-event occupancy without stats syncs."""

        if not isinstance(enabled, bool):
            raise TypeError("sparse objective accounting must be enabled by a bool")
        self._sparse_objective_accounting_enabled = enabled
        self._capacity_execution.collect_sparse_telemetry = enabled or (
            self._sparse_access_stats_enabled or self._sparse_memory_norms_enabled
        )
        if not enabled:
            self.last_sparse_occupancy = None

    def _prepare_memory_telemetry(
        self,
        execution_context,
        k_val: torch.Tensor,
        q_val: torch.Tensor,
    ) -> None:
        occupancy = self._capacity_execution.prepare_telemetry(
            execution_context,
            k_val,
            q_val,
        )
        self.last_sparse_occupancy = occupancy
        if occupancy is None:
            return
        if not (
            self._sparse_access_stats_enabled or self._sparse_memory_norms_enabled
        ):
            return
        self._append_sparse_stat(
            "write_unique_pct",
            100.0 * occupancy.hard_final_fraction.mean().item(),
        )
        self._append_sparse_stat(
            "read_unique_pct",
            100.0 * occupancy.read_final_fraction.mean().item(),
        )
        self._append_sparse_stat(
            "read_slot_entropy_normalized",
            occupancy.read_address_entropy_normalized.mean().item(),
        )
        self._append_sparse_stat(
            "write_slot_entropy_normalized",
            occupancy.write_address_entropy_normalized.mean().item(),
        )
        self._append_sparse_stat("read_weight_max", q_val.max().item())
        self._append_sparse_stat(
            "write_soft_occupied_pct",
            100.0 * occupancy.soft_final_fraction.mean().item(),
        )
        self._append_sparse_stat(
            "first_touches",
            occupancy.first_touch_count_by_position.sum().item(),
        )
        self._append_sparse_stat(
            "repeated_writes",
            occupancy.repeated_write_count_by_position.sum().item(),
        )
        self._append_sparse_stat(
            "private_read_fraction",
            (occupancy.private_read_count_by_position.float() / max(1, q_val.shape[-1]))
            .mean()
            .item(),
        )
        self._append_sparse_stat(
            "private_read_weight",
            occupancy.private_read_weight_by_position.mean().item(),
        )
        self._append_sparse_stat(
            "write_route_entropy",
            occupancy.write_route_entropy_by_position.mean().item(),
        )

    def _append_sparse_stat(self, name: str, value: float) -> None:
        self._sparse_stat_rows.setdefault(name, []).append(float(value))

    def _record_gate_stats(self, g: torch.Tensor, beta: torch.Tensor) -> None:
        """Replace released S-wide telemetry with selected-event scalars."""

        if not (self.training and self._sparse_access_stats_enabled):
            return
        with torch.no_grad():
            decay = torch.exp(g.float())
            for name, value in (
                ("forget_gate_min", decay.min()),
                ("forget_gate_mean", decay.mean()),
                ("forget_gate_max", decay.max()),
                ("input_gate_min", beta.min()),
                ("input_gate_mean", beta.mean()),
                ("input_gate_max", beta.max()),
            ):
                self._append_sparse_stat(name, value.item())

    def _record_read_stats(self, q_idx, q_val, BH, H, sph, ulysses_active) -> None:
        del q_idx, q_val, BH, H, sph, ulysses_active
        # Read/write routes are combined after compaction in _execute_memory so
        # first-touch and private-read ordering share one tuple namespace.

    def _record_memory_norms_pre(self, memory: torch.Tensor) -> None:
        if not self._sparse_memory_norms_enabled:
            return
        with torch.no_grad():
            selected = self._selected_memory_rows(memory)
            self._record_selected_memory_norms(selected, prefix="")

    def _selected_memory_rows(self, memory: torch.Tensor) -> torch.Tensor:
        """Exclude reserved slab rows and compact padding from norm telemetry."""

        context = self._active_memory_execution
        if context is None:
            return memory
        if context.cache is not None:
            return memory[context.cache.packed.owners >= 0]
        if context.layout is not None:
            compact = memory.view(
                context.layout.banks,
                context.layout.max_rows,
                self.head_dim,
            )
            return compact[context.layout.valid_mask()]
        return memory

    def _record_selected_memory_norms(
        self,
        memory: torch.Tensor,
        *,
        prefix: str,
    ) -> None:
        names = (
            f"{prefix}memory_norm",
            f"{prefix}memory_max",
            f"{prefix}slot_norm_mean",
            f"{prefix}slot_norm_max",
        )
        if not memory.numel():
            for name in names:
                self._append_sparse_stat(name, 0.0)
            return
        rows = memory.float()
        norms = rows.norm(dim=-1)
        values = (
            rows.norm().item(),
            rows.abs().max().item(),
            norms.mean().item(),
            norms.max().item(),
        )
        for name, value in zip(names, values):
            self._append_sparse_stat(name, value)

    def _record_write_stats(
        self, k_idx, k_val, memory, output, BH, H, sph, ulysses_active
    ) -> None:
        del k_idx, BH, H, sph, ulysses_active
        if not (self._sparse_access_stats_enabled or self._sparse_memory_norms_enabled):
            return
        with torch.no_grad():
            if self._sparse_access_stats_enabled:
                self._append_sparse_stat("write_strength_max", k_val.max().item())
                self._append_sparse_stat("write_strength_mean", k_val.mean().item())
                self._append_sparse_stat(
                    "output_norm", output.float().norm(dim=-1).mean().item()
                )
            if self._sparse_memory_norms_enabled:
                self._record_selected_memory_norms(
                    self._selected_memory_rows(memory),
                    prefix="updated_",
                )

    def collect_grad_stats(self) -> None:
        if not self._sparse_access_stats_enabled:
            return
        prior_grads = [
            parameter.grad.float().norm()
            for parameter in self.selected_row_prior.parameters()
            if parameter.grad is not None
        ]
        if prior_grads:
            self._append_sparse_stat(
                "memory_grad_norm",
                torch.stack(prior_grads).norm().item(),
            )
        qk_grads = []
        for projection in (self.Wq_read, self.Wk_write):
            for parameter in projection.parameters():
                if parameter.grad is not None:
                    qk_grads.append(parameter.grad.float().norm())
                    break
        if qk_grads:
            self._append_sparse_stat(
                "qk_proj_grad_norm", torch.stack(qk_grads).mean().item()
            )
        for parameter in self.Wo.parameters():
            if parameter.grad is not None:
                self._append_sparse_stat(
                    "output_proj_grad_norm", parameter.grad.float().norm().item()
                )
                break

    def get_memory_access_stats(self, reset: bool = True) -> dict[str, float] | None:
        if not (self._sparse_access_stats_enabled or self._sparse_memory_norms_enabled):
            return None
        stats = {
            name: sum(values) / len(values)
            for name, values in self._sparse_stat_rows.items()
            if values
        }
        if reset:
            self._sparse_stat_rows.clear()
        return stats

    def _finalize_memory_execution(
        self,
        memory: torch.Tensor,
        execution_context,
        cache,
        *,
        seq_len: int,
    ) -> CapacityIndependentLayerState:
        del memory
        try:
            return self._capacity_execution.finalize(
                execution_context,
                cache,
                seq_len=seq_len,
            )
        finally:
            self._active_memory_execution = None

    def create_kv_cache(
        self,
        bsz: int,
        seq_len: int,
        dtype: torch.dtype,
        device: str | torch.device | None = None,
    ) -> CapacityIndependentLayerState:
        return self._capacity_execution.create_cache(
            batch_size=bsz,
            seq_len=seq_len,
            dtype=dtype,
            device=device,
        )

    def init_weights(
        self,
        init_std: float | None = None,
        factor: float = 1.0,
        width_scaling: float | None = None,
        *,
        role_seed: int | None = None,
        selected_prior_seed: int | None = None,
    ) -> None:
        if selected_prior_seed is not None and selected_prior_seed < 0:
            raise ValueError("selected_prior_seed must be non-negative")
        super().init_weights(
            init_std,
            factor,
            width_scaling,
            role_seed=role_seed,
        )

        def prior_generator(parameter: torch.Tensor):
            if selected_prior_seed is None:
                return None
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(selected_prior_seed)
            return generator

        if isinstance(self.selected_row_prior, DenseTablePrior):
            selected_std = (init_std or (self.dim**-0.5)) / factor
            torch.nn.init.trunc_normal_(
                self.selected_row_prior.table,
                mean=0.0,
                std=selected_std,
                a=-3.0 * selected_std,
                b=3.0 * selected_std,
                generator=prior_generator(self.selected_row_prior.table),
            )
        elif isinstance(self.selected_row_prior, AdditiveProductPrior):
            selected_std = (init_std or (self.dim**-0.5)) / factor
            self.selected_row_prior.reset_parameters(
                full_table_std=selected_std,
                generator=prior_generator(self.selected_row_prior.factor_tables),
            )

    def execution_storage_contract(self) -> dict[str, str | int]:
        return {
            "logical_capacity": self.slots_per_head,
            "product_key_factors": self._selected_product_key_geometry.factors,
            "product_key_codebook_size": (
                self._selected_product_key_geometry.codebook_size
            ),
            "selector_backend": getattr(self, "_last_selector_backend", "not_run"),
            "logical_capacity_tensor_elements": 0,
            "initial_state": type(self.selected_row_prior).__name__,
            "prefill_state": "selected_read_write_union",
            "retained_state": "first_written_tuple_rows",
            "decode_state": "packed_copy_on_write_rows",
        }


CapacityIndependentSDM = CapacityIndependentSparseDeltaMemory

__all__ = [
    "CapacityIndependentSDM",
    "CapacityIndependentSparseDeltaMemory",
    "SDM_EXECUTION_SEAMS",
]
