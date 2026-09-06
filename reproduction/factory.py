"""Narrow constructor adapter for the released SDM model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .config import MemoryGeometry


@dataclass(frozen=True, slots=True)
class HigherDimensionalSDMFactory:
    """Build the released SDM layer for one logical geometry."""

    geometry: MemoryGeometry
    growth_quantum_rows: int = 256

    def __post_init__(self) -> None:
        if self.growth_quantum_rows < 0:
            raise ValueError("growth quantum cannot be negative")

    def __call__(self, args: Any, layer_id: int):
        # Imports remain inside the seam so geometry/accounting tools do not
        # require a prepared CUDA/runtime overlay merely to load the package.
        from capacity_independent_sdm.extensions import AdditiveProductPrior
        from capacity_independent_sdm.production import (
            CapacityIndependentSparseDeltaMemory,
        )

        if args.dim <= 0 or args.dim % self.geometry.memory_heads:
            raise ValueError("released SDM width is incompatible with memory heads")
        expected = {
            "num_heads": self.geometry.memory_heads,
            "num_reads": self.geometry.reads,
            "num_writes": self.geometry.writes,
            "slots_per_head": self.geometry.logical_capacity,
        }
        observed = {name: getattr(args, name) for name in expected}
        if observed != expected:
            raise ValueError(
                f"released model supplied a different memory geometry: {observed}"
            )
        if not args.backprop_on_memory:
            raise ValueError("the reproduction requires learned product-key initial memory")

        layer_args = replace(
            args,
            # The host flag requests learned initialization. The production
            # subclass suppresses the dense table and evaluates this provider
            # only at selected tuples.
            backprop_on_memory=True,
        )
        prior = AdditiveProductPrior(
            templates=self.geometry.memory_heads,
            factors=self.geometry.factors,
            codebook_size=self.geometry.codebook_size,
            value_width=self.geometry.value_width(args.dim),
        )
        # Preserve model-size accounting: learned untouched state is
        # model state read sparsely, not an active per-token parameter surface.
        for parameter in prior.parameters():
            parameter._sdm_memory_bank = True
        return CapacityIndependentSparseDeltaMemory(
            layer_args,
            layer_id,
            selected_row_prior=prior,
            product_key_factors=self.geometry.factors,
            product_key_codebook_size=self.geometry.codebook_size,
            growth_quantum_rows=self.growth_quantum_rows,
        )

    def contract(self) -> dict[str, object]:
        return {
            "host": "released_lingua_sdm_factory_seam",
            "layer": "CapacityIndependentSparseDeltaMemory",
            "prior": "AdditiveProductPrior",
            "subclasses_capacity_layer": False,
            "owns_forward": False,
            "owns_selector": False,
            "owns_recurrence": False,
            "owns_cuda_kernel": False,
            "geometry": self.geometry.as_dict(),
        }


__all__ = ["HigherDimensionalSDMFactory"]
