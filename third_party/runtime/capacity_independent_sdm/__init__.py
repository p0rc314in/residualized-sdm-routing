"""Capacity-independent execution for released Sparse Delta Memory."""

from .accounting import SparseOccupancy, sparse_occupancy
from .addresses import SelectedAddressLayout, compact_selected_addresses
from .execution import ExecutionResult, execute_compact
from .integration import CapacityIndependentExecutionAdapter
from .kernels import CompactKernelBackend, CompactKernelReport
from .priors import DenseTablePrior, SelectedRowPrior, ZeroPrior
from .state import (
    DEFAULT_GROWTH_QUANTUM_ROWS,
    CapacityIndependentLayerState,
    PackedMemoryState,
)

__all__ = [
    "CapacityIndependentLayerState",
    "CapacityIndependentExecutionAdapter",
    "CompactKernelBackend",
    "CompactKernelReport",
    "DEFAULT_GROWTH_QUANTUM_ROWS",
    "DenseTablePrior",
    "ExecutionResult",
    "PackedMemoryState",
    "SelectedAddressLayout",
    "SelectedRowPrior",
    "SparseOccupancy",
    "ZeroPrior",
    "compact_selected_addresses",
    "execute_compact",
    "sparse_occupancy",
]
