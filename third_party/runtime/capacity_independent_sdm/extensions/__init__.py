"""Optional model-side mechanisms layered above the C execution runtime."""

from .factorized_prior import AdditiveProductPrior
from .geometry import ProductKeyGeometry
from .objectives import (
    AllocationObjective,
    AllocationObjectiveResult,
    ProjectedDualController,
    elastic_sdm_loss,
)
from .routing import fused_product_key_scores, product_key_topk

__all__ = [
    "AdditiveProductPrior",
    "AllocationObjective",
    "AllocationObjectiveResult",
    "ProductKeyGeometry",
    "ProjectedDualController",
    "elastic_sdm_loss",
    "fused_product_key_scores",
    "product_key_topk",
]
