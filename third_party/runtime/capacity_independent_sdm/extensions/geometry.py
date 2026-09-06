"""Canonical p-factor address geometry for M5 composition."""

from __future__ import annotations

from dataclasses import dataclass

from ..addresses import logical_capacity


@dataclass(frozen=True)
class ProductKeyGeometry:
    factors: int
    codebook_size: int

    def __post_init__(self) -> None:
        logical_capacity(self.codebook_size, self.factors)

    @property
    def logical_capacity(self) -> int:
        return logical_capacity(self.codebook_size, self.factors)

    @property
    def score_width_per_head(self) -> int:
        return self.factors * self.codebook_size

    def projection_width(self, heads: int) -> int:
        if heads <= 0:
            raise ValueError("memory heads must be positive")
        return heads * self.score_width_per_head

    def validate_access(self, *, reads: int, writes: int) -> None:
        if not 1 <= reads <= self.logical_capacity:
            raise ValueError("selected reads must lie in the logical address space")
        if not 1 <= writes <= self.logical_capacity:
            raise ValueError("selected writes must lie in the logical address space")


__all__ = ["ProductKeyGeometry"]
