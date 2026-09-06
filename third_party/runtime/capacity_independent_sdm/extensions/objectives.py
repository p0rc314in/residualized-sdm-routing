"""M7 loss policies built on B+'s selected-event first-touch estimate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..accounting import SparseOccupancy
from .geometry import ProductKeyGeometry


OBJECTIVE_KINDS = (
    "unpriced",
    "elastic_u_over_s",
    "linear_u_over_c",
    "concave_u_over_c",
    "dimension_u_root_over_c",
    "budget_rows",
)


@dataclass(frozen=True)
class AllocationObjectiveResult:
    penalty: torch.Tensor
    hard_rows_per_request: torch.Tensor
    surrogate_rows_per_request: torch.Tensor
    normalized_per_request: torch.Tensor
    hard_rows_by_layer: torch.Tensor
    kind: str


@dataclass(frozen=True)
class AllocationObjective:
    """One explicitly normalized first-touch objective.

    Nonlinear policies are applied to each request's mean rows per memory bank
    before the batch reduction. Total request rows and physical bytes remain
    reporting quantities rather than being hidden inside this loss helper.
    """

    kind: str
    coefficient: float | tuple[float, ...] = 0.0
    concave_exponent: float = 0.75
    target_rows: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in OBJECTIVE_KINDS:
            raise ValueError(f"unknown allocation objective: {self.kind}")
        coefficients = (
            self.coefficient
            if isinstance(self.coefficient, tuple)
            else (self.coefficient,)
        )
        if not coefficients or any(value < 0.0 for value in coefficients):
            raise ValueError("allocation prices must be nonnegative")
        if not 0.0 < self.concave_exponent <= 1.0:
            raise ValueError("concave exponent must lie in (0,1]")
        if self.kind == "budget_rows" and (
            self.target_rows is None or self.target_rows < 0.0
        ):
            raise ValueError("budget_rows requires a nonnegative row target")
        if self.kind != "budget_rows" and self.target_rows is not None:
            raise ValueError("only budget_rows accepts a row target")

    def evaluate(
        self,
        occupancies: Sequence[SparseOccupancy],
        *,
        batch_size: int,
        heads: int,
        geometry: ProductKeyGeometry,
    ) -> AllocationObjectiveResult:
        if not occupancies:
            raise ValueError("an allocation objective requires at least one layer")
        if batch_size <= 0 or heads <= 0:
            raise ValueError("batch size and memory heads must be positive")
        expected_banks = batch_size * heads
        hard_layers = []
        surrogate_layers = []
        fraction_layers = []
        for occupancy in occupancies:
            if occupancy.logical_capacity != geometry.logical_capacity:
                raise ValueError("occupancy and product-key geometry disagree")
            if occupancy.hard_final_fraction.numel() != expected_banks:
                raise ValueError("occupancy banks do not match batch and heads")
            hard_rows = occupancy.hard_unique_by_position[:, -1].reshape(
                batch_size, heads
            ).float()
            soft_rows = occupancy.soft_final_fraction.reshape(
                batch_size, heads
            ) * float(occupancy.logical_capacity)
            surrogate_rows = hard_rows.detach() - soft_rows.detach() + soft_rows
            hard_layers.append(hard_rows)
            surrogate_layers.append(surrogate_rows)
            fraction_layers.append(
                surrogate_rows / float(occupancy.logical_capacity)
            )

        hard = torch.stack(hard_layers)
        surrogate = torch.stack(surrogate_layers)
        fractions = torch.stack(fraction_layers)
        hard_per_request = hard.mean(dim=(0, 2))
        surrogate_per_request = surrogate.mean(dim=(0, 2))
        coefficient = self._coefficient_tensor(
            layers=len(occupancies),
            reference=surrogate,
        )

        if self.kind == "unpriced":
            normalized = surrogate_per_request
            penalty = surrogate_per_request.mean() * 0.0
        elif self.kind == "elastic_u_over_s":
            normalized = fractions.mean(dim=(0, 2))
            penalty = coefficient[0] * normalized.mean()
        elif self.kind == "linear_u_over_c":
            normalized = surrogate_per_request / float(geometry.codebook_size)
            penalty = coefficient[0] * normalized.mean()
        elif self.kind == "concave_u_over_c":
            scaled = surrogate_per_request / float(geometry.codebook_size)
            normalized = scaled.clamp_min(0.0).pow(self.concave_exponent)
            penalty = coefficient[0] * normalized.mean()
        elif self.kind == "dimension_u_root_over_c":
            positive = surrogate_per_request.clamp_min(0.0)
            normalized = torch.where(
                positive > 0,
                positive.pow(1.0 / geometry.factors)
                / float(geometry.codebook_size),
                torch.zeros_like(positive),
            )
            penalty = coefficient[0] * normalized.mean()
        else:
            target = float(self.target_rows)
            constraint_by_layer = surrogate.mean(dim=2) - target
            normalized = surrogate_per_request - target
            penalty = (coefficient[:, None] * constraint_by_layer).mean()

        return AllocationObjectiveResult(
            penalty=penalty,
            hard_rows_per_request=hard_per_request,
            surrogate_rows_per_request=surrogate_per_request,
            normalized_per_request=normalized,
            hard_rows_by_layer=hard.mean(dim=(1, 2)),
            kind=self.kind,
        )

    def apply(
        self,
        task_loss: torch.Tensor,
        occupancies: Sequence[SparseOccupancy],
        *,
        batch_size: int,
        heads: int,
        geometry: ProductKeyGeometry,
    ) -> tuple[torch.Tensor, AllocationObjectiveResult]:
        result = self.evaluate(
            occupancies,
            batch_size=batch_size,
            heads=heads,
            geometry=geometry,
        )
        return task_loss + result.penalty, result

    def _coefficient_tensor(
        self,
        *,
        layers: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        values = (
            self.coefficient
            if isinstance(self.coefficient, tuple)
            else (self.coefficient,)
        )
        if len(values) not in (1, layers):
            raise ValueError("allocation prices must be scalar or one per layer")
        tensor = reference.new_tensor(values)
        if tensor.numel() == 1:
            tensor = tensor.expand(layers)
        if self.kind != "budget_rows" and tensor.numel() != 1 and not bool(
            (tensor == tensor[0]).all()
        ):
            raise ValueError("non-budget objectives use one shared coefficient")
        return tensor


@dataclass
class ProjectedDualController:
    """Update nonnegative global or per-layer prices between optimizer steps."""

    layers: int
    target_rows: float
    learning_rate: float
    ema_beta: float = 0.95
    maximum_price: float = 1.0
    per_layer: bool = False

    def __post_init__(self) -> None:
        if self.layers <= 0 or self.target_rows < 0.0:
            raise ValueError("layers must be positive and target rows nonnegative")
        if self.learning_rate < 0.0 or self.maximum_price <= 0.0:
            raise ValueError("dual learning rate and price bound are invalid")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("dual EMA beta must lie in [0,1)")
        count = self.layers if self.per_layer else 1
        self.prices = [0.0] * count
        self.ema_rows = [self.target_rows] * count
        self.updates = 0

    def price_by_layer(self) -> tuple[float, ...]:
        if self.per_layer:
            return tuple(self.prices)
        return (self.prices[0],) * self.layers

    def objective(self) -> AllocationObjective:
        return AllocationObjective(
            kind="budget_rows",
            coefficient=self.price_by_layer(),
            target_rows=self.target_rows,
        )

    def update(self, hard_rows_by_layer: Sequence[float] | torch.Tensor) -> None:
        values = [float(value) for value in hard_rows_by_layer]
        if len(values) != self.layers:
            raise ValueError("hard rows do not match the physical layer count")
        observations = values if self.per_layer else [sum(values) / self.layers]
        for index, observed in enumerate(observations):
            self.ema_rows[index] = (
                self.ema_beta * self.ema_rows[index]
                + (1.0 - self.ema_beta) * observed
            )
            proposed = self.prices[index] + self.learning_rate * (
                self.ema_rows[index] - self.target_rows
            )
            self.prices[index] = min(self.maximum_price, max(0.0, proposed))
        self.updates += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "layers": self.layers,
            "target_rows": self.target_rows,
            "per_layer": self.per_layer,
            "prices": list(self.prices),
            "ema_rows": list(self.ema_rows),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if (
            int(state.get("layers", -1)) != self.layers
            or float(state.get("target_rows", -1.0)) != self.target_rows
            or bool(state.get("per_layer", not self.per_layer)) != self.per_layer
        ):
            raise ValueError("dual-controller identity changed across recovery")
        prices = [float(value) for value in state.get("prices", [])]
        ema_rows = [float(value) for value in state.get("ema_rows", [])]
        if len(prices) != len(self.prices) or len(ema_rows) != len(self.ema_rows):
            raise ValueError("dual-controller state shape changed")
        self.prices = prices
        self.ema_rows = ema_rows
        self.updates = int(state.get("updates", 0))


def elastic_sdm_loss(
    task_loss: torch.Tensor,
    occupancies: Sequence[SparseOccupancy],
    *,
    coefficient: float,
    batch_size: int,
    heads: int,
    geometry: ProductKeyGeometry,
) -> tuple[torch.Tensor, AllocationObjectiveResult]:
    """Apply the historical Elastic SDM λ × U/S loss and nothing else."""

    return AllocationObjective(
        kind="elastic_u_over_s",
        coefficient=coefficient,
    ).apply(
        task_loss,
        occupancies,
        batch_size=batch_size,
        heads=heads,
        geometry=geometry,
    )


__all__ = [
    "OBJECTIVE_KINDS",
    "AllocationObjective",
    "AllocationObjectiveResult",
    "ProjectedDualController",
    "elastic_sdm_loss",
]
