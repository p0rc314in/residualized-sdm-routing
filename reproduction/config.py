"""Model geometry and topology definitions used by the reproduction."""

from __future__ import annotations

from dataclasses import dataclass


MAX_SIGNED_ADDRESS = 2**63 - 1


@dataclass(frozen=True, slots=True)
class MemoryGeometry:
    """One balanced product-key geometry."""

    arm: str
    factors: int
    codebook_size: int
    memory_heads: int = 1
    reads: int = 8
    writes: int = 8

    def __post_init__(self) -> None:
        if not self.arm or not self.arm.replace("_", "").isalnum():
            raise ValueError("arm must be a non-empty identifier")
        if self.factors < 2 or self.codebook_size <= 0:
            raise ValueError("product-key geometry is invalid")
        if self.memory_heads <= 0:
            raise ValueError("memory head count must be positive")
        if self.reads <= 0 or self.writes != self.reads:
            raise ValueError("balanced sparse access requires W = R > 0")
        if self.reads > self.codebook_size:
            raise ValueError("selected access exceeds a factor codebook")
        if self.logical_capacity > MAX_SIGNED_ADDRESS:
            raise ValueError("logical addresses must fit signed 64-bit IDs")

    @property
    def logical_capacity(self) -> int:
        return self.codebook_size**self.factors

    @property
    def routing_width(self) -> int:
        return self.memory_heads * self.factors * self.codebook_size

    def value_width(self, model_width: int) -> int:
        if model_width <= 0 or model_width % self.memory_heads:
            raise ValueError("model width must be divisible by memory heads")
        return model_width // self.memory_heads

    def prior_parameters_per_layer(self, model_width: int) -> int:
        return (
            self.memory_heads
            * self.factors
            * self.codebook_size
            * self.value_width(model_width)
        )

    def routing_parameters_per_layer(self, model_width: int) -> int:
        return 2 * self.routing_width * (model_width + 1)

    def as_dict(self) -> dict[str, int | str]:
        return {
            "arm": self.arm,
            "factors": self.factors,
            "codebook_size": self.codebook_size,
            "logical_capacity": self.logical_capacity,
            "memory_heads": self.memory_heads,
            "routing_width": self.routing_width,
            "reads": self.reads,
            "writes": self.writes,
        }


@dataclass(frozen=True, slots=True)
class ModelProfile:
    width: int = 128
    layers: int = 8
    attention_layers: tuple[int, ...] = (7,)
    attention_heads: int = 4
    maximum_sequence_length: int = 2_048
    feed_forward_multiple: int = 256
    norm_epsilon: float = 1e-5
    allow_all_sdm: bool = False
    allow_all_attention: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.layers <= 0:
            raise ValueError("width and layer count must be positive")
        if self.attention_heads <= 0 or self.width % self.attention_heads:
            raise ValueError("width must be divisible by attention heads")
        if self.allow_all_sdm and self.allow_all_attention:
            raise ValueError("all-SDM and all-attention modes are mutually exclusive")
        if not self.attention_layers and not self.allow_all_sdm:
            raise ValueError("at least one attention layer is required")
        if self.attention_layers and self.allow_all_sdm:
            raise ValueError("all-SDM mode cannot list attention layers")
        if len(set(self.attention_layers)) != len(self.attention_layers):
            raise ValueError("attention layer indices must be unique")
        if any(index < 0 or index >= self.layers for index in self.attention_layers):
            raise ValueError("attention layer index lies outside the model")
        if len(self.attention_layers) == self.layers and not self.allow_all_attention:
            raise ValueError("all-attention mode must be explicit")
        if self.allow_all_attention and len(self.attention_layers) != self.layers:
            raise ValueError("all-attention mode requires every layer")

    @property
    def memory_layers(self) -> tuple[int, ...]:
        attention = set(self.attention_layers)
        return tuple(index for index in range(self.layers) if index not in attention)

    @property
    def layout(self) -> str:
        attention = set(self.attention_layers)
        return "".join("A" if index in attention else "B" for index in range(self.layers))

    def as_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "layers": self.layers,
            "layout": self.layout,
            "attention_layers": list(self.attention_layers),
            "memory_layers": list(self.memory_layers),
            "attention_heads": self.attention_heads,
            "maximum_sequence_length": self.maximum_sequence_length,
            "feed_forward_multiple": self.feed_forward_multiple,
            "norm_epsilon": self.norm_epsilon,
            "allow_all_sdm": self.allow_all_sdm,
            "allow_all_attention": self.allow_all_attention,
        }


CANONICAL_PROFILE = ModelProfile()
SDM_ONLY_PROFILE = ModelProfile(attention_layers=(), allow_all_sdm=True)
ATTENTION_ONLY_PROFILE = ModelProfile(
    attention_layers=tuple(range(8)), allow_all_attention=True
)


__all__ = [
    "ATTENTION_ONLY_PROFILE",
    "CANONICAL_PROFILE",
    "SDM_ONLY_PROFILE",
    "MemoryGeometry",
    "ModelProfile",
]
