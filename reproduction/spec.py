"""Exact arm and optimization specification for the small reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .config import (
    ATTENTION_ONLY_PROFILE,
    CANONICAL_PROFILE,
    SDM_ONLY_PROFILE,
    MemoryGeometry,
    ModelProfile,
)


CAMPAIGN_ID = "residualized-routing-wikitext103-seed0-v1"
PASSES = 3
TOTAL_STEPS = 21_603
WARMUP_STEPS = 540
EXPECTED_MANIFEST_SHA256 = (
    "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6"
)
EXPECTED_PAYLOAD_SHA256 = (
    "f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e"
)
EXPECTED_INVENTORY_SHA256 = (
    "efa0a0b857d184dccdecf6adafda5d646ec45bae91f01a15cd1165f9fe9ad6b3"
)
EXPECTED_COMMON_SDM_SHA256 = MappingProxyType(
    {
        "BBBBBBBA": "c5ebb90d8a51e7a3ff584d77ac8d63c9f3e3a995571469116e7068b5931f3241",
        "BBBBBBBB": "47f42b13c4b3cd3ebcb7a01b42b96d15d2d482a00ffd2debc1cab1cbc9b654fa",
    }
)
EXPECTED_NLL_RANGES = MappingProxyType(
    {
        "dense_attention": MappingProxyType(
            {"validation": (4.32, 4.40), "test": (4.32, 4.40)}
        ),
        "b7a1_native_sdm": MappingProxyType(
            {"validation": (4.40, 4.49), "test": (4.40, 4.49)}
        ),
        "b8_native_sdm": MappingProxyType(
            {"validation": (4.43, 4.53), "test": (4.43, 4.53)}
        ),
        "b7a1_residualized_sdm": MappingProxyType(
            {"validation": (4.31, 4.40), "test": (4.31, 4.40)}
        ),
        "b8_residualized_sdm": MappingProxyType(
            {"validation": (4.33, 4.42), "test": (4.33, 4.42)}
        ),
    }
)
EXPECTED_GAP_CLOSURE_RANGES = MappingProxyType(
    {
        "b7a1": (0.75, 1.20),
        "b8": (0.65, 1.05),
    }
)


@dataclass(frozen=True, slots=True)
class Arm:
    identifier: str
    label: str
    profile: ModelProfile
    router: str
    micro_batch_size: int
    expected_trainable_parameters: int

    @property
    def has_sdm(self) -> bool:
        return bool(self.profile.memory_layers)


_ARM_ROWS = (
    Arm(
        "dense_attention",
        "Dense attention",
        ATTENTION_ONLY_PROFILE,
        "not_applicable",
        8,
        14_965_120,
    ),
    Arm(
        "b7a1_native_sdm",
        "7:1 native SDM",
        CANONICAL_PROFILE,
        "independent_read_write",
        1,
        15_029_660,
    ),
    Arm(
        "b8_native_sdm",
        "8-layer native SDM",
        SDM_ONLY_PROFILE,
        "independent_read_write",
        1,
        15_038_880,
    ),
    Arm(
        "b7a1_residualized_sdm",
        "7:1 residualized SDM",
        CANONICAL_PROFILE,
        "residualized_read_write",
        1,
        15_087_452,
    ),
    Arm(
        "b8_residualized_sdm",
        "8-layer residualized SDM",
        SDM_ONLY_PROFILE,
        "residualized_read_write",
        1,
        15_104_928,
    ),
)
ARMS: Mapping[str, Arm] = MappingProxyType({arm.identifier: arm for arm in _ARM_ROWS})
ARM_ORDER = tuple(arm.identifier for arm in _ARM_ROWS)
GEOMETRY = MemoryGeometry(
    "two_factor_c32_r8_w8",
    factors=2,
    codebook_size=32,
    reads=8,
    writes=8,
)


__all__ = [
    "ARMS",
    "ARM_ORDER",
    "CAMPAIGN_ID",
    "EXPECTED_COMMON_SDM_SHA256",
    "EXPECTED_GAP_CLOSURE_RANGES",
    "EXPECTED_INVENTORY_SHA256",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_NLL_RANGES",
    "EXPECTED_PAYLOAD_SHA256",
    "GEOMETRY",
    "PASSES",
    "TOTAL_STEPS",
    "WARMUP_STEPS",
    "Arm",
]
