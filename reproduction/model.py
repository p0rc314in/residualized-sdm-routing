"""Construct and inspect the released Lingua SDM model."""

from __future__ import annotations

import hashlib
from typing import Any, Iterator

import torch

from .config import MemoryGeometry, ModelProfile
from .factory import HigherDimensionalSDMFactory


def build_language_model(
    *,
    vocab_size: int,
    geometry: MemoryGeometry,
    profile: ModelProfile,
    telemetry: bool = False,
    memory_block_size: int = 256,
):
    """Return the released LMTransformer with accepted SDM layers injected."""

    if vocab_size <= 0:
        raise ValueError("vocabulary size must be positive")
    if memory_block_size < 16:
        raise ValueError("released WY memory blocks must contain at least 16 tokens")
    geometry.value_width(profile.width)

    from apps.main.transformer import LMTransformer, LMTransformerArgs
    from lingua.sparse_delta_memory import SparseDeltaMemoryArgs

    factory = HigherDimensionalSDMFactory(geometry)
    sdm_args = SparseDeltaMemoryArgs(
        dim=profile.width,
        num_heads=geometry.memory_heads,
        slots_per_head=geometry.logical_capacity,
        num_reads=geometry.reads,
        num_writes=geometry.writes,
        memory_block_size=memory_block_size,
        read_act="Softmax",
        write_act="Softmax",
        read_act_scale=1.0,
        write_act_scale=1.0,
        normalize_readings=True,
        norm_eps=1e-6,
        backprop_on_memory=True,
        output_gate=True,
        query_batchnorm=False,
        log_memory_access_stats=telemetry,
        log_memory_norms=False,
        key_weighted_decay=False,
        snapshot_quant="none",
    )
    args = LMTransformerArgs(
        dim=profile.width,
        n_layers=profile.layers,
        head_dim=profile.width // profile.attention_heads,
        n_heads=profile.attention_heads,
        n_kv_heads=profile.attention_heads,
        ffn_dim_multiplier=None,
        multiple_of=profile.feed_forward_multiple,
        norm_eps=profile.norm_epsilon,
        init_std_factor="disabled",
        max_seqlen=profile.maximum_sequence_length,
        attn_at=list(profile.attention_layers),
        swa_at=[],
        sdm_at=list(profile.memory_layers),
        sdm_args=sdm_args,
        sdm_factory=factory,
        vocab_size=vocab_size,
        weight_tying=False,
    )
    model = LMTransformer(args)
    model._m5_geometry = geometry
    model._m5_profile = profile
    model._m5_factory_contract = factory.contract()
    return model


def iter_sdm_mixers(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    for kind, layer in zip(model.layer_types, model.layers, strict=True):
        if kind == "sdm":
            yield layer.attention


def _depth_factor(model: torch.nn.Module, depth: int) -> float:
    from lingua.transformer import InitStdFactor

    return {
        InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
        InitStdFactor.GLOBAL_DEPTH: (2 * (len(model.layers) + 1)) ** 0.5,
        InitStdFactor.DIM_RATIO: model.dim / 4096,
        InitStdFactor.DISABLED: 1.0,
    }[model.init_std_factor]


def initialize_role_keyed(model: torch.nn.Module, seed: int) -> None:
    """Call released initializers with independent semantic-role RNG streams."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    base = seed * 100_000
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base + 100)
        model.reset_parameters()

    for depth, layer in enumerate(model.layers):
        factor = _depth_factor(model, depth)
        if layer._is_sdm:
            layer.attention.init_weights(
                model.init_base_std,
                factor,
                role_seed=base + 6_000 + depth,
                selected_prior_seed=base + 8_000 + depth,
            )
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base + 6_000 + depth)
                layer.attention.reset_parameters(model.init_base_std, factor)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base + 1_000 + depth)
            layer.feed_forward.reset_parameters(model.init_base_std, factor)
        layer.attention_norm.reset_parameters()
        layer.ffn_norm.reset_parameters()


def install_sdm_generation_caches(
    model: torch.nn.Module,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    """Install the cache surface consumed by released TransformerBlock.forward."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for mixer in iter_sdm_mixers(model):
        mixer._gen_cache = mixer.create_kv_cache(
            bsz=batch_size,
            seq_len=0,
            dtype=dtype,
            device=device,
        )


def clear_sdm_generation_caches(model: torch.nn.Module) -> None:
    for mixer in iter_sdm_mixers(model):
        if hasattr(mixer, "_gen_cache"):
            del mixer._gen_cache


def _parameter_digest(selected: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(selected):
        tensor = selected[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def named_common_parameters(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    """Yield parameters common to a matched native/residualized topology pair."""

    memory_layers = {
        index for index, kind in enumerate(model.layer_types) if kind == "sdm"
    }
    for name, parameter in model.named_parameters():
        if name.startswith("layers."):
            layer_index = int(name.split(".", 2)[1])
            if layer_index in memory_layers and ".attention." in name:
                mixer_name = name.split(".attention.", 1)[1]
                if mixer_name.startswith(
                    (
                        "Wq_read.",
                        "Wk_write.",
                        "query_bn.",
                        "key_bn.",
                        "memory",
                        "selected_row_prior.",
                    )
                ):
                    continue
        yield name, parameter


def common_parameter_sha256(model: torch.nn.Module) -> str:
    """Fingerprint every shape-common parameter that must match across p arms."""

    return _parameter_digest(dict(named_common_parameters(model)))


def parameter_accounting(model: torch.nn.Module) -> dict[str, Any]:
    geometry: MemoryGeometry = model._m5_geometry
    profile: ModelProfile = model._m5_profile
    mixers = list(iter_sdm_mixers(model))
    prior_ids = {
        id(parameter)
        for mixer in mixers
        for parameter in mixer.selected_row_prior.parameters()
    }
    mixer_ids = {id(parameter) for mixer in mixers for parameter in mixer.parameters()}
    routing_ids = {
        id(parameter)
        for mixer in mixers
        for projection in (mixer.Wq_read, mixer.Wk_write)
        for parameter in projection.parameters()
    }
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    input_embedding = sum(
        parameter.numel()
        for parameter in model.tok_embeddings.parameters()
        if parameter.requires_grad
    )
    output_projection = sum(
        parameter.numel()
        for parameter in model.output.parameters()
        if parameter.requires_grad
    )
    learned_prior = sum(
        parameter.numel() for parameter in trainable if id(parameter) in prior_ids
    )
    active = sum(
        parameter.numel() for parameter in trainable if id(parameter) not in prior_ids
    )
    recurrent_active = sum(
        parameter.numel()
        for parameter in trainable
        if id(parameter) in mixer_ids and id(parameter) not in prior_ids
    )
    routing = sum(
        parameter.numel() for parameter in trainable if id(parameter) in routing_ids
    )
    expected_prior = len(mixers) * geometry.prior_parameters_per_layer(profile.width)
    expected_routing = len(mixers) * geometry.routing_parameters_per_layer(
        profile.width
    )
    if learned_prior != expected_prior:
        raise RuntimeError(
            f"product prior has {learned_prior} parameters, expected {expected_prior}"
        )
    if routing != expected_routing:
        raise RuntimeError(
            f"routing projections have {routing} parameters, expected {expected_routing}"
        )
    if any(mixer.memory is not None for mixer in mixers):
        raise RuntimeError("a logical recurrent table was allocated")

    total = active + learned_prior
    logical_rows = len(mixers) * geometry.memory_heads * geometry.logical_capacity
    logical_values = logical_rows * geometry.value_width(profile.width)
    return {
        "active_parameters_excluding_product_prior": active,
        "fixed_shell_active_parameters": active - recurrent_active,
        "recurrent_mixer_active_parameters": recurrent_active,
        "routing_projection_parameters": routing,
        "learned_product_prior_parameters": learned_prior,
        "input_embedding_parameters": input_embedding,
        "output_projection_parameters": output_projection,
        "total_trainable_parameters": total,
        "adamw_moment_elements": 2 * total,
        "logical_rows_per_sequence": logical_rows,
        "logical_value_elements_per_sequence": logical_values,
        "logical_capacity_tensor_elements": 0,
        "retained_request_state": "materialized_tuple_rows",
        "product_prior_names": [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in prior_ids
        ],
    }


def validate_model_contract(model: torch.nn.Module) -> dict[str, Any]:
    from apps.main.transformer import LMTransformer
    from capacity_independent_sdm.extensions import AdditiveProductPrior
    from capacity_independent_sdm.production import (
        CapacityIndependentSparseDeltaMemory,
        SparseDeltaMemory,
    )
    from lingua.transformer import BaseTransformer, TransformerBlock

    geometry: MemoryGeometry = model._m5_geometry
    profile: ModelProfile = model._m5_profile
    mixers = list(iter_sdm_mixers(model))
    if not mixers:
        if not profile.allow_all_attention:
            raise RuntimeError("model contains no SDM layers")
        return {
            "schema_version": 1,
            "status": "passed",
            "geometry": geometry.as_dict(),
            "profile": profile.as_dict(),
            "model_type": f"{type(model).__module__}.{type(model).__name__}",
            "released_lm_forward_inherited": type(model).forward
            is LMTransformer.forward,
            "released_base_forward_inherited": BaseTransformer.forward.__module__
            == "lingua.transformer",
            "released_block_forward_inherited": TransformerBlock.forward.__module__
            == "lingua.transformer",
            "released_sdm_forward_inherited": "not_applicable_no_sdm_layers",
            "factory_contract": model._m5_factory_contract,
            "mixers": [],
        }
    rows = []
    for index, mixer in enumerate(mixers):
        if type(mixer) is not CapacityIndependentSparseDeltaMemory:
            raise RuntimeError("factory did not construct the accepted capacity layer")
        if type(mixer).forward is not SparseDeltaMemory.forward:
            raise RuntimeError("capacity layer replaced released SDM forward")
        if not isinstance(mixer.selected_row_prior, AdditiveProductPrior):
            raise RuntimeError("the reproduction requires additive product-key initial memory")
        contract = mixer.execution_storage_contract()
        if contract["logical_capacity_tensor_elements"] != 0:
            raise RuntimeError("capacity-axis runtime storage was allocated")
        rows.append({"mixer_index": index, **contract})
    return {
        "schema_version": 1,
        "status": "passed",
        "geometry": geometry.as_dict(),
        "model_type": f"{type(model).__module__}.{type(model).__name__}",
        "released_lm_forward_inherited": type(model).forward is LMTransformer.forward,
        "released_base_forward_inherited": BaseTransformer.forward.__module__
        == "lingua.transformer",
        "released_block_forward_inherited": TransformerBlock.forward.__module__
        == "lingua.transformer",
        "released_sdm_forward_inherited": True,
        "factory_contract": model._m5_factory_contract,
        "mixers": rows,
    }


__all__ = [
    "build_language_model",
    "clear_sdm_generation_caches",
    "common_parameter_sha256",
    "initialize_role_keyed",
    "install_sdm_generation_caches",
    "iter_sdm_mixers",
    "named_common_parameters",
    "parameter_accounting",
    "validate_model_contract",
]
